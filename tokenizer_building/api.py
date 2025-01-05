from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json
import torch
import chess
from fen_utils import tokenize_fen
import random

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
DEFAULT_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
MODEL_NAME = "amazingvince/chess-llama-pretrain-phase"

# Model initialization
print(f"Loading model {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=128,
    do_sample=True,
    pad_token_id=tokenizer.pad_token_id,
)


class Message(BaseModel):
    role: str
    content: str


class GenerateRequest(BaseModel):
    messages: List[Message]
    model: str
    temperature: float


def convert_uci_to_san(fen: str, uci_move: str) -> str:
    """Convert UCI move to SAN notation."""
    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(uci_move)
        return board.san(move)
    except (chess.InvalidMoveError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid move: {str(e)}")


def convert_san_to_uci(fen: str, san_moves: str) -> List[str]:
    """Convert SAN moves to UCI format."""
    try:
        board = chess.Board(fen)
        uci_moves = []

        # Handle empty moves
        if not san_moves or san_moves.isspace() or san_moves == "Opening position":
            return []

        san_moves_list = san_moves.split()

        for san_move in san_moves_list:
            try:
                move = board.parse_san(san_move)
                uci_moves.append(move.uci())
                board.push(move)
            except chess.InvalidMoveError:
                raise HTTPException(
                    status_code=400, detail=f"Invalid SAN move: {san_move}"
                )

        return uci_moves
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing moves: {str(e)}")


def build_input_text(fen: str, moves: str, from_fen: bool = False) -> str:
    """Build the input text for the model."""
    if from_fen:
        return f"<|start|> <|above_2000|> <|standard|> {tokenize_fen(fen)} <|sep|>"
    else:
        return f"<|start|> <|above_2000|> <|standard|> {tokenize_fen(DEFAULT_FEN)} <|sep|> {moves} {'<|turn|>' if moves else ''}"


def process_moves(moves: list) -> str:
    """Format moves list into the expected string format."""
    formatted_moves = []
    for move in moves:
        from_square = move[:2]
        to_square = move[2:4]
        promotion = f" {move[4].lower()}" if len(move) > 4 else ""
        formatted_moves.append(f"{from_square} {to_square}{promotion}")

    return " <|turn|> ".join(formatted_moves)


def parse_move(decoded_output: str) -> Optional[str]:
    """Parse UCI move from model output."""
    try:
        parts = decoded_output.split("<|turn|>")[0].strip().split()
        return "".join(parts)
    except Exception:
        return None


def format_prompt(messages: List[Message], from_fen=False) -> Tuple[str, str, str]:
    """Format messages into a prompt string."""
    try:
        user_message = next(msg for msg in messages if msg.role == "user")
        chess_data = json.loads(user_message.content)

        if "history" not in chess_data or "FEN" not in chess_data:
            raise KeyError("Missing required chess data fields")

        moves = process_moves(convert_san_to_uci(DEFAULT_FEN, chess_data["history"]))
        prompt = build_input_text(chess_data["FEN"], moves, from_fen=from_fen)

        return prompt, chess_data["FEN"], chess_data["history"]
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid JSON in message content: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error formatting prompt: {str(e)}"
        )


@app.post("/generate")
async def generate(request: GenerateRequest):
    """Generate a chess move."""
    fen = DEFAULT_FEN  # Initialize with default FEN

    try:
        # Format the prompt from messages
        from_fen = random.choice([True, False])
        prompt, fen, history = format_prompt(request.messages, from_fen=from_fen)

        # Generate response
        outputs = generator(
            prompt,
            temperature=request.temperature,
            max_new_tokens=50,
            do_sample=True,
            num_return_sequences=1,
        )

        # Extract and validate the generated response
        generated_text = outputs[0]["generated_text"][len(prompt) :]
        move = parse_move(generated_text)

        if not move:
            raise ValueError("Failed to parse move from model output")

        # Convert the move to SAN notation
        san_move = convert_uci_to_san(fen, move)

        return {
            "content": json.dumps(
                {
                    "move": san_move,
                    "reasoning": f"Model generated move. {'From fen' if from_fen else 'From history'}",
                }
            )
        }

    except Exception as e:
        # Fallback to random move
        try:
            print(f"Fallback to random move: {str(e)}")
            board = chess.Board(fen)
            legal_moves = list(board.legal_moves)
            if legal_moves:
                random_move = legal_moves[0].uci()
                san_move = convert_uci_to_san(fen, random_move)
                return {
                    "content": json.dumps(
                        {"move": san_move, "reasoning": "Fallback: random valid move"}
                    )
                }
        except Exception as fallback_error:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate move and fallback also failed: {str(fallback_error)}",
            )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8008)
