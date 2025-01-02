from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from fastapi.staticfiles import StaticFiles
import json
import torch
from typing import Dict, List
import chess
from fastapi.responses import HTMLResponse, FileResponse
from fen_utils import tokenize_fen

app = FastAPI()

# Add CORS middleware to allow requests from the chess UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model and tokenizer initialization
MODEL_NAME = "amazingvince/chess-llama-full-2048"  # "amazingvince/chess-llama-decoder-2048"  # Replace with your preferred model

print(f"Loading model {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Create a pipeline for text generation
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


def convert_uci_to_san(fen, uci_move):
    # Create a board from the FEN string
    board = chess.Board(fen)

    # Parse the UCI move
    move = chess.Move.from_uci(uci_move)

    # Convert to SAN
    san = board.san(move)

    return san


def build_input_text(fen: str, moves: str) -> str:
    return f"<|start|> <|above_2000|> <|standard|> {tokenize_fen(fen)} <|sep|> {moves} {'<|turn|>' if moves else ''}"


def parse_move(decoded_output: str) -> Optional[str]:
    """Attempt to parse a UCI move from the model's raw output."""
    try:
        parts = decoded_output.split("<|turn|>")[0]
        parts = parts.strip().split(" ")

        return "".join(parts)
    except Exception:
        return None


def format_prompt(messages: List[Message]) -> str:
    """Format messages into a single prompt string."""
    try:
        # Get the user message content which contains the JSON data
        user_message = next(msg for msg in messages if msg.role == "user")
        chess_data = json.loads(user_message.content)

        # {chess_data['legal_moves']}
        # start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        # will need to change the history to the moves
        # prompt = build_input_text(, chess_data['history'])

        prompt = build_input_text(chess_data["FEN"], "")

        return prompt, chess_data["FEN"], chess_data["history"]
    except json.JSONDecoder as e:
        raise HTTPException(status_code=400, detail="Invalid JSON in message content")
    except KeyError as e:
        raise HTTPException(
            status_code=400, detail=f"Missing required chess data field: {str(e)}"
        )


@app.post("/generate")
async def generate(request: GenerateRequest):
    try:
        # Format the prompt from messages
        prompt, fen, history = format_prompt(request.messages)

        # Generate response with the local model
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

        # Convert the move to SAN notation
        san_move = convert_uci_to_san(fen, move)

        json_response = {"move": san_move, "reasoning": "Because the model said so!"}

        # Return the response in the expected format
        return {"content": json.dumps(json_response)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
