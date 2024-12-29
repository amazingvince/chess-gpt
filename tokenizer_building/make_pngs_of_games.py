import json
import chess
import chess.pgn
import datetime
import io
import os


def create_game_from_moves(moves, metadata, stockfish_level):
    """Create a chess.pgn.Game object from a list of moves and metadata."""
    game = chess.pgn.Game()
    if metadata["played_as_white"] and metadata["winner"] == "model":
        winner = "1-0"
    elif metadata["played_as_white"] and metadata["winner"] == "stockfish":
        winner = "0-1"
    elif not metadata["played_as_white"] and metadata["winner"] == "model":
        winner = "0-1"
    else:
        winner = "1-0"

    # Set game metadata
    game.headers["Event"] = f"AI Chess Game (Stockfish Level {stockfish_level})"
    game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")
    game.headers["White"] = "model" if metadata["played_as_white"] else "stockfish"
    game.headers["Black"] = "stockfish" if metadata["played_as_white"] else "model"
    game.headers["Result"] = winner
    game.headers["EndReason"] = metadata["reason"]
    game.headers["Duration"] = str(metadata["duration"])
    game.headers["FailedMovesLocal"] = str(metadata["failed_moves_local"])
    game.headers["FailedMovesStockfish"] = str(metadata["failed_moves_stockfish"])
    game.headers["StockfishLevel"] = str(stockfish_level)

    # Add moves to game
    node = game
    board = chess.Board()

    for move in moves:
        node = node.add_variation(chess.Move.from_uci(move))

    return game


def generate_game_pgns(json_data, output_dir="chess_games"):
    """Generate PGN files for each chess game in the JSON data."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create a single PGN file for each Stockfish level
    for level, games in json_data.items():
        stockfish_level = int(level)  # Convert string key to integer
        pgn_path = os.path.join(
            output_dir, f"games_stockfish_level_{stockfish_level}.pgn"
        )

        # Process each game in the level group
        with open(pgn_path, "w") as pgn_file:
            for i, game_data in enumerate(games):
                try:
                    # Create game object with stockfish level
                    game = create_game_from_moves(
                        game_data["moves"], game_data, stockfish_level
                    )

                    # Add game number
                    game.headers["GameNumber"] = str(i + 1)

                    # Write the game to the PGN file
                    print(game, file=pgn_file)
                    print("\n", file=pgn_file)  # Add blank line between games

                except Exception as e:
                    print(
                        f"Error processing game {i} at level {stockfish_level}: {str(e)}"
                    )
                    continue


def main():
    # Read the JSON data from file
    with open(
        "/home/vince/code/chess-gpt/tokenizer_building/results/chess_results.json", "r"
    ) as f:
        data = json.load(f)

    # Generate PGNs
    generate_game_pgns(data)
    print("PGN files have been generated!")


if __name__ == "__main__":
    main()
