game = {
'Moves': ["c2c4", "c7c6", "g1f3", "d7d5", "g2g3", "g7g6", 
          "c4d5", "c6d5", "d2d4", "g8f6", "f1g2", "f8g7", 
          "b1c3", "e8g8", "f3e5", "b8c6", "e1g1", "e7e6", 
          "c1f4", "f6h5", "e5c6", "b7c6", "f4e3", "f7f5", 
          "d1d2", "f5f4", "e3f4", "h5f4", "g3f4", "f8f5", 
          "e2e3", "d8h4", "c3e2", "c8d7", "h2h3", "f5f7", 
          "g1h2", "g6g5", "f4g5", "f7f2", "f1f2", "h4f2", 
          "a1f1", "f2h4", "e2f4", "h4g5", "f1g1", "g5h4", 
          "d2a5", "e6e5", "a5c7", "h4d8", "g2d5", "g8h8", 
          "c7d8", "a8d8", "f4e6", "d7e6", "d5e6", "e5d4",
          "e3d4", "g7d4", "g1g2", "d4e5", "h2g1", "d8d1", 
          "g1f2", "d1d2", "f2f3", "d2g2", "f3g2", "e5b2", 
          "g2f3", "h8g7", "f3e4", "g7h6", "e6f5", "b2f6", 
          "e4d3", "c6c5", "d3c4", "f6d4", "f5e4", "h6g5", 
          "e4h7", "g5f4", "h7b1", "f4g5", "b1h7", "g5h4", 
          "h7d3", "h4h3", "d3f5", "h3g3", "c4d3", "g3g2", 
          "f5e4", "g2g1", "d3c4", "d4f6", "c4d3", "c5c4", 
          "d3c4", "f6h8", "c4d3", "g1f2", "e4c6", "h8f6", 
          "d3e4", "f2g3", "e4f5", "f6d8", "a2a4", "a7a6", 
          "c6d5", "g3h4", "f5g6", "d8b6", "d5b3", "b6c5", 
          "b3d1", "h4h3", "d1b3", "h3g2", "b3c2", "g2f1", 
          "c2b3", "f1e2", "b3c4", "e2e3", "c4d5", "e3f4", 
          "d5c6", "c5d4", "c6d5", "f4g4", "d5b7", "d4c5", 
          "b7d5", "c5d6", "d5e4", "g4h3", "g6h5", "d6b8", 
          "e4d5", "b8d6", "d5b7", "d6b8", "b7a6", "b8c7", 
          "a6b7", "h3g3", "a4a5", "c7a5", "b7d5", "a5b6", 
          "d5b7", "b6c7", "b7a8", "c7a5", "h5g6", "a5c7", 
          "a8c6", "c7b6", "c6g2", "b6c7", "g2f3", "g3f4", 
          "g6h5", "c7d8", "f3a8", "f4g3", "h5g6", "d8b6", 
          "g6f5", "b6d8", "a8b7", "d8b6", "b7f3", "b6c7", 
          "f5f6", "c7d8", "f6f5", "d8c7", "f5e6", "c7d8", 
          "f3g4", "g3h4", "e6d6", "d8a5", "d6d5", "h4g5", 
          "d5d6", "a5c7", "d6c6", "c7b6", "g4f5", "b6c7", 
          "f5g4", "g5f6", "c6d5", "c7d8", "g4e6", "f6e7", 
          "e6f5", "d8a5", "d5e5", "a5d8", "e5d5", "d8b6", 
          "f5h3", "b6c7", "h3f5", "c7b6", "f5h3", "e7f6", 
          "h3f5", "b6c7", "f5h3", "c7a5", "h3g4", "f6e7", 
          "g4f5", "a5d8", "f5g4", "e7f6", "g4e6", "d8b6", 
          "e6f5", "b6a5", "d5d6", "a5c7", "d6d5", "c7b6", 
          "f5b1", "f6f7", "b1h7", "b6c7", "h7g8", "f7f6", 
          "g8e6", "c7d8", "e6g4", "d8c7", "g4d7", "f6f7", 
          "d7g4", "c7d8", "g4f5", "f7e7", "f5e6", "d8c7", 
          "e6g4", "c7h2", "g4d1", "h2g1", "d1h5", "g1a7", 
          "h5f7", "a7f2", "f7h5", "f2e1", "h5d1", "e1d2", 
          "d1c2", "d2c3", "c2b1", "c3a1", "b1h7", "a1b2", 
          "h7f5", "b2a1", "f5e4", "a1b2", "e4b1", "b2d4", 
          "b1g6", "d4a1", "g6b1", "a1b2", "b1h7", "b2a3", 
          "h7b1", "a3b2", "b1d3", "b2d4", "d3e2", "e7f8", 
          "e2d1", "d4a1", "d1c2", "a1b2", "c2d1", "b2a1", 
          "d1c2", "a1b2", "c2b3", "b2a1", "b3a2", "a1b2", 
          "a2b1", "b2a1", "b1a2", "a1b2" ],
'Termination': 'SEVENTYFIVE_MOVES',
'Result': '1/2-1/2',
}

def make_output_string(game):
    
    moves = []
    for move in game['Moves']:
        m = move[:2]
        end_token = move[2:]
        m += " " + end_token[:2]
        if len(end_token) == 3:
            m += " " + end_token[2].upper()
        moves.append(m)
    turns =  " <|turn|> ".join(moves)
    return f"<|start|> {game['Result']} {turns} {game['Termination']} <|end|>"

out = make_output_string(game)    
print(out)

from transformers import AutoTokenizer

t = AutoTokenizer.from_pretrained("chess-llama")
print(len(t(out)['input_ids']))
print(t.decode(t(out)['input_ids'], skip_special_tokens=True))