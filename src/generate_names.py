import argparse
import torch
from pathlib import Path

from .wavenet import WaveNetLM

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = Path("models/wavenet_indian_names.pt")
BLOCK_SIZE = 12  # equal to the training block size

# Main execution
def load_model():
    """Loads the WaveNet model and associated artifacts from the checkpoint."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found at '{MODEL_PATH}'")

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]
    vocab_size = len(stoi)

    model = WaveNetLM(
        vocab_size=vocab_size,
        emb_dim=64,
        channels=128,
        kernel_size=3,
        dilations=[1, 2, 4, 8, 16, 32],
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, stoi, itos

def generate_names(model, stoi, itos, n, temperature=1.0, start_char='.'):
    """Generates a list of names using the provided model."""
    
    def decode(ids):
        return "".join(itos[i] for i in ids)

    def sample(start_char, max_tokens=50):
        if temperature <= 0:
            raise ValueError("Temperature must be greater than 0")
        # Initialize context
        if start_char != '.':
            # Start with a specific letter
            start_id = stoi.get(start_char.lower())
            if start_id is None:
                # Fallback for characters not in vocab
                return f"'{start_char}' not in vocab"
            
            # Create a context with the starting character
            ctx = torch.full((1, BLOCK_SIZE), 0, dtype=torch.long, device=DEVICE)
            ctx[0, -1] = start_id
            out = [start_id]
        else:
            # Default behavior: start with padding tokens
            ctx = torch.zeros((1, BLOCK_SIZE), dtype=torch.long, device=DEVICE)
            out = []

        with torch.no_grad():
            for _ in range(max_tokens - len(out)):
                logits = model(ctx)[:, -1] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, 1)

                if next_id.item() == 0:  # End of string token
                    break
                
                out.append(next_id.item())
                ctx = torch.cat([ctx[:, 1:], next_id], dim=1)
        
        return decode(out)

    names = []
    for _ in range(n):
        names.append(sample(start_char))
    return names

def main_cli():
    """Command-line interface for generating names."""
    parser = argparse.ArgumentParser(
        description="Generate Indian names using a pre-trained WaveNet model."
    )
    parser.add_argument(
        "--n", type=int, default=20, help="Number of names to generate."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (e.g., >1.0 for more creative, <1.0 for more conservative names).",
    )
    parser.add_argument(
        "--start-char",
        type=str,
        default='.',
        help="A character to start the name with (e.g., 'A'). Defaults to random.",
    )
    args = parser.parse_args()

    try:
        model, stoi, itos = load_model()
        names = generate_names(model, stoi, itos, args.n, args.temperature, args.start_char)
        
        print(f"\nGenerating {args.n} names with temperature {args.temperature} starting with '{args.start_char}':\n")
        for i, name in enumerate(names):
            print(f"  {i+1}. {name}")
        print()

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the training script first: python src/train_wavenet.py")

if __name__ == "__main__":
    main_cli()
