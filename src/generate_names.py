import argparse
import torch
from pathlib import Path

from wavenet import WaveNetLM

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = Path("models/wavenet_indian_names.pt")
BLOCK_SIZE = 12  # equal to the training block size

# Main execution
def main(n: int, temperature: float):
    """
    Loads a pre-trained WaveNet model and generates a specified number of names.
    """
    if not MODEL_PATH.exists():
        print(f"Error: Model checkpoint not found at '{MODEL_PATH}'")
        print("Please run the training script first: python -m src.train_wavenet")
        return

    # Load model, stoi, and itos from checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]
    vocab_size = len(stoi)

    # Re-create the model with the same architecture as during training
    model = WaveNetLM(
        vocab_size=vocab_size,
        emb_dim=64,
        channels=128,
        kernel_size=3,
        dilations=[1, 2, 4, 8, 16, 32],
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    def decode(ids):
        return "".join(itos[i] for i in ids)

    def sample(max_tokens=50):
        # Start with the same context as in training: all padding ('.')
        ctx = torch.zeros(1, BLOCK_SIZE, dtype=torch.long, device=DEVICE)
        out = []
        with torch.no_grad():
            for _ in range(max_tokens):
                logits = model(ctx)[:, -1] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, 1)
                ctx = torch.cat([ctx[:, 1:], next_id], dim=1)  # slide window
                if next_id.item() == 0:  # End of string token
                    break
                out.append(next_id.item())
        return decode(out)

    print(f"\nGenerating {n} names with temperature {temperature}:\n")
    for i in range(n):
        print(f"  {i+1}. {sample()}")
    print()


if __name__ == "__main__":
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
    args = parser.parse_args()
    main(args.n, args.temperature)
