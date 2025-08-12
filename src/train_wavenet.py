import random
import math
import torch
import torch.nn.functional as F
from pathlib import Path

from .wavenet import WaveNetLM  # noqa: F401

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = Path("data/processed/indian_first_names_cleaned.csv")  # one name per line
SAVE_DIR  = Path("models")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ------------- 1.  Load & tokenize names ----------------
with open(DATA_PATH, encoding="utf8") as f:
    words = [w.strip().lower() for w in f if w.strip()]

chars = sorted(set("".join(words)))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}

def encode(s: str):
    return [stoi[c] for c in s + "."]

def decode(ids):
    return "".join(itos[i] for i in ids)

# ------------- 2.  Dataset preparation ------------------
BLOCK_SIZE = 12   # receptive field window
def build_dataset(names):
    X, Y = [], []
    for w in names:
        ids = encode(w)
        # Left pad for initial context (if shorter than BLOCK_SIZE)
        ids = [0] * (BLOCK_SIZE - 1) + ids
        for i in range(len(ids) - BLOCK_SIZE):
            context = ids[i : i + BLOCK_SIZE]        # always length BLOCK_SIZE
            target = ids[i + BLOCK_SIZE]             # predict next char
            X.append(context)
            Y.append(target)
    X = torch.tensor(X, dtype=torch.long)
    Y = torch.tensor(Y, dtype=torch.long)
    return X, Y

random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words)); n2 = int(0.9 * len(words))
train_set = build_dataset(words[:n1])
val_set   = build_dataset(words[n1:n2])

# ------------- 3.  Model -------------------------------
model = WaveNetLM(
    vocab_size=len(stoi),
    emb_dim=64,
    channels=128,
    kernel_size=3,
    dilations=[1, 2, 4, 8, 16, 32],
).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)

# ------------- 4.  Training loop -----------------------
BATCH_SIZE = 256
STEPS = 5_000
PRINT_EVERY = 500


def get_batch(dataset):
    X, Y = dataset
    ix = torch.randint(0, X.size(0), (BATCH_SIZE,))
    return X[ix].to(DEVICE), Y[ix].to(DEVICE)


best_val_loss = float('inf')
patience = 1500     # Number of steps to wait for improvement
wait = 0            # Counter for early stopping
best_model_path = SAVE_DIR / "wavenet_indian_names.pt"


for step in range(1, STEPS + 1):
    xb, yb = get_batch(train_set)
    logits = model(xb)
    loss = F.cross_entropy(logits[:, -1, :], yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % PRINT_EVERY == 0 or step == 1:
        with torch.no_grad():
            val_x, val_y = get_batch(val_set)
            val_logits = model(val_x)
            val_loss = F.cross_entropy(val_logits[:, -1, :], val_y)

        print(
            f"step {step:>6} | train loss {loss.item():.3f} | val loss {val_loss.item():.3f}"
        )

        # Early stopping logic
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            wait = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "stoi": stoi,
                    "itos": itos,
                },
                best_model_path,
            )
            print(f"  ↳ New best val loss. Model saved.")
        else:
            wait += PRINT_EVERY

        if wait >= patience:
            print(f"\nEarly stopping triggered after {step} steps (no val loss improvement for {patience} steps).")
            break

# ------------- 5.  Save checkpoint ---------------------

# The best model is saved during validation loss improvements above.
# Avoid overwriting that checkpoint unless none was created.
if not best_model_path.exists():
    torch.save(
        {
            "model_state": model.state_dict(),
            "stoi": stoi,
            "itos": itos,
        },
        best_model_path,
    )



# ------------- 6.  Sampling ----------------------------
def sample(model, block_size=12, max_tokens=50, temperature=1.0):
    if temperature <= 0:
        raise ValueError("Temperature must be greater than 0")
    model.eval()
    # Start with the same context as in training: all padding ('.')
    ctx = torch.zeros(1, block_size, dtype=torch.long, device=DEVICE)
    out = []
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(ctx)[:, -1] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            ctx = torch.cat([ctx[:, 1:], next_id], dim=1)  # slide window
            if next_id.item() == 0:
                break
            out.append(next_id.item())
    model.train()
    return decode(out)

print("\nSome samples:")
for _ in range(20):
    print("•", sample(model, 40))
