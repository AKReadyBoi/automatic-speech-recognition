import torch
from model import Conformer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = Conformer(
        d_model=100,
        num_blocks=3,
        vocab_size=79,
        n_mels=80,
        dropout=0.0
    ).to(device)

    # sanity check forward pass
    x = torch.randn(10, 200, 80).to(device)  # (batch, time, n_mels)
    out = model(x)
    print(f"Output shape: {out.shape}")       # should be (10, 50, 79)

if __name__ == "__main__":
    main()