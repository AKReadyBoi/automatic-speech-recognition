import argparse
import os
import re

import torch
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import Conformer
from dictionary import Dictionary
from wer import parse_corpus, evaluate_model


def find_checkpoints(checkpoint_dir):
    # checkpoint_epoch<N>.pt sorted by epoch number
    pattern = re.compile(r"^checkpoint_epoch(\d+)\.pt$")
    found = []

    for name in os.listdir(checkpoint_dir):
        match = pattern.match(name)
        if match:
            found.append((int(match.group(1)), os.path.join(checkpoint_dir, name)))

    return sorted(found)


def average_state_dicts(paths):
    # Load one checkpoint at a time
    averaged = None

    for path in paths:
        # weights_only=False: some checkpoints store dev_wer as a numpy scalar
        state_dict = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]

        if averaged is None:
            averaged = {k: v.clone().float() if v.is_floating_point() else v.clone()
                        for k, v in state_dict.items()}
            continue

        # Refuse checkpoints from different model versions
        if state_dict.keys() != averaged.keys():
            raise ValueError(f"{path} does not have the same parameters as the first "
                             "checkpoint. Averaging requires the same model architecture.")

        for key, tensor in state_dict.items():
            if averaged[key].is_floating_point():
                averaged[key] += tensor.float()

    for key, tensor in averaged.items():
        if tensor.is_floating_point():
            tensor /= len(paths)

    return averaged


def build_model(state_dict, num_heads, device):
    # Collect config from tensor shapes.
    d_model = state_dict["frontend.linear.weight"].shape[0]
    n_mels = state_dict["frontend.linear.weight"].shape[1] // 16
    vocab_size = state_dict["output_linear.weight"].shape[0]
    conv_channels = state_dict["blocks.0.convolutionalBlock.pointwise_conv1.weight"].shape[0] // 2

    num_blocks = 0
    while f"blocks.{num_blocks}.layernorm.weight" in state_dict:
        num_blocks += 1

    # Recover number of heads.
    qk_norm_key = "blocks.0.attentionBlock.mha.q_norm.weight"
    if qk_norm_key in state_dict:
        num_heads = d_model // state_dict[qk_norm_key].shape[0]

    print(f"Config from checkpoint: d_model={d_model}, num_blocks={num_blocks}, "
          f"conv_channels={conv_channels}, num_heads={num_heads}, vocab_size={vocab_size}")

    model = Conformer(
        d_model=d_model,
        num_blocks=num_blocks,
        vocab_size=vocab_size,
        dropout=0.0,
        conv_channels=conv_channels,
        n_mels=n_mels,
        num_heads=num_heads,
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()
    return model


def write_report(rows, report_dir):
    # rows: list of (n, epochs, wer). n=1 is baseline.
    plot_path = os.path.join(report_dir, "ensemble_wer.png")
    ns = [n for n, _, _ in rows]
    wers = [wer * 100 for _, _, wer in rows]
    baseline = next((wer for n, _, wer in rows if n == 1), None)
    best_n, _, best_wer = min(rows, key=lambda r: r[2])

    plt.figure()
    if baseline is not None:
        plt.axhline(baseline * 100, color='#888888', linestyle='--', linewidth=1)
        plt.annotate("single checkpoint (n=1)", xy=(ns[0], baseline * 100),
                     xytext=(4, 5), textcoords="offset points",
                     ha="left", fontsize=9, color='#666666')
    plt.plot(ns, wers, marker='o', color='#1f77b4', linewidth=2)
    plt.annotate(f"best: n={best_n} ({best_wer:.2%})",
                 xy=(best_n, best_wer * 100), xytext=(0, -20),
                 textcoords="offset points", ha="center", fontsize=9)
    plt.margins(y=0.18)
    plt.xlabel("Number of averaged checkpoints (n)")
    plt.ylabel("Dev WER (%)")
    plt.title("Impact of Checkpoint Averaging on WER")
    plt.xticks(ns)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

    lines = [
        "# Checkpoint Averaging Report\n",
        "Weight averaging over the last n checkpoints of a single training run. "
        "The averaged weights form one regular model, so decoding cost is unchanged. "
        "n=1 is the last single checkpoint and serves as the baseline.\n",
        "| n averaged | Epochs | Dev WER |",
        "| :---: | :---: | :---: |",
    ]
    for n, epochs, wer in rows:
        span = f"{epochs[0]}-{epochs[-1]}" if len(epochs) > 1 else str(epochs[0])
        lines.append(f"| {n} | {span} | {wer:.2%} |")

    lines.append(f"\n![Checkpoint Averaging Impact](ensemble_wer.png)\n")
    if baseline is not None and best_n != 1:
        lines.append(f"Best result: n={best_n} at {best_wer:.2%} WER, "
                     f"an improvement of {(baseline - best_wer) * 100:.2f} points over the baseline.\n")

    report_path = os.path.join(report_dir, "ensemble_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nReport written to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Average the last n checkpoints and evaluate the WER")
    parser.add_argument("--n", type=int, default=5, help="How many of the last checkpoints to average")
    parser.add_argument("--sweep", type=int, nargs="+", default=None,
                        help="Test several values of n in one run and write a report, e.g. --sweep 1 2 4 6 8")
    parser.add_argument("--num-heads", type=int, default=4, help="Must match training")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--xml-corpus", type=str, default="dev_corpus.xml")
    parser.add_argument("--feature-path", type=str, default="spect.npy")
    parser.add_argument("--save", action="store_true", help="Save the averaged checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_checkpoints = find_checkpoints(args.checkpoint_dir)
    if not all_checkpoints:
        raise SystemExit(f"No checkpoints found in {args.checkpoint_dir}")
    print(f"Found {len(all_checkpoints)} checkpoints "
          f"(epochs {all_checkpoints[0][0]}-{all_checkpoints[-1][0]})")

    # Loaded once and reused for every n
    dictionary = Dictionary()
    dictionary.load("ResultsA2/dictionary.json")
    features = np.load(args.feature_path, allow_pickle=True)[()]
    corpus_data = parse_corpus(args.xml_corpus)

    rows = []
    for n in sorted(set(args.sweep)) if args.sweep else [args.n]:
        if n > len(all_checkpoints):
            print(f"Skipping n={n}: only {len(all_checkpoints)} checkpoints available")
            continue

        selected = all_checkpoints[-n:]
        epochs = [e for e, _ in selected]
        print(f"\nAveraging {n} checkpoint(s): epochs {epochs}")

        averaged = average_state_dicts([p for _, p in selected])

        if args.save:
            out_path = os.path.join(args.checkpoint_dir, f"averaged_last{n}.pt")
            torch.save({"model_state_dict": averaged}, out_path)
            print(f"Saved: {out_path}")

        model = build_model(averaged, args.num_heads, device)
        wer = evaluate_model(model, corpus_data, features, dictionary, device)
        print(f"n={n} | Dev WER: {wer:.2%}")
        rows.append((n, epochs, wer))

    if not rows:
        raise SystemExit("Nothing evaluated.")

    if args.sweep:
        os.makedirs("ResultsA2", exist_ok=True)
        write_report(rows, "ResultsA2")


if __name__ == "__main__":
    main()
