"""Generate transcriptions for submission.

Decodes every segment of a corpus with the trained model and writes a
{segment_name: transcription} dict as JSON, following the format required
for the final submission:

    "loquacious-dev-commonvoice/common_voice_en_533152/0":
        "SUDDENLY A HUGE QUANTITY OF GREEN SMOKE ..."

The key is built as  corpus_name / recording_name / segment_name.
No reference (<orth>) is needed, so this also works on the unlabelled
submission corpus. Model configuration is read back from the checkpoint;
only --num-heads must be passed to match training.
"""
import argparse
import glob
import json
import os
import re
import xml.etree.ElementTree as ET

import numpy as np
import torch

from model import Conformer
from dictionary import Dictionary
from wer import greedy_decode


def resolve_checkpoint(args):
    """Decide which checkpoint to load.

    Priority:
      1. --model-checkpoint <path>  (explicit override)
      2. --best-wer                 -> checkpoints/best_wer_checkpoint.pt
      3. default                    -> the checkpoint_epoch<N>.pt with the
                                       highest epoch number (our per-epoch saves)
    """
    if args.model_checkpoint:
        return args.model_checkpoint

    if args.best_wer:
        path = os.path.join(args.checkpoint_dir, "best_wer_checkpoint.pt")
        if not os.path.exists(path):
            raise SystemExit(f"--best-wer requested but {path} does not exist.")
        print(f"Using best-WER checkpoint: {path}")
        return path

    # highest-epoch checkpoint_epochN.pt (mid-epoch _stepN saves are ignored)
    if not os.path.isdir(args.checkpoint_dir):
        raise SystemExit(
            f"Checkpoint dir {args.checkpoint_dir!r} does not exist. "
            "Train first, or pass --model-checkpoint / --checkpoint-dir."
        )
    pattern = re.compile(r"^checkpoint_epoch(\d+)\.pt$")
    found = []
    for name in os.listdir(args.checkpoint_dir):
        match = pattern.match(name)
        if match:
            found.append((int(match.group(1)), os.path.join(args.checkpoint_dir, name)))
    if not found:
        raise SystemExit(
            f"No checkpoint_epoch<N>.pt found in {args.checkpoint_dir}. "
            "Pass --model-checkpoint or --best-wer."
        )
    epoch, path = max(found)
    print(f"Using last epoch checkpoint: {path} (epoch {epoch})")
    return path


def parse_corpus_segments(xml_path):
    """Return list of (segment_key, audio_path, start, end).

    segment_key = corpus_name/recording_name/segment_name.
    start/end are floats in seconds, or None if the recording has no
    <segment> children (then the whole file is one utterance).
    """
    root = ET.parse(xml_path).getroot()
    corpus_name = root.attrib.get("name", "")

    items = []
    for recording in root.findall(".//recording"):
        rec_name = recording.attrib.get("name")
        audio_path = recording.attrib.get("audio")
        segments = recording.findall(".//segment")

        if not segments:
            # No segment elements: treat the whole recording as one segment.
            key = f"{corpus_name}/{rec_name}/{rec_name}"
            items.append((key, audio_path, None, None))
            continue

        multi = len(segments) > 1
        for segment in segments:
            seg_name = segment.attrib.get("name")
            key = f"{corpus_name}/{rec_name}/{seg_name}"
            # Only slice when a recording actually holds several segments;
            # a lone segment spans the whole file, so we keep all frames
            # (features were computed per audio file, not per segment).
            if multi:
                start = float(segment.attrib["start"]) if "start" in segment.attrib else None
                end = float(segment.attrib["end"]) if "end" in segment.attrib else None
            else:
                start = end = None
            items.append((key, audio_path, start, end))

    return items


def parse_audio_folder(audio_dir, key_prefix, pattern="*.wav"):
    """Build items straight from a folder of audio files (no corpus XML).

    Each file is treated as a single segment with index 0, matching the
    slide example  <prefix>/<filename>/0 , e.g.
        loquacious-dev-commonvoice/common_voice_en_533152/0
    The audio_path IS the real file path, so features are read from disk here.
    start/end are None (whole file).
    """
    files = sorted(glob.glob(os.path.join(audio_dir, "**", pattern), recursive=True))
    items = []
    seen = {}
    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        parts = [p for p in (key_prefix, stem, "0") if p != ""]
        key = "/".join(parts)
        # Same filename in different subfolders would collide and silently drop
        # a segment; warn loudly instead of losing it unnoticed.
        if key in seen:
            print(f"WARNING: duplicate key {key!r} from {path} and {seen[key]} "
                  "- one transcription will be lost. Use unique filenames.")
        seen[key] = path
        items.append((key, path, None, None))
    return items


def build_model(state_dict, num_heads, device):
    """Reconstruct the Conformer from tensor shapes in the checkpoint."""
    d_model = state_dict["frontend.linear.weight"].shape[0]
    n_mels = state_dict["frontend.linear.weight"].shape[1] // 16
    vocab_size = state_dict["output_linear.weight"].shape[0]
    conv_channels = state_dict["blocks.0.convolutionalBlock.pointwise_conv1.weight"].shape[0] // 2

    num_blocks = 0
    while f"blocks.{num_blocks}.layernorm.weight" in state_dict:
        num_blocks += 1

    use_attention = any("attentionBlock" in key for key in state_dict)

    # Heads aren't inferable from most shapes, but q_norm pins them down if present.
    qk_norm_key = "blocks.0.attentionBlock.mha.q_norm.weight"
    if qk_norm_key in state_dict:
        num_heads = d_model // state_dict[qk_norm_key].shape[0]

    print(f"Model config from checkpoint: d_model={d_model}, num_blocks={num_blocks}, "
          f"conv_channels={conv_channels}, num_heads={num_heads}, n_mels={n_mels}, "
          f"vocab_size={vocab_size}, use_attention={use_attention}")

    model = Conformer(
        d_model=d_model,
        num_blocks=num_blocks,
        vocab_size=vocab_size,
        dropout=0.0,
        conv_channels=conv_channels,
        n_mels=n_mels,
        num_heads=num_heads,
        use_attention=use_attention,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def transcribe(model, items, features, dictionary, device):
    # features: dict {audio_path: array} for the XML flow, or None for the
    # audio-folder flow (then log-mel is computed from disk on the fly).
    transcriptions = {}
    total = len(items)
    skipped = 0

    if features is None:
        from mel_feature import logmel_from_audio

    use_amp = (device.type == "cuda")
    for i, (key, audio_path, start, end) in enumerate(items):
        if features is None:
            try:
                feats = logmel_from_audio(audio_path)
            except Exception as e:
                print(f"Could not read {audio_path} ({e}), skipping {key}.")
                transcriptions[key] = ""
                skipped += 1
                continue
        elif audio_path not in features:
            print(f"No features for {audio_path}, skipping {key}.")
            transcriptions[key] = ""   # keep the key so no segment is missing
            skipped += 1
            continue
        else:
            feats = features[audio_path]

        if start is not None and end is not None:
            # 10 ms hop -> 100 frames per second (see mel_feature.py)
            f0 = max(0, round(start * 100))
            f1 = min(feats.shape[0], round(end * 100))
            feats = feats[f0:f1]

        if feats.shape[0] < 4:   # frontend pools by 4; too short to decode
            transcriptions[key] = ""
            skipped += 1
            continue

        audio_tensor = torch.from_numpy(np.ascontiguousarray(feats)).float().unsqueeze(0).to(device)
        input_lengths = torch.tensor([audio_tensor.shape[1] // 4], dtype=torch.long, device=device)

        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(audio_tensor, input_lengths)
        logits = logits.float().squeeze(0)[:input_lengths.item()]
        transcriptions[key] = greedy_decode(logits, dictionary)

        if (i + 1) % 250 == 0:
            print(f"Decoded {i + 1}/{total} segments ({(i + 1) / total * 100:.1f}%)")

    if skipped:
        print(f"Warning: {skipped} segment(s) had no/too-short features -> empty transcription.")
    return transcriptions


def main():
    parser = argparse.ArgumentParser(description="Transcribe a corpus into a submission JSON dict")
    # --- input: either a corpus XML, or a plain folder of audio files ---
    parser.add_argument("--xml-corpus", type=str, default="dev_corpus.xml",
                        help="Corpus XML to transcribe (used if --audio-dir is not given)")
    parser.add_argument("--audio-dir", type=str, default=None,
                        help="Folder of audio files (audio-only mode, no XML). "
                             "Keys are built as <key-prefix>/<filename>/0")
    parser.add_argument("--key-prefix", type=str, default=None,
                        help="Prefix for the segment key in --audio-dir mode. "
                             "Defaults to the audio folder's name (i6 corpus convention). "
                             "VERIFY the printed key matches the submission format!")
    parser.add_argument("--audio-glob", type=str, default="*.wav",
                        help="Filename pattern in --audio-dir mode")
    # --- checkpoint selection ---
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--best-wer", action="store_true",
                        help="Use best_wer_checkpoint.pt instead of the last epoch checkpoint")
    parser.add_argument("--model-checkpoint", type=str, default=None,
                        help="Explicit checkpoint path (overrides the default selection)")
    # --- features (XML mode only) ---
    parser.add_argument("--feature-path", type=str, default="spect.npy",
                        help="Precomputed features (XML mode). Ignored in --audio-dir mode.")
    parser.add_argument("--num-heads", type=int, default=8, help="Must match training; only used if not inferable")
    parser.add_argument("--output", type=str, default="ResultsA2/transcriptions.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dictionary = Dictionary()
    dictionary.load("ResultsA2/dictionary.json")

    if args.audio_dir:
        # audio-only mode: keys from filenames, features computed on the fly.
        # Default prefix = the folder's own name (i6 corpus convention), so a
        # folder named "loquacious-dev-commonvoice" yields exactly that prefix.
        key_prefix = args.key_prefix
        if key_prefix is None:
            key_prefix = os.path.basename(os.path.normpath(args.audio_dir))
        items = parse_audio_folder(args.audio_dir, key_prefix, args.audio_glob)
        features = None
        print(f"Found {len(items)} audio file(s) in {args.audio_dir}")
        print(f"Key format: {items[0][0] if items else '<none>'}  "
              f"(prefix={key_prefix!r}; override with --key-prefix)")
    else:
        # XML mode: keys from the corpus, features from the precomputed .npy
        items = parse_corpus_segments(args.xml_corpus)
        features = np.load(args.feature_path, allow_pickle=True)[()]
        print(f"Parsed {len(items)} segment(s) from {args.xml_corpus}")

    checkpoint_path = resolve_checkpoint(args)
    # weights_only=False: best-checkpoints store dev_wer as a numpy scalar
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(checkpoint["model_state_dict"], args.num_heads, device)

    transcriptions = transcribe(model, items, features, dictionary, device)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(transcriptions, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {len(transcriptions)} transcriptions to {args.output}")


if __name__ == "__main__":
    main()
