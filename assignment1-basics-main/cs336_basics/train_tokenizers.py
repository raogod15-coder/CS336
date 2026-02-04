from __future__ import annotations

import argparse
import pickle
import json
from pathlib import Path

from cs336_basics.train_bpe import train_bpe


def _save_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def _save_vocab_json(vocab: dict[int, bytes], path: Path) -> None:
    # JSON-friendly: id -> utf-8 string with replacement for invalid bytes
    payload = {str(k): v.decode("utf-8", errors="replace") for k, v in vocab.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def _save_merges_json(merges: list[tuple[bytes, bytes]], path: Path) -> None:
    payload = [
        [
            a.decode("utf-8", errors="replace"),
            b.decode("utf-8", errors="replace"),
        ]
        for a, b in merges
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def train_and_save(input_path: Path, vocab_size: int, special_tokens: list[str], out_dir: Path, name: str) -> None:
    vocab, merges = train_bpe(
        input_path=str(input_path),
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
    _save_pickle(vocab, out_dir / f"{name}_vocab.pkl")
    _save_pickle(merges, out_dir / f"{name}_merges.pkl")
    _save_vocab_json(vocab, out_dir / f"{name}_vocab.json")
    _save_merges_json(merges, out_dir / f"{name}_merges.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BPE tokenizers for TinyStories and OpenWebText.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing dataset text files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/tokenizers"),
        help="Directory to write trained tokenizer artifacts.",
    )
    parser.add_argument(
        "--tinystories-vocab-size",
        type=int,
        default=10_000,
        help="Vocabulary size for TinyStories tokenizer.",
    )
    parser.add_argument(
        "--owt-vocab-size",
        type=int,
        default=32_000,
        help="Vocabulary size for OpenWebText tokenizer.",
    )
    parser.add_argument(
        "--special-token",
        type=str,
        default="<|endoftext|>",
        help="Special token to include in the vocabulary.",
    )
    parser.add_argument(
        "--only-tinystories",
        action="store_true",
        help="Train only the TinyStories tokenizer.",
    )
    args = parser.parse_args()

    tinystories_train = args.data_dir / "TinyStoriesV2-GPT4-train.txt"
    owt_train = args.data_dir / "owt_train.txt"

    if not tinystories_train.exists():
        raise FileNotFoundError(f"Missing TinyStories train file: {tinystories_train}")
    if not args.only_tinystories and not owt_train.exists():
        raise FileNotFoundError(f"Missing OWT train file: {owt_train}")

    special_tokens = [args.special_token]

    train_and_save(
        input_path=tinystories_train,
        vocab_size=args.tinystories_vocab_size,
        special_tokens=special_tokens,
        out_dir=args.out_dir,
        name="tinystories",
    )

    if not args.only_tinystories:
        train_and_save(
            input_path=owt_train,
            vocab_size=args.owt_vocab_size,
            special_tokens=special_tokens,
            out_dir=args.out_dir,
            name="owt",
        )


if __name__ == "__main__":
    main()
