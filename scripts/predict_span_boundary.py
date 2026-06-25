from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from magkg.span_boundary import (  # noqa: E402
    build_model,
    choose_device,
    load_checkpoint,
    load_tokenizer,
    predict_rows,
    read_jsonl,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MAGKG span-boundary inference with a trained checkpoint.")
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL with `text` or `sentence` fields.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint, e.g. best_model.pt.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", type=str, default="allenai/scibert_scivocab_uncased")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-span-len", type=int, default=24)
    parser.add_argument("--start-thr", type=float, default=0.08)
    parser.add_argument("--end-thr", type=float, default=0.08)
    parser.add_argument("--decode-thr", type=float, default=0.08)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--return-token-probs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]
    normalized_rows = []
    for index, row in enumerate(rows):
        text = row.get("text") or row.get("sentence")
        if not text:
            continue
        normalized_rows.append({"id": str(row.get("id") or f"row_{index:06d}"), "text": str(text)})

    device = choose_device(args.device)
    tokenizer = load_tokenizer(args.model_name, local_files_only=args.local_files_only)
    model = build_model(args.model_name, dropout=args.dropout, local_files_only=args.local_files_only)
    load_checkpoint(model, args.checkpoint, map_location=device)
    model.to(device)
    model.eval()

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    predictions = predict_rows(
        normalized_rows,
        model,
        tokenizer,
        device,
        max_len=args.max_len,
        batch_size=args.batch_size,
        start_thr=args.start_thr,
        end_thr=args.end_thr,
        decode_thr=args.decode_thr,
        max_span_len=args.max_span_len,
        return_token_probs=args.return_token_probs,
    )
    write_jsonl(args.output, predictions)
    print(json.dumps({"input_rows": len(normalized_rows), "predictions": len(predictions), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
