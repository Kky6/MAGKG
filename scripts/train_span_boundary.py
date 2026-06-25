from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from magkg.span_boundary import (  # noqa: E402
    SpanBoundaryDataset,
    build_model,
    choose_device,
    load_checkpoint,
    load_samples,
    load_tokenizer,
    masked_weighted_bce_loss,
    set_seed,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or continue training the MAGKG span-boundary model.")
    parser.add_argument("--train", type=Path, required=True, help="Boundary training JSONL file.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", type=str, default="allenai/scibert_scivocab_uncased")
    parser.add_argument("--init-checkpoint", type=Path, help="Existing best_model.pt to continue training from.")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--pos-weight", type=float, default=8.0)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260416)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp-dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--freeze-encoder", action="store_true", help="Train only start/end heads after loading the encoder.")
    parser.add_argument("--checkpoint-name", type=str, default="model.pt")
    return parser.parse_args()


def json_safe_args(args: argparse.Namespace) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in vars(args).items():
        output[key] = str(value) if isinstance(value, Path) else value
    return output


def resolve_amp_context(device: torch.device, requested: str):
    if device.type != "cuda":
        return "fp32", nullcontext()
    if requested == "auto":
        requested = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if requested == "bf16":
        return "bf16", torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if requested == "fp16":
        return "fp16", torch.autocast(device_type="cuda", dtype=torch.float16)
    return "fp32", nullcontext()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    train_samples = load_samples(args.train, limit=args.limit_train if args.limit_train > 0 else None)
    if not train_samples:
        raise SystemExit("No training samples were loaded.")

    tokenizer = load_tokenizer(args.model_name, local_files_only=args.local_files_only)
    train_dataset = SpanBoundaryDataset(train_samples, tokenizer, max_len=args.max_len)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = build_model(args.model_name, dropout=args.dropout, local_files_only=args.local_files_only)
    if args.init_checkpoint:
        load_checkpoint(model, args.init_checkpoint, map_location=device)
    if args.freeze_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
    model.to(device)

    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=max(0, args.num_workers),
        pin_memory=(device.type == "cuda"),
    )
    total_steps = max(1, len(train_loader) * max(args.epochs, 1))
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)
    amp_dtype, _ = resolve_amp_context(device, args.amp_dtype)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and amp_dtype == "fp16"))

    for epoch_index in range(args.epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"epoch {epoch_index + 1}/{args.epochs}")
        for batch in progress:
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_labels = batch["start_labels"].to(device)
            end_labels = batch["end_labels"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            sample_weights = batch["sample_weight"].to(device)

            _, autocast_context = resolve_amp_context(device, args.amp_dtype)
            with autocast_context:
                start_logits, end_logits = model(input_ids, attention_mask)
                start_loss = masked_weighted_bce_loss(
                    start_logits,
                    start_labels,
                    valid_mask,
                    sample_weights,
                    pos_weight=args.pos_weight,
                )
                end_loss = masked_weighted_bce_loss(
                    end_logits,
                    end_labels,
                    valid_mask,
                    sample_weights,
                    pos_weight=args.pos_weight,
                )
                loss = 0.5 * (start_loss + end_loss)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

    checkpoint_path = args.output_dir / args.checkpoint_name
    torch.save(model.state_dict(), checkpoint_path)
    write_json(
        args.output_dir / "training_run.json",
        {
            "config": json_safe_args(args),
            "device": str(device),
            "amp_dtype": amp_dtype,
            "train_size": len(train_samples),
            "loaded_init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
            "checkpoint_path": str(checkpoint_path),
        },
    )
    print(json.dumps({"checkpoint_path": str(checkpoint_path)}, indent=2))


if __name__ == "__main__":
    main()

