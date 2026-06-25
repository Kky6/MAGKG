from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HF_HOME = PROJECT_ROOT / "cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Model training requires transformers. Install torch, transformers, numpy, and tqdm.") from exc


@dataclass
class SpanSample:
    sample_id: str
    text: str
    spans: list[dict[str, Any]]
    sample_weight: float = 1.0


class SciBERTSpanBoundary(nn.Module):
    """SciBERT encoder with independent start/end token boundary heads."""

    def __init__(self, model_name: str, dropout: float = 0.1, local_files_only: bool = False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.start_head = nn.Linear(hidden_size, 1)
        self.end_head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        return_hidden: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        dropped = self.dropout(hidden)
        start_logits = self.start_head(dropped).squeeze(-1)
        end_logits = self.end_head(dropped).squeeze(-1)
        if return_hidden:
            return start_logits, end_logits, hidden
        return start_logits, end_logits


class SpanBoundaryDataset(Dataset):
    def __init__(self, samples: Sequence[SpanSample], tokenizer: Any, max_len: int):
        self.samples = list(samples)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _span_to_token_labels(
        spans: Sequence[dict[str, Any]],
        offsets: Sequence[Sequence[int]],
        attention_mask: Sequence[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        start_labels = np.zeros(len(offsets), dtype=np.float32)
        end_labels = np.zeros(len(offsets), dtype=np.float32)
        valid_mask = np.zeros(len(offsets), dtype=np.float32)

        for idx, (start_char, end_char) in enumerate(offsets):
            if attention_mask[idx] == 1 and not (start_char == 0 and end_char == 0):
                valid_mask[idx] = 1.0

        for span in spans:
            span_start = int(span["start"])
            span_end = int(span["end"])
            token_ids: list[int] = []
            for idx, (token_start, token_end) in enumerate(offsets):
                if attention_mask[idx] == 0 or (token_start == 0 and token_end == 0):
                    continue
                if token_end > 0 and max(token_start, span_start) < min(token_end, span_end):
                    token_ids.append(idx)
            if token_ids:
                start_labels[token_ids[0]] = 1.0
                end_labels[token_ids[-1]] = 1.0

        return start_labels, end_labels, valid_mask

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        encoded = self.tokenizer(
            sample.text,
            return_offsets_mapping=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
        )
        start_labels, end_labels, valid_mask = self._span_to_token_labels(
            sample.spans,
            encoded["offset_mapping"],
            encoded["attention_mask"],
        )
        return {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "start_labels": torch.tensor(start_labels, dtype=torch.float32),
            "end_labels": torch.tensor(end_labels, dtype=torch.float32),
            "valid_mask": torch.tensor(valid_mask, dtype=torch.float32),
            "sample_weight": torch.tensor(sample.sample_weight, dtype=torch.float32),
            "sample_id": sample.sample_id,
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_spans(text: str, spans: Sequence[dict[str, Any]], max_span_chars: int | None = None) -> list[dict[str, Any]]:
    clean: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for span in spans:
        try:
            start = int(span["start"])
            end = int(span["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 <= start < end <= len(text)):
            continue
        if max_span_chars is not None and (end - start) > max_span_chars:
            continue
        if "text" in span and str(span["text"]) != text[start:end]:
            continue
        key = (start, end)
        if key in seen:
            continue
        seen.add(key)
        clean.append({"text": text[start:end], "start": start, "end": end})
    clean.sort(key=lambda item: (item["start"], item["end"]))
    return clean


def row_to_sample(row: dict[str, Any], index: int = 0) -> SpanSample:
    text = str(row.get("text") or row.get("sentence") or "")
    if not text:
        raise ValueError("Each boundary row must contain `text` or `sentence`.")
    sample_id = str(row.get("id") or row.get("sample_id") or f"sample_{index:06d}")
    spans = validate_spans(text, row.get("spans", []))
    return SpanSample(
        sample_id=sample_id,
        text=text,
        spans=spans,
        sample_weight=float(row.get("sample_weight", 1.0)),
    )


def load_samples(path: str | Path, limit: int | None = None) -> list[SpanSample]:
    rows = read_jsonl(path)
    if limit is not None and limit > 0:
        rows = rows[:limit]
    return [row_to_sample(row, index) for index, row in enumerate(rows)]



def load_tokenizer(model_name: str, local_files_only: bool = False) -> Any:
    return AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only, use_fast=True)


def build_model(model_name: str, dropout: float = 0.1, local_files_only: bool = False) -> SciBERTSpanBoundary:
    return SciBERTSpanBoundary(model_name=model_name, dropout=dropout, local_files_only=local_files_only)


def load_state_dict(path: str | Path, map_location: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    if isinstance(payload, dict):
        if "model_state_dict" in payload:
            return payload["model_state_dict"]
        if "state_dict" in payload:
            return payload["state_dict"]
    return payload


def load_checkpoint(
    model: SciBERTSpanBoundary,
    checkpoint_path: str | Path,
    *,
    strict: bool = True,
    map_location: torch.device | str = "cpu",
) -> None:
    state_dict = load_state_dict(checkpoint_path, map_location=map_location)
    model.load_state_dict(state_dict, strict=strict)


def masked_weighted_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    sample_weights: torch.Tensor,
    pos_weight: float,
) -> torch.Tensor:
    bce = nn.BCEWithLogitsLoss(
        reduction="none",
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=logits.device),
    )
    per_token = bce(logits, labels)
    denom = valid_mask.sum(dim=1).clamp(min=1.0)
    per_sample = (per_token * valid_mask).sum(dim=1) / denom
    return (per_sample * sample_weights).mean()


def compute_valid_mask(attention_mask: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    non_special = ~((offsets[..., 0] == 0) & (offsets[..., 1] == 0))
    return (attention_mask.bool() & non_special).to(dtype=torch.float32)


def _token_overlap(candidate: tuple[int, int], accepted: tuple[int, int]) -> bool:
    return not (candidate[1] < accepted[0] or accepted[1] < candidate[0])


def decode_span_candidates(
    text: str,
    offsets: Sequence[Sequence[int]],
    start_probs: Sequence[float],
    end_probs: Sequence[float],
    valid_mask: Sequence[float],
    *,
    start_thr: float,
    end_thr: float,
    decode_thr: float,
    max_span_len: int,
) -> list[dict[str, Any]]:
    start_indices = [idx for idx, value in enumerate(start_probs) if value >= start_thr and valid_mask[idx] > 0.5]
    end_indices = [idx for idx, value in enumerate(end_probs) if value >= end_thr and valid_mask[idx] > 0.5]
    candidates: list[dict[str, Any]] = []
    for start_index in start_indices:
        for end_index in end_indices:
            if end_index < start_index or (end_index - start_index + 1) > max_span_len:
                continue
            start_char = int(offsets[start_index][0])
            end_char = int(offsets[end_index][1])
            if end_char <= start_char:
                continue
            score = float(start_probs[start_index] * end_probs[end_index])
            if score < decode_thr:
                continue
            candidates.append(
                {
                    "text": text[start_char:end_char],
                    "start": start_char,
                    "end": end_char,
                    "token_start": int(start_index),
                    "token_end": int(end_index),
                    "start_prob": float(start_probs[start_index]),
                    "end_prob": float(end_probs[end_index]),
                    "score": score,
                }
            )
    candidates.sort(key=lambda item: (-item["score"], item["start"], item["end"]))
    accepted: list[dict[str, Any]] = []
    accepted_token_ranges: list[tuple[int, int]] = []
    for candidate in candidates:
        token_range = (candidate["token_start"], candidate["token_end"])
        if any(_token_overlap(token_range, existing) for existing in accepted_token_ranges):
            continue
        accepted.append(candidate)
        accepted_token_ranges.append(token_range)
    accepted.sort(key=lambda item: (item["start"], item["end"]))
    return accepted


def predict_rows(
    rows: Sequence[dict[str, Any]],
    model: SciBERTSpanBoundary,
    tokenizer: Any,
    device: torch.device,
    *,
    max_len: int,
    batch_size: int,
    start_thr: float,
    end_thr: float,
    decode_thr: float,
    max_span_len: int,
    return_token_probs: bool = False,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    model.eval()
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        texts = [str(row.get("text") or row.get("sentence") or "") for row in batch_rows]
        encoded = tokenizer(
            texts,
            return_offsets_mapping=True,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        offsets = encoded["offset_mapping"].to(device)
        valid_mask = compute_valid_mask(attention_mask, offsets)

        with torch.inference_mode():
            start_logits, end_logits = model(input_ids, attention_mask)

        start_probs = torch.sigmoid(start_logits).detach().cpu().numpy()
        end_probs = torch.sigmoid(end_logits).detach().cpu().numpy()
        offsets_cpu = encoded["offset_mapping"].tolist()
        valid_mask_cpu = valid_mask.detach().cpu().numpy()

        for index, row in enumerate(batch_rows):
            text = texts[index]
            spans = decode_span_candidates(
                text,
                offsets_cpu[index],
                start_probs[index],
                end_probs[index],
                valid_mask_cpu[index],
                start_thr=start_thr,
                end_thr=end_thr,
                decode_thr=decode_thr,
                max_span_len=max_span_len,
            )
            prediction = {
                "id": str(row.get("id") or row.get("sample_id") or f"row_{start + index:06d}"),
                "text": text,
                "pred_spans": spans,
            }
            if return_token_probs:
                prediction["start_probs"] = start_probs[index].tolist()
                prediction["end_probs"] = end_probs[index].tolist()
                prediction["valid_mask"] = valid_mask_cpu[index].tolist()
            predictions.append(prediction)
    return predictions

