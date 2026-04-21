#!/usr/bin/env python3
"""Train and evaluate a lightweight BERT acceptance classifier.

This script trains on PeerRead-style review JSON files (title/abstract/accepted)
and evaluates on review JSON files. It is optimized for modest AWS instances:
- default model: distilbert-base-uncased
- default max length: 256
- mixed precision enabled automatically on CUDA
"""

import numpy as np
import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, get_last_checkpoint


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _paper_id_from_filename(path: Path) -> str:
    name = path.name
    if name.endswith(".pdf.json"):
        return name[: -len(".pdf.json")]
    if name.endswith(".json"):
        return name[: -len(".json")]
    return path.stem


def _safe_text(x: Optional[str]) -> str:
    if not x:
        return ""
    return str(x).strip()


def _extract_abstract_from_sections(data: Dict) -> str:
    metadata = data.get("metadata", {})
    sections = metadata.get("sections", [])
    for sec in sections:
        heading = _safe_text(sec.get("heading")).lower()
        if "abstract" in heading:
            return _safe_text(sec.get("text"))
    return ""


def _extract_title_abstract_from_parsed_pdf(data: Dict) -> Tuple[str, str]:
    # PeerRead ScienceParse-like format
    metadata = data.get("metadata", {})
    title = _safe_text(metadata.get("title"))
    abstract = _safe_text(metadata.get("abstract"))
    if not abstract:
        abstract = _extract_abstract_from_sections(data)

    # Raw Docling format fallback
    if not title or not abstract:
        texts = data.get("texts", [])
        if isinstance(texts, list):
            if not title:
                for item in texts:
                    if _safe_text(item.get("label")).lower() == "title":
                        title = _safe_text(item.get("text"))
                        break
            if not abstract:
                abstract_chunks: List[str] = []
                in_abstract = False
                for item in texts:
                    label = _safe_text(item.get("label")).lower()
                    text = _safe_text(item.get("text"))
                    if not text:
                        continue
                    if label == "section_header" and "abstract" in text.lower():
                        in_abstract = True
                        continue
                    if in_abstract and label == "section_header":
                        break
                    if in_abstract and label in {"text", "list_item", "caption"}:
                        abstract_chunks.append(text)
                if abstract_chunks:
                    abstract = " ".join(abstract_chunks)

    return title, abstract


def _load_json(path: Path) -> Optional[Dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _extract_train_example(data: Dict) -> Optional[Tuple[str, int]]:
    title = _safe_text(data.get("title"))
    abstract = _safe_text(data.get("abstract"))
    accepted = data.get("accepted")
    if title and abstract and isinstance(accepted, bool):
        text = f"Title: {title}\n\nAbstract: {abstract}"
        return text, int(accepted)
    return None


def collect_peerread_examples(
    data_root: Path, split_keyword: Optional[str] = None
) -> Tuple[List[str], List[int], List[str]]:
    texts: List[str] = []
    labels: List[int] = []
    ids: List[str] = []
    for path in data_root.rglob("*.json"):
        parts = {p.lower() for p in path.parts}
        if split_keyword and split_keyword.lower() not in parts:
            continue
        if 'reviews' not in parts:
            continue
        data = _load_json(path)
        if not data:
            continue
        ex = _extract_train_example(data)
        if ex is None:
            continue
        text, label = ex
        texts.append(text)
        labels.append(label)
        ids.append(_paper_id_from_filename(path))
    return texts, labels, ids


def collect_parsed_pdf_examples(parsed_pdfs_dir: Path) -> Tuple[List[str], List[str]]:
    texts: List[str] = []
    ids: List[str] = []
    for path in sorted(parsed_pdfs_dir.glob("*.json")):
        data = _load_json(path)
        if not data:
            continue
        title, abstract = _extract_title_abstract_from_parsed_pdf(data)
        if not title and not abstract:
            continue
        text = f"Title: {title}\n\nAbstract: {abstract}"
        texts.append(text)
        ids.append(_paper_id_from_filename(path))
    return texts, ids


def collect_labels_for_ids(
    reviews_dir: Path, paper_ids: List[str]
) -> Dict[str, int]:
    label_map: Dict[str, int] = {}
    for pid in paper_ids:
        review_path = reviews_dir / f"{pid}.json"
        data = _load_json(review_path)
        if not data:
            continue
        accepted = data.get("accepted")
        if isinstance(accepted, bool):
            label_map[pid] = int(accepted)
    return label_map


class PaperDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[int]],
        tokenizer: AutoTokenizer,
        max_length: int,
    ):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        self.labels = labels

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "accuracy": accuracy_score(labels, preds),
        "false_positive_rate": fpr,
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }


@dataclass
class TrainBundle:
    train_texts: List[str]
    train_labels: List[int]
    dev_texts: List[str]
    dev_labels: List[int]

def get_balanced_dataset(texts: List[str], labels: List[int]) -> Tuple[List[str], List[int]]:
    idxs = list(range(len(texts)))
    accepted_idxs = [i for i in idxs if labels[i] == 1]
    rejected_idxs = [i for i in idxs if labels[i] == 0]
    random.shuffle(accepted_idxs)
    random.shuffle(rejected_idxs)
    n_samples = min(len(accepted_idxs), len(rejected_idxs))
    accepted_idxs = accepted_idxs[:n_samples]
    rejected_idxs = rejected_idxs[:n_samples]
    balanced_texts = [texts[i] for i in accepted_idxs + rejected_idxs]
    balanced_labels = [labels[i] for i in accepted_idxs + rejected_idxs]
    return balanced_texts, balanced_labels

def load_train_dev(peerread_root: Path, balanced) -> TrainBundle:
    train_texts, train_labels, _ = collect_peerread_examples(peerread_root, "train")
    dev_texts, dev_labels, _ = collect_peerread_examples(peerread_root, "test")

    # Fallback if split folders are absent: random split from all labeled examples.
    if not train_texts or not dev_texts:
        all_texts, all_labels, _ = collect_peerread_examples(peerread_root, None)
        if len(all_texts) < 20:
            raise RuntimeError(
                "Not enough labeled examples found under PeerRead/data/. "
                "Need JSONs with title, abstract, and accepted fields."
            )
        idxs = list(range(len(all_texts)))
        random.shuffle(idxs)
        cut = int(0.9 * len(idxs))
        train_idx, dev_idx = idxs[:cut], idxs[cut:]
        train_texts = [all_texts[i] for i in train_idx]
        train_labels = [all_labels[i] for i in train_idx]
        dev_texts = [all_texts[i] for i in dev_idx]
        dev_labels = [all_labels[i] for i in dev_idx]

    if balanced:
        train_texts, train_labels = get_balanced_dataset(train_texts, train_labels)
        dev_texts, dev_labels = get_balanced_dataset(dev_texts, dev_labels)

    return TrainBundle(
        train_texts=train_texts,
        train_labels=train_labels,
        dev_texts=dev_texts,
        dev_labels=dev_labels,
    )


def run_train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    peerread_root = Path(args.peerread_data).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_train_dev(peerread_root, args.balanced)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    )

    train_ds = PaperDataset(
        bundle.train_texts, bundle.train_labels, tokenizer, args.max_length
    )
    dev_ds = PaperDataset(bundle.dev_texts, bundle.dev_labels, tokenizer, args.max_length)
    print(f"Train Size: {len(bundle.train_labels)}")
    print(f"Dev Size: {len(bundle.dev_labels)}")
    
    use_fp16 = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size * 2),
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        save_total_limit=2,
        report_to="none",
        fp16=use_fp16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    last_checkpoint = get_last_checkpoint(output_dir) if os.path.isdir(output_dir) else None
    if last_checkpoint is not None:
        print(f"Resuming from checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("No checkpoint found, starting training from scratch.")
        trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metrics_path = output_dir / "dev_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved model/tokenizer to {output_dir}")
    print(f"Dev metrics saved to {metrics_path}")


def collect_review_examples_with_labels(
    reviews_dir: Path, balanced
) -> Tuple[List[str], List[int], List[str]]:
    texts: List[str] = []
    labels: List[int] = []
    ids: List[str] = []
    for path in sorted(reviews_dir.rglob("*.json")):
        parts = {p.lower() for p in path.parts}
        if "dev" not in parts:
            continue
        if 'reviews' not in parts:
            continue
        data = _load_json(path)
        if not data:
            continue
        ex = _extract_train_example(data)
        if ex is None:
            continue
        text, label = ex
        texts.append(text)
        labels.append(label)
        ids.append(_paper_id_from_filename(path))
    
    if balanced:
        texts, labels = get_balanced_dataset(texts, labels)
    return texts, labels, ids[:len(texts)]

def run_eval_batched(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    model_dir = Path(args.model_dir).resolve()
    eval_reviews_dir = Path(args.eval_reviews_dir).resolve()
    output_path = Path(args.predictions_out).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    texts, y_true, _ = collect_review_examples_with_labels(eval_reviews_dir, balanced=True)
    print(f"Total Accepts: {sum(y_true)} / {len(y_true)}")
    N = len(texts)
    idxs =  list(range(len(texts)))
        
    for k in range(args.repeat):
        random.seed(k)
        sampled_idxs = random.choices(idxs, k=N)
        sampled_texts = [texts[i] for i in sampled_idxs]
        sampled_y = [y_true[i] for i in sampled_idxs]

        dataset = PaperDataset(
            sampled_texts, labels=None, tokenizer=tokenizer, max_length=args.max_length
        )
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=str(model_dir / "tmp_eval"),
                per_device_eval_batch_size=max(1, args.batch_size * 2),
                report_to="none",
                fp16=torch.cuda.is_available(),
            ),
            tokenizer=tokenizer,
        )
        pred_out = trainer.predict(dataset)
        logits = pred_out.predictions
        pred_labels = np.argmax(logits, axis=-1)
        y_pred = [int(x) for x in pred_labels]
        tn, fp, fn, tp = confusion_matrix(sampled_y, y_pred, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        metrics = {
            "num_labeled_examples": len(sampled_y),
            "accuracy": accuracy_score(sampled_y, y_pred),
            "false_positive_rate": fpr,
            "precision": precision_score(sampled_y, y_pred, zero_division=0),
            "recall": recall_score(sampled_y, y_pred, zero_division=0),
            "f1": f1_score(sampled_y, y_pred, zero_division=0),
        }
        with output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")


def run_eval(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    model_dir = Path(args.model_dir).resolve()
    eval_reviews_dir = Path(args.eval_reviews_dir).resolve()
    output_path = Path(args.predictions_out).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    texts, y_true, ids = collect_review_examples_with_labels(eval_reviews_dir)
    if not texts:
        raise RuntimeError(
            f"No usable review JSON files found in {eval_reviews_dir}. "
            "Need title, abstract, and accepted fields."
        )
    print(f"Total Accepts: {sum(y_true)} / {len(y_true)}")

    dataset = PaperDataset(
        texts, labels=None, tokenizer=tokenizer, max_length=args.max_length
    )
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(model_dir / "tmp_eval"),
            per_device_eval_batch_size=max(1, args.batch_size * 2),
            report_to="none",
            fp16=torch.cuda.is_available(),
        ),
        tokenizer=tokenizer,
    )
    pred_out = trainer.predict(dataset)
    logits = pred_out.predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    pred_labels = np.argmax(logits, axis=-1)

    rows = []
    for i, pid in enumerate(ids):
        rows.append(
            {
                "paper_id": pid,
                "predicted_label": int(pred_labels[i]),
                "predicted_accepted": bool(int(pred_labels[i])),
                "prob_reject": float(probs[i][0]),
                "prob_accept": float(probs[i][1]),
            }
        )

    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(rows)} predictions to {output_path}")

    y_pred = [int(x) for x in pred_labels]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    metrics = {
        "num_labeled_examples": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "false_positive_rate": fpr,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    metrics_path = output_path.with_suffix(".metrics.json")
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved labeled-set metrics to {metrics_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train/evaluate a BERT-style acceptance classifier on PeerRead."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train model on PeerRead/data")
    train_p.add_argument("--peerread-data", default="PeerRead/data")
    train_p.add_argument("--output-dir", default="models/bert_acceptance")
    train_p.add_argument("--model-name", default="distilbert-base-uncased")
    train_p.add_argument("--max-length", type=int, default=256)
    train_p.add_argument("--batch-size", type=int, default=8)
    train_p.add_argument("--epochs", type=float, default=3.0)
    train_p.add_argument("--learning-rate", type=float, default=2e-5)
    train_p.add_argument("--seed", type=int, default=42)
    train_p.add_argument("--balanced", action="store_true", default=True)
    eval_p = sub.add_parser("eval", help="Run inference on review JSON files")
    eval_p.add_argument("--model-dir", default="models/bert_acceptance")
    eval_p.add_argument(
        "--eval-reviews-dir",
        default="output/neurips_2025_split/test/reviews",
        help="Directory containing review JSONs with title/abstract/accepted.",
    )
    eval_p.add_argument("--predictions-out", default="outputs/bert_predictions.jsonl")
    eval_p.add_argument("--max-length", type=int, default=256)
    eval_p.add_argument("--batch-size", type=int, default=8)
    eval_p.add_argument("--seed", type=int, default=42)
    eval_p.add_argument("--repeat", type=int, default=100)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train":
        run_train(args)
    elif args.command == "eval" and args.repeat > 0:
        run_eval_batched(args)
    elif args.command == "eval":
        run_eval(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()


'''
python3 bert_model.py train \
  --peerread-data PeerRead/data \
  --output-dir models/stat_sig/bert_peerread_2 \
  --seed=2

  python3 bert_model.py train \
  --peerread-data output/neurips_total_split \
  --output-dir models/bert_acceptance_neurips_5e_5 \
  --max-length=512 \
  --epochs=5.0

TODO: need to use "test" not "dev"
python3 bert_model.py eval \
  --model-dir models/bert_acceptance/checkpoint-2067 \
  --eval-reviews-dir output/neurips_total_split \
  --predictions-out results/bert_predictions_old_on_new.jsonl

TODO: need to use "dev" not "test"
  python3 bert_model.py eval \
  --model-dir models/bert_acceptance/checkpoint-2067 \
  --eval-reviews-dir PeerRead/data \
  --predictions-out results/bert_predictions_old_on_old.jsonl

TODO: need to use "dev" not "test"
python3 bert_model.py eval \
  --model-dir models/bert_acceptance_neurips_5e_5/checkpoint-390 \
  --eval-reviews-dir PeerRead/data \
  --predictions-out results/bert_predictions_new_on_old_tester.jsonl \
  --max-length=512

TODO: need to use "test" not "dev"
python3 bert_model.py eval \
  --model-dir models/bert_acceptance_neurips/checkpoint-234 \
  --eval-reviews-dir output/neurips_total_split  \
  --predictions-out results/bert_predictions_new_on_new.jsonl \
  --max-length=512


repetition eval metrics
python bert_model.py eval \
  --model-dir models/bert_acceptance/checkpoint-2067 \
  --eval-reviews-dir PeerRead/data \
  --predictions-out repetition_results/o_o_repeats.json

python bert_model.py eval \
  --model-dir models/bert_acceptance/checkpoint-2067 \
  --eval-reviews-dir output/neurips_total_split \
  --predictions-out repetition_results/o_n_repeats.json \
  
python bert_model.py eval \
  --model-dir models/bert_acceptance_neurips/checkpoint-234 \
  --eval-reviews-dir output/neurips_total_split  \
  --predictions-out repetition_results/n_n_repeats.jsonl \
  --max-length=512

python bert_model.py eval \
  --model-dir models/bert_acceptance_neurips/checkpoint-234 \
  --eval-reviews-dir PeerRead/data \
  --predictions-out repetition_results/n_o_repeats.json \
  --max-length=512
'''