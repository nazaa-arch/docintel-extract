"""Curated local benchmark evaluating DocIntel Extract across 15 synthetic multi-layout PDFs."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from docintel.config import Settings  # noqa: E402
from docintel.pipeline import DocumentIntelligencePipeline  # noqa: E402

SAMPLES_DIR = REPO_ROOT / "data" / "samples"
REPORT_PATH = REPO_ROOT / "eval" / "latest_report.json"


def _normalize_scalar(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return round(float(val), 2)
    return " ".join(str(val).strip().lower().split())


def _scalars_match(pred: Any, truth: Any) -> bool:
    if isinstance(pred, (int, float)) and isinstance(truth, (int, float)):
        return abs(float(pred) - float(truth)) <= 0.02
    return _normalize_scalar(pred) == _normalize_scalar(truth)


def _line_item_key(item: dict[str, Any]) -> tuple[str, float, float, float]:
    return (
        " ".join(str(item.get("description", "")).strip().lower().split()),
        round(float(item.get("quantity", 0.0)), 2),
        round(float(item.get("unit_price", 0.0)), 2),
        round(float(item.get("line_total", 0.0)), 2),
    )


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return ordered[int(k)]
    return ordered[f] * (c - k) + ordered[c] * (k - f)


def _safe_div(num: float, den: float) -> float:
    return num / den if den > 0 else 0.0


def _f1(precision: float, recall: float) -> float:
    return _safe_div(2.0 * precision * recall, precision + recall)


def run_benchmark(
    samples_dir: Path = SAMPLES_DIR,
    report_path: Path = REPORT_PATH,
    provider: str = "local",
) -> dict[str, Any]:
    """Execute the end-to-end pipeline on every sample PDF and compute benchmark metrics."""
    gt_path = samples_dir / "ground_truth.json"
    if not gt_path.is_file():
        raise FileNotFoundError(
            f"Ground truth not found at {gt_path}. Run `python scripts/generate_samples.py` first."
        )

    dataset: list[dict[str, Any]] = json.loads(gt_path.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory() as tmp_dir:
        base_settings = Settings.from_env()
        settings = base_settings.model_copy(
            update={
                "provider": provider,
                "db_path": Path(tmp_dir) / "eval_review.db",
                "upload_dir": Path(tmp_dir) / "uploads",
                "samples_dir": samples_dir,
                "review_confidence_threshold": 0.80,
            }
        )
        pipeline = DocumentIntelligencePipeline(settings)

        doc_type_correct = 0
        recon_correct = 0

        field_tp = 0
        field_fp = 0
        field_fn = 0

        item_tp = 0
        item_fp = 0
        item_fn = 0

        # HITL Triage: positive class = NEEDS_REVIEW (degraded/mismatched doc)
        hitl_tp = 0
        hitl_fp = 0
        hitl_fn = 0
        hitl_tn = 0

        latencies_ms: list[float] = []
        per_doc_details: list[dict[str, Any]] = []

        t_wall_start = time.perf_counter()

        for record in dataset:
            pdf_path = samples_dir / record["filename"]
            result = pipeline.process_file(pdf_path, doc_type="auto", persist=True)
            latencies_ms.append(result.latency_ms)

            gt = record["ground_truth"]
            pred_dict = result.data.model_dump()

            type_ok = result.doc_type == record["doc_type"]
            if type_ok:
                doc_type_correct += 1

            recon_ok = result.reconciliation_passed == record["expected_reconciliation_passed"]
            if recon_ok:
                recon_correct += 1

            # Scalar field precision / recall / F1
            doc_field_mismatches: list[dict[str, Any]] = []
            for key, truth_val in gt.items():
                if key == "line_items":
                    continue
                pred_val = pred_dict.get(key)
                has_truth = truth_val is not None
                has_pred = pred_val is not None

                if has_truth and has_pred:
                    if _scalars_match(pred_val, truth_val):
                        field_tp += 1
                    else:
                        field_fp += 1
                        field_fn += 1
                        doc_field_mismatches.append({"field": key, "expected": truth_val, "predicted": pred_val})
                elif has_truth and not has_pred:
                    field_fn += 1
                    doc_field_mismatches.append({"field": key, "expected": truth_val, "predicted": None})
                elif not has_truth and has_pred:
                    field_fp += 1
                    doc_field_mismatches.append({"field": key, "expected": None, "predicted": pred_val})

            # Line item exact-match F1
            gt_items = [_line_item_key(it) for it in gt.get("line_items", [])]
            pred_items = [_line_item_key(it) for it in pred_dict.get("line_items", [])]
            gt_pool = list(gt_items)
            matched_here = 0
            for p_item in pred_items:
                if p_item in gt_pool:
                    gt_pool.remove(p_item)
                    matched_here += 1
            item_tp += matched_here
            item_fp += len(pred_items) - matched_here
            item_fn += len(gt_items) - matched_here

            # HITL triage confusion matrix
            expected_flag = record["expected_status"] == "NEEDS_REVIEW"
            predicted_flag = result.status == "NEEDS_REVIEW"
            if expected_flag and predicted_flag:
                hitl_tp += 1
            elif not expected_flag and predicted_flag:
                hitl_fp += 1
            elif expected_flag and not predicted_flag:
                hitl_fn += 1
            else:
                hitl_tn += 1

            per_doc_details.append(
                {
                    "filename": record["filename"],
                    "expected_doc_type": record["doc_type"],
                    "predicted_doc_type": result.doc_type,
                    "expected_status": record["expected_status"],
                    "predicted_status": result.status,
                    "expected_reconciliation_passed": record["expected_reconciliation_passed"],
                    "reconciliation_passed": result.reconciliation_passed,
                    "overall_confidence": result.overall_confidence,
                    "latency_ms": result.latency_ms,
                    "line_items_matched": f"{matched_here}/{len(gt_items)}",
                    "field_mismatches": doc_field_mismatches,
                }
            )

        total_wall_sec = max(time.perf_counter() - t_wall_start, 1e-6)

    total_docs = len(dataset)
    doc_type_acc = _safe_div(doc_type_correct, total_docs)
    recon_acc = _safe_div(recon_correct, total_docs)

    field_prec = _safe_div(field_tp, field_tp + field_fp)
    field_rec = _safe_div(field_tp, field_tp + field_fn)
    field_f1 = _f1(field_prec, field_rec)

    item_prec = _safe_div(item_tp, item_tp + item_fp)
    item_rec = _safe_div(item_tp, item_tp + item_fn)
    item_f1 = _f1(item_prec, item_rec)

    hitl_prec = _safe_div(hitl_tp, hitl_tp + hitl_fp)
    hitl_rec = _safe_div(hitl_tp, hitl_tp + hitl_fn)
    hitl_f1 = _f1(hitl_prec, hitl_rec)

    throughput_dps = total_docs / total_wall_sec
    latency_p50 = _percentile(latencies_ms, 50.0)
    latency_p95 = _percentile(latencies_ms, 95.0)
    latency_mean = sum(latencies_ms) / total_docs if total_docs else 0.0

    report: dict[str, Any] = {
        "dataset_size": total_docs,
        "provider": provider,
        "metrics": {
            "doc_type_classification_accuracy": round(doc_type_acc, 4),
            "field_precision": round(field_prec, 4),
            "field_recall": round(field_rec, 4),
            "field_f1": round(field_f1, 4),
            "line_item_precision": round(item_prec, 4),
            "line_item_recall": round(item_rec, 4),
            "line_item_f1": round(item_f1, 4),
            "arithmetic_reconciliation_accuracy": round(recon_acc, 4),
            "hitl_triage_precision": round(hitl_prec, 4),
            "hitl_triage_recall": round(hitl_rec, 4),
            "hitl_triage_f1": round(hitl_f1, 4),
            "throughput_docs_per_sec": round(throughput_dps, 2),
            "latency_mean_ms": round(latency_mean, 2),
            "latency_p50_ms": round(latency_p50, 2),
            "latency_p95_ms": round(latency_p95, 2),
        },
        "counts": {
            "doc_type_correct": f"{doc_type_correct}/{total_docs}",
            "reconciliation_correct": f"{recon_correct}/{total_docs}",
            "field_tp": field_tp,
            "field_fp": field_fp,
            "field_fn": field_fn,
            "line_item_tp": item_tp,
            "line_item_fp": item_fp,
            "line_item_fn": item_fn,
            "hitl_confusion": {"tp": hitl_tp, "fp": hitl_fp, "fn": hitl_fn, "tn": hitl_tn},
        },
        "documents": per_doc_details,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def print_markdown_summary(report: dict[str, Any]) -> None:
    """Print a portfolio-ready Markdown benchmark table."""
    m = report["metrics"]
    print(f"\n## DocIntel Extract Curated Benchmark (15 Synthetic Multi-Layout PDFs, provider={report['provider']})\n")
    print("| Metric | Measured Result |")
    print("| :--- | ---: |")
    print(f"| Doc-Type Classification Accuracy | **{m['doc_type_classification_accuracy'] * 100:.2f}%** ({report['counts']['doc_type_correct']}) |")
    print(f"| Field-Level Precision / Recall / F1 | **{m['field_precision'] * 100:.2f}% / {m['field_recall'] * 100:.2f}% / {m['field_f1'] * 100:.2f}%** |")
    print(f"| Line-Item Extraction Precision / Recall / F1 | **{m['line_item_precision'] * 100:.2f}% / {m['line_item_recall'] * 100:.2f}% / {m['line_item_f1'] * 100:.2f}%** |")
    print(f"| Arithmetic Reconciliation Accuracy | **{m['arithmetic_reconciliation_accuracy'] * 100:.2f}%** ({report['counts']['reconciliation_correct']}) |")
    print(f"| HITL Triage Precision / Recall / F1 | **{m['hitl_triage_precision'] * 100:.2f}% / {m['hitl_triage_recall'] * 100:.2f}% / {m['hitl_triage_f1'] * 100:.2f}%** |")
    print(f"| Throughput | **{m['throughput_docs_per_sec']:.2f} docs/sec** |")
    print(f"| Latency (p50 / p95) | **{m['latency_p50_ms']:.2f} ms / {m['latency_p95_ms']:.2f} ms** |")
    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the 15-PDF DocIntel Extract benchmark suite.")
    parser.add_argument(
        "--provider",
        choices=["local", "ollama", "gemini", "openai"],
        default="local",
        help="Extraction backend to evaluate (default: local deterministic layout engine).",
    )
    args = parser.parse_args()
    result_report = run_benchmark(provider=args.provider)
    print_markdown_summary(result_report)
