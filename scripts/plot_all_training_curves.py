#!/usr/bin/env python3
"""Plot training curves for all fairseq-style experiment logs.

The script scans results/logs and results/repro_logs, extracts JSON metrics from
train_inner / valid records, and writes per-experiment plots plus overview
figures. It intentionally keeps labels in ASCII so plots render consistently on
headless servers without CJK fonts.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LINE_RE = re.compile(r"\]\[(?P<tag>[\w_]+)\]\[INFO\] - (?P<body>\{.*\})\s*$")


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value):
    number = _to_float(value)
    return None if number is None else int(number)


def parse_log(path: Path) -> Dict:
    train_steps: List[int] = []
    train_loss: List[float] = []
    train_acc: List[float] = []
    train_nll: List[float] = []
    valid_steps: List[int] = []
    valid_loss: List[float] = []
    valid_acc: List[float] = []
    valid_nll: List[float] = []
    done_training = False

    with path.open("r", errors="ignore") as f:
        for line in f:
            if "done training" in line:
                done_training = True
            match = LINE_RE.search(line.strip())
            if not match:
                continue
            tag = match.group("tag")
            try:
                record = json.loads(match.group("body"))
            except json.JSONDecodeError:
                continue

            if tag == "train_inner":
                step = _to_int(record.get("num_updates"))
                loss = _to_float(record.get("loss"))
                acc = _to_float(record.get("accuracy"))
                nll = _to_float(record.get("nll_loss"))
                if step is None or loss is None or acc is None:
                    continue
                train_steps.append(step)
                train_loss.append(loss)
                train_acc.append(acc)
                train_nll.append(nll if nll is not None else float("nan"))
            elif tag == "valid":
                step = _to_int(record.get("valid_num_updates"))
                loss = _to_float(record.get("valid_loss"))
                acc = _to_float(record.get("valid_accuracy"))
                nll = _to_float(record.get("valid_nll_loss"))
                if step is None or loss is None or acc is None:
                    continue
                valid_steps.append(step)
                valid_loss.append(loss)
                valid_acc.append(acc)
                valid_nll.append(nll if nll is not None else float("nan"))

    tag = path.stem
    last_update = train_steps[-1] if train_steps else (valid_steps[-1] if valid_steps else 0)
    best_valid_acc = max(valid_acc) if valid_acc else None
    best_idx = valid_acc.index(best_valid_acc) if valid_acc else None

    return {
        "tag": tag,
        "log_path": str(path),
        "group": classify(tag),
        "done_training": done_training,
        "last_update": last_update,
        "num_train_points": len(train_steps),
        "num_valid_points": len(valid_steps),
        "train_steps": train_steps,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "train_nll": train_nll,
        "valid_steps": valid_steps,
        "valid_loss": valid_loss,
        "valid_acc": valid_acc,
        "valid_nll": valid_nll,
        "final_train_loss": train_loss[-1] if train_loss else None,
        "final_train_acc": train_acc[-1] if train_acc else None,
        "final_valid_loss": valid_loss[-1] if valid_loss else None,
        "final_valid_acc": valid_acc[-1] if valid_acc else None,
        "best_valid_acc": best_valid_acc,
        "best_valid_loss": valid_loss[best_idx] if best_idx is not None else None,
        "best_valid_update": valid_steps[best_idx] if best_idx is not None else None,
    }


def classify(tag: str) -> str:
    if tag.startswith("sweep_5_2"):
        return "5.2 sweep"
    if tag.startswith("default_5_1") or "enhanced_loss" in tag:
        return "5.1 loss"
    if tag.startswith("default_5_2") or "modality_dropout" in tag:
        return "5.2 modality dropout"
    if tag.startswith("default_5_3") or "feature_mask" in tag:
        return "5.3 feature mask"
    if tag in {"train_v2t", "train_av2t", "train_repro_v2t", "train_repro_av2t"}:
        return "4 baseline"
    return "other"


def discover_logs(root: Path) -> List[Path]:
    candidates = []
    for subdir in [root / "results" / "logs", root / "results" / "repro_logs"]:
        if subdir.exists():
            candidates.extend(sorted(subdir.glob("*.log")))
    return candidates


def plot_one(metrics: Dict, out_dir: Path) -> None:
    tag = metrics["tag"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].plot(
        metrics["train_steps"],
        metrics["train_loss"],
        label="train loss",
        color="tab:blue",
        alpha=0.85,
        linewidth=1.5,
    )
    if metrics["valid_steps"]:
        axes[0].plot(
            metrics["valid_steps"],
            metrics["valid_loss"],
            "o-",
            label="valid loss",
            color="tab:red",
            markersize=3,
        )
    axes[0].set_title(f"{tag} loss")
    axes[0].set_xlabel("updates")
    axes[0].set_ylabel("loss")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].plot(
        metrics["train_steps"],
        metrics["train_acc"],
        label="train acc",
        color="tab:green",
        alpha=0.85,
        linewidth=1.5,
    )
    if metrics["valid_steps"]:
        axes[1].plot(
            metrics["valid_steps"],
            metrics["valid_acc"],
            "o-",
            label="valid acc",
            color="tab:orange",
            markersize=3,
        )
    axes[1].set_title(f"{tag} accuracy")
    axes[1].set_xlabel("updates")
    axes[1].set_ylabel("accuracy (%)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.suptitle(f"{tag} ({metrics['group']})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / f"{tag}_curves.png", dpi=150)
    plt.close(fig)


def plot_overview(experiments: List[Dict], out_dir: Path) -> None:
    full = [m for m in experiments if m["valid_steps"] and m["last_update"] >= 1000]
    if not full:
        return

    fig, ax = plt.subplots(figsize=(16, 9))
    for metrics in full:
        line_width = 2.4 if metrics["group"] == "4 baseline" else 1.35
        alpha = 0.95 if metrics["group"] == "4 baseline" else 0.65
        ax.plot(
            metrics["valid_steps"],
            metrics["valid_acc"],
            marker="o",
            markersize=3,
            linewidth=line_width,
            alpha=alpha,
            label=metrics["tag"],
        )
    ax.set_title("All experiments: validation accuracy")
    ax.set_xlabel("updates")
    ax.set_ylabel("valid accuracy (%)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(out_dir / "overview_valid_accuracy_curves.png", dpi=150)
    plt.close(fig)

    sorted_exp = sorted(full, key=lambda m: (m["best_valid_acc"] is None, m["best_valid_acc"] or -1))
    fig_h = max(6, 0.38 * len(sorted_exp))
    fig, ax = plt.subplots(figsize=(12, fig_h))
    labels = [m["tag"] for m in sorted_exp]
    values = [m["best_valid_acc"] for m in sorted_exp]
    colors = [group_color(m["group"]) for m in sorted_exp]
    ax.barh(labels, values, color=colors)
    ax.set_title("Best validation accuracy by experiment")
    ax.set_xlabel("best valid accuracy (%)")
    ax.grid(axis="x", alpha=0.25)
    for idx, value in enumerate(values):
        ax.text(value + 0.05, idx, f"{value:.3f}", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "overview_best_valid_accuracy.png", dpi=150)
    plt.close(fig)

    sweep = [m for m in full if m["group"] == "5.2 sweep"]
    if sweep:
        fig, ax = plt.subplots(figsize=(12, 7))
        for metrics in sweep:
            ax.plot(
                metrics["valid_steps"],
                metrics["valid_acc"],
                marker="o",
                linewidth=1.6,
                markersize=3,
                label=metrics["tag"].replace("sweep_5_2_", ""),
            )
        ax.set_title("5.2 sweep: validation accuracy")
        ax.set_xlabel("updates")
        ax.set_ylabel("valid accuracy (%)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(out_dir / "overview_5_2_sweep_valid_accuracy.png", dpi=150)
        plt.close(fig)


def group_color(group: str) -> str:
    return {
        "4 baseline": "tab:blue",
        "5.1 loss": "tab:green",
        "5.2 modality dropout": "tab:orange",
        "5.2 sweep": "tab:purple",
        "5.3 feature mask": "tab:red",
    }.get(group, "tab:gray")


def write_summary(experiments: List[Dict], metric_dir: Path) -> None:
    metric_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for metrics in experiments:
        rows.append(
            {
                "tag": metrics["tag"],
                "group": metrics["group"],
                "log_path": metrics["log_path"],
                "done_training": metrics["done_training"],
                "last_update": metrics["last_update"],
                "num_train_points": metrics["num_train_points"],
                "num_valid_points": metrics["num_valid_points"],
                "final_train_loss": metrics["final_train_loss"],
                "final_train_acc": metrics["final_train_acc"],
                "final_valid_loss": metrics["final_valid_loss"],
                "final_valid_acc": metrics["final_valid_acc"],
                "best_valid_loss": metrics["best_valid_loss"],
                "best_valid_acc": metrics["best_valid_acc"],
                "best_valid_update": metrics["best_valid_update"],
            }
        )

    csv_path = metric_dir / "all_experiments_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = metric_dir / "all_experiments_summary.json"
    with json_path.open("w") as f:
        json.dump(rows, f, indent=2)


def filter_train_logs(logs: Iterable[Path], root: Path) -> List[Dict]:
    experiments = []
    for path in logs:
        metrics = parse_log(path)
        if metrics["num_train_points"] == 0 and metrics["num_valid_points"] == 0:
            continue
        try:
            metrics["log_path"] = str(path.relative_to(root))
        except ValueError:
            metrics["log_path"] = str(path)
        experiments.append(metrics)
    return sorted(experiments, key=lambda m: (m["group"], m["tag"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out-dir", type=Path, default=Path("results/plots/all_experiments"))
    parser.add_argument("--metric-dir", type=Path, default=Path("results/metrics"))
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = (root / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir
    metric_dir = (root / args.metric_dir).resolve() if not args.metric_dir.is_absolute() else args.metric_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = filter_train_logs(discover_logs(root), root)
    if not experiments:
        raise SystemExit("No training metrics found.")

    for metrics in experiments:
        plot_one(metrics, out_dir)
    plot_overview(experiments, out_dir)
    write_summary(experiments, metric_dir)

    print(f"Plotted {len(experiments)} training logs.")
    print(f"Plots: {out_dir}")
    print(f"Summary: {metric_dir / 'all_experiments_summary.csv'}")


if __name__ == "__main__":
    main()
