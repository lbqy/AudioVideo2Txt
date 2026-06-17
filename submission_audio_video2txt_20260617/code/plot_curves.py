"""Parse fairseq json training logs and plot loss / accuracy curves.

Usage:
    python plot_curves.py <train_log> <tag> <out_dir>
e.g.
    python plot_curves.py results/logs/train_v2t.log v2t results
"""
import json
import re
import sys
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LINE_RE = re.compile(r"\]\[(?P<tag>[\w_]+)\]\[INFO\] - (?P<body>\{.*\})\s*$")


def parse(log_path):
    """Return dict of lists for train_inner / valid / train aggregates."""
    train_steps, train_loss, train_acc, train_nll = [], [], [], []
    train_ctc = []
    valid_steps, valid_loss, valid_acc, valid_nll = [], [], [], []
    valid_ctc = []
    with open(log_path, "r", errors="ignore") as f:
        for line in f:
            m = LINE_RE.search(line.strip())
            if not m:
                continue
            tag = m.group("tag")
            try:
                d = json.loads(m.group("body"))
            except json.JSONDecodeError:
                continue
            if tag == "train_inner":
                train_steps.append(int(float(d["num_updates"])))
                train_loss.append(float(d["loss"]))
                train_acc.append(float(d["accuracy"]))
                train_nll.append(float(d["nll_loss"]))
                train_ctc.append(float(d["ctc_loss"]) if "ctc_loss" in d else None)
            elif tag == "valid":
                valid_steps.append(int(float(d["valid_num_updates"])))
                valid_loss.append(float(d["valid_loss"]))
                valid_acc.append(float(d["valid_accuracy"]))
                valid_nll.append(float(d["valid_nll_loss"]))
                valid_ctc.append(float(d["valid_ctc_loss"]) if "valid_ctc_loss" in d else None)
    return {
        "train_steps": train_steps, "train_loss": train_loss,
        "train_acc": train_acc, "train_nll": train_nll, "train_ctc": train_ctc,
        "valid_steps": valid_steps, "valid_loss": valid_loss,
        "valid_acc": valid_acc, "valid_nll": valid_nll, "valid_ctc": valid_ctc,
    }


def plot(metrics, tag, out_dir):
    plot_dir = os.path.join(out_dir, "plots")
    metric_dir = os.path.join(out_dir, "metrics")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(metric_dir, exist_ok=True)

    # ---- loss curve ----
    plt.figure(figsize=(7, 5))
    plt.plot(metrics["train_steps"], metrics["train_loss"], label="train loss", color="tab:blue", alpha=0.8)
    if metrics["valid_steps"]:
        plt.plot(metrics["valid_steps"], metrics["valid_loss"], "o-", label="valid loss", color="tab:red")
    plt.xlabel("updates"); plt.ylabel("loss (label-smoothed, base-2, per sentence)")
    plt.title(f"{tag}: training / validation loss")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{tag}_loss.png"), dpi=150)
    plt.close()

    # ---- accuracy curve ----
    plt.figure(figsize=(7, 5))
    plt.plot(metrics["train_steps"], metrics["train_acc"], label="train acc", color="tab:blue", alpha=0.8)
    if metrics["valid_steps"]:
        plt.plot(metrics["valid_steps"], metrics["valid_acc"], "o-", label="valid acc", color="tab:red")
    plt.xlabel("updates"); plt.ylabel("token accuracy (%)")
    plt.title(f"{tag}: training / validation accuracy")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{tag}_accuracy.png"), dpi=150)
    plt.close()

    # ---- CTC auxiliary loss curve, if present ----
    has_train_ctc = any(v is not None for v in metrics["train_ctc"])
    has_valid_ctc = any(v is not None for v in metrics["valid_ctc"])
    if has_train_ctc or has_valid_ctc:
        plt.figure(figsize=(7, 5))
        if has_train_ctc:
            train_pairs = [
                (step, value)
                for step, value in zip(metrics["train_steps"], metrics["train_ctc"])
                if value is not None
            ]
            plt.plot(
                [step for step, _ in train_pairs],
                [value for _, value in train_pairs],
                label="train CTC loss",
                color="tab:purple",
                alpha=0.8,
            )
        if has_valid_ctc:
            valid_pairs = [
                (step, value)
                for step, value in zip(metrics["valid_steps"], metrics["valid_ctc"])
                if value is not None
            ]
            plt.plot(
                [step for step, _ in valid_pairs],
                [value for _, value in valid_pairs],
                "o-",
                label="valid CTC loss",
                color="tab:orange",
            )
        plt.xlabel("updates"); plt.ylabel("CTC loss (base-2, per sentence)")
        plt.title(f"{tag}: auxiliary CTC loss")
        plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{tag}_ctc_loss.png"), dpi=150)
        plt.close()

    # ---- combined 2x1 ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(metrics["train_steps"], metrics["train_loss"], label="train", color="tab:blue", alpha=0.8)
    if metrics["valid_steps"]:
        axes[0].plot(metrics["valid_steps"], metrics["valid_loss"], "o-", label="valid", color="tab:red")
    axes[0].set_title(f"{tag} loss"); axes[0].set_xlabel("updates"); axes[0].set_ylabel("loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(metrics["train_steps"], metrics["train_acc"], label="train", color="tab:blue", alpha=0.8)
    if metrics["valid_steps"]:
        axes[1].plot(metrics["valid_steps"], metrics["valid_acc"], "o-", label="valid", color="tab:red")
    axes[1].set_title(f"{tag} accuracy (%)"); axes[1].set_xlabel("updates"); axes[1].set_ylabel("accuracy"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{tag}_curves.png"), dpi=150)
    plt.close()

    # ---- dump metrics + summary ----
    with open(os.path.join(metric_dir, f"{tag}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    summary = {}
    if metrics["train_steps"]:
        summary["final_train_loss"] = metrics["train_loss"][-1]
        summary["final_train_acc"] = metrics["train_acc"][-1]
        summary["final_train_nll"] = metrics["train_nll"][-1]
        if metrics["train_ctc"][-1] is not None:
            summary["final_train_ctc_loss"] = metrics["train_ctc"][-1]
        summary["last_update"] = metrics["train_steps"][-1]
    if metrics["valid_steps"]:
        best_i = max(range(len(metrics["valid_acc"])), key=lambda i: metrics["valid_acc"][i])
        summary["best_valid_acc"] = metrics["valid_acc"][best_i]
        summary["best_valid_loss_at_best_acc"] = metrics["valid_loss"][best_i]
        summary["best_valid_acc_update"] = metrics["valid_steps"][best_i]
        summary["final_valid_loss"] = metrics["valid_loss"][-1]
        summary["final_valid_acc"] = metrics["valid_acc"][-1]
        if metrics["valid_ctc"][-1] is not None:
            summary["final_valid_ctc_loss"] = metrics["valid_ctc"][-1]
    with open(os.path.join(metric_dir, f"{tag}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{tag}] summary:", json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    log_path, tag, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    metrics = parse(log_path)
    plot(metrics, tag, out_dir)
