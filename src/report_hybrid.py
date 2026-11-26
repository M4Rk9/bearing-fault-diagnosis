"""Generate plots, tables and evidence-based discussion from measured hybrid results."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

VARIANTS = ("Clean", "Noisy")
MODELS = ("hybrid", "baseline")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def curves(history, name, target):
    for metric in ("accuracy", "loss"):
        fig, ax = plt.subplots(figsize=(7, 4))
        for key, label in ((metric, "Training"), ("val_" + metric, "Validation")):
            values = history[key]
            ax.plot(range(1, len(values)+1), values, label=label)
        best = int(np.argmin(history["val_loss"])) + 1
        ax.axvline(best, linestyle="--", color="gray", label="Minimum validation loss")
        ax.set(xlabel="Epoch", ylabel=metric.capitalize(), title=f"{name}: {metric}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(target / f"{name}_{metric}.png", dpi=180)
        plt.close(fig)


def matrix_plot(cm, classes, title, path):
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    for i, j in np.ndindex(cm.shape):
        ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                color="white" if cm[i,j] > cm.max()/2 else "black")
    ax.set(xticks=range(len(classes)), yticks=range(len(classes)),
           xticklabels=classes, yticklabels=classes,
           xlabel="Predicted", ylabel="Actual", title=title)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def generate(source, target):
    results = json.loads(Path(source).read_text())
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    scores, by_class, discussions = [], [], []
    models = [m for m in MODELS if m in results]
    if "hybrid" not in models:
        raise ValueError("Missing measured hybrid results")
    for model in models:
        h = results["history" if model == "hybrid" else "baseline_history"]
        curves(h, model, target)
        best = int(np.argmin(h["val_loss"]))
        gap = h["accuracy"][best] - h["val_accuracy"][best]
        discussions.append(
            f"**{model}:** Minimum validation loss was {h['val_loss'][best]:.4f} "
            f"at epoch {best+1}. Training and validation accuracy at that epoch "
            f"were {h['accuracy'][best]:.2%} and {h['val_accuracy'][best]:.2%}; "
            f"the observed gap was {gap*100:+.2f} percentage points. "
            "Inspect both learning curves; a gap alone cannot establish overfitting "
            "because validation recordings also differ in load.")
        for variant in VARIANTS:
            data = results[model][variant]
            for level in ("windows", "recordings"):
                m = data[level]
                row = {"model": model, "condition": variant, "level": level,
                       "count": m["count"], "accuracy": m["accuracy"],
                       "macro_f1": m["macro_f1"], "loss": m["loss"]}
                scores.append(row)
                classes = data["class_order"]
                matrix_plot(m["confusion_matrix"], classes,
                            f"{model} / {variant} / {level}",
                            target / f"{model}_{variant.lower()}_{level}_confusion.png")
                for name in classes:
                    by_class.append({"model": model, "condition": variant, "level": level,
                                     "class": name, **m["classification_report"][name]})
            cm = np.array(data["windows"]["confusion_matrix"])
            np.fill_diagonal(cm, 0)
            if cm.max() > 0:
                i,j = np.unravel_index(cm.argmax(), cm.shape)
                discussions.append(f"{model} / {variant}: largest off-diagonal window error "
                                   f"was {data['class_order'][i]} classified as "
                                   f"{data['class_order'][j]} ({cm[i,j]} windows).")
        clean = results[model]["Clean"]["windows"]["accuracy"]
        noisy = results[model]["Noisy"]["windows"]["accuracy"]
        discussions.append(f"{model}: clean-to-noisy accuracy change was "
                           f"{(noisy-clean)*100:+.2f} percentage points.")
    write_csv(target / "metrics.csv", scores)
    write_csv(target / "per_class_metrics.csv", by_class)
    if "baseline" in models:
        for variant in VARIANTS:
            for level in ("windows", "recordings"):
                a = results["hybrid"][variant][level]
                b = results["baseline"][variant][level]
                discussions.append(f"{variant} {level}: hybrid minus baseline accuracy "
                                   f"{(a['accuracy']-b['accuracy'])*100:+.2f} percentage points; "
                                   f"macro-F1 change {a['macro_f1']-b['macro_f1']:+.4f}.")
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(VARIANTS))
        for i, model in enumerate(models):
            ax.bar(x+(i-.5)*.34,
                   [results[model][v]["windows"]["accuracy"]*100 for v in VARIANTS],
                   width=.34, label=model)
        ax.set(xticks=x, xticklabels=VARIANTS, ylabel="Held-out window accuracy (%)", ylim=(0,100))
        ax.legend()
        fig.tight_layout()
        fig.savefig(target / "model_comparison.png", dpi=180)
        plt.close(fig)
    lines = ["# Results and discussion", "",
             "All values below are derived from the saved experiment; no simulated scores.", "",
             "## Test performance", "",
             "| Model | Condition | Level | N | Accuracy | Macro-F1 | Log loss |",
             "|---|---|---|---:|---:|---:|---:|"]
    for s in scores:
        lines.append(f"| {s['model']} | {s['condition']} | {s['level']} | {s['count']} | "
                     f"{s['accuracy']:.2%} | {s['macro_f1']:.4f} | {s['loss']:.4f} |")
    lines += ["", "## Observations", ""] + ["- "+d for d in discussions]
    lines += ["", "## Limitations", "",
              "The test set is a held-out motor load from the same CWRU laboratory cohort. "
              "Overlapping windows and clean/noisy pairs are correlated; the number of "
              "independent recordings is much smaller than the window count. "
              "The experiment uses one seed and does not establish statistical significance "
              "or deployment accuracy. Hybrid-versus-baseline differences combine "
              "architecture and feature changes; an ablation study is required to isolate "
              "the contribution of Envelope or LSTM alone. Report the actual noise "
              "configuration only when independently verified. A small gap or high "
              "accuracy does not by itself rule out overfitting or underfitting.", ""]
    (target / "results_and_discussion.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    generate(a.results, a.output)


if __name__ == "__main__":
    main()
