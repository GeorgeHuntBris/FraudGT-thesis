import json, os
import numpy as np

results_dir = os.path.expanduser("~/FraudGT-thesis/results")

def get_metrics(model_path, max_seeds=5):
    seed_dirs = sorted([d for d in os.listdir(model_path) if d.isdigit()])[:max_seeds]
    metrics = {k: [] for k in ["f1", "precision", "recall", "accuracy", "macro_f1", "auc",
                                "test_f1", "test_precision", "test_recall", "test_accuracy", "test_auc"]}
    for seed in seed_dirs:
        val_path = os.path.join(model_path, seed, "val", "stats.json")
        test_path = os.path.join(model_path, seed, "test", "stats.json")
        if not os.path.exists(val_path) or not os.path.exists(test_path):
            continue
        val_stats = [json.loads(l) for l in open(val_path)]
        best = max(val_stats, key=lambda x: x.get("f1", 0))
        best_epoch = best["epoch"]
        metrics["f1"].append(best.get("f1", 0))
        metrics["precision"].append(best.get("precision", 0))
        metrics["recall"].append(best.get("recall", 0))
        metrics["accuracy"].append(best.get("accuracy", 0))
        metrics["macro_f1"].append(best.get("macro-f1", 0))
        metrics["auc"].append(best.get("auc", 0))
        test_stats = [json.loads(l) for l in open(test_path)]
        t = next((s for s in test_stats if s["epoch"] == best_epoch), None)
        if t is None:
            t = max(test_stats, key=lambda x: x.get("f1", 0))
        metrics["test_f1"].append(t.get("f1", 0))
        metrics["test_precision"].append(t.get("precision", 0))
        metrics["test_recall"].append(t.get("recall", 0))
        metrics["test_accuracy"].append(t.get("accuracy", 0))
        metrics["test_auc"].append(t.get("auc", 0))
    return metrics

def fmt(vals):
    if not vals:
        return "   -       "
    return f"{np.mean(vals):.4f} ±{np.std(vals):.4f}"

def analyse(prefix):
    models = sorted([d for d in os.listdir(results_dir) if d.startswith(prefix)])

    print(f"\n{'Model':<30} {'F1 (Val)':<16} {'Prec (Val)':<16} {'Rec (Val)':<16} {'Acc (Val)':<16} {'Macro-F1':<16} {'AUC (Val)':<16}  |  {'F1 (Test)':<16}")
    print("-" * 155)

    for model in models:
        model_path = os.path.join(results_dir, model)
        m = get_metrics(model_path)
        name = model.replace("-gpu0", "").replace(prefix.rstrip("-"), "").lstrip("-")
        if m["f1"]:
            print(f"{name:<30} {fmt(m['f1']):<16} {fmt(m['precision']):<16} {fmt(m['recall']):<16} "
                  f"{fmt(m['accuracy']):<16} {fmt(m['macro_f1']):<16} {fmt(m['auc']):<16}  |  {fmt(m['test_f1']):<16}")
        else:
            print(f"{name:<30} no results")

print("=" * 155)
print("ELLIPTIC RESULTS (Val metrics @ best val F1 epoch, up to 5 seeds)")
print("=" * 155)
analyse("Elliptic-")

print()
print("=" * 155)
print("ETH RESULTS (Val metrics @ best val F1 epoch, up to 5 seeds)")
print("=" * 155)
analyse("ETH-")
