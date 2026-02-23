import json, os
import numpy as np

results_dir = os.path.expanduser("~/FraudGT-thesis/results")
models = sorted([d for d in os.listdir(results_dir) if d.startswith("Elliptic-")])

print(f"{'Model':<35} {'Val F1':<20} {'Test F1':<20} {'Val Macro-F1':<15} {'Val AUC':<10}")
print("-" * 102)

for model in models:
    model_path = os.path.join(results_dir, model)
    seed_dirs = [d for d in os.listdir(model_path) if d.isdigit()]
    val_f1s, test_f1s, val_macro_f1s, val_aucs = [], [], [], []
    for seed in seed_dirs:
        val_path = os.path.join(model_path, seed, "val", "stats.json")
        test_path = os.path.join(model_path, seed, "test", "stats.json")
        if not os.path.exists(val_path) or not os.path.exists(test_path):
            continue
        val_stats = [json.loads(l) for l in open(val_path)]
        best = max(val_stats, key=lambda x: x.get("f1", 0))
        best_epoch = best["epoch"]
        val_f1s.append(best.get("f1", 0))
        val_macro_f1s.append(best.get("macro-f1", 0))
        val_aucs.append(best.get("auc", 0))
        test_stats = [json.loads(l) for l in open(test_path)]
        test_at_best = next((s for s in test_stats if s["epoch"] == best_epoch), None)
        if test_at_best is None:
            test_at_best = max(test_stats, key=lambda x: x.get("f1", 0))
        test_f1s.append(test_at_best.get("f1", 0))
    name = model.replace("-gpu0", "").replace("Elliptic-", "")
    if val_f1s:
        print(f"{name:<35} {np.mean(val_f1s):.4f} +/- {np.std(val_f1s):.4f}  "
              f"{np.mean(test_f1s):.4f} +/- {np.std(test_f1s):.4f}  "
              f"{np.mean(val_macro_f1s):.4f}          {np.mean(val_aucs):.4f}")
    else:
        print(f"{name:<35} no results")
