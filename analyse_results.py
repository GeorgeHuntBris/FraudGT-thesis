import json, os
import numpy as np

results_dir = os.path.expanduser("~/FraudGT-thesis/results")
models = sorted([d for d in os.listdir(results_dir) if d.startswith("Elliptic-")])

print(f"{'Model':<40} {'F1 (illicit)':<20} {'Macro-F1':<15} {'AUC':<10}")
print("-" * 88)

for model in models:
    model_path = os.path.join(results_dir, model)
    seed_dirs = [d for d in os.listdir(model_path) if d.isdigit()]
    seed_f1s, seed_macro_f1s, seed_aucs = [], [], []
    for seed in seed_dirs:
        val_path = os.path.join(model_path, seed, "val", "stats.json")
        test_path = os.path.join(model_path, seed, "test", "stats.json")
        if not os.path.exists(val_path) or not os.path.exists(test_path):
            continue
        val_stats = [json.loads(l) for l in open(val_path)]
        best_epoch = max(val_stats, key=lambda x: x.get("f1", 0))["epoch"]
        test_stats = [json.loads(l) for l in open(test_path)]
        test_at_best = next((s for s in test_stats if s["epoch"] == best_epoch), None)
        if test_at_best is None:
            test_at_best = max(test_stats, key=lambda x: x.get("f1", 0))
        seed_f1s.append(test_at_best.get("f1", 0))
        seed_macro_f1s.append(test_at_best.get("macro-f1", 0))
        seed_aucs.append(test_at_best.get("auc", 0))
    name = model.replace("-gpu0", "").replace("Elliptic-", "")
    if seed_f1s:
        print(f"{name:<40} {np.mean(seed_f1s):.4f} +/- {np.std(seed_f1s):.4f}  "
              f"{np.mean(seed_macro_f1s):.4f}          {np.mean(seed_aucs):.4f}")
    else:
        print(f"{name:<40} no results")
