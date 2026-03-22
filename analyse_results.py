import json, os
import numpy as np

results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

BATCH_SIZE = 2048  # same across all models/datasets


def get_metrics(model_path, max_seeds=5, best_metric="f1"):
    seed_dirs = sorted([d for d in os.listdir(model_path) if d.isdigit()])[:max_seeds]
    metrics = {k: [] for k in ["f1", "precision", "recall", "accuracy", "macro_f1", "auc", "ap",
                                "test_f1", "test_precision", "test_recall", "test_accuracy", "test_auc", "test_ap"]}
    for seed in seed_dirs:
        val_path = os.path.join(model_path, seed, "val", "stats.json")
        test_path = os.path.join(model_path, seed, "test", "stats.json")
        if not os.path.exists(val_path) or not os.path.exists(test_path):
            continue
        val_stats = [json.loads(l) for l in open(val_path)]
        best = max(val_stats, key=lambda x: x.get(best_metric, 0))
        best_epoch = best["epoch"]
        metrics["f1"].append(best.get("f1", 0))
        metrics["precision"].append(best.get("precision", 0))
        metrics["recall"].append(best.get("recall", 0))
        metrics["accuracy"].append(best.get("accuracy", 0))
        metrics["macro_f1"].append(best.get("macro-f1", 0))
        metrics["auc"].append(best.get("auc", 0))
        metrics["ap"].append(best.get("ap", 0))
        test_stats = [json.loads(l) for l in open(test_path)]
        t = next((s for s in test_stats if s["epoch"] == best_epoch), None)
        if t is None:
            t = max(test_stats, key=lambda x: x.get(best_metric, 0))
        metrics["test_f1"].append(t.get("f1", 0))
        metrics["test_precision"].append(t.get("precision", 0))
        metrics["test_recall"].append(t.get("recall", 0))
        metrics["test_accuracy"].append(t.get("accuracy", 0))
        metrics["test_auc"].append(t.get("auc", 0))
        metrics["test_ap"].append(t.get("ap", 0))
    return metrics


def get_throughput_latency(model_path, max_seeds=5):
    """
    Compute inference throughput (trans/s) and per-batch latency (ms).
    Methodology from FraudGT paper Fig 3:
      - Latency l = average per-batch inference time (time_iter from val stats)
      - Throughput = T / l, where T = batch_size
    We average time_iter across all val epochs and seeds.
    """
    seed_dirs = sorted([d for d in os.listdir(model_path) if d.isdigit()])[:max_seeds]
    all_time_iters = []
    for seed in seed_dirs:
        val_path = os.path.join(model_path, seed, "val", "stats.json")
        if not os.path.exists(val_path):
            continue
        val_stats = [json.loads(l) for l in open(val_path)]
        # Skip first few epochs (warm-up) if enough data
        if len(val_stats) > 10:
            val_stats = val_stats[5:]
        for s in val_stats:
            if s.get("time_iter", 0) > 0:
                all_time_iters.append(s["time_iter"])
    if not all_time_iters:
        return None, None, None, None
    latency_s = np.mean(all_time_iters)
    latency_std = np.std(all_time_iters)
    throughput = BATCH_SIZE / latency_s
    latency_ms = latency_s * 1000
    latency_ms_std = latency_std * 1000
    return throughput, latency_ms, latency_ms_std, len(all_time_iters)


def fmt(vals):
    if not vals:
        return "   -       "
    return f"{np.mean(vals):.4f} ±{np.std(vals):.4f}"


def analyse(prefix, best_metric="f1"):
    models = sorted([d for d in os.listdir(results_dir) if d.startswith(prefix)])

    print(f"\n{'Model':<30} {'F1 (Val)':<16} {'Prec (Val)':<16} {'Rec (Val)':<16} {'AUC (Val)':<16} {'AP (Val)':<16}  |  {'F1 (Test)':<16} {'Prec (Test)':<16} {'Rec (Test)':<16} {'AUC (Test)':<16} {'AP (Test)':<16}")
    print("-" * 215)

    for model in models:
        model_path = os.path.join(results_dir, model)
        m = get_metrics(model_path, best_metric=best_metric)
        name = model.replace("-gpu0", "").replace(prefix.rstrip("-"), "").lstrip("-")
        if m["f1"]:
            print(f"{name:<30} {fmt(m['f1']):<16} {fmt(m['precision']):<16} {fmt(m['recall']):<16} "
                  f"{fmt(m['auc']):<16} {fmt(m['ap']):<16}  |  {fmt(m['test_f1']):<16} {fmt(m['test_precision']):<16} "
                  f"{fmt(m['test_recall']):<16} {fmt(m['test_auc']):<16} {fmt(m['test_ap']):<16}")
        else:
            print(f"{name:<30} no results")


def analyse_throughput(prefix):
    """Print throughput and latency table for all models with a given prefix."""
    models = sorted([d for d in os.listdir(results_dir) if d.startswith(prefix)])
    results = []
    for model in models:
        model_path = os.path.join(results_dir, model)
        tp, lat_ms, lat_std, n = get_throughput_latency(model_path)
        name = model.replace("-gpu0", "").replace(prefix.rstrip("-"), "").lstrip("-")
        results.append((name, tp, lat_ms, lat_std))

    # Sort by throughput descending
    results.sort(key=lambda x: x[1] if x[1] else 0, reverse=True)

    print(f"\n{'Model':<30} {'Throughput (trans/s)':<25} {'Latency (ms/batch)':<25}")
    print("-" * 80)
    for name, tp, lat_ms, lat_std in results:
        if tp is not None:
            print(f"{name:<30} {tp:<25.1f} {lat_ms:.1f} ±{lat_std:.1f}")
        else:
            print(f"{name:<30} no results")


# =============================================================================
# ACCURACY RESULTS
# =============================================================================

print("=" * 185)
print("ELLIPTIC RESULTS (Val metrics @ best val F1 epoch, up to 5 seeds)")
print("=" * 185)
analyse("Elliptic-", best_metric="f1")

print()
print("=" * 185)
print("ETH RESULTS (Val metrics @ best val F1 epoch, up to 5 seeds)")
print("=" * 185)
analyse("ETH-", best_metric="f1")

print()
print("=" * 185)
print("DGRAPH RESULTS (Val metrics @ best val F1 epoch, up to 5 seeds)")
print("=" * 185)
analyse("DGraph-", best_metric="f1")

print()
print("=" * 185)
print("BITCOIN-M RESULTS (Val metrics @ best val F1 epoch, up to 5 seeds)")
print("=" * 185)
analyse("BitcoinM-", best_metric="f1")

print()
print("=" * 185)
print("ETHEREUM-P RESULTS (Val metrics @ best val F1 epoch, up to 5 seeds)")
print("=" * 185)
analyse("EthereumP-", best_metric="f1")

# =============================================================================
# THROUGHPUT AND LATENCY  (FraudGT paper Fig 3 methodology)
# Throughput = batch_size / mean_per_batch_inference_time  (trans/s)
# Latency    = mean_per_batch_inference_time               (ms/batch)
# =============================================================================

print()
print("=" * 80)
print("THROUGHPUT AND LATENCY — Elliptic")
print("=" * 80)
analyse_throughput("Elliptic-")

print()
print("=" * 80)
print("THROUGHPUT AND LATENCY — ETH")
print("=" * 80)
analyse_throughput("ETH-")

print()
print("=" * 80)
print("THROUGHPUT AND LATENCY — DGraph")
print("=" * 80)
analyse_throughput("DGraph-")

print()
print("=" * 80)
print("THROUGHPUT AND LATENCY — Bitcoin-M")
print("=" * 80)
analyse_throughput("BitcoinM-")

print()
print("=" * 80)
print("THROUGHPUT AND LATENCY — Ethereum-P")
print("=" * 80)
analyse_throughput("EthereumP-")
