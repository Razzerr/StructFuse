"""The keys the callbacks actually read must carry the cluster-balanced value.

`ModelCheckpoint`/`EarlyStopping` monitor `val/f1_long` (configs/callbacks/
model_checkpoint.yaml). Selecting the threshold on a cluster-balanced curve is
not enough if `_log_range_metrics` still writes the POOLED F1 under that key —
that was a real regression, caught only because these assertions inspect the
emitted keys rather than the helper in isolation.
"""
import sys
import types
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    import faiss  # noqa: F401
except ModuleNotFoundError:  # faiss is absent in the local env; unused here
    sys.modules["faiss"] = types.ModuleType("faiss")

from src.models.esm2_only_lit_module import ESM2OnlyLitModule  # noqa: E402
from src.models.template_only_lit_module import TemplateOnlyLitModule  # noqa: E402

MODULES = (ESM2OnlyLitModule, TemplateOnlyLitModule)
T = 20
THRESHOLDS = torch.linspace(0.05, 0.99, T)


def _fake(chain_rows, pooled, cls):
    """Build a stand-in `self` carrying only what _log_range_metrics touches."""
    logged = {}
    obj = types.SimpleNamespace(
        _val_thresholds=THRESHOLDS,
        log=lambda k, v, **kw: logged.__setitem__(k, float(v)),
        _val_chain_cids={}, _val_chain_tp={}, _val_chain_fp={}, _val_chain_fn={},
    )
    for rname, (cids, tp, fp, fn) in chain_rows.items():
        obj._val_chain_cids[rname] = cids
        obj._val_chain_tp[rname] = [np.asarray(tp, dtype=float)]
        obj._val_chain_fp[rname] = [np.asarray(fp, dtype=float)]
        obj._val_chain_fn[rname] = [np.asarray(fn, dtype=float)]
    dicts = {k: {} for k in ("tp", "fp", "fn", "tn")}
    # bind the real aggregation helper so the test exercises production code
    obj._val_cluster_macro_by_range = cls._val_cluster_macro_by_range.__get__(obj)
    for rname, (tp, fp, fn, tn) in pooled.items():
        dicts["tp"][rname] = torch.tensor(tp, dtype=torch.float)
        dicts["fp"][rname] = torch.tensor(fp, dtype=torch.float)
        dicts["fn"][rname] = torch.tensor(fn, dtype=torch.float)
        dicts["tn"][rname] = torch.tensor(tn, dtype=torch.float)
    return obj, logged, dicts


def _disagreeing_case():
    """One huge chain in its own cluster vs five small ones in five clusters.

    Threshold index 0 wins on pooled counts, index 1 wins per family.
    """
    big = ([1000.0] + [400.0] * (T - 1), [0.0] * T, [0.0] + [600.0] * (T - 1))
    small = ([1.0] + [10.0] * (T - 1), [9.0] + [0.0] * (T - 1), [9.0] + [0.0] * (T - 1))
    tp = [big[0]] + [small[0]] * 5
    fp = [big[1]] + [small[1]] * 5
    fn = [big[2]] + [small[2]] * 5
    cids = [1, 2, 3, 4, 5, 6]
    chain_rows = {"long": (cids, tp, fp, fn)}
    pooled = {"long": (
        np.sum(tp, axis=0), np.sum(fp, axis=0), np.sum(fn, axis=0), np.zeros(T),
    )}
    return chain_rows, pooled


def test_canonical_f1_long_is_cluster_macro_not_pooled():
    chain_rows, pooled = _disagreeing_case()
    for cls in MODULES:
        obj, logged, d = _fake(chain_rows, pooled, cls)
        cls._log_range_metrics(obj, "val", d["tp"], d["fp"], d["fn"], d["tn"], {})

        f1 = np.asarray(
            2 * np.asarray(pooled["long"][0])
            / (2 * np.asarray(pooled["long"][0])
               + np.asarray(pooled["long"][1]) + np.asarray(pooled["long"][2]) + 1e-8)
        )
        pooled_best = float(f1.max())

        assert "val/f1_long" in logged, cls.__name__
        assert "val/f1_long_micro" in logged, cls.__name__
        assert abs(logged["val/f1_long_micro"] - pooled_best) < 1e-6, cls.__name__
        # the canonical key must NOT be the pooled value in a case where they differ
        assert abs(logged["val/f1_long"] - pooled_best) > 1e-3, (
            cls.__name__, logged["val/f1_long"], pooled_best)


def test_canonical_and_micro_thresholds_differ_when_curves_disagree():
    chain_rows, pooled = _disagreeing_case()
    for cls in MODULES:
        obj, logged, d = _fake(chain_rows, pooled, cls)
        cls._log_range_metrics(obj, "val", d["tp"], d["fp"], d["fn"], d["tn"], {})
        assert logged["val/threshold_long"] != logged["val/threshold_long_micro"], (
            cls.__name__, logged["val/threshold_long"])
        assert abs(logged["val/threshold_long"] - THRESHOLDS[1].item()) < 1e-9
        assert abs(logged["val/threshold_long_micro"] - THRESHOLDS[0].item()) < 1e-9


def test_precision_and_recall_follow_the_same_estimand():
    chain_rows, pooled = _disagreeing_case()
    for cls in MODULES:
        obj, logged, d = _fake(chain_rows, pooled, cls)
        cls._log_range_metrics(obj, "val", d["tp"], d["fp"], d["fn"], d["tn"], {})
        for key in ("precision_long", "recall_long"):
            assert f"val/{key}" in logged, (cls.__name__, key)
            assert f"val/{key}_micro" in logged, (cls.__name__, key)


def test_cluster_balanced_marker_and_count_are_emitted():
    chain_rows, pooled = _disagreeing_case()
    for cls in MODULES:
        obj, logged, d = _fake(chain_rows, pooled, cls)
        cls._log_range_metrics(obj, "val", d["tp"], d["fp"], d["fn"], d["tn"], {})
        assert logged["val/cluster_balanced"] == 1.0, cls.__name__
        assert logged["val/n_clusters"] == 6.0, (cls.__name__, logged["val/n_clusters"])


def test_falls_back_to_pooled_and_flags_it_when_clusters_unknown():
    chain_rows, pooled = _disagreeing_case()
    chain_rows = {"long": ([-1] * 6, *chain_rows["long"][1:])}
    for cls in MODULES:
        obj, logged, d = _fake(chain_rows, pooled, cls)
        cls._log_range_metrics(obj, "val", d["tp"], d["fp"], d["fn"], d["tn"], {})
        assert logged["val/cluster_balanced"] == 0.0, cls.__name__
        # falling back means canonical == micro, and the marker says so
        assert abs(logged["val/f1_long"] - logged["val/f1_long_micro"]) < 1e-9, cls.__name__


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
