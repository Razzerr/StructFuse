"""cost_benchmark.py: the accounting, not the numbers.

The timing itself needs a GPU, the FAISS index and the NPZs, so it is exercised
first by a 3-chain smoke on the server. What CAN be pinned here is the part that
would silently produce a wrong cost table: the additive decomposition (and the
fact that the explanatory probes are excluded from it), the summary statistic,
and the deterministic frozen chain set.
"""

import importlib.util
import json
import math
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "cost_benchmark.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))       # main() imports src.data.components.dataset

# omegaconf is NEVER stubbed: a stub made `OmegaConf.create` raise, the main()
# test swallowed it and reported a pass while main() never ran. If omegaconf is
# missing the import fails loudly, which is the correct outcome.
from omegaconf import OmegaConf  # noqa: E402  (must be the real one)

# Hydra itself is NOT stubbed either. A hand-rolled stub is a second
# implementation that can disagree with production in shape — it already did,
# exposing `hydra_config.get` where the script calls
# `hydra_config.HydraConfig.get()`. Using the real package removes that class of
# bug entirely; only `rootutils` (which would rewrite sys.path) is stubbed, and
# the decorated entry point is reached through `__wrapped__`.
import hydra  # noqa: E402
import hydra.core.hydra_config  # noqa: E402

if "rootutils" not in sys.modules:
    _ru = types.ModuleType("rootutils")
    _ru.setup_root = lambda *a, **k: ROOT
    sys.modules["rootutils"] = _ru

spec = importlib.util.spec_from_file_location("cb", SCRIPT)
cb = importlib.util.module_from_spec(spec)
sys.modules["cb"] = cb
spec.loader.exec_module(cb)


def test_only_the_real_path_stages_are_additive():
    """`retrieval` and `prior_build` repeat work already inside `data`. If they
    ever join STAGES the totals silently double-count the FAISS search. `read`
    and `to_device` must BE stages, or their known cost lands in the residual."""
    assert cb.STAGES == ("read", "data", "to_device", "predict")
    for probe in ("retrieval", "prior_build", "esm_cache_read"):
        assert probe in cb.COMPONENTS and probe not in cb.STAGES, probe


def test_residual_is_reported_not_absorbed():
    r = cb.reconcile(1.0, {"data": 0.6, "head": 0.3})
    assert abs(r["stage_sum_s"] - 0.9) < 1e-12
    assert abs(r["residual_s"] - 0.1) < 1e-12
    assert abs(r["residual_frac"] - 0.1) < 1e-12
    # a decomposition that overshoots must show a negative residual, not zero
    r = cb.reconcile(1.0, {"data": 0.8, "head": 0.5})
    assert r["residual_s"] < 0


def test_reconcile_on_zero_total_does_not_divide_by_zero():
    r = cb.reconcile(0.0, {"data": 0.0, "head": 0.0})
    assert math.isnan(r["residual_frac"])


def test_summary_is_median_with_spread_not_a_bare_mean():
    """One outlier must not move the reported figure; a mean would."""
    xs = [1.0] * 99 + [1000.0]
    s = cb.summarise(xs)
    assert s["n"] == 100
    assert abs(s["median_s"] - 1.0) < 1e-9
    assert s["p90_s"] < 2.0            # the outlier sits above p90
    assert abs(s["total_s"] - (99 + 1000)) < 1e-9
    assert s["min_s"] == 1.0


def test_summary_of_nothing_is_nan_not_zero():
    s = cb.summarise([])
    assert s["n"] == 0 and math.isnan(s["median_s"])


def test_frozen_chain_set_is_deterministic_and_spans_the_LENGTH_distribution():
    """Cost is dominated by length, and ids are not ordered by length: a stride
    over sorted ids can miss the long tail entirely. Here id order is the
    REVERSE of length order, so an id-stride would be wrong."""
    class D:
        ids = [f"c{i:04d}" for i in range(1000)]
        cached_lengths = [1000 - i for i in range(1000)]
    a = cb._frozen_chain_set(D, 10)
    assert a == cb._frozen_chain_set(D, 10), "must not move between variants"
    assert len(a) == 10 and len(set(a)) == 10
    lens = dict(zip(D.ids, D.cached_lengths))
    picked = [lens[p] for p in a]
    assert picked == sorted(picked), "selection must walk the length ordering"
    assert min(picked) <= 20 and max(picked) >= 900, "both tails must be represented"
    assert cb._frozen_chain_set(D, 0) == sorted(D.ids, key=lambda p: (lens[p], p))


def test_param_split_separates_frozen_from_trainable():
    import torch
    m = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    for p in m[0].parameters():
        p.requires_grad_(False)
    c = cb.count_params(m)
    assert c["frozen"] == 4 * 4 + 4
    assert c["trainable"] == 4 * 2 + 2
    assert c["total"] == c["frozen"] + c["trainable"]


def test_artifact_sizes_are_reported_not_folded_into_per_chain_cost():
    """Index and cache size are the one-off preparation cost; a missing path is
    0, never an exception that would abort a finished measurement."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "a.bin").write_bytes(b"x" * 100)
        (root / "sub").mkdir()
        (root / "sub" / "b.bin").write_bytes(b"y" * 50)
        assert cb.dir_bytes(root) == 150
        assert cb.dir_bytes(root / "a.bin") == 100
    assert cb.dir_bytes(None) == 0
    assert cb.dir_bytes(Path("/definitely/not/here")) == 0




# ── the real main(), driven against stubs ──────────────────────────────────
# These are the tests that catch an attribute that does not exist in
# production (`builder.index` vs `builder.faiss_index`, `dataset.lengths` vs
# `cached_lengths`). No GPU, no FAISS, no data — but the real call graph.

def _stub_world(tmp: Path, n_ids=6):
    import torch as T

    class Idx:
        def __init__(self): self.calls = 0
        def topk_precomputed(self, pid, k, filter_holdout=True):
            self.calls += 1
            return [(f"t{i}", 0.9) for i in range(k)]

    class Builder:
        def __init__(self):
            self.faiss_index = Idx()
            self._tpl_cache = {"seed": 1}
            self.clears = 0
            self.topk = 4
        def build_one(self, pid, seq, s, e, filter_holdout=True):
            return None

    class DS:
        def __init__(self):
            self.ids = [f"c{i}" for i in range(n_ids)]
            self._len = {p: 100 + 10 * i for i, p in enumerate(self.ids)}
        @property
        def cached_lengths(self): return [self._len[p] for p in self.ids]
        def __getitem__(self, i):
            p = self.ids[i]
            return {"pid": p, "seq": "A" * self._len[p], "L": self._len[p]}

    class DM:
        def __init__(self):
            self.dset_test = DS(); self._prior_builder = Builder()
        def setup(self, stage=None): pass
        def _collate_eval(self, items):
            L = items[0]["L"]
            return {"pid": [items[0]["pid"]], "seq": [items[0]["seq"]],
                    "crop_bounds": T.zeros(1, 2, dtype=T.long),
                    "pair_mask": T.ones(1, L, L), "long_mask": T.ones(1, L, L),
                    "prior": T.zeros(1, 1, L, L), "count": T.zeros(1, 1, L, L),
                    "h_esm": T.zeros(1, L, 8), "esm_contacts": T.zeros(1, 1, L, L)}

    class Net(T.nn.Module):
        def forward(self, *a, **k): return T.zeros(1, 1, 4, 4)

    class Model(T.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = Net(); self.use_tpl_dist_bins = False
        def eval(self): return self
        def to(self, *a, **k): return self

    cfg = {"data": {"topk": 4, "crop_size": 384, "crop_mode": "center",
                    "num_workers": 0, "max_tpl_cache": 5000, "index_dir": str(tmp),
                    "esm_embeddings_dir": None},
           "model": {}, "trainer": {"precision": "32"},
           "cost": {"n_chains": 3, "repeats": 1, "warmup": 1}}
    return DM(), Model(), cfg


def test_real_main_runs_and_uses_production_attribute_names(monkeypatch=None):
    """Drives cb.main() end to end. A wrong attribute name (builder.index,
    dataset.lengths) raises here, without a GPU or the FAISS index."""
    import tempfile, types as _t
    import torch as T
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        dm, model, cfg_d = _stub_world(tmp)
        cfg = OmegaConf.create(cfg_d)          # real; no try/except, no skip

        def fake_instantiate(node, *a, **k):
            return dm if "topk" in node else model

        HC = hydra.core.hydra_config.HydraConfig
        real_instantiate, real_get = hydra.utils.instantiate, HC.get
        real_predict = cb.predict_only
        hydra.utils.instantiate = fake_instantiate
        HC.get = staticmethod(lambda: _t.SimpleNamespace(
            runtime=_t.SimpleNamespace(output_dir=str(tmp))))
        cb.predict_only = lambda m, b, dev, dt: T.zeros(1)
        try:
            entry = getattr(cb.main, "__wrapped__", cb.main)
            entry(cfg)                      # the real function under @hydra.main
        finally:
            hydra.utils.instantiate = real_instantiate
            HC.get = real_get
            cb.predict_only = real_predict

        rows = (tmp / "cost_benchmark.tsv").read_text().splitlines()
        assert len(rows) == 3, "header + cold + warm"
        chains = (tmp / "chains.tsv").read_text().splitlines()
        assert len(chains) == 4, "header + 3 chains"
        assert all(line.split("\t")[1] != "-1" for line in chains[1:]), \
            "lengths must come from cached_lengths, not a missing .lengths"
        man = json.loads((tmp / "manifest.json").read_text())
        assert man["cache_preparation"]["preparation_time_measured"] is False
        assert man["frozen"]["topk"] == 4


def test_cold_clears_per_query_and_warm_is_primed_over_every_chain():
    """Two failure modes, one test.

    cold: clearing once before the loop lets the warmup repopulate it.
    warm: priming only the first 8 chains (or relying on the GPU warmup) leaves
    the rest missing on the first measured repeat and hitting later, so the row
    averages two different cache states.
    """
    src = SCRIPT.read_text()
    loop = src[src.index('for cache_state in ("cold", "warm")'):]
    assert loop.count("_clear_template_cache(builder)") >= 3, \
        "cold must re-clear before the query and before each probe"
    # the GPU warmup must cover every chain: kernel choice is per SHAPE, and an
    # 8-chain warmup leaked first-encounter cost into cold `predict` (measured)
    gpu = loop[loop.index("for _ in range(warmup)"):loop.index('if cache_state == "warm":')]
    assert "for pid in chains:" in gpu and "chains[:" not in gpu
    # ... but the warm priming must walk ALL chains and build priors
    prime = loop[loop.index('if cache_state == "warm":'):]
    prime = prime[:prime.index("for _ in range(repeats)")]
    assert "for pid in chains:" in prime and "build_one" in prime, \
        "warm must prime the template cache over every selected chain"


def test_predict_path_excludes_training_bookkeeping():
    src = SCRIPT.read_text()
    start = src.index("def predict_only")
    body = src[start:src.index("\ndef ", start + 1)]   # not the FIRST @contextmanager
    for forbidden in ("masked_bce", "tversky", "precision_at_k", "self.log", "_step("):
        assert forbidden not in body, forbidden
    assert "torch.autocast" in body, "precision must be applied, not merely read"


def test_transfer_and_read_are_stages_not_residual():
    assert "read" in cb.STAGES and "to_device" in cb.STAGES
    for probe in ("retrieval", "prior_build", "esm_cache_read"):
        assert probe in cb.COMPONENTS and probe not in cb.STAGES


def test_launcher_appends_cost_overrides():
    """Hydra refuses a plain `cost.x=y`: the group is not declared."""
    sh = (ROOT / "configs/experiment/launch_preflight.sh").read_text()
    assert "+cost.n_chains" in sh
    import re
    assert not re.search(r"(?<![+])\bcost\.\w+=", sh), "bare cost.* override would abort"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
