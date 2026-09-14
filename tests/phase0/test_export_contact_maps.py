"""export_contact_maps.py: the real main(), real forward path, stub data.

What must hold for Figure 4's arrays to mean what the caption says: the
weights come from the checkpoint the evaluation fingerprinted (not "the newest
file", not an initialised model), the forward is the benchmarked
`predict_only`, the sigmoid is taken in the evaluation's dtype (bf16 under
bf16-mixed — upcasting first changes the ranking's ties), soft distance
histograms survive the write, the threshold is the evaluation's own at full
precision, and every metric the evaluation reported for a chain is reproduced
or the script refuses.
"""

import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "export_contact_maps.py"
LAUNCHER = ROOT / "configs" / "experiment" / "launch_preflight.sh"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf  # noqa: E402  real, never stubbed
import hydra  # noqa: E402
import hydra.core.hydra_config  # noqa: E402
import torch as T  # noqa: E402

if "rootutils" not in sys.modules:
    _ru = types.ModuleType("rootutils")
    _ru.setup_root = lambda *a, **k: ROOT
    sys.modules["rootutils"] = _ru

spec = importlib.util.spec_from_file_location("ecm", SCRIPT)
ecm = importlib.util.module_from_spec(spec)
sys.modules["ecm"] = ecm
spec.loader.exec_module(ecm)

from src.utils.audit import _file_identity  # noqa: E402

THRESHOLD = 0.7921052575111389   # the real headline value, deliberately unrounded


# ── stub world: production batch contract, tiny sizes ──────────────────────

class _Net(T.nn.Module):
    """Logits = 4*signal + bias. `bias` is the only weight, so a loaded
    checkpoint visibly changes the output. Under autocast the logits are
    emitted in bf16, as a real conv head would be."""
    def __init__(self):
        super().__init__()
        self.bias = T.nn.Parameter(T.zeros(()))
        self.last_kwargs = None

    def forward(self, h, prior, count, rel, esm_contacts, pair_mask=None, tpl_dist_bins=None):
        self.last_kwargs = {"pair_mask": pair_mask, "tpl_dist_bins": tpl_dist_bins}
        out = 4.0 * esm_contacts + self.bias
        if T.is_autocast_enabled(esm_contacts.device.type):
            out = out.to(T.bfloat16)
        return out                                    # (1,1,L,L)


class _Model(T.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = _Net()
        self.use_tpl_dist_bins = True

    def to(self, *a, **k):
        return self


def _batch(pid, L, seed):
    rng = np.random.RandomState(seed)
    c = (rng.rand(L, L) < 0.08).astype(np.float32)
    c = np.triu(c, 1); c = c + c.T
    contact = T.tensor(c)[None]
    idx = T.arange(L)
    sep = (idx[None, :] - idx[:, None]).abs()
    long_mask = (sep >= 6).float()[None]
    prior = T.zeros(1, 1, L, L); prior[0, 0, 3, 40] = prior[0, 0, 40, 3] = 0.7
    esm = T.tensor(c + 0.1 * rng.rand(L, L).astype(np.float32))[None, None]
    bins = T.zeros(1, 9, L, L)
    bins[0, 0, 3, 40] = 0.5; bins[0, 1, 3, 40] = 0.25; bins[0, 2, 3, 40] = 0.125  # soft mass
    return {"pid": [pid], "seq": ["A" * L], "subset": ["gold"], "cluster_id": [17],
            "crop_bounds": T.tensor([[0, L]], dtype=T.long),
            "seq_len": T.tensor([L]), "contact": contact,
            "pair_mask": T.ones(1, L, L), "long_mask": long_mask,
            "prior": prior, "count": T.ones(1, 1, L, L),
            "n_templates_retrieved": T.tensor([4]), "best_tpl_sim": T.tensor([0.99]),
            "tpl_dist_bins": bins,
            "h_esm": T.zeros(1, L, 8), "esm_contacts": esm}


class _DS:
    def __init__(self, lens):
        self.ids = list(lens)
        self._len = dict(lens)
    def __getitem__(self, i):
        p = self.ids[i]
        return {"pid": p, "seq": "A" * self._len[p], "L": self._len[p], "crop_bounds": (0, self._len[p])}


class _DM:
    def __init__(self, lens, crop_override=None):
        self.dset_test = _DS(lens)
        self.crop_override = crop_override
    def setup(self, stage=None):
        pass
    def _collate_eval(self, items):
        it = items[0]
        b = _batch(it["pid"], it["L"], seed=sum(map(ord, it["pid"])))
        if self.crop_override:
            b["crop_bounds"] = T.tensor([list(self.crop_override)], dtype=T.long)
        return b


def _write_requests(path, rows):
    with open(path, "w") as fh:
        fh.write("category\trole\tsample_id\tpdb_id\tchain_id\tcrop_start\tcrop_end\ttop1_template_id\n")
        for cat, pid, L in rows:
            fh.write(f"{cat}\tprimary\t{pid}\t{pid[:4]}\t{pid[5:]}\t0\t{L}\ttpl_A\n")


def _write_ckpt(path, bias=2.5, thr=0.55):
    T.save({"state_dict": {"net.bias": T.tensor(bias), "pred_threshold": thr},
            "epoch": 25, "global_step": 1234}, path)


def _write_audit(path, ckpt: Path, *, fingerprint=None):
    ident = _file_identity(ckpt)
    if fingerprint is not None:
        ident = {**ident, "fingerprint": fingerprint}
    path.write_text(json.dumps({"run_id": "n87663ed", "task_name": "paper_650m_trufor_s42_bs1",
                                "git_commit": "abc", "ckpt_path": str(ckpt),
                                "checkpoint_identity": ident}))


def _write_eval_run(run_dir: Path, per_sample_rows: dict, threshold=THRESHOLD):
    """per_sample_rows: sample_id -> {metric: value}; columns = union of keys."""
    run_dir.mkdir(parents=True, exist_ok=True)
    cols = sorted({k for r in per_sample_rows.values() for k in r})
    with (run_dir / "per_sample_metrics.tsv").open("w") as fh:
        fh.write("\t".join(["sample_id", *cols]) + "\n")
        for sid, r in per_sample_rows.items():
            fh.write("\t".join([sid, *(repr(float(r[c])) for c in cols)]) + "\n")
    (run_dir / "eval_metrics.json").write_text(json.dumps({"test/threshold_used": threshold,
                                                          "test/P@L_long": 0.5}))


def _run(tmp: Path, *, lens, requests, eval_run, audit, ckpt_path=None, bs=1, tol=0.0,
         precision="32", crop_override=None, extra=None):
    dm, model = _DM(lens, crop_override), _Model()
    cfg = OmegaConf.create({
        "data": {"topk": 4, "crop_size": 384, "crop_mode": "center", "num_workers": 0,
                 "index_dir": str(tmp), "eval_batch_size": bs},
        "model": {"use_template_features": True},
        "trainer": {"precision": precision, "deterministic": True},
        "ckpt_path": str(ckpt_path) if ckpt_path else None, "task_name": "stub",
        "export": {"requests": str(requests), "tag": "structfuse",
                   "eval_run_dir": str(eval_run) if eval_run else None,
                   "eval_audit_manifest": str(audit) if audit else None,
                   "metric_tol": tol, **(extra or {})},
    })
    HC = hydra.core.hydra_config.HydraConfig
    real_inst, real_get = hydra.utils.instantiate, HC.get
    hydra.utils.instantiate = lambda node, *a, **k: dm if "topk" in node else model
    out = tmp / "out"; out.mkdir(exist_ok=True)
    HC.get = staticmethod(lambda: types.SimpleNamespace(
        runtime=types.SimpleNamespace(output_dir=str(out))))
    try:
        entry = getattr(ecm.main, "__wrapped__", ecm.main)
        entry(cfg)
    finally:
        hydra.utils.instantiate = real_inst
        HC.get = real_get
    return out, model


def _exported(out: Path):
    hdr, *rows = (out / "exported_cases.tsv").read_text().splitlines()
    cols = hdr.split("\t")
    return {r.split("\t")[cols.index("sample_id")]: dict(zip(cols, r.split("\t"))) for r in rows}


def _world(tmp: Path, ids=(("largest_gain", "1abc_A", 96), ("failure", "2xyz_B", 80)),
           precision="32"):
    """A consistent stub world whose evaluation table was produced by the same
    stub forward: run once with an empty-metric table to learn the values,
    then write the real one. Returns (paths, reported)."""
    req = tmp / "req.tsv"; _write_requests(req, list(ids))
    ck = tmp / "epoch25-f1long0.602.ckpt"; _write_ckpt(ck)
    audit = tmp / "manifest.json"; _write_audit(audit, ck)
    lens = {pid: L for _, pid, L in ids}
    boot = tmp / "eval_boot"; _write_eval_run(boot, {pid: {} for pid in lens})
    out, _ = _run(tmp, lens=lens, requests=req, eval_run=boot, audit=audit, precision=precision)
    got = _exported(out)
    reported = {pid: {m: float(got[pid][m]) for m in ecm.GATE_METRICS if m in got[pid]}
                for pid in lens}
    run = tmp / "eval_run"; _write_eval_run(run, reported)
    return {"req": req, "ckpt": ck, "audit": audit, "run": run, "lens": lens}, reported


# ── tests ──────────────────────────────────────────────────────────────────

def test_real_main_exports_from_the_fingerprinted_checkpoint_at_the_evaluations_threshold():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        w, reported = _world(tmp)
        # no ckpt_path on the command line: it comes from the audit manifest
        out, model = _run(tmp, lens=w["lens"], requests=w["req"], eval_run=w["run"], audit=w["audit"])

        assert model.net.bias.item() == 2.5, "checkpoint weights were not loaded"
        files = sorted(p.name for p in (out / "maps_structfuse").glob("*.npz"))
        assert files == ["1abc_A.npz", "2xyz_B.npz"], files

        z = np.load(out / "maps_structfuse" / "1abc_A.npz")
        L = 96
        for k, shape, dt in (("contact", (L, L), np.uint8), ("pair_mask", (L, L), np.uint8),
                             ("valid_mask", (L, L), np.uint8), ("prior", (L, L), np.float32),
                             ("count", (L, L), np.float32), ("prob", (L, L), np.float32),
                             ("tpl_dist_bins", (9, L, L), np.float32)):
            assert z[k].shape == shape and z[k].dtype == dt, (k, z[k].shape, z[k].dtype)
        assert z["prob"].min() >= 0 and z["prob"].max() <= 1
        assert np.array_equal(z["contact"], z["contact"].T)
        assert z["valid_mask"][0, 3] == 0 and z["valid_mask"][0, 30] == 1, "|i-j|<6 excluded"
        assert str(z["sample_id"]) == "1abc_A" and int(z["seq_len"]) == L
        assert float(z["threshold"]) == THRESHOLD, "full-precision eval_metrics.json value"
        assert str(z["prob_dtype"]) == "torch.float32"

        man = json.loads((out / "manifest.json").read_text())
        assert man["checkpoint"]["sha256"] == ecm.sha256(w["ckpt"])
        assert man["checkpoint"]["stored_pred_threshold"] == 0.55
        assert man["threshold_used"] == THRESHOLD and man["threshold_used"] != 0.7921
        assert man["checkpoint_identity_vs_evaluation"]["status"] == "MATCH"
        assert man["checkpoint_identity_vs_evaluation"]["eval_run_id"] == "n87663ed"
        assert man["gate"]["status"] == "PASS" and man["gate"]["mismatches"] == []
        assert "AUC-PR_long" in man["gate"]["metrics"] and "f1_long" in man["gate"]["metrics"]
        assert man["deterministic"] == "True"
        assert "predict_only" in man["forward"]
        assert model.net.last_kwargs["pair_mask"].shape == (1, 1, 80, 80), "(B,1,L,L) as in _step"


def test_soft_distance_histograms_survive_the_write():
    """tpl_dist_bins is per-pair MASS over bins, not bin ids. [0.5, 0.25, 0.125]
    written as uint8 would be [0, 0, 0] — and the P@L gate would still pass,
    because the prediction is made before the write."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        w, _ = _world(tmp)
        out, _ = _run(tmp, lens=w["lens"], requests=w["req"], eval_run=w["run"], audit=w["audit"])
        z = np.load(out / "maps_structfuse" / "1abc_A.npz")
        got = z["tpl_dist_bins"][:3, 3, 40]
        assert np.array_equal(got, np.array([0.5, 0.25, 0.125], dtype=np.float32)), got
        assert z["tpl_dist_bins"].sum() > 0


def test_sigmoid_is_taken_in_the_evaluations_dtype_not_after_upcasting():
    """Under bf16-mixed the evaluation computed torch.sigmoid(bf16 logits): the
    probabilities are bf16-quantised and the top-L ties follow that grid.
    sigmoid(logits.float()) is a different, finer set of values."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        w, reported = _world(tmp, precision="bf16-mixed")
        out, _ = _run(tmp, lens=w["lens"], requests=w["req"], eval_run=w["run"], audit=w["audit"],
                      precision="bf16-mixed")
        z = np.load(out / "maps_structfuse" / "1abc_A.npz")
        assert str(z["prob_dtype"]) == "torch.bfloat16" and str(z["logits_dtype"]) == "torch.bfloat16"
        prob = T.tensor(z["prob"])
        assert T.equal(prob.to(T.bfloat16).float(), prob), "prob must lie on the bf16 grid"
        # the upcast-first path gives different numbers on this very input
        b = _batch("1abc_A", 96, seed=sum(map(ord, "1abc_A")))
        logits32 = 4.0 * b["esm_contacts"][0, 0] + 2.5
        fine = T.sigmoid(logits32)
        assert not T.equal(fine, prob), "export reproduced sigmoid(logits.float()), not the eval"
        assert T.equal(T.sigmoid(logits32.to(T.bfloat16)).float(), prob)
        man = json.loads((out / "manifest.json").read_text())
        assert man["gate"]["status"] == "PASS"


def test_gate_covers_ranking_and_threshold_metrics_not_only_top_L():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        w, reported = _world(tmp)
        # f1_long depends on the threshold; P@L_long does not. Perturb ONLY f1_long.
        bad = dict(reported); bad["1abc_A"] = {**reported["1abc_A"],
                                              "f1_long": reported["1abc_A"]["f1_long"] + 0.01}
        run_bad = tmp / "eval_bad"; _write_eval_run(run_bad, bad)
        try:
            _run(tmp, lens=w["lens"], requests=w["req"], eval_run=run_bad, audit=w["audit"])
        except SystemExit as exc:
            assert "metric gate FAILED" in str(exc) and "f1_long" in str(exc)
        else:
            raise AssertionError("a threshold-visible mismatch was accepted")
        man = json.loads((tmp / "out" / "manifest.json").read_text())
        assert man["gate"]["status"] == "FAIL"
        # ... and a ranking-visible one (AUC-PR_long) as well
        bad2 = dict(reported); bad2["2xyz_B"] = {**reported["2xyz_B"],
                                                "AUC-PR_long": reported["2xyz_B"]["AUC-PR_long"] - 0.02}
        run_bad2 = tmp / "eval_bad2"; _write_eval_run(run_bad2, bad2)
        try:
            _run(tmp, lens=w["lens"], requests=w["req"], eval_run=run_bad2, audit=w["audit"])
        except SystemExit as exc:
            assert "AUC-PR_long" in str(exc)
        else:
            raise AssertionError("a ranking-visible mismatch was accepted")
        # a chain missing from the table is a population error
        run_abs = tmp / "eval_abs"; _write_eval_run(run_abs, {"1abc_A": reported["1abc_A"]})
        try:
            _run(tmp, lens=w["lens"], requests=w["req"], eval_run=run_abs, audit=w["audit"])
        except SystemExit as exc:
            assert "absent from" in str(exc)
        else:
            raise AssertionError("nothing raised")


def test_checkpoint_identity_is_verified_against_the_evaluations_fingerprint():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        w, _ = _world(tmp)
        # an audit manifest that fingerprints a DIFFERENT file than the one loaded
        other = tmp / "other.ckpt"; _write_ckpt(other, bias=9.0)
        wrong = tmp / "wrong_manifest.json"; _write_audit(wrong, other)
        try:
            _run(tmp, lens=w["lens"], requests=w["req"], eval_run=w["run"], audit=wrong,
                 ckpt_path=w["ckpt"])
        except ValueError as exc:
            assert "checkpoint identity mismatch" in str(exc)
        else:
            raise AssertionError("a checkpoint the evaluation never saw was accepted")
        # explicit ckpt_path pointing at the fingerprinted file (moved copy) is fine
        moved = tmp / "moved.ckpt"; moved.write_bytes(w["ckpt"].read_bytes())
        out, _ = _run(tmp, lens=w["lens"], requests=w["req"], eval_run=w["run"], audit=w["audit"],
                      ckpt_path=moved)
        man = json.loads((out / "manifest.json").read_text())
        assert man["checkpoint_identity_vs_evaluation"]["status"] == "MATCH"
        assert man["checkpoint_identity_vs_evaluation"]["loaded_ckpt_path"] == str(moved)


def test_refusals_hand_typed_threshold_batch_size_unknown_chain_and_crop():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        w, _ = _world(tmp)
        base = dict(lens=w["lens"], requests=w["req"], eval_run=w["run"], audit=w["audit"])
        for kw, needle, exc_type in (
            (dict(extra={"threshold": 0.7921}), "no longer accepted", SystemExit),
            (dict(extra={"per_sample_tsv": "x.tsv"}), "no longer accepted", SystemExit),
            (dict(bs=12), "eval_batch_size=1", SystemExit),
            (dict(eval_run=None), "+export.eval_run_dir is required", SystemExit),
            (dict(audit=None), "+export.eval_audit_manifest is required", SystemExit),
            (dict(lens={"9other_A": 96}), "not in the evaluated test set", SystemExit),
            (dict(crop_override=(10, 96)), "whole-chain", ValueError),
        ):
            args = {**base, **kw}
            try:
                _run(tmp, **args)
            except exc_type as exc:
                assert needle in str(exc), (needle, str(exc))
            else:
                raise AssertionError(f"nothing raised for {needle}")


def test_forward_is_the_benchmarked_predict_only_and_sigmoid_is_not_upcast_first():
    src = SCRIPT.read_text()
    assert "from scripts.cost_benchmark import" in src and "predict_only" in src
    assert "_step(" not in src, "the export must not go through the training step"
    assert "torch.sigmoid(logits)" in src and "sigmoid(logits.float())" not in src
    assert "per_sample_metric_rows" in src
    line = src.split('arrays["tpl_dist_bins"] = ')[1].split("\n")[0]
    assert "float32" in line and "uint8" not in line, "histograms are soft mass, never an integer dtype"
    assert "weights_only=False" in src


def test_launcher_pins_the_evaluation_runs_instead_of_discovering_files():
    src = LAUNCHER.read_text()
    block = src[src.index("run_export()"):src.index('run_selected "ceiling"')]
    assert "data.eval_batch_size=1" in block
    for key in ("+export.requests=", "+export.tag=", "+export.eval_run_dir=", "+export.eval_audit_manifest="):
        assert key in block, key
    for gone in ("+export.threshold=", "+export.per_sample_tsv=", "newest_ckpt", "find "):
        assert gone not in block, f"{gone!r} must not be used: provenance is pinned"
    assert '"n87663ed"' in src and '"qux75yw4"' in src, "the two bs=1 evaluation run ids"
    assert "2026-09-11_08-45-58" in src and "2026-09-11_08-47-11" in src, "their hydra run dirs"
    assert 'export-maps* ) && "${MODE}" == "all"' in src, "export is opt-in, not part of --all"


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
