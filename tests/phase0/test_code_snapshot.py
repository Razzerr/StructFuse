"""configs/experiment/_launch_common.sh: `use_code_snapshot` on a real git repository.

Pins what parallel branches rely on: the snapshot holds the requested commit and not
the checkout, later commits do not reach it, data and logs are shared through
symlinks, and the job records that commit. Also pins the validation-only path.

Plain-python: `python tests/phase0/test_code_snapshot.py`.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMMON = ROOT / "configs/experiment/_launch_common.sh"


def _git(repo, *args):
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t", "PATH": "/usr/bin:/bin"}
    return subprocess.run(["git", "-C", str(repo), *args], check=True, env=env,
                          capture_output=True, text=True).stdout.strip()


def _repo(tmp):
    repo = Path(tmp) / "project"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "code.py").write_text("v1\n")
    _git(repo, "add", "code.py")
    _git(repo, "commit", "-q", "-m", "one")
    for name in ("data", "logs"):
        (repo / name).mkdir()
    return repo


def _snapshot(repo, ref):
    script = f'export PROJECT_DIR="{repo}"; source "{COMMON}"; use_code_snapshot "{ref}"; echo "$JOB_DIR $JOB_COMMIT"'
    out = subprocess.run(["bash", "-c", script], check=True, capture_output=True, text=True).stdout
    job_dir, commit = out.split()
    return Path(job_dir), commit


def test_snapshot_holds_the_commit_and_shares_data():
    with tempfile.TemporaryDirectory() as tmp:
        repo = _repo(tmp)
        first = _git(repo, "rev-parse", "HEAD")
        job_dir, commit = _snapshot(repo, "HEAD")
        assert commit == first and job_dir.name == first
        assert (job_dir / "code.py").read_text() == "v1\n"
        assert (job_dir / "data").is_symlink() and (job_dir / "data").resolve() == (repo / "data").resolve()
        (repo / "logs" / "x").write_text("shared")
        assert (job_dir / "logs" / "x").read_text() == "shared"


def test_later_commits_do_not_reach_an_existing_snapshot():
    with tempfile.TemporaryDirectory() as tmp:
        repo = _repo(tmp)
        first = _git(repo, "rev-parse", "HEAD")
        job_dir, _ = _snapshot(repo, "HEAD")
        (repo / "code.py").write_text("v2\n")
        _git(repo, "commit", "-q", "-am", "two")
        again, commit = _snapshot(repo, first)
        assert again == job_dir and commit == first and (again / "code.py").read_text() == "v1\n"
        newer, commit = _snapshot(repo, "HEAD")
        assert newer != job_dir and (newer / "code.py").read_text() == "v2\n"


def test_uncommitted_changes_never_enter_a_snapshot():
    with tempfile.TemporaryDirectory() as tmp:
        repo = _repo(tmp)
        (repo / "code.py").write_text("dirty\n")
        job_dir, _ = _snapshot(repo, "HEAD")
        assert (job_dir / "code.py").read_text() == "v1\n"


def test_unknown_ref_stops_the_launch():
    with tempfile.TemporaryDirectory() as tmp:
        repo = _repo(tmp)
        script = f'export PROJECT_DIR="{repo}"; source "{COMMON}"; use_code_snapshot "origin/nope"'
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        assert r.returncode != 0, r.stdout


def test_selection_modes_never_read_the_test_set():
    src = (ROOT / "configs/experiment/launch_paper_8M.sh").read_text()
    gate = src[src.index("submit_aligner_gate() {"):src.index("submit_trufor_controls() {")]
    assert gate.count("submit_training") == 4 and gate.count('"" false') == 4, gate
    assert 'submit_training "${SELECT_EXPERIMENT}" "${SELECT_TASK}" "48:00:00" "" false' in src
    assert 'submit_final_eval "$1" "$2" "valbs1" "run_test=false"' in src
    rows = src[src.index("submit_select_rows() {"):src.index("submit_select_rows_aligner_gate() {")]
    assert "data.test_ids=${PROJECT_DIR}/data/output_splits_2026cb/val_holdout_ids.txt" in rows, rows
    assert "test_ids.txt" not in rows.replace("val_holdout_ids.txt", ""), "rows mode must not point at the test split"
    ev = (ROOT / "src/eval.py").read_text()
    i = ev.index('cfg.get("run_test", True)')
    assert "trainer.test(" in ev[i:i + 260], "the test call must sit under the run_test guard"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
