"""Fetch and display experiment info from a W&B run.

Usage:
    python scripts/fetch_wandb_run.py <run_id>
    python scripts/fetch_wandb_run.py <run_id> --max-rows 30

Output is saved to .temp/experiments/<run_id>.md
"""

import argparse
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd
import wandb
from rich.console import Console
from rich.table import Table
from rich import box
from rich.text import Text

STATE_COLOR = {"finished": "green", "running": "yellow", "crashed": "red", "failed": "red"}


def fetch_run(run_id: str) -> wandb.apis.public.Run:
    api = wandb.Api(timeout=60)
    return api.run(f"ryzzr/StructFuse/{run_id}")


def render_run_info(run: wandb.apis.public.Run, console: Console) -> None:
    state_color = STATE_COLOR.get(run.state, "white")
    state_text = run.state
    if run.state == "crashed":
        state_text += " (probably stopped by user)"

    t = Table(box=box.SIMPLE, show_header=False, pad_edge=False)
    t.add_column(style="bold cyan", no_wrap=True)
    t.add_column()

    t.add_row("ID", run.id)
    t.add_row("Name", run.name or "—")
    t.add_row("State", Text(state_text, style=state_color))
    t.add_row("Created", str(run.created_at)[:19])
    t.add_row("Tags", ", ".join(run.tags) if run.tags else "—")
    t.add_row("Group", run.group or "—")
    t.add_row("URL", run.url)

    console.rule("[bold]Run Info[/bold]")
    console.print(t)

    # Config — skip internal/boilerplate keys
    SKIP = {"wandb_version", "_wandb"}
    cfg = {k: v for k, v in sorted(run.config.items()) if k not in SKIP}
    if cfg:
        ct = Table(box=box.SIMPLE, show_header=False, pad_edge=False)
        ct.add_column(style="dim cyan", no_wrap=True)
        ct.add_column()
        for k, v in cfg.items():
            ct.add_row(k, str(v))
        console.rule("[bold]Config[/bold]")
        console.print(ct)


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _render_metric_table(
    df: pd.DataFrame,
    cols: list[str],
    index_col: str,
    title: str,
    title_prefix: str,
    console: Console,
) -> None:
    """Render one metric table.

    `title_prefix` is stripped from each col header for display (e.g. "val/" → empty header text).
    """
    subset = df[[index_col] + cols].dropna(how="all", subset=cols)
    if subset.empty:
        return

    t = Table(
        title=f"[bold]{title}[/bold]",
        box=box.SIMPLE_HEAD,
        show_lines=False,
        pad_edge=False,
        title_justify="left",
    )
    t.add_column(index_col, style="dim", no_wrap=True, min_width=6)
    for c in cols:
        short = c[len(title_prefix) + 1:] if title_prefix and c.startswith(title_prefix + "/") else c
        t.add_column(short, justify="right", no_wrap=True)

    for _, row in subset.iterrows():
        raw = row[index_col]
        idx = f"{int(raw)}" if pd.notna(raw) else "?"
        values = [_fmt(row[c]) for c in cols]
        t.add_row(idx, *values)

    console.print(t)


def build_metric_tables(run: wandb.apis.public.Run, max_rows: int | None, console: Console) -> None:
    # scan_history pages through data (page_size rows per request) — avoids single large HTTP call
    rows = []
    for row in run.scan_history(page_size=500):
        rows.append(row)
        if max_rows is not None and len(rows) >= max_rows:
            break

    if not rows:
        console.print("[dim]No metric history available.[/dim]")
        return

    history = pd.DataFrame(rows)

    # Test metrics are logged via `on_test_epoch_end` after fit finishes — they don't have an
    # `epoch` value, so `groupby("epoch", dropna=True).last()` would silently drop them.
    # Pull them out before the epoch aggregation and render separately.
    test_cols = sorted(c for c in history.columns if c.startswith("test/"))
    test_history: pd.DataFrame | None = None
    if test_cols:
        test_mask = history[test_cols].notna().any(axis=1)
        test_history = history.loc[test_mask, test_cols + ["_step"]].copy()
        history = history.drop(columns=test_cols)

    # Aggregate by epoch (last value per epoch per column)
    if "epoch" in history.columns:
        history = (
            history.groupby("epoch", sort=True)
            .last()
            .reset_index()
        )
        index_col = "epoch"
    else:
        index_col = "_step"

    # Group non-test columns by top-level prefix
    categories: dict[str, list[str]] = defaultdict(list)
    for col in history.columns:
        if col.startswith("_") or col == "epoch":
            continue
        prefix = col.split("/")[0]
        categories[prefix].append(col)

    console.rule("[bold]Metrics[/bold]")

    for category in sorted(categories):
        cols = sorted(categories[category])
        _render_metric_table(history, cols, index_col, category, category, console)

    # Test rows in scan_history are typically empty (`on_test_epoch_end` writes to summary,
    # not to step-level history). The dedicated test section is rendered from `run.summary`
    # in `render_test_summary`.


def render_test_summary(run: wandb.apis.public.Run, console: Console) -> None:
    """Render final test metrics from `run.summary`.

    `trainer.test` writes via `self.log(..., on_epoch=True)` which lands in summary, not in
    step-level history (so `scan_history` doesn't see them). We pull them straight from summary.
    """
    test_keys = sorted(k for k in run.summary.keys() if k.startswith("test/"))
    if not test_keys:
        return

    # Group by sub-prefix: `test/P@L_long` → "test"; `test/casp16/P@L_long` → "test/casp16".
    groups: dict[str, list[str]] = defaultdict(list)
    for k in test_keys:
        parts = k.split("/")
        grp = "/".join(parts[:-1]) if len(parts) >= 3 else parts[0]
        groups[grp].append(k)

    console.rule("[bold]Test (final)[/bold]")
    for grp in sorted(groups):
        items = sorted(groups[grp])
        t = Table(
            title=f"[bold]{grp}[/bold]",
            box=box.SIMPLE_HEAD,
            show_lines=False,
            pad_edge=False,
            title_justify="left",
        )
        t.add_column("metric", style="dim", no_wrap=True)
        t.add_column("value", justify="right", no_wrap=True)
        for k in items:
            short = k[len(grp) + 1:] if k.startswith(grp + "/") else k
            t.add_row(short, _fmt(run.summary[k]))
        console.print(t)


def _epoch_frame(run: wandb.apis.public.Run, max_rows: int | None) -> pd.DataFrame:
    """Epoch-aggregated history, test columns dropped. Shared by both renderers."""
    rows = []
    for row in run.scan_history(page_size=500):
        rows.append(row)
        if max_rows is not None and len(rows) >= max_rows:
            break
    if not rows:
        return pd.DataFrame()
    history = pd.DataFrame(rows)
    history = history.drop(columns=[c for c in history.columns if c.startswith("test/")])
    if "epoch" in history.columns:
        history = history.groupby("epoch", sort=True).last().reset_index()
    return history


def render_selected_keys(
    runs: list[wandb.apis.public.Run], patterns: list[str], max_rows: int | None,
    console: Console,
) -> None:
    """Print only the columns matching `patterns`, so nothing gets truncated.

    The default per-prefix tables render every logged key at once; once a run
    carries ~40 val columns, rich abbreviates the headers (`cluster_bala…`,
    `f1_long_m…`) and the numbers become unreadable. Selecting a handful of keys
    keeps the table narrow enough to print in full. With several runs it also
    emits a side-by-side last/best comparison, which is what a paired gate or an
    ablation cell actually needs.
    """
    per_run: dict[str, pd.DataFrame] = {}
    for run in runs:
        history = _epoch_frame(run, max_rows)
        if history.empty:
            console.print(f"[dim]{run.id}: no metric history.[/dim]")
            continue
        cols = [
            c for c in history.columns
            if not c.startswith("_") and c != "epoch"
            and any(pat in c for pat in patterns)
        ]
        if not cols:
            console.print(
                f"[dim]{run.id}: no column matches {patterns}. "
                f"Available prefixes: "
                f"{sorted({c.split('/')[0] for c in history.columns if not c.startswith('_')})}[/dim]"
            )
            continue
        index_col = "epoch" if "epoch" in history.columns else "_step"
        frame = history[[index_col] + sorted(cols)].dropna(how="all", subset=sorted(cols))
        per_run[run.id] = frame
        console.rule(f"[bold]{run.id} — {run.name} ({run.state})[/bold]")
        _render_metric_table(frame, sorted(cols), index_col, run.id, run.id, console)

    if len(per_run) < 2:
        return

    console.rule("[bold]Comparison — last / best epoch[/bold]")
    table = Table(show_header=True, header_style="bold")
    table.add_column("run")
    table.add_column("epoch")
    all_cols = sorted({c for f in per_run.values() for c in f.columns if c not in ("epoch", "_step")})
    for col in all_cols:
        table.add_column(col, justify="right")
    for rid, frame in per_run.items():
        idx = "epoch" if "epoch" in frame.columns else "_step"
        last = frame.iloc[-1]
        table.add_row(f"{rid} last", _fmt(last[idx]),
                      *[_fmt(last.get(c)) for c in all_cols])
    console.print(table)


def list_recent_runs(limit: int, state: str | None, name_filter: str | None,
                     console: Console) -> None:
    """List recent runs so a run id never has to be copied out of the browser.

    Ordered newest-first. `state` filters server-side (finished / running /
    failed / crashed); `name_filter` is a case-insensitive substring match on the
    task name, which is how our runs are actually identified (`paper_8m_*`).
    """
    import os
    entity = os.environ.get("WANDB_ENTITY", "ryzzr")
    project = os.environ.get("WANDB_PROJECT", "StructFuse")
    api = wandb.Api(timeout=120)
    filters = {"state": state} if state else None
    runs = api.runs(f"{entity}/{project}", filters=filters,
                    order="-created_at", per_page=min(limit * 3, 200))

    table = Table(show_header=True, header_style="bold", box=box.SIMPLE)
    for col, just in (("id", "left"), ("task_name", "left"), ("state", "left"),
                      ("created", "left"), ("epochs", "right"),
                      ("test/P@L_long", "right"), ("tags", "left")):
        table.add_column(col, justify=just)

    shown = 0
    for run in runs:
        task = run.config.get("task_name") or run.display_name or run.name
        if name_filter and name_filter.lower() not in str(task).lower():
            continue
        sm = run.summary_metrics or {}
        table.add_row(
            run.id, str(task), run.state, str(run.created_at)[:16],
            _fmt(sm.get("epoch")), _fmt(sm.get("test/P@L_long")),
            ",".join(run.tags[:3]),
        )
        shown += 1
        if shown >= limit:
            break
    console.print(table)
    if shown == 0:
        console.print("[dim]No run matched. Try a wider --recent or drop --name.[/dim]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch W&B run info and metrics.")
    parser.add_argument("run_id", nargs="*",
                        help="One or more W&B run IDs. Omit with --recent to list instead.")
    parser.add_argument("--recent", type=int, nargs="?", const=15, default=None,
                        help="List the N most recent runs (default 15) instead of fetching. "
                             "Saves copying run ids out of the browser.")
    parser.add_argument("--state", default=None,
                        choices=["finished", "running", "failed", "crashed"],
                        help="Filter --recent by run state.")
    parser.add_argument("--name", default=None,
                        help="Filter --recent by task-name substring, e.g. paper_8m.")
    parser.add_argument(
        "--keys",
        default=None,
        help="Comma-separated substrings; print ONLY matching metric columns, to stdout, "
             "without rich truncating the headers. E.g. --keys f1_long,P@L_long,cluster_balanced. With several run ids, adds a side-by-side comparison.",
    )
    parser.add_argument(
        "--width", type=int, default=200,
        help="Console width for --keys output (default 200); a bare terminal is "
             "usually 80, which truncates the headers again.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Maximum number of raw history rows to fetch before epoch-aggregation (default: unlimited)",
    )
    args = parser.parse_args()

    if args.recent is not None:
        list_recent_runs(args.recent, args.state, args.name,
                         Console(width=args.width, highlight=False, no_color=True))
        return
    if not args.run_id:
        raise SystemExit("Give a run id, or --recent to list what is available.")

    runs = [fetch_run(rid) for rid in args.run_id]

    if args.keys:
        patterns = [k.strip() for k in args.keys.split(",") if k.strip()]
        # Explicit width: a bare Console() inherits the terminal's, which is
        # usually 80 and re-introduces the very truncation --keys exists to avoid.
        render_selected_keys(
            runs, patterns, args.max_rows,
            Console(width=args.width, highlight=False, no_color=True),
        )
        return

    for run in runs:
        # Render into a string via a file console
        out_path = Path(".temp/experiments") / f"{run.id}.md"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w") as f:
            file_console = Console(file=f, width=500, highlight=False, markup=True, no_color=True)
            render_run_info(run, file_console)
            build_metric_tables(run, max_rows=args.max_rows, console=file_console)
            render_test_summary(run, file_console)

        print(f"Saved in {out_path}")


if __name__ == "__main__":
    main()
