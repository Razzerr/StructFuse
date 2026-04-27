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

    # Group columns by prefix
    categories: dict[str, list[str]] = defaultdict(list)
    for col in history.columns:
        if col.startswith("_") or col == "epoch":
            continue
        prefix = col.split("/")[0]
        categories[prefix].append(col)

    console.rule("[bold]Metrics[/bold]")

    for category in sorted(categories):  # type: ignore[assignment]
        cols = sorted(categories[category])
        subset = history[[index_col] + cols].dropna(how="all", subset=cols)
        subset = subset.dropna(how="all", subset=cols)

        t = Table(
            title=f"[bold]{category}[/bold]",
            box=box.SIMPLE_HEAD,
            show_lines=False,
            pad_edge=False,
            title_justify="left",
        )
        t.add_column(index_col, style="dim", no_wrap=True, min_width=6)
        short_names = []
        for c in cols:
            short = c[len(category) + 1:] if c.startswith(category + "/") else c
            short_names.append(short)
            t.add_column(short, justify="right", no_wrap=True)

        for _, row in subset.iterrows():
            idx = f"{int(row[index_col])}" if not math.isnan(float(row[index_col])) else "?"
            values = [_fmt(row[c]) for c in cols]
            t.add_row(idx, *values)

        console.print(t)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch W&B run info and metrics.")
    parser.add_argument("run_id", help="W&B run ID (e.g. abc123xy)")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Maximum number of raw history rows to fetch before epoch-aggregation (default: unlimited)",
    )
    args = parser.parse_args()

    run = fetch_run(args.run_id)

    # Render into a string via a file console
    out_path = Path(".temp/experiments") / f"{args.run_id}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        file_console = Console(file=f, width=500, highlight=False, markup=True, no_color=True)
        render_run_info(run, file_console)
        build_metric_tables(run, max_rows=args.max_rows, console=file_console)

    print(f"Saved in {out_path}")


if __name__ == "__main__":
    main()
