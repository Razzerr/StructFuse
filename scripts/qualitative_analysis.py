#!/usr/bin/env python3
"""
Qualitative analysis: Find rescue cases where StructFuse outperforms ESM2-only.

Identifies proteins where:
- ESM2-only baseline: F1 < 0.3 (failed)
- StructFuse: F1 > 0.6 (success)

These "rescue cases" show the value of template retrieval.

Output:
- TSV file with protein IDs, metrics, and retrieval statistics
- PyMOL session files for visualization (optional)

Usage:
    python scripts/qualitative_analysis.py \
        --esm2_results results/esm2_only_metrics.tsv \
        --structfuse_results results/structfuse_metrics.tsv \
        --data_dir data/output_splits \
        --output_dir results/qualitative \
        --esm2_threshold 0.3 \
        --structfuse_threshold 0.6
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Find rescue cases where StructFuse outperforms ESM2-only"
    )
    parser.add_argument(
        "--esm2_results",
        type=str,
        required=True,
        help="Path to ESM2-only per-sample metrics TSV",
    )
    parser.add_argument(
        "--structfuse_results",
        type=str,
        required=True,
        help="Path to StructFuse per-sample metrics TSV",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/output_splits",
        help="Directory with NPZ files (for additional info)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/qualitative",
        help="Output directory for results",
    )
    parser.add_argument(
        "--esm2_threshold",
        type=float,
        default=0.3,
        help="Maximum F1 for ESM2-only to be considered 'failed'",
    )
    parser.add_argument(
        "--structfuse_threshold",
        type=float,
        default=0.6,
        help="Minimum F1 for StructFuse to be considered 'success'",
    )
    parser.add_argument(
        "--generate_pymol",
        action="store_true",
        help="Generate PyMOL session files for top rescue cases",
    )
    parser.add_argument(
        "--pdb_dir",
        type=str,
        default=None,
        help="Directory with PDB/mmCIF files for PyMOL visualization",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top rescue cases to analyze in detail",
    )
    return parser.parse_args()


def load_metrics(path: str) -> pd.DataFrame:
    """Load per-sample metrics from TSV file."""
    df = pd.read_csv(path, sep="\t")
    return df


def find_rescue_cases(
    esm2_df: pd.DataFrame,
    structfuse_df: pd.DataFrame,
    esm2_threshold: float,
    structfuse_threshold: float,
) -> pd.DataFrame:
    """Find samples where ESM2 failed but StructFuse succeeded."""
    
    # Merge on sample_id
    merged = esm2_df.merge(
        structfuse_df,
        on="sample_id",
        suffixes=("_esm2", "_sf"),
    )
    
    # Filter rescue cases
    rescue = merged[
        (merged["f1_esm2"] < esm2_threshold) & 
        (merged["f1_sf"] > structfuse_threshold)
    ].copy()
    
    # Calculate improvement
    rescue["f1_improvement"] = rescue["f1_sf"] - rescue["f1_esm2"]
    rescue["relative_improvement"] = (
        rescue["f1_improvement"] / rescue["f1_esm2"].clip(lower=0.01)
    )
    
    # Sort by improvement
    rescue = rescue.sort_values("f1_improvement", ascending=False)
    
    return rescue


def find_failure_cases(
    esm2_df: pd.DataFrame,
    structfuse_df: pd.DataFrame,
) -> pd.DataFrame:
    """Find cases where both methods failed (for error analysis)."""
    
    merged = esm2_df.merge(
        structfuse_df,
        on="sample_id",
        suffixes=("_esm2", "_sf"),
    )
    
    # Both methods have low F1
    failures = merged[
        (merged["f1_esm2"] < 0.3) & 
        (merged["f1_sf"] < 0.3)
    ].copy()
    
    return failures


def find_regression_cases(
    esm2_df: pd.DataFrame,
    structfuse_df: pd.DataFrame,
) -> pd.DataFrame:
    """Find cases where StructFuse performed worse than ESM2-only."""
    
    merged = esm2_df.merge(
        structfuse_df,
        on="sample_id",
        suffixes=("_esm2", "_sf"),
    )
    
    # StructFuse worse than ESM2
    regressions = merged[
        merged["f1_sf"] < merged["f1_esm2"] - 0.1  # At least 10% worse
    ].copy()
    
    regressions["f1_regression"] = regressions["f1_esm2"] - regressions["f1_sf"]
    regressions = regressions.sort_values("f1_regression", ascending=False)
    
    return regressions


def get_sample_info(sample_id: str, data_dir: Path) -> Dict:
    """Get additional information about a sample."""
    npz_path = data_dir / f"{sample_id}.npz"
    if not npz_path.exists():
        return {}
    
    data = np.load(npz_path, allow_pickle=True)
    info = {
        "seq_len": data["seq_emb"].shape[0] if "seq_emb" in data else None,
    }
    
    # Count contacts
    if "contact_map" in data:
        contact_map = data["contact_map"]
        n_contacts = np.sum(contact_map > 0) // 2  # Symmetric, count once
        info["n_contacts"] = n_contacts
        info["contact_density"] = n_contacts / (info["seq_len"] ** 2) if info["seq_len"] else None
    
    return info


def generate_pymol_script(
    sample_id: str,
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    output_path: Path,
    pdb_path: Optional[Path] = None,
) -> str:
    """Generate PyMOL script for contact visualization."""
    
    script = f"""# PyMOL script for {sample_id}
# Generated by qualitative_analysis.py

# Load structure
"""
    
    if pdb_path and pdb_path.exists():
        script += f"load {pdb_path}\n"
    else:
        script += f"# Structure file not found, please load manually\n"
        script += f"# fetch {sample_id.split('_')[0]}\n"
    
    script += """
# Visualization settings
hide everything
show cartoon
color gray80

# True positives (correct predictions) - green
"""
    
    # Find TP, FP, FN
    threshold = 0.5
    pred_binary = predictions > threshold
    
    tp_pairs = np.argwhere((pred_binary == 1) & (ground_truth == 1))
    fp_pairs = np.argwhere((pred_binary == 1) & (ground_truth == 0))
    fn_pairs = np.argwhere((pred_binary == 0) & (ground_truth == 1))
    
    # Limit number of pairs to visualize
    max_pairs = 50
    
    for i, (r1, r2) in enumerate(tp_pairs[:max_pairs]):
        if abs(r1 - r2) > 6:  # Skip short-range
            script += f"distance tp_{i}, resi {r1+1} and name CA, resi {r2+1} and name CA\n"
    
    script += "\n# False positives (incorrect predictions) - red\n"
    for i, (r1, r2) in enumerate(fp_pairs[:max_pairs]):
        if abs(r1 - r2) > 6:
            script += f"distance fp_{i}, resi {r1+1} and name CA, resi {r2+1} and name CA\n"
    
    script += "\n# False negatives (missed contacts) - yellow\n"
    for i, (r1, r2) in enumerate(fn_pairs[:max_pairs]):
        if abs(r1 - r2) > 6:
            script += f"distance fn_{i}, resi {r1+1} and name CA, resi {r2+1} and name CA\n"
    
    script += """
# Color distances
color green, tp_*
color red, fp_*
color yellow, fn_*

# Clean up display
hide labels
set dash_gap, 0.3
set dash_width, 2

# Save session
"""
    script += f"save {output_path.with_suffix('.pse')}\n"
    
    return script


def analyze_by_length_bins(rescue_df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Analyze rescue cases by sequence length bins."""
    
    length_data = []
    for _, row in rescue_df.iterrows():
        info = get_sample_info(row["sample_id"], data_dir)
        if info.get("seq_len"):
            length_data.append({
                "sample_id": row["sample_id"],
                "seq_len": info["seq_len"],
                "f1_improvement": row["f1_improvement"],
            })
    
    if not length_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(length_data)
    
    # Create length bins
    bins = [0, 100, 200, 300, 500, 1000, float("inf")]
    labels = ["<100", "100-200", "200-300", "300-500", "500-1000", ">1000"]
    df["length_bin"] = pd.cut(df["seq_len"], bins=bins, labels=labels)
    
    # Aggregate by bin
    summary = df.groupby("length_bin").agg({
        "sample_id": "count",
        "f1_improvement": ["mean", "std"],
    }).round(3)
    
    return summary


def main():
    args = parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metrics
    print("Loading metrics...")
    esm2_df = load_metrics(args.esm2_results)
    structfuse_df = load_metrics(args.structfuse_results)
    
    print(f"ESM2-only samples: {len(esm2_df)}")
    print(f"StructFuse samples: {len(structfuse_df)}")
    
    # Find rescue cases
    print("\nFinding rescue cases...")
    rescue_df = find_rescue_cases(
        esm2_df, structfuse_df,
        args.esm2_threshold, args.structfuse_threshold
    )
    print(f"Found {len(rescue_df)} rescue cases")
    
    # Save rescue cases
    rescue_path = output_dir / "rescue_cases.tsv"
    rescue_df.to_csv(rescue_path, sep="\t", index=False)
    print(f"Saved to {rescue_path}")
    
    # Find failure cases (both methods fail)
    print("\nFinding failure cases (both methods fail)...")
    failure_df = find_failure_cases(esm2_df, structfuse_df)
    print(f"Found {len(failure_df)} failure cases")
    failure_path = output_dir / "failure_cases.tsv"
    failure_df.to_csv(failure_path, sep="\t", index=False)
    
    # Find regression cases
    print("\nFinding regression cases (StructFuse worse than ESM2)...")
    regression_df = find_regression_cases(esm2_df, structfuse_df)
    print(f"Found {len(regression_df)} regression cases")
    regression_path = output_dir / "regression_cases.tsv"
    regression_df.to_csv(regression_path, sep="\t", index=False)
    
    # Analyze by length
    data_dir = Path(args.data_dir)
    if len(rescue_df) > 0:
        print("\nAnalyzing rescue cases by length...")
        length_analysis = analyze_by_length_bins(rescue_df, data_dir)
        if not length_analysis.empty:
            print(length_analysis)
            length_analysis.to_csv(output_dir / "rescue_by_length.tsv", sep="\t")
    
    # Top rescue cases detailed analysis
    print(f"\n=== Top {args.top_k} Rescue Cases ===")
    top_rescue = rescue_df.head(args.top_k)
    for _, row in top_rescue.iterrows():
        info = get_sample_info(row["sample_id"], data_dir)
        print(f"\n{row['sample_id']}:")
        print(f"  ESM2-only F1: {row['f1_esm2']:.3f}")
        print(f"  StructFuse F1: {row['f1_sf']:.3f}")
        print(f"  Improvement: +{row['f1_improvement']:.3f}")
        if info:
            print(f"  Sequence length: {info.get('seq_len', 'N/A')}")
            print(f"  Contact density: {info.get('contact_density', 'N/A'):.4f}" 
                  if info.get('contact_density') else "")
    
    # Summary statistics
    print("\n=== Summary Statistics ===")
    merged = esm2_df.merge(structfuse_df, on="sample_id", suffixes=("_esm2", "_sf"))
    
    n_improved = len(merged[merged["f1_sf"] > merged["f1_esm2"]])
    n_worse = len(merged[merged["f1_sf"] < merged["f1_esm2"]])
    n_same = len(merged[merged["f1_sf"] == merged["f1_esm2"]])
    
    print(f"Total samples: {len(merged)}")
    print(f"StructFuse better: {n_improved} ({100*n_improved/len(merged):.1f}%)")
    print(f"StructFuse worse: {n_worse} ({100*n_worse/len(merged):.1f}%)")
    print(f"Same performance: {n_same} ({100*n_same/len(merged):.1f}%)")
    print(f"\nRescue cases (ESM2 F1<{args.esm2_threshold}, SF F1>{args.structfuse_threshold}): {len(rescue_df)}")
    
    avg_improvement = (merged["f1_sf"] - merged["f1_esm2"]).mean()
    print(f"Average F1 improvement: {avg_improvement:+.3f}")
    
    # Save summary
    summary = {
        "total_samples": len(merged),
        "n_improved": n_improved,
        "n_worse": n_worse,
        "n_rescue": len(rescue_df),
        "n_failure": len(failure_df),
        "n_regression": len(regression_df),
        "avg_f1_improvement": float(avg_improvement),
        "esm2_mean_f1": float(merged["f1_esm2"].mean()),
        "structfuse_mean_f1": float(merged["f1_sf"].mean()),
    }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
