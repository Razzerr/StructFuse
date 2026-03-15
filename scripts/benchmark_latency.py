#!/usr/bin/env python3
"""
Latency benchmark for StructFuse model.

Measures timing breakdown:
- t_retrieval: FAISS template retrieval time
- t_forward: Model forward pass time  
- t_total: End-to-end inference time

Output: TSV file with per-sample timings for accuracy vs speed analysis.

Usage:
    python scripts/benchmark_latency.py \
        --checkpoint path/to/checkpoint.ckpt \
        --data_dir data/output_splits \
        --index_dir data/index_t6 \
        --output_file results/latency_benchmark.tsv \
        --num_samples 100 \
        --warmup_runs 5
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

# Ensure proper imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.contact_lit_module import ContactLitModule
from src.models.utils.faiss import FaissIndex


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark StructFuse latency")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/output_splits",
        help="Directory with NPZ files",
    )
    parser.add_argument(
        "--index_dir",
        type=str,
        default="data/index_t6",
        help="Directory with FAISS index",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="results/latency_benchmark.tsv",
        help="Output TSV file with timings",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to benchmark",
    )
    parser.add_argument(
        "--warmup_runs",
        type=int,
        default=5,
        help="Number of warmup runs before timing",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=4,
        help="Number of templates to retrieve",
    )
    return parser.parse_args()


def load_model(checkpoint_path: str, device: str) -> ContactLitModule:
    """Load model from checkpoint."""
    model = ContactLitModule.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    model.to(device)
    return model


def load_sample(npz_path: Path) -> Dict[str, torch.Tensor]:
    """Load a single sample from NPZ file."""
    data = np.load(npz_path, allow_pickle=True)
    return {
        "seq_emb": torch.from_numpy(data["seq_emb"]).float(),
        "sequence": str(data["sequence"]) if "sequence" in data else None,
        "seq_len": data["seq_emb"].shape[0],
    }


def benchmark_retrieval(
    index: FaissIndex,
    query_emb: np.ndarray,
    topk: int,
    num_runs: int = 10,
) -> Tuple[float, float]:
    """Benchmark FAISS retrieval time."""
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        _ = index.topk(query_emb, k=topk)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        times.append(time.perf_counter() - start)
    return np.mean(times), np.std(times)


def benchmark_forward(
    model: ContactLitModule,
    batch: Dict[str, torch.Tensor],
    device: str,
    num_runs: int = 10,
) -> Tuple[float, float]:
    """Benchmark model forward pass time."""
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(batch)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        times.append(time.perf_counter() - start)
    return np.mean(times), np.std(times)


def benchmark_e2e(
    model: ContactLitModule,
    index: FaissIndex,
    sample: Dict[str, torch.Tensor],
    topk: int,
    device: str,
    num_runs: int = 10,
) -> Tuple[float, float]:
    """Benchmark end-to-end inference time (retrieval + forward)."""
    times = []
    seq_emb = sample["seq_emb"].numpy()
    
    for _ in range(num_runs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        
        # Retrieval
        hits = index.topk(seq_emb, k=topk)
        
        # Prepare batch and forward
        batch = {
            "seq_emb": sample["seq_emb"].unsqueeze(0).to(device),
            # Add other required fields based on model
        }
        with torch.no_grad():
            _ = model.model(batch["seq_emb"])  # Simplified forward
            
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        times.append(time.perf_counter() - start)
    
    return np.mean(times), np.std(times)


def main():
    args = parse_args()
    
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, args.device)
    
    print(f"Loading FAISS index from {args.index_dir}...")
    index = FaissIndex(args.index_dir)
    
    # Find NPZ files
    data_dir = Path(args.data_dir)
    npz_files = list(data_dir.glob("*.npz"))[:args.num_samples]
    print(f"Found {len(npz_files)} samples to benchmark")
    
    if len(npz_files) == 0:
        print("No NPZ files found!")
        return
    
    # Warmup runs
    print(f"Running {args.warmup_runs} warmup iterations...")
    sample = load_sample(npz_files[0])
    seq_emb = sample["seq_emb"].numpy()
    
    for _ in range(args.warmup_runs):
        _ = index.topk(seq_emb, k=args.topk)
        batch = {"seq_emb": sample["seq_emb"].unsqueeze(0).to(args.device)}
        with torch.no_grad():
            _ = model.model(batch["seq_emb"])
    
    # Benchmark each sample
    print("Benchmarking...")
    results: List[Dict] = []
    
    for npz_file in npz_files:
        sample = load_sample(npz_file)
        seq_len = sample["seq_len"]
        seq_emb = sample["seq_emb"].numpy()
        
        # Measure retrieval time
        t_retrieval_mean, t_retrieval_std = benchmark_retrieval(
            index, seq_emb, args.topk, num_runs=5
        )
        
        # Measure forward time
        batch = {"seq_emb": sample["seq_emb"].unsqueeze(0).to(args.device)}
        t_forward_mean, t_forward_std = benchmark_forward(
            model, batch, args.device, num_runs=5
        )
        
        # Total time
        t_total = t_retrieval_mean + t_forward_mean
        
        results.append({
            "sample_id": npz_file.stem,
            "seq_len": seq_len,
            "t_retrieval_ms": t_retrieval_mean * 1000,
            "t_retrieval_std_ms": t_retrieval_std * 1000,
            "t_forward_ms": t_forward_mean * 1000,
            "t_forward_std_ms": t_forward_std * 1000,
            "t_total_ms": t_total * 1000,
        })
        
        print(f"  {npz_file.stem}: L={seq_len}, "
              f"retrieval={t_retrieval_mean*1000:.2f}ms, "
              f"forward={t_forward_mean*1000:.2f}ms, "
              f"total={t_total*1000:.2f}ms")
    
    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        # Header
        f.write("sample_id\tseq_len\tt_retrieval_ms\tt_retrieval_std_ms\t"
                "t_forward_ms\tt_forward_std_ms\tt_total_ms\n")
        # Data
        for r in results:
            f.write(f"{r['sample_id']}\t{r['seq_len']}\t"
                    f"{r['t_retrieval_ms']:.3f}\t{r['t_retrieval_std_ms']:.3f}\t"
                    f"{r['t_forward_ms']:.3f}\t{r['t_forward_std_ms']:.3f}\t"
                    f"{r['t_total_ms']:.3f}\n")
    
    print(f"\nResults saved to {output_path}")
    
    # Summary statistics
    all_retrieval = [r["t_retrieval_ms"] for r in results]
    all_forward = [r["t_forward_ms"] for r in results]
    all_total = [r["t_total_ms"] for r in results]
    
    print("\n=== Summary ===")
    print(f"Retrieval: {np.mean(all_retrieval):.2f} ± {np.std(all_retrieval):.2f} ms")
    print(f"Forward:   {np.mean(all_forward):.2f} ± {np.std(all_forward):.2f} ms")
    print(f"Total:     {np.mean(all_total):.2f} ± {np.std(all_total):.2f} ms")
    print(f"Throughput: {1000 / np.mean(all_total):.1f} samples/sec")


if __name__ == "__main__":
    main()
