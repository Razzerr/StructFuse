import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pathlib import Path
from typing import Optional


def plot_contact_map_comparison(
    pred_prob: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    seq: str,
    pid: str,
    threshold: float = 0.5,
    save_path: Optional[Path] = None,
    show: bool = False,
) -> Figure:
    """
    Plot 2x2 grid comparison of predicted and ground truth contact maps.

    Args:
        pred_prob: (L, L) predicted contact probabilities [0, 1]
        target: (L, L) ground truth contact map {0, 1}
        mask: (L, L) valid region mask
        seq: Protein sequence (length L)
        pid: Protein ID for title
        threshold: Threshold for binary prediction (default: 0.5)
        save_path: Optional path to save figure
        show: Whether to display figure

    Returns:
        matplotlib Figure object
    """
    # Convert to numpy
    pred_np = pred_prob.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()

    # Binary prediction
    pred_binary = (pred_np > threshold).astype(np.float32)

    # Apply mask (set invalid regions to NaN for cleaner visualization)
    pred_masked = pred_np.copy()
    target_masked = target_np.copy()
    pred_binary_masked = pred_binary.copy()
    pred_masked[mask_np == 0] = np.nan
    target_masked[mask_np == 0] = np.nan
    pred_binary_masked[mask_np == 0] = np.nan

    # Difference: ground truth - binary prediction
    diff = target_masked - pred_binary_masked

    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    L = len(seq)
    extent = [0, L, 0, L]  # For imshow orientation (lower-left origin)

    # (0, 0): Ground truth
    im00 = axes[0, 0].imshow(
        target_masked, cmap="Blues", vmin=0, vmax=1, extent=extent, origin="lower"
    )
    axes[0, 0].set_title("Ground Truth", fontsize=12)
    axes[0, 0].set_xlabel("Residue position", fontsize=11)
    axes[0, 0].set_ylabel("Residue position", fontsize=11)
    axes[0, 0].plot([0, L], [0, L], "k--", alpha=0.3, linewidth=1)  # Diagonal
    plt.colorbar(im00, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # (0, 1): Prediction probabilities
    im01 = axes[0, 1].imshow(
        pred_masked, cmap="Blues", vmin=0, vmax=1, extent=extent, origin="lower"
    )
    axes[0, 1].set_title("Prediction (Probabilities)", fontsize=12)
    axes[0, 1].set_xlabel("Residue position", fontsize=11)
    axes[0, 1].set_ylabel("Residue position", fontsize=11)
    axes[0, 1].plot([0, L], [0, L], "k--", alpha=0.3, linewidth=1)
    plt.colorbar(im01, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # (1, 0): Binary prediction
    im10 = axes[1, 0].imshow(
        pred_binary_masked, cmap="Blues", vmin=0, vmax=1, extent=extent, origin="lower"
    )
    axes[1, 0].set_title(f"Binary Prediction (threshold={threshold:.3f})", fontsize=12)
    axes[1, 0].set_xlabel("Residue position", fontsize=11)
    axes[1, 0].set_ylabel("Residue position", fontsize=11)
    axes[1, 0].plot([0, L], [0, L], "k--", alpha=0.3, linewidth=1)
    plt.colorbar(im10, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # (1, 1): Difference (Ground Truth - Binary Prediction)
    im11 = axes[1, 1].imshow(
        diff, cmap="RdBu_r", vmin=-1, vmax=1, extent=extent, origin="lower"
    )
    axes[1, 1].set_title("Difference (Truth - Pred)\nRed=FN, Blue=FP", fontsize=12)
    axes[1, 1].set_xlabel("Residue position", fontsize=11)
    axes[1, 1].set_ylabel("Residue position", fontsize=11)
    axes[1, 1].plot([0, L], [0, L], "k--", alpha=0.3, linewidth=1)
    plt.colorbar(im11, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # Add sequence ticks (every 10 residues)
    tick_positions = list(range(0, L, max(1, L // 10)))
    tick_labels = [f"{seq[i]}{i+1}" if i < len(seq) else "" for i in tick_positions]

    for ax in axes.flat:
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels, fontsize=9)
        ax.set_aspect("equal")

    # Add overall title with metrics
    # Compute P@L, P@L/2, P@L/5 (same as training metrics)
    from src.models.utils.metrics import precision_at_k_masked

    # Convert back to torch for metric computation (add batch dimension)
    pred_torch = torch.from_numpy(pred_np).unsqueeze(0)  # (1, L, L)
    target_torch = torch.from_numpy(target_np).unsqueeze(0)  # (1, L, L)
    mask_torch = torch.from_numpy(mask_np).unsqueeze(0)  # (1, L, L)

    # Compute P@L metrics
    pL = precision_at_k_masked(pred_torch, target_torch, mask_torch, k_mode="L")
    pL2 = precision_at_k_masked(pred_torch, target_torch, mask_torch, k_mode="L/2")
    pL5 = precision_at_k_masked(pred_torch, target_torch, mask_torch, k_mode="L/5")

    # Compute precision, recall, F1 at the given threshold
    valid_pred = pred_binary[mask_np > 0]
    valid_target = target_np[mask_np > 0]

    tp = np.sum((valid_pred == 1) & (valid_target == 1))
    fp = np.sum((valid_pred == 1) & (valid_target == 0))
    fn = np.sum((valid_pred == 0) & (valid_target == 1))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # Also show simple counts
    num_true = np.nansum(target_masked)
    num_pred = np.nansum(pred_binary_masked)

    # Combined title with automatic spacing
    title_line1 = (
        f"True contacts: {int(num_true)} | Binary pred: {int(num_pred)} | "
        f"P@L={pL:.3f} | P@L/2={pL2:.3f} | P@L/5={pL5:.3f}"
    )
    title_line2 = f"Precision={precision:.3f} | Recall={recall:.3f} | F1={f1:.3f}"

    fig.suptitle(
        pid,
        fontsize=14,
        style="normal",
        fontweight='bold',
        y=0.97
    )
    fig.text(0.5,
        0.95,
        f"{title_line1}\n{title_line2}",
        ha='center', 
        va='top',
        fontsize=11
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            save_path, dpi=100, bbox_inches="tight"
        ) 
        print(f"Saved visualization to {save_path}")

    if not show:
        plt.close(fig)

    return fig


def plot_contact_map_with_sequence(
    contact_map: torch.Tensor,
    seq: str,
    pid: str,
    mask: Optional[torch.Tensor] = None,
    title: str = "Contact Map",
    figsize: tuple = (10, 10),
    cmap: str = "Blues",
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> Figure:
    """
    Plot a single contact map with sequence labels on axes.

    Args:
        contact_map: (L, L) contact probabilities or binary map
        seq: protein sequence string (length L)
        pid: protein ID for title
        mask: optional (L, L) mask to apply
        title: plot title
        figsize: figure size
        cmap: colormap name
        vmin, vmax: color scale limits

    Returns:
        matplotlib Figure object
    """
    contact_map = contact_map.detach().cpu().numpy()

    if mask is not None:
        mask = mask.detach().cpu().numpy()
        contact_map = np.where(mask > 0, contact_map, np.nan)

    L = len(seq)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(contact_map, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
    ax.set_title(f"{title}\n{pid} (L={L})")
    ax.set_xlabel("Residue position")
    ax.set_ylabel("Residue position")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Add residue sequence as tick labels
    if L <= 50:
        tick_spacing = 5
    elif L <= 150:
        tick_spacing = 10
    else:
        tick_spacing = 20

    tick_positions = list(range(0, L, tick_spacing))
    tick_labels = [f"{seq[i]}{i+1}" for i in tick_positions]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=8)
    ax.set_aspect("equal")

    plt.tight_layout()
    return fig


def plot_precision_recall_curve(
    pred_prob: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    pid: str,
    figsize: tuple = (7, 7),  # Square figure
) -> Figure:
    """
    Plot precision-recall curve for contact prediction.

    Args:
        pred_prob: (L, L) predicted contact probabilities
        target: (L, L) ground truth binary contacts
        mask: (L, L) valid pair mask
        pid: protein ID for title
        figsize: figure size

    Returns:
        matplotlib Figure object
    """
    pred_prob = pred_prob.detach().cpu().numpy().flatten()
    target = target.detach().cpu().numpy().flatten()
    mask = mask.detach().cpu().numpy().flatten()

    # Filter to valid pairs
    valid = mask > 0
    pred_prob = pred_prob[valid]
    target = target[valid]

    # Sort by prediction confidence (descending)
    sorted_idx = np.argsort(-pred_prob)
    pred_sorted = pred_prob[sorted_idx]
    target_sorted = target[sorted_idx]

    # Compute cumulative precision and recall
    n_positives = target.sum()
    if n_positives == 0:
        # No positives, can't compute meaningful PR curve
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No positive contacts", ha="center", va="center")
        ax.set_title(f"Precision-Recall\n{pid}")
        return fig

    tp_cumsum = np.cumsum(target_sorted)
    n_predicted = np.arange(1, len(target_sorted) + 1)

    precision = tp_cumsum / n_predicted
    recall = tp_cumsum / n_positives

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(recall, precision, linewidth=2)
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"Precision-Recall Curve\n{pid}", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect("equal", adjustable="box")  # Force square aspect ratio

    # Mark L, L/2, L/5 points
    L = int(np.sqrt(len(pred_prob) * 2))  # approximate sequence length
    for k, label in [(L, "L"), (L // 2, "L/2"), (L // 5, "L/5")]:
        if k < len(precision):
            ax.plot(recall[k], precision[k], "ro", markersize=8)
            ax.text(recall[k], precision[k], f" {label}", fontsize=10, va="bottom")

    plt.tight_layout()
    return fig
