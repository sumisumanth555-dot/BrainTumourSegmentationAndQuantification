import os
import csv
import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def compute_tissue_quantification(mask: np.ndarray) -> Dict[str, float]:
    """
    Compute percentage of each tumor sub-region relative to all non-background voxels.

    Classes: 0 = background, 1 = edema, 2 = necrotic core, 3 = enhancing tumor.
    Returns a dict with keys: 'Edema', 'Necrotic core', 'Enhancing tumor', 'Total lesion load'.
    """
    if mask is None:
        raise ValueError("mask is None")
    if mask.ndim != 3:
        raise ValueError("mask must be a 3D array")

    counts = {cls: int(np.count_nonzero(mask == cls)) for cls in (0, 1, 2, 3)}
    non_bg = counts[1] + counts[2] + counts[3]
    if non_bg == 0:
        return {
            "Edema": 0.0,
            "Necrotic core": 0.0,
            "Enhancing tumor": 0.0,
            "Total lesion load": 0.0,
        }

    edema_pct = 100.0 * counts[1] / non_bg
    necrotic_pct = 100.0 * counts[2] / non_bg
    enhancing_pct = 100.0 * counts[3] / non_bg
    total_pct = 100.0 * (counts[1] + counts[2] + counts[3]) / non_bg

    return {
        "Edema": edema_pct,
        "Necrotic core": necrotic_pct,
        "Enhancing tumor": enhancing_pct,
        "Total lesion load": total_pct,
    }


def quantification_dataframe(mask: np.ndarray) -> pd.DataFrame:
    metrics = compute_tissue_quantification(mask)
    rows = [
        {"Region": key, "Percent": float(value)}
        for key, value in metrics.items()
        if key != "Total lesion load"
    ]
    return pd.DataFrame(rows, columns=["Region", "Percent"])


def _make_rgba_overlay(mask_slice: np.ndarray, opacity: float) -> np.ndarray:
    """Create an RGBA overlay for a 2D mask slice with specified opacity."""
    if mask_slice.ndim != 2:
        raise ValueError("mask_slice must be 2D")
    h, w = mask_slice.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    alpha = int(max(0.0, min(1.0, float(opacity))) * 255)

    # Class colors: 1 cyan, 2 yellow, 3 red
    # Edema (1): cyan (0,255,255)
    m1 = mask_slice == 1
    rgba[m1, 0] = 0
    rgba[m1, 1] = 255
    rgba[m1, 2] = 255
    rgba[m1, 3] = alpha

    # Necrotic core (2): yellow (255,255,0)
    m2 = mask_slice == 2
    rgba[m2, 0] = 255
    rgba[m2, 1] = 255
    rgba[m2, 2] = 0
    rgba[m2, 3] = alpha

    # Enhancing tumor (3): red (255,0,0)
    m3 = mask_slice == 3
    rgba[m3, 0] = 255
    rgba[m3, 1] = 0
    rgba[m3, 2] = 0
    rgba[m3, 3] = alpha

    return rgba


def overlay_segmentation(image_volume: np.ndarray, mask_volume: np.ndarray, slice_index: Optional[int] = None, opacity: float = 0.45) -> go.Figure:
    """
    Create an overlay figure (2D slice) of image and segmentation mask.
    - image_volume: 3D or 4D (use first channel if 4D)
    - mask_volume: 3D (same spatial dims as image)
    - slice: axial slice index (defaults to center)
    - colormap: 1 cyan, 2 yellow, 3 red
    """
    if image_volume.ndim == 4:
        image_volume = image_volume[..., 0]
    if image_volume.ndim != 3:
        raise ValueError("image_volume must be 3D or 4D")
    if mask_volume.ndim != 3:
        raise ValueError("mask_volume must be 3D")

    if image_volume.shape != mask_volume.shape:
        raise ValueError("image_volume and mask_volume must have the same shape")

    depth = image_volume.shape[0]
    if slice_index is None:
        slice_index = depth // 2
    slice_index = int(max(0, min(depth - 1, slice_index)))

    img_slice = image_volume[slice_index]
    # Normalize to 0-255 for visualization
    img_min = float(np.min(img_slice))
    img_max = float(np.max(img_slice))
    if img_max > img_min:
        base = ((img_slice - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)
    else:
        base = np.zeros_like(img_slice, dtype=np.uint8)
    base_rgb = np.stack([base, base, base], axis=-1)

    overlay_rgba = _make_rgba_overlay(mask_volume[slice_index], opacity=opacity)

    fig = go.Figure()
    fig.add_trace(go.Image(z=base_rgb))
    fig.add_trace(go.Image(z=overlay_rgba))
    fig.update_layout(
        title="Segmentation Overlay (Axial slice)",
        template="plotly_dark",
        width=600,
        height=600,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    return fig


def bar_chart_from_metrics(metrics: Dict[str, float]) -> go.Figure:
    categories = [k for k in ("Edema", "Necrotic core", "Enhancing tumor")]
    values = [float(metrics.get(k, 0.0)) for k in categories]
    fig = go.Figure(
        data=[go.Bar(x=categories, y=values, marker_color=["cyan", "yellow", "red"])],
        layout=go.Layout(template="plotly_dark", title="Tissue Quantification (%)", yaxis_title="Percent")
    )
    return fig


def pie_chart_from_metrics(metrics: Dict[str, float]) -> go.Figure:
    labels = [k for k in ("Edema", "Necrotic core", "Enhancing tumor")]
    values = [float(metrics.get(k, 0.0)) for k in labels]
    fig = go.Figure(
        data=[go.Pie(labels=labels, values=values, marker=dict(colors=["cyan", "yellow", "red"]))],
        layout=go.Layout(template="plotly_dark", title="Tissue Composition")
    )
    return fig


def save_quantification_csv(metrics: Dict[str, float], out_path: str, extra_context: Optional[Dict[str, str]] = None) -> None:
    """Append quantification results to CSV, creating file with header if needed."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    row = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        **({} if extra_context is None else extra_context),
        **{k: float(v) for k, v in metrics.items()},
    }

    file_exists = os.path.exists(out_path)
    with open(out_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


