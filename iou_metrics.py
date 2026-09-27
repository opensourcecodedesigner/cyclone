"""
===============================================================================
AEGIS: TRUE GRID-BASED SPATIAL IOU EVALUATION ENGINE
===============================================================================
Replaces hard-coded static polygon IoU calculations with a true grid-level
raster evaluation pipeline against Copernicus EMSR357 / Sentinel-1 ground truth.

Features:
1. Ingests simulated 2D water depth matrices from the Julia Cellular Automata solver.
2. Applies a physical depth threshold (e.g. depth > 0.15m) to produce a dynamic boolean
   prediction mask of flooded cells across the 100x100 domain.
3. Ingests Sentinel-1 / Copernicus EMS EMSR357 radar delineation, projecting it onto
   the exact matching 100x100 spatial domain grid.
4. Computes true mathematical Intersection-over-Union (Jaccard Index), Spatial Recall
   (True Positive Rate), Precision, and F1-Score over individual grid cells.
===============================================================================
"""

import os
import json
import numpy as np
from typing import Tuple, Dict, Any, Optional

try:
    from shapely.geometry import shape, Point, Polygon, MultiPolygon
    from shapely.prepared import prep
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GT_GEOJSON = os.path.join(BASE_DIR, "fani_ground_truth_flood_extent.geojson")
CACHED_GT_MASK_PATH = os.path.join(BASE_DIR, "perception_cache", "sentinel_ground_truth_mask_100x100.npy")

# Spatial Bounds for the Puri Littoral Domain (Matches app.py & backtest coordinates)
PURI_BOUNDS = {
    "lat_min": 19.730,
    "lat_max": 19.860,
    "lon_min": 85.730,
    "lon_max": 85.930,
    "cell_resolution_m": 30.0
}


def generate_prediction_mask(
    water_depth_grid: np.ndarray,
    flood_threshold: float = 0.15
) -> np.ndarray:
    """
    Converts a continuous 2D simulated water depth matrix into a boolean inundation mask.
    Args:
        water_depth_grid: 2D numpy array of depths in meters (e.g. 100x100)
        flood_threshold: Critical depth threshold (meters) for cell to be flooded (default 0.15m)
    Returns:
        Boolean 2D numpy array of shape matching water_depth_grid
    """
    if isinstance(water_depth_grid, list):
        water_depth_grid = np.array(water_depth_grid, dtype=np.float64)

    if water_depth_grid.ndim != 2:
        raise ValueError(f"Expected 2D water depth grid, got {water_depth_grid.ndim}D array")

    return water_depth_grid >= flood_threshold


def load_sentinel_ground_truth_mask(
    source_path: Optional[str] = None,
    grid_shape: Tuple[int, int] = (100, 100),
    force_recompute: bool = False
) -> np.ndarray:
    """
    Loads or rasterizes the Copernicus EMS EMSR357 / Sentinel-1 SAR ground truth
    onto the matching 2D simulation grid.

    Returns:
        Boolean 2D numpy array of shape grid_shape.
    """
    # 1. Use pre-rasterized cache if available and not recomputing
    if not force_recompute and source_path is None and os.path.exists(CACHED_GT_MASK_PATH):
        try:
            cached = np.load(CACHED_GT_MASK_PATH)
            if cached.shape == grid_shape and cached.dtype == bool:
                return cached
        except Exception:
            pass

    # 2. Check if source path is already a numpy file
    if source_path and source_path.endswith(".npy") and os.path.exists(source_path):
        data = np.load(source_path)
        if data.shape == grid_shape:
            return data.astype(bool)

    # 3. Rasterize from GeoJSON ground truth
    geojson_file = source_path if source_path and source_path.endswith(".geojson") else DEFAULT_GT_GEOJSON
    if not os.path.exists(geojson_file):
        raise FileNotFoundError(f"Ground truth GeoJSON file not found: {geojson_file}")

    if not HAS_SHAPELY:
        raise RuntimeError("Shapely is required to rasterize GeoJSON ground truth to grid.")

    with open(geojson_file, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    # Extract polygon from features
    features = gt_data.get("features", [])
    if not features:
        raise ValueError("GeoJSON feature collection is empty")

    gt_geom = shape(features[0]["geometry"])
    prep_geom = prep(gt_geom)

    nx, ny = grid_shape
    # Coordinate orientation for the Puri coastal sector:
    # y (0..99) is along-coast: SW (19.740, 85.740) to NE (19.850, 85.910)
    # x (0..99) is cross-shore inland distance: 0 = shoreline, 99 = ~3.5 km inland (NW direction)
    p_sw = np.array([85.740, 19.740])
    p_ne = np.array([85.910, 19.850])
    along_vec = p_ne - p_sw

    # Inland normal vector (pointing North-West perpendicular to coast)
    norm_vec = np.array([-along_vec[1], along_vec[0]])
    norm_vec = norm_vec / np.linalg.norm(norm_vec) * 0.035  # ~3.5 km domain depth

    gt_mask = np.zeros(grid_shape, dtype=bool)

    # Fast point-in-polygon containment
    for i in range(nx):
        u_inland = i / float(nx - 1)
        for j in range(ny):
            v_along = j / float(ny - 1)
            pt_coords = p_sw + v_along * along_vec + u_inland * norm_vec
            pt = Point(pt_coords[0], pt_coords[1])
            if prep_geom.contains(pt):
                gt_mask[i, j] = True

    # Cache for subsequent runs
    os.makedirs(os.path.dirname(CACHED_GT_MASK_PATH), exist_ok=True)
    try:
        np.save(CACHED_GT_MASK_PATH, gt_mask)
    except Exception:
        pass

    return gt_mask


def calculate_spatial_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    cell_resolution_m: float = 30.0
) -> Dict[str, Any]:
    """
    Computes rigorous cell-level spatial overlap metrics between predicted
    simulation flood mask and true Sentinel-1 / radar ground truth mask.
    """
    if pred_mask.shape != gt_mask.shape:
        raise ValueError(f"Shape mismatch: pred_mask {pred_mask.shape} vs gt_mask {gt_mask.shape}")

    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())

    total_cells = int(pred.size)
    intersection_cells = tp
    union_cells = tp + fp + fn

    iou = float(tp / union_cells) if union_cells > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1_score = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
    accuracy = float((tp + tn) / total_cells)

    # Geographic area equivalents (km2)
    cell_area_km2 = (cell_resolution_m ** 2) / 1e6
    pred_area_km2 = round(pred.sum() * cell_area_km2, 2)
    gt_area_km2 = round(gt.sum() * cell_area_km2, 2)
    intersection_km2 = round(tp * cell_area_km2, 2)
    union_km2 = round(union_cells * cell_area_km2, 2)

    return {
        "intersection_over_union_pct": round(iou * 100.0, 2),
        "overlap_recall_pct": round(recall * 100.0, 2),
        "precision_pct": round(precision * 100.0, 2),
        "f1_score_pct": round(f1_score * 100.0, 2),
        "accuracy_pct": round(accuracy * 100.0, 2),
        "confusion_matrix": {
            "true_positives_cells": tp,
            "false_positives_cells": fp,
            "false_negatives_cells": fn,
            "true_negatives_cells": tn,
            "total_cells": total_cells
        },
        "spatial_extents_km2": {
            "predicted_flood_area": pred_area_km2,
            "ground_truth_flood_area": gt_area_km2,
            "intersection_area": intersection_km2,
            "union_area": union_km2
        }
    }


def evaluate_simulation_iou(
    water_depth_grid: np.ndarray,
    ground_truth_source: Optional[str] = None,
    flood_threshold: float = 0.15,
    cell_resolution_m: float = 30.0
) -> Dict[str, Any]:
    """
    End-to-end evaluation: Takes raw Julia water depth grid, computes prediction mask,
    loads ground truth, and calculates authentic spatial IoU metrics.
    """
    pred_mask = generate_prediction_mask(water_depth_grid, flood_threshold=flood_threshold)
    gt_mask = load_sentinel_ground_truth_mask(source_path=ground_truth_source, grid_shape=pred_mask.shape)
    metrics = calculate_spatial_metrics(pred_mask, gt_mask, cell_resolution_m=cell_resolution_m)
    metrics["flood_depth_threshold_m"] = flood_threshold
    return metrics


if __name__ == "__main__":
    print("Testing true grid-based IoU evaluation engine...")
    gt = load_sentinel_ground_truth_mask()
    print(f"Loaded ground truth mask: shape {gt.shape}, flooded cells: {gt.sum()} ({gt.mean()*100:.1f}%)")

    # Mock test simulation grid
    mock_depth = np.zeros((100, 100), dtype=float)
    mock_depth[:25, :] = 1.2  # simulate 25 cells inland flood

    res = evaluate_simulation_iou(mock_depth, flood_threshold=0.15)
    print("Evaluation Result:")
    print(json.dumps(res, indent=2))
