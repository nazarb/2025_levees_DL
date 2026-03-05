from typing import Dict, Optional

import numpy as np
import rasterio
from scipy.ndimage import binary_dilation, binary_erosion


from typing import Dict, Optional, Union
from pathlib import Path
import geopandas as gpd
from shapely.ops import unary_union

def compare_linear_vectors(
    reference_path: Union[str, gpd.GeoDataFrame],
    predicted_path: Union[str, gpd.GeoDataFrame],
    buffer_distance: float,
    output_path: Optional[str] = None,
    crs: Optional[Union[str, int]] = None
) -> Dict[str, float]:
    """
    Compare linear features (e.g. levees, channels) by buffering BOTH reference
    and predicted lines, then computing overlap-based precision, recall, F1, IoU.
    
    Unlike compare_vectors (which buffers only the reference), this buffers both
    input geometries so area-based metrics are meaningful for line data.
    
    The comparison output contains polygons with the following classification:
    - 1: True Positive (TP) - overlap of both buffers
    - -1: False Positive (FP) - predicted buffer outside reference buffer
    - 2: False Negative (FN) - reference buffer outside predicted buffer
    
    Args:
        reference_path: Path to reference vector file, or a GeoDataFrame.
        predicted_path: Path to predicted vector file, or a GeoDataFrame.
        buffer_distance: Buffer distance in CRS units (meters for EPSG:3857).
        output_path: Optional path for comparison vector output.
        crs: Optional CRS to reproject to (e.g., 3857 for meters).
    
    Returns:
        Dictionary with precision, recall, f1_score, iou, tp_area, fp_area, fn_area.
    
    Example:
        >>> metrics = compare_linear_vectors(
        ...     reference_path="ground_truth.gpkg",
        ...     predicted_path="predicted_edges.gpkg",
        ...     buffer_distance=900,
        ...     crs=3857
        ... )
    """
    # Load vector data (file path or GeoDataFrame)
    if isinstance(reference_path, gpd.GeoDataFrame):
        ref = reference_path.copy()
    else:
        ref = gpd.read_file(reference_path)
    
    if isinstance(predicted_path, gpd.GeoDataFrame):
        pred = predicted_path.copy()
    else:
        pred = gpd.read_file(predicted_path)
    
    if len(ref) == 0 or ref.geometry.is_empty.all():
        raise ValueError("Reference vector is empty or has no valid geometry.")
    if len(pred) == 0 or pred.geometry.is_empty.all():
        raise ValueError("Predicted vector is empty or has no valid geometry.")
    
    target_crs = crs if crs else ref.crs
    ref = ref.to_crs(target_crs)
    pred = pred.to_crs(target_crs)
    
    ref_union = unary_union(ref.geometry)
    pred_union = unary_union(pred.geometry)
    
    # Buffer BOTH reference and predicted (critical for linear data)
    buffered_ref = ref_union.buffer(buffer_distance)
    buffered_pred = pred_union.buffer(buffer_distance)
    
    # Overlap-based areas
    tp_geom = buffered_ref.intersection(buffered_pred)
    fp_geom = buffered_pred.difference(buffered_ref)
    fn_geom = buffered_ref.difference(buffered_pred)
    
    tp_area = tp_geom.area if not tp_geom.is_empty else 0
    fp_area = fp_geom.area if not fp_geom.is_empty else 0
    fn_area = fn_geom.area if not fn_geom.is_empty else 0
    
    precision = tp_area / (tp_area + fp_area) if (tp_area + fp_area) > 0 else 0
    recall = tp_area / (tp_area + fn_area) if (tp_area + fn_area) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    iou = tp_area / (tp_area + fp_area + fn_area) if (tp_area + fp_area + fn_area) > 0 else 0
    
    print(f"\n--- Evaluation Metrics (Linear, buffer both) ---")
    print(f"Buffer distance: {buffer_distance} units")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"F1 Score:   {f1_score:.4f}")
    print(f"IoU:        {iou:.4f}")
    print(f"\nAreas (sq units):")
    print(f"  TP: {tp_area:,.2f}")
    print(f"  FP: {fp_area:,.2f}")
    print(f"  FN: {fn_area:,.2f}")
    
    comparison_records = []
    if not tp_geom.is_empty:
        comparison_records.append({"geometry": tp_geom, "class": 1, "label": "TP", "area": tp_area})
    if not fp_geom.is_empty:
        comparison_records.append({"geometry": fp_geom, "class": -1, "label": "FP", "area": fp_area})
    if not fn_geom.is_empty:
        comparison_records.append({"geometry": fn_geom, "class": 2, "label": "FN", "area": fn_area})
    
    comparison_gdf = gpd.GeoDataFrame(comparison_records, crs=target_crs)
    
    if output_path is None:
        if isinstance(predicted_path, str):
            pred_path = Path(predicted_path)
            output_path = str(pred_path.parent / f"{pred_path.stem}_linear_comp_buf{buffer_distance}.gpkg")
        else:
            output_path = f"linear_comparison_buf{buffer_distance}.gpkg"
    
    if 'fid' in comparison_gdf.columns:
        comparison_gdf = comparison_gdf.rename(columns={'fid': 'orig_fid'})
    
    comparison_gdf.to_file(output_path, driver="GPKG")
    print(f"\nComparison saved to: {output_path}")
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "iou": float(iou),
        "tp_area": float(tp_area),
        "fp_area": float(fp_area),
        "fn_area": float(fn_area)
    }
    
def compare_vectors(
    reference_path: Union[str, gpd.GeoDataFrame],
    predicted_path: Union[str, gpd.GeoDataFrame],
    buffer_distance: float,
    output_path: Optional[str] = None,
    crs: Optional[Union[str, int]] = None,
    pred_buffer: Optional[float] = None,
    ref_buffer: Optional[float] = None
) -> Dict[str, float]:
    """
    Compare predicted vector with reference vector and compute evaluation metrics.
    
    Args:
        reference_path: Path to reference vector file OR a GeoDataFrame.
        predicted_path: Path to predicted vector file OR a GeoDataFrame.
        buffer_distance: Tolerance buffer applied to reference for matching.
        output_path: Optional path for comparison vector output.
        crs: CRS to reproject to (use projected CRS like 3857 for accuracy).
        pred_buffer: Buffer to apply to predictions (required for points/lines).
        ref_buffer: Buffer to apply to reference (required for points/lines).
    
    Returns:
        Dictionary with precision, recall, f1_score, iou, and areas.
    """
    # Load or use GeoDataFrames directly
    if isinstance(reference_path, gpd.GeoDataFrame):
        ref = reference_path.copy()
    else:
        ref = gpd.read_file(reference_path)
    
    if isinstance(predicted_path, gpd.GeoDataFrame):
        pred = predicted_path.copy()
    else:
        pred = gpd.read_file(predicted_path)
    
    if len(ref) == 0 or ref.geometry.is_empty.all():
        raise ValueError("Reference vector is empty or has no valid geometry.")
    if len(pred) == 0 or pred.geometry.is_empty.all():
        raise ValueError("Predicted vector is empty or has no valid geometry.")
    
    # Reproject to common CRS
    target_crs = crs if crs else ref.crs
    ref = ref.to_crs(target_crs)
    pred = pred.to_crs(target_crs)
    
    # Report geometry types
    ref_types = ref.geom_type.value_counts().to_dict()
    pred_types = pred.geom_type.value_counts().to_dict()
    print(f"Reference geometry types: {ref_types}")
    print(f"Predicted geometry types: {pred_types}")
    
    # Check if geometries need buffering (points/lines have no area)
    ref_union = unary_union(ref.geometry)
    pred_union = unary_union(pred.geometry)
    
    # Auto-detect if buffering needed
    if ref_union.area == 0 and ref_buffer is None:
        print("⚠️  Reference has zero area (points/lines). Set ref_buffer parameter.")
        ref_buffer = buffer_distance  # Use tolerance as default
        print(f"    Auto-applying ref_buffer={ref_buffer}")
    
    if pred_union.area == 0 and pred_buffer is None:
        print("⚠️  Predictions have zero area (points/lines). Set pred_buffer parameter.")
        pred_buffer = 10  # Default 10m for predictions
        print(f"    Auto-applying pred_buffer={pred_buffer}")
    
    # Apply buffers if specified
    if ref_buffer:
        ref_union = ref_union.buffer(ref_buffer)
        print(f"Reference buffered by {ref_buffer} units")
    
    if pred_buffer:
        pred_union = pred_union.buffer(pred_buffer)
        print(f"Predictions buffered by {pred_buffer} units")
    
    # Apply tolerance buffer to reference
    buffered_ref = ref_union.buffer(buffer_distance)
    
    # Compute TP, FP, FN geometries
    tp_geom = pred_union.intersection(buffered_ref)
    fp_geom = pred_union.difference(buffered_ref)
    fn_geom = buffered_ref.difference(pred_union)
    
    # Calculate areas
    tp_area = tp_geom.area if not tp_geom.is_empty else 0
    fp_area = fp_geom.area if not fp_geom.is_empty else 0
    fn_area = fn_geom.area if not fn_geom.is_empty else 0
    
    # Compute metrics
    precision = tp_area / (tp_area + fp_area) if (tp_area + fp_area) > 0 else 0
    recall = tp_area / (tp_area + fn_area) if (tp_area + fn_area) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    iou = tp_area / (tp_area + fp_area + fn_area) if (tp_area + fp_area + fn_area) > 0 else 0
    
    print(f"\n--- Evaluation Metrics (Vector) ---")
    print(f"Tolerance buffer: {buffer_distance} units")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"F1 Score:   {f1_score:.4f}")
    print(f"IoU:        {iou:.4f}")
    print(f"\nAreas:")
    print(f"  TP: {tp_area:,.2f} sq units")
    print(f"  FP: {fp_area:,.2f} sq units")
    print(f"  FN: {fn_area:,.2f} sq units")
    
    # Build comparison GeoDataFrame
    comparison_records = []
    
    if not tp_geom.is_empty:
        comparison_records.append({"geometry": tp_geom, "class": 1, "label": "TP", "area": tp_area})
    if not fp_geom.is_empty:
        comparison_records.append({"geometry": fp_geom, "class": -1, "label": "FP", "area": fp_area})
    if not fn_geom.is_empty:
        comparison_records.append({"geometry": fn_geom, "class": 2, "label": "FN", "area": fn_area})
    
    comparison_gdf = gpd.GeoDataFrame(comparison_records, crs=target_crs)
    
    # Save if output_path provided or can be derived
    if output_path is None and isinstance(predicted_path, str):
        pred_path = Path(predicted_path)
        output_path = str(pred_path.parent / f"{pred_path.stem}_comparison_buf{buffer_distance}.gpkg")
    
    if output_path:
        if 'fid' in comparison_gdf.columns:
            comparison_gdf = comparison_gdf.rename(columns={'fid': 'orig_fid'})
        comparison_gdf.to_file(output_path, driver="GPKG")
        print(f"\nComparison vector saved to:\n{output_path}")
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "iou": float(iou),
        "tp_area": float(tp_area),
        "fp_area": float(fp_area),
        "fn_area": float(fn_area),
        "comparison_gdf": comparison_gdf
    }


   
def compare_networks(
    reference_path: Union[str, gpd.GeoDataFrame],
    predicted_path: Union[str, gpd.GeoDataFrame],
    buffer_distance: float,
    output_path: Optional[str] = None,
    crs: Optional[Union[str, int]] = None,
) -> Dict[str, float]:
    """
    Compare predicted and reference line networks using symmetric buffering and IoU.

    Both networks are buffered by the same distance, converting lines to polygons.
    Metrics are computed purely from area overlap between the two buffered zones:
      - TP: intersection of buffered prediction and buffered reference
      - FP: buffered prediction minus buffered reference
      - FN: buffered reference minus buffered prediction

    Args:
        reference_path: Path to reference network file (.gpkg/.shp) or GeoDataFrame.
        predicted_path: Path to predicted network file (.gpkg/.shp) or GeoDataFrame.
        buffer_distance: Buffer radius applied symmetrically to both networks (map units).
        output_path: Optional path to save comparison GeoPackage with TP/FP/FN polygons.
        crs: CRS to reproject to before comparison (use a projected CRS, e.g. EPSG:3857).

    Returns:
        Dictionary with keys: precision, recall, f1_score, iou,
        tp_area, fp_area, fn_area, comparison_gdf.
    """
    if isinstance(reference_path, gpd.GeoDataFrame):
        ref = reference_path.copy()
    else:
        ref = gpd.read_file(reference_path)

    if isinstance(predicted_path, gpd.GeoDataFrame):
        pred = predicted_path.copy()
    else:
        pred = gpd.read_file(predicted_path)

    if len(ref) == 0 or ref.geometry.is_empty.all():
        raise ValueError("Reference network is empty or has no valid geometry.")
    if len(pred) == 0 or pred.geometry.is_empty.all():
        raise ValueError("Predicted network is empty or has no valid geometry.")

    target_crs = crs if crs else ref.crs
    ref = ref.to_crs(target_crs)
    pred = pred.to_crs(target_crs)

    ref_union = unary_union(ref.geometry)
    pred_union = unary_union(pred.geometry)

    buffered_ref = ref_union.buffer(buffer_distance)
    buffered_pred = pred_union.buffer(buffer_distance)

    tp_geom = buffered_pred.intersection(buffered_ref)
    fp_geom = buffered_pred.difference(buffered_ref)
    fn_geom = buffered_ref.difference(buffered_pred)

    tp_area = tp_geom.area if not tp_geom.is_empty else 0.0
    fp_area = fp_geom.area if not fp_geom.is_empty else 0.0
    fn_area = fn_geom.area if not fn_geom.is_empty else 0.0

    precision = tp_area / (tp_area + fp_area) if (tp_area + fp_area) > 0 else 0.0
    recall    = tp_area / (tp_area + fn_area) if (tp_area + fn_area) > 0 else 0.0
    f1_score  = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    iou       = tp_area / (tp_area + fp_area + fn_area) if (tp_area + fp_area + fn_area) > 0 else 0.0

    print(f"\n--- Network Evaluation Metrics ---")
    print(f"Buffer distance: {buffer_distance} units")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"F1 Score:   {f1_score:.4f}")
    print(f"IoU:        {iou:.4f}")
    print(f"\nAreas:")
    print(f"  TP: {tp_area:,.2f} sq units")
    print(f"  FP: {fp_area:,.2f} sq units")
    print(f"  FN: {fn_area:,.2f} sq units")

    records = []
    if not tp_geom.is_empty:
        records.append({"geometry": tp_geom, "class": 1,  "label": "TP", "area": tp_area})
    if not fp_geom.is_empty:
        records.append({"geometry": fp_geom, "class": -1, "label": "FP", "area": fp_area})
    if not fn_geom.is_empty:
        records.append({"geometry": fn_geom, "class": 2,  "label": "FN", "area": fn_area})

    comparison_gdf = gpd.GeoDataFrame(records, crs=target_crs)

    if output_path is None and isinstance(predicted_path, str):
        pred_p = Path(predicted_path)
        output_path = str(pred_p.parent / f"{pred_p.stem}_network_eval_buf{buffer_distance}.gpkg")

    if output_path:
        if "fid" in comparison_gdf.columns:
            comparison_gdf = comparison_gdf.rename(columns={"fid": "orig_fid"})
        comparison_gdf.to_file(output_path, driver="GPKG")
        print(f"\nComparison saved to:\n{output_path}")

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "iou": float(iou),
        "tp_area": float(tp_area),
        "fp_area": float(fp_area),
        "fn_area": float(fn_area),
        "comparison_gdf": comparison_gdf,
    }


def compare_rasters(
    reference_tif: str,
    predicted_tif: str,
    IoU_buf: int,
    output_path: Optional[str] = None
) -> Dict[str, float]:
    """
    Compare predicted raster with reference raster and compute evaluation metrics.

    Performs a buffered comparison between reference and predicted binary rasters
    using morphological operations (dilation on reference, erosion on prediction).
    Computes precision, recall, F1 score, and IoU metrics.

    The comparison raster output uses the following encoding:
    - 1: True Positive (TP)
    - -1: False Positive (FP)
    - 2: False Negative (FN)
    - 0: True Negative (background)

    Args:
        reference_tif: Path to the reference (ground truth) raster file.
        predicted_tif: Path to the predicted raster file.
        IoU_buf: Buffer size for morphological operations (dilation/erosion kernel size).
        output_path: Optional path for comparison raster output. If None, generates
            path based on predicted_tif with "_comparison_buf{IoU_buf}.tif" suffix.

    Returns:
        Dictionary containing evaluation metrics:
        - precision: TP / (TP + FP)
        - recall: TP / (TP + FN)
        - f1_score: 2 * (precision * recall) / (precision + recall)
        - iou: TP / (TP + FP + FN)

    Raises:
        ValueError: If rasters have different dimensions.

    Example:
        >>> metrics = compare_rasters(
        ...     reference_tif="ground_truth.tif",
        ...     predicted_tif="prediction.tif",
        ...     IoU_buf=4
        ... )
        >>> print(f"IoU: {metrics['iou']:.4f}")
    """
    with rasterio.open(reference_tif) as ref_src:
        ref_data = ref_src.read(1)
        profile = ref_src.profile

    with rasterio.open(predicted_tif) as pred_src:
        pred_data = pred_src.read(1)

    if ref_data.shape != pred_data.shape:
        raise ValueError("Rasters must have the same dimensions and alignment.")

    struct_element = np.ones((IoU_buf, IoU_buf))
    buffered_ref = binary_dilation(ref_data, structure=struct_element).astype(ref_data.dtype)

    TP = (buffered_ref == 1) & (pred_data == 1)
    FP = (buffered_ref == 0) & (pred_data == 1)
    FN = (buffered_ref == 1) & (pred_data == 0)

    output_data = np.zeros_like(pred_data, dtype=np.int8)
    output_data[TP] = 1
    output_data[FP] = -1
    output_data[FN] = 2

    tp_count = np.sum(TP)
    fp_count = np.sum(FP)
    fn_count = np.sum(FN)

    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    iou = tp_count / (tp_count + fp_count + fn_count) if (tp_count + fp_count + fn_count) > 0 else 0

    print(f"\n--- Evaluation Metrics ---")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"F1 Score:   {f1_score:.4f}")
    print(f"IoU:        {iou:.4f}")

    # Determine output path
    if output_path is None:
        output_path = predicted_tif.replace(".tif", f"_comparison_buf{IoU_buf}.tif")
    
    profile.update(dtype=rasterio.int8, count=1)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(output_data, 1)

    print(f"Comparison raster saved to:\n{output_path}")

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "iou": float(iou)
    }

