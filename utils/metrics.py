from typing import Dict, Optional

import numpy as np
import rasterio
from scipy.ndimage import binary_dilation, binary_erosion

from typing import Dict, Optional, Union
from pathlib import Path
import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union
from shapely.geometry import LineString, MultiLineString

__all__ = [
    "compare_vectors",
    "compare_linear_vectors",
    "compare_rasters",
    "segmentize_line",
    "compare_line_segments",
    "add_vector_comparison_to_map",
    "display_vector_comparison",
    "compare_networks"
]


def compare_vectors(
    reference_path: Union[str, gpd.GeoDataFrame],
    predicted_path: Union[str, gpd.GeoDataFrame],
    buffer_distance: float,
    output_path: Optional[str] = None,
    crs: Optional[Union[str, int]] = None
) -> Dict[str, float]:
    """
    Compare predicted vector with reference vector and compute evaluation metrics.
    
    Performs a buffered comparison between reference and predicted geometries
    using geometric buffer operations. Computes precision, recall, F1 score, 
    and IoU metrics based on area overlap.
    
    The comparison output contains polygons with the following classification:
    - 1: True Positive (TP) - predicted area within buffered reference
    - -1: False Positive (FP) - predicted area outside buffered reference
    - 2: False Negative (FN) - buffered reference area not covered by prediction
    
    Args:
        reference_path: Path to the reference (ground truth) vector file, or a GeoDataFrame.
        predicted_path: Path to the predicted vector file, or a GeoDataFrame.
        buffer_distance: Buffer distance in CRS units (meters for projected CRS).
            Applied to reference geometry for tolerance matching.
        output_path: Optional path for comparison vector output. If None, generates
            path based on predicted_path when it is a file path; otherwise defaults to
            "comparison_buf{buffer_distance}.gpkg" when predicted_path is a GeoDataFrame.
        crs: Optional CRS to reproject both datasets to before comparison.
            Recommended to use a projected CRS (e.g., EPSG:3857) for accurate
            area calculations. If None, uses reference CRS.
    
    Returns:
        Dictionary containing evaluation metrics:
        - precision: TP_area / (TP_area + FP_area)
        - recall: TP_area / (TP_area + FN_area)
        - f1_score: 2 * (precision * recall) / (precision + recall)
        - iou: TP_area / (TP_area + FP_area + FN_area)
        - tp_area: Total true positive area
        - fp_area: Total false positive area
        - fn_area: Total false negative area
    
    Raises:
        ValueError: If either input file is empty or has no valid geometry.
    
    Example:
        >>> metrics = compare_vectors(
        ...     reference_path="ground_truth.gpkg",
        ...     predicted_path="prediction.gpkg",
        ...     buffer_distance=10,  # 10 meter tolerance
        ...     crs=3857
        ... )
        >>> print(f"IoU: {metrics['iou']:.4f}")
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
        raise ValueError("Reference vector file is empty or has no valid geometry.")
    if len(pred) == 0 or pred.geometry.is_empty.all():
        raise ValueError("Predicted vector file is empty or has no valid geometry.")
    
    # Reproject to common CRS
    target_crs = crs if crs else ref.crs
    ref = ref.to_crs(target_crs)
    pred = pred.to_crs(target_crs)
    
    # Dissolve to single geometries
    ref_union = unary_union(ref.geometry)
    pred_union = unary_union(pred.geometry)
    
    # Buffer reference (equivalent to dilation in raster)
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
    print(f"Buffer distance: {buffer_distance} units")
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
        comparison_records.append({
            "geometry": tp_geom,
            "class": 1,
            "label": "TP",
            "area": tp_area
        })
    if not fp_geom.is_empty:
        comparison_records.append({
            "geometry": fp_geom,
            "class": -1,
            "label": "FP",
            "area": fp_area
        })
    if not fn_geom.is_empty:
        comparison_records.append({
            "geometry": fn_geom,
            "class": 2,
            "label": "FN",
            "area": fn_area
        })
    
    comparison_gdf = gpd.GeoDataFrame(comparison_records, crs=target_crs)
    
    # Determine output path
    if output_path is None:
        if isinstance(predicted_path, str):
            pred_path = Path(predicted_path)
            output_path = str(pred_path.parent / f"{pred_path.stem}_comparison_buf{buffer_distance}.gpkg")
        else:
            output_path = f"comparison_buf{buffer_distance}.gpkg"
    
    # Handle fid column conflict for GPKG
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
        "fn_area": float(fn_area)
    }


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


def segmentize_line(geom, segment_length):
    """
    Split a line into segments of approximate length.
    
    Parameters
    ----------
    geom : shapely.geometry.LineString or MultiLineString
        The geometry to segmentize
    segment_length : float
        Target length for each segment in CRS units
    
    Returns
    -------
    list
        List of LineString segments
    """
    if geom.is_empty:
        return []
    
    if geom.geom_type == 'MultiLineString':
        segments = []
        for line in geom.geoms:
            segments.extend(segmentize_line(line, segment_length))
        return segments
    
    if geom.geom_type != 'LineString':
        return []
    
    total_length = geom.length
    if total_length <= segment_length:
        return [geom]
    
    num_segments = int(np.ceil(total_length / segment_length))
    segments = []
    
    for i in range(num_segments):
        start_frac = i / num_segments
        end_frac = (i + 1) / num_segments
        
        start_pt = geom.interpolate(start_frac, normalized=True)
        end_pt = geom.interpolate(end_frac, normalized=True)
        
        # Extract segment between two points
        start_dist = start_frac * total_length
        end_dist = end_frac * total_length
        
        coords = list(geom.coords)
        segment_coords = [start_pt.coords[0]]
        
        cumulative = 0
        for j in range(len(coords) - 1):
            p1, p2 = coords[j], coords[j + 1]
            seg_len = LineString([p1, p2]).length
            
            if cumulative + seg_len > start_dist and cumulative < end_dist:
                if cumulative >= start_dist:
                    segment_coords.append(p1)
                if cumulative + seg_len <= end_dist:
                    segment_coords.append(p2)
            
            cumulative += seg_len
        
        segment_coords.append(end_pt.coords[0])
        
        if len(segment_coords) >= 2:
            seg = LineString(segment_coords)
            if seg.length > 0:
                segments.append(seg)
    
    return segments


def compare_line_segments(
    reference_path: Union[str, gpd.GeoDataFrame],
    predicted_path: Union[str, gpd.GeoDataFrame],
    segment_length: float,
    buffer_distance: float,
    output_path: Optional[str] = None,
    crs: Optional[Union[str, int]] = None,
    reference_layer: Optional[str] = None,
    predicted_layer: Optional[str] = None
) -> Dict[str, float]:
    """
    Compare predicted line segments with reference line segments using length-based metrics.
    
    This function segments both reference and predicted lines into smaller segments,
    then classifies each segment as True Positive (TP), False Positive (FP), or 
    False Negative (FN) based on buffer distance matching. Metrics are computed
    based on segment lengths rather than areas, making it suitable for linear 
    features like levees, channels, or road networks.
    
    The comparison output contains line segments with the following classification:
    - 1: True Positive (TP) - predicted segment within buffered reference
    - -1: False Positive (FP) - predicted segment outside buffered reference
    - 2: False Negative (FN) - reference segment not detected by prediction
    
    Parameters
    ----------
    reference_path : str or GeoDataFrame
        Path to the reference (ground truth) vector file, or a GeoDataFrame.
        If a file path, can be any format supported by geopandas (GPKG, GeoJSON, etc.).
    predicted_path : str or GeoDataFrame
        Path to the predicted vector file, or a GeoDataFrame.
        If a file path and GPKG format, specify layer name if needed.
    segment_length : float
        Target length for segmenting lines, in CRS units (meters for projected CRS).
        Lines are split into segments of approximately this length for comparison.
    buffer_distance : float
        Buffer distance in CRS units (meters for projected CRS).
        Used to determine if predicted segments match reference segments.
    output_path : str, optional
        Path for comparison vector output. If None, generates path based on 
        predicted_path with "_comp_segm_{segment_length}_dist_{buffer_distance}" suffix.
    crs : str or int, optional
        CRS to reproject both datasets to before comparison.
        Recommended to use a projected CRS (e.g., EPSG:3857) for accurate
        length calculations. If None, uses reference CRS.
    reference_layer : str, optional
        Layer name if reference_path is a GPKG file with multiple layers.
        Defaults to first layer.
    predicted_layer : str, optional
        Layer name if predicted_path is a GPKG file with multiple layers.
        Defaults to 'edges' layer if it exists, otherwise first layer.
    
    Returns
    -------
    dict
        Dictionary containing evaluation metrics:
        - precision: TP_length / (TP_length + FP_length)
        - recall: TP_length / (TP_length + FN_length)
        - f1_score: 2 * (precision * recall) / (precision + recall)
        - iou: TP_length / (TP_length + FP_length + FN_length)
        - tp_length: Total true positive length
        - fp_length: Total false positive length
        - fn_length: Total false negative length
        - tp_count: Number of TP segments
        - fp_count: Number of FP segments
        - fn_count: Number of FN segments
    
    Raises
    ------
    ValueError
        If either input file is empty or has no valid geometry.
    
    Example
    -------
    >>> from utils import compare_line_segments
    >>> metrics = compare_line_segments(
    ...     reference_path="ground_truth.gpkg",
    ...     predicted_path="prediction.gpkg",
    ...     segment_length=250,  # 250 meter segments
    ...     buffer_distance=250,  # 250 meter tolerance
    ...     crs=3857
    ... )
    >>> print(f"F1 Score: {metrics['f1_score']:.4f}")
    >>> print(f"Total TP length: {metrics['tp_length']/1000:.2f} km")
    """
    # Load vector files
    if isinstance(reference_path, gpd.GeoDataFrame):
        ref = reference_path.copy()
    else:
        ref = gpd.read_file(reference_path, layer=reference_layer)
    
    if isinstance(predicted_path, gpd.GeoDataFrame):
        pred = predicted_path.copy()
    else:
        # Try 'edges' layer first (common for network outputs), fallback to default
        try:
            pred = gpd.read_file(predicted_path, layer=predicted_layer or 'edges')
        except (ValueError, KeyError):
            pred = gpd.read_file(predicted_path, layer=predicted_layer)
    
    if len(ref) == 0 or ref.geometry.is_empty.all():
        raise ValueError("Reference vector file is empty or has no valid geometry.")
    if len(pred) == 0 or pred.geometry.is_empty.all():
        raise ValueError("Predicted vector file is empty or has no valid geometry.")
    
    # Reproject to common CRS
    target_crs = crs if crs else ref.crs
    ref = ref.to_crs(epsg=target_crs if isinstance(target_crs, int) else target_crs)
    pred = pred.to_crs(epsg=target_crs if isinstance(target_crs, int) else target_crs)
    
    # Segment reference lines
    print(f"Segmenting reference lines into {segment_length}m sections...")
    ref_segments = []
    for idx, row in ref.iterrows():
        segs = segmentize_line(row.geometry, segment_length)
        for seg in segs:
            ref_segments.append({'geometry': seg, 'orig_idx': idx})
    
    ref_segmented = gpd.GeoDataFrame(ref_segments, crs=ref.crs)
    print(f"Reference: {len(ref)} lines → {len(ref_segmented)} segments")
    
    # Segment predicted lines
    print(f"Segmenting predicted lines into {segment_length}m sections...")
    pred_segments = []
    for idx, row in pred.iterrows():
        segs = segmentize_line(row.geometry, segment_length)
        for seg in segs:
            pred_segments.append({'geometry': seg, 'orig_idx': idx})
    
    pred_segmented = gpd.GeoDataFrame(pred_segments, crs=pred.crs)
    print(f"Predicted: {len(pred)} lines → {len(pred_segmented)} segments")
    
    # Create buffered geometries for matching
    ref_union = unary_union(ref_segmented.geometry)
    ref_buffered = ref_union.buffer(buffer_distance)
    
    pred_union = unary_union(pred_segmented.geometry)
    pred_buffered = pred_union.buffer(buffer_distance)
    
    # Classify predicted segments: TP or FP
    pred_segmented['near_ref'] = pred_segmented.geometry.intersects(ref_buffered)
    pred_segmented['class'] = pred_segmented['near_ref'].map({True: 1, False: -1})
    pred_segmented['label'] = pred_segmented['near_ref'].map({True: 'TP', False: 'FP'})
    
    # Classify reference segments: TP or FN
    ref_segmented['detected'] = ref_segmented.geometry.intersects(pred_buffered)
    ref_segmented['class'] = ref_segmented['detected'].map({True: 1, False: 2})
    ref_segmented['label'] = ref_segmented['detected'].map({True: 'TP', False: 'FN'})
    
    # Get only FN reference segments
    ref_fn = ref_segmented[ref_segmented['class'] == 2].copy()
    
    # Combine
    comparison_lines = gpd.GeoDataFrame(
        pd.concat([
            pred_segmented[['geometry', 'class', 'label']],
            ref_fn[['geometry', 'class', 'label']]
        ], ignore_index=True),
        crs=pred.crs
    )
    
    comparison_lines['length'] = comparison_lines.geometry.length
    
    # Determine output path
    if output_path is None:
        if isinstance(predicted_path, str):
            pred_path = Path(predicted_path)
            output_path = str(pred_path.parent / f"{pred_path.stem}_comp_segm_{segment_length}_dist_{buffer_distance}.gpkg")
        else:
            output_path = f"comparison_segm_{segment_length}_dist_{buffer_distance}.gpkg"
    
    # Handle fid column conflict for GPKG
    if 'fid' in comparison_lines.columns:
        comparison_lines = comparison_lines.rename(columns={'fid': 'orig_fid'})
    
    comparison_lines.to_file(output_path, driver="GPKG")
    
    # Calculate metrics
    tp_length = comparison_lines[comparison_lines['class'] == 1]['length'].sum()
    fp_length = comparison_lines[comparison_lines['class'] == -1]['length'].sum()
    fn_length = comparison_lines[comparison_lines['class'] == 2]['length'].sum()
    
    tp_count = len(comparison_lines[comparison_lines['class'] == 1])
    fp_count = len(comparison_lines[comparison_lines['class'] == -1])
    fn_count = len(comparison_lines[comparison_lines['class'] == 2])
    
    precision = tp_length / (tp_length + fp_length) if (tp_length + fp_length) > 0 else 0
    recall = tp_length / (tp_length + fn_length) if (tp_length + fn_length) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    iou = tp_length / (tp_length + fp_length + fn_length) if (tp_length + fp_length + fn_length) > 0 else 0
    
    print(f"\n--- Line Comparison Summary ({segment_length}m segments) ---")
    for cls, label in [(1, 'TP'), (-1, 'FP'), (2, 'FN')]:
        subset = comparison_lines[comparison_lines['class'] == cls]
        total_length = subset['length'].sum()
        print(f"{label}: {len(subset)} segments, {total_length/1000:,.2f} km")
    
    print(f"\n--- Evaluation Metrics (Length-based) ---")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"F1 Score:   {f1_score:.4f}")
    print(f"IoU:        {iou:.4f}")
    
    print(f"\nSaved to: {output_path}")
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "iou": float(iou),
        "tp_length": float(tp_length),
        "fp_length": float(fp_length),
        "fn_length": float(fn_length),
        "tp_count": int(tp_count),
        "fp_count": int(fp_count),
        "fn_count": int(fn_count)
    }


def add_vector_comparison_to_map(
    m,
    vector_path: Union[str, gpd.GeoDataFrame],
    class_column: str = "class",
    layer_name: str = "Vector Comparison"
):
    """
    Add vector comparison data to an existing leafmap Map.
    
    Parameters
    ----------
    m : leafmap.Map
        Existing leafmap Map object to add layers to
    vector_path : str or GeoDataFrame
        Path to vector file or GeoDataFrame containing comparison results
    class_column : str, default "class"
        Column name containing classification values (1, -1, 2)
    layer_name : str, default "Vector Comparison"
        Base name for the vector layers
    
    Returns
    -------
    leafmap.Map
        The map object with added layers
    """
    # Load vector data
    if isinstance(vector_path, gpd.GeoDataFrame):
        gdf = vector_path.copy()
    else:
        gdf = gpd.read_file(vector_path)
    
    if len(gdf) == 0:
        raise ValueError("Vector data is empty.")
    
    # Ensure we have the class column
    if class_column not in gdf.columns:
        raise ValueError(f"Column '{class_column}' not found in vector data.")
    
    # Reproject to WGS84 for display if needed
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf_display = gdf.to_crs(epsg=4326)
    else:
        gdf_display = gdf.copy()
    
    # Color mapping
    color_map = {
        1: "#0dff0d",   # TP - Green
        -1: "#e5350e",  # FP - Red
        2: "#729bff",   # FN - Blue
    }
    
    # Add each class as a separate layer for better control
    for class_val, color in color_map.items():
        subset = gdf_display[gdf_display[class_column] == class_val]
        if len(subset) > 0:
            # Convert to GeoJSON for leafmap
            geojson_data = subset.to_json()
            
            label_map = {1: "True Positives", -1: "False Positives", 2: "False Negatives"}
            layer_label = f"{layer_name} - {label_map[class_val]}"
            
            m.add_geojson(
                geojson_data,
                layer_name=layer_label,
                style={"color": color, "weight": 3, "opacity": 0.8}
            )
    
    return m


def display_vector_comparison(
    vector_path: Union[str, gpd.GeoDataFrame],
    center: Optional[list] = None,
    zoom: Optional[int] = None,
    class_column: str = "class",
    layer_name: str = "Vector Comparison",
    auto_center: bool = True,
    map_obj: Optional[object] = None
):
    """
    Display vector comparison data using leafmap with color-coded TP/FP/FN classification.
    
    Creates an interactive map showing line segments colored by their classification:
    - TP (True Positives): Green
    - FP (False Positives): Red  
    - FN (False Negatives): Blue
    
    Parameters
    ----------
    vector_path : str or GeoDataFrame
        Path to vector file (GPKG, GeoJSON, etc.) or GeoDataFrame containing comparison results.
        Must have a 'class' column with values: 1 (TP), -1 (FP), 2 (FN)
    center : list, optional
        Map center as [latitude, longitude]. If None and auto_center=True, 
        calculates from data bounds.
    zoom : int, optional
        Initial zoom level. If None and auto_center=True, calculates from data bounds.
    class_column : str, default "class"
        Column name containing classification values (1, -1, 2)
    layer_name : str, default "Vector Comparison"
        Name for the vector layer in the map
    auto_center : bool, default True
        If True, automatically center and zoom to data bounds
    map_obj : leafmap.Map, optional
        Existing map object to add layers to. If provided, center/zoom are ignored.
    
    Returns
    -------
    leafmap.Map
        Interactive map object
    
    Example
    -------
    >>> from utils import compare_line_segments, display_vector_comparison
    >>> 
    >>> # Run comparison
    >>> metrics = compare_line_segments(
    ...     reference_path="ref.gpkg",
    ...     predicted_path="pred.gpkg",
    ...     segment_length=250,
    ...     buffer_distance=250,
    ...     output_path="comparison.gpkg"
    ... )
    >>> 
    >>> # Display results
    >>> m = display_vector_comparison(
    ...     vector_path="comparison.gpkg",
    ...     center=[31.52806, 65.24722],
    ...     zoom=14
    ... )
    >>> m
    """
    try:
        import leafmap
    except ImportError:
        raise ImportError(
            "leafmap is required for visualization. Install with: pip install leafmap"
        )
    
    # Use existing map if provided
    if map_obj is not None:
        return add_vector_comparison_to_map(map_obj, vector_path, class_column, layer_name)
    
    # Load vector data for bounds calculation
    if isinstance(vector_path, gpd.GeoDataFrame):
        gdf = vector_path.copy()
    else:
        gdf = gpd.read_file(vector_path)
    
    if len(gdf) == 0:
        raise ValueError("Vector data is empty.")
    
    # Ensure we have the class column
    if class_column not in gdf.columns:
        raise ValueError(f"Column '{class_column}' not found in vector data.")
    
    # Reproject to WGS84 for display if needed
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf_display = gdf.to_crs(epsg=4326)
    else:
        gdf_display = gdf.copy()
    
    # Calculate center and zoom if auto_center
    if auto_center and (center is None or zoom is None):
        bounds = gdf_display.total_bounds  # [minx, miny, maxx, maxy]
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2
        center = [center_lat, center_lon]
        
        # Estimate zoom based on bounds
        if zoom is None:
            lat_range = bounds[3] - bounds[1]
            lon_range = bounds[2] - bounds[0]
            max_range = max(lat_range, lon_range)
            
            # Rough zoom estimation
            if max_range > 10:
                zoom = 8
            elif max_range > 5:
                zoom = 9
            elif max_range > 1:
                zoom = 11
            elif max_range > 0.5:
                zoom = 12
            elif max_range > 0.1:
                zoom = 13
            else:
                zoom = 14
    
    # Initialize map
    if center is None:
        center = [0, 0]
    if zoom is None:
        zoom = 10
    
    m = leafmap.Map(center=center, zoom=zoom)
    
    # Add vector layers
    add_vector_comparison_to_map(m, vector_path, class_column, layer_name)
    
    # Add legend
    legend_dict = {
        "True Positives (1)": "#0dff0d",
        "False Positives (-1)": "#e5350e",
        "False Negatives (2)": "#729bff",
    }
    m.add_legend(title="Vector Comparison", legend_dict=legend_dict)
    
    # Add layer control
    m.add_layer_control()
    
    return m

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
