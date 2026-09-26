"""
Outputs 5 km polygon rings, bounding boxes, and alert zones for GIS mapping (GeoJSON RFC 7946).
Compatible with Leaflet, Mapbox, QGIS, ArcGIS, and emergency response platforms.
"""

from typing import Dict, List, Any, Optional
import json
import numpy as np


def generate_circle_polygon(
    center_lat: float,
    center_lon: float,
    radius_km: float,
    num_points: int = 64
) -> List[List[float]]:
    """
    Generate geodesic circular polygon coordinates [lon, lat] in GeoJSON format.
    """
    angles = np.linspace(0, 2 * np.pi, num_points + 1)
    # Earth radius ~6371 km
    lat_r = np.radians(center_lat)
    lon_r = np.radians(center_lon)
    d_r = radius_km / 6371.0

    coords = []
    for angle in angles:
        point_lat_r = np.arcsin(
            np.sin(lat_r) * np.cos(d_r) + np.cos(lat_r) * np.sin(d_r) * np.cos(angle)
        )
        point_lon_r = lon_r + np.arctan2(
            np.sin(angle) * np.sin(d_r) * np.cos(lat_r),
            np.cos(d_r) - np.sin(lat_r) * np.sin(point_lat_r)
        )
        coords.append([
            round(float(np.degrees(point_lon_r)), 5),
            round(float(np.degrees(point_lat_r)), 5)
        ])
    return coords


def generate_bounding_box_geojson(
    bbox: List[float],
    properties: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate GeoJSON polygon feature for bounding box [min_lon, min_lat, max_lon, max_lat].
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    coordinates = [[
        [min_lon, min_lat],
        [max_lon, min_lat],
        [max_lon, max_lat],
        [min_lon, max_lat],
        [min_lon, min_lat]  # Closed ring
    ]]
    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": coordinates
        },
        "properties": properties or {
            "layer_type": "bounding_box",
            "description": "Cyclone Tracking & Downscaling Bounding Box"
        }
    }


def generate_cyclone_geojson(
    centroid_lat: float,
    centroid_lon: float,
    min_pressure_hpa: float,
    timestamp: str,
    intensity_info: Dict[str, Any],
    bbox: Optional[List[float]] = None,
    ring_radii_km: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Generate complete GIS FeatureCollection containing:
    1. Cyclone Eye Centroid Point
    2. Concentric 5 km, 25 km, 65 km, 150 km, and 250 km Risk Ring Polygons
    3. Bounding Box Polygon (if provided)
    """
    if ring_radii_km is None:
        # High-res concentric rings: 5 km core, 25 km eye, 65 km destructive, 150 km gale, 250 km outer
        ring_radii_km = [5.0, 25.0, 65.0, 150.0, 250.0]

    features = []

    # 1. Centroid Point Feature
    features.append({
        "type": "Feature",
        "id": "cyclone-centroid",
        "geometry": {
            "type": "Point",
            "coordinates": [round(centroid_lon, 5), round(centroid_lat, 5)]
        },
        "properties": {
            "feature_type": "cyclone_centroid",
            "timestamp": timestamp,
            "min_pressure_hpa": min_pressure_hpa,
            "category": intensity_info.get("category", "Cyclone"),
            "alert_level": intensity_info.get("alert_level", "CODE_RED"),
            "vmax_knots": intensity_info.get("vmax_knots", 0.0),
            "vmax_kmh": intensity_info.get("vmax_kmh", 0.0),
            "source": "Meteora Cyclone Tracking Engine"
        }
    })

    # Ring styling metadata
    ring_styles = {
        5.0: {
            "name": "5 km Inner Eye Core",
            "stroke": "#9c27b0",
            "fill": "#ba68c8",
            "fill_opacity": 0.45,
            "risk_level": "Catastrophic Pressure Minimum"
        },
        25.0: {
            "name": "25 km Eyewall Radius",
            "stroke": "#d32f2f",
            "fill": "#ef5350",
            "fill_opacity": 0.35,
            "risk_level": "Maximum Sustained Wind Swath"
        },
        65.0: {
            "name": "65 km Destructive Wind Ring",
            "stroke": "#f57c00",
            "fill": "#ffb74d",
            "fill_opacity": 0.25,
            "risk_level": "Destructive Wind / Storm Surge"
        },
        150.0: {
            "name": "150 km Gale Force Buffer",
            "stroke": "#fbc02d",
            "fill": "#fff176",
            "fill_opacity": 0.18,
            "risk_level": "Gale Force Winds & Torrential Rain"
        },
        250.0: {
            "name": "250 km Outer Feeder Band Ring",
            "stroke": "#0288d1",
            "fill": "#81d4fa",
            "fill_opacity": 0.12,
            "risk_level": "Peripheral Convective Bands"
        },
    }

    # 2. Concentric Polygon Rings
    for radius in ring_radii_km:
        coords = generate_circle_polygon(centroid_lat, centroid_lon, radius_km=radius)
        style = ring_styles.get(radius, {
            "name": f"{radius} km Ring",
            "stroke": "#ff9800",
            "fill": "#ffcc80",
            "fill_opacity": 0.2,
            "risk_level": "Cyclone Risk Buffer"
        })

        features.append({
            "type": "Feature",
            "id": f"risk-ring-{int(radius)}km",
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords]
            },
            "properties": {
                "feature_type": "risk_polygon_ring",
                "radius_km": radius,
                "zone_name": style["name"],
                "risk_level": style["risk_level"],
                "stroke": style["stroke"],
                "stroke_width": 2,
                "fill": style["fill"],
                "fill_opacity": style["fill_opacity"],
                "timestamp": timestamp
            }
        })

    # 3. Bounding Box Polygon
    if bbox is not None:
        features.append(generate_bounding_box_geojson(
            bbox,
            properties={
                "feature_type": "tracking_bounding_box",
                "bbox": bbox,
                "stroke": "#00e676",
                "stroke_width": 2,
                "stroke_dasharray": "5,5",
                "fill": "#00e676",
                "fill_opacity": 0.05,
                "description": "5 km downscaled tracking patch"
            }
        ))

    return {
        "type": "FeatureCollection",
        "generator": "Meteora GIS Engine v1.0",
        "timestamp": timestamp,
        "features": features
    }


def export_geojson_string(geojson_data: Dict[str, Any], indent: int = 2) -> str:
    """Format GeoJSON dict to pretty JSON string."""
    return json.dumps(geojson_data, indent=indent)
