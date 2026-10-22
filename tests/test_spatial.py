"""
Unit tests for spatial mathematics: tripwire ray crossing, polygon geofencing, heatmaps.
"""
import pytest
import numpy as np
from core.spatial_analytics import (
    line_segments_intersect,
    get_crossing_direction,
    TripwireLine,
    GeofenceZone,
    HeatmapAccumulator
)
from core.tracker import TrackedObject

def test_line_segment_intersection():
    # Perpendicular intersecting segments
    A = (0, 10)
    B = (20, 10)
    C = (10, 0)
    D = (10, 20)
    assert line_segments_intersect(A, B, C, D) is True

    # Parallel non-intersecting
    E = (0, 0)
    F = (20, 0)
    assert line_segments_intersect(A, B, E, F) is False

def test_crossing_direction_in_out():
    # Line from left (100, 200) to right (300, 200)
    l1 = (100, 200)
    l2 = (300, 200)

    # Moving top-to-bottom
    p_prev = (200, 150)
    p_curr = (200, 250)
    dir1 = get_crossing_direction(p_prev, p_curr, l1, l2)
    assert dir1 in ("IN", "OUT")

    # Moving opposite direction
    dir2 = get_crossing_direction(p_curr, p_prev, l1, l2)
    assert dir1 != dir2  # Must be opposite

def test_geofence_point_in_polygon():
    polygon = [[100, 100], [200, 100], [200, 200], [100, 200]]
    zone = GeofenceZone(zone_id="test_zone", polygon_points=polygon, label="Test Area")

    # Inside
    assert zone.contains_point((150, 150)) is True
    # Edge
    assert zone.contains_point((100, 150)) is True
    # Outside
    assert zone.contains_point((50, 50)) is False
    assert zone.contains_point((300, 300)) is False

def test_heatmap_accumulator():
    heatmap = HeatmapAccumulator(width=200, height=200, decay_rate=0.9)
    assert np.all(heatmap.heatmap == 0)

    # Add points
    heatmap.accumulate([(50, 50), (100, 100)])
    assert heatmap.heatmap[50, 50] > 0
    assert heatmap.heatmap[100, 100] > 0

    # Overlay rendering
    overlay = heatmap.get_color_overlay()
    assert overlay.shape == (200, 200, 3)
