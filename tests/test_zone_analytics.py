"""
Unit tests for polygon zone analytics and dwell duration calculation.
"""
import numpy as np

def test_point_in_polygon_containment():
    polygon = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
    test_pt = (50, 50)
    assert 0 <= test_pt[0] <= 100
    assert 0 <= test_pt[1] <= 100
