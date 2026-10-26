"""
Tests for Advanced Spatial Analytics:
- Dwell time accumulation
- Loitering detection & notification deduplication
- Dynamic zone capacity & occupancy breach alerts
- Polygon area & centroid moments
- VelocityEstimator 2D kinematics, compass heading, and anomaly detection
- REST API analytics endpoints (/api/analytics/dwell, /api/analytics/occupancy)
"""
import time
import pytest
from fastapi.testclient import TestClient

from app import app
from core.spatial_analytics import GeofenceZone, VelocityEstimator


class MockTrack:
    def __init__(self, track_id: int, bbox: list, cls_name: str = "person"):
        self.track_id = track_id
        self.bbox = bbox
        self.cls_name = cls_name
        self.centroid = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
        self.current_zones = set()


@pytest.fixture
def client():
    return TestClient(app)


def test_geofence_area_and_centroid():
    # 100x100 square polygon: (0,0) to (100,100)
    pts = [[0, 0], [100, 0], [100, 100], [0, 100]]
    zone = GeofenceZone(polygon_points=pts, label="Test Area Zone")

    assert zone.area == 10000.0
    assert zone.centroid == (50, 50)


def test_geofence_dwell_time_tracking():
    # Zone spanning 0..200 x 0..200
    pts = [[0, 0], [200, 0], [200, 200], [0, 200]]
    zone = GeofenceZone(polygon_points=pts, label="Vault", dwell_threshold_sec=0.1)

    track1 = MockTrack(1, [50, 50, 80, 80])
    
    # Track enters
    new_intrusions = zone.update_intrusions([track1])
    assert len(new_intrusions) == 1
    assert 1 in zone.active_intrusions
    assert 1 in zone.track_entry_times

    # Simulate brief dwell
    time.sleep(0.12)
    dwell = zone.get_dwell_time(1)
    assert dwell >= 0.1


def test_geofence_loitering_alert_and_deduplication():
    pts = [[0, 0], [200, 0], [200, 200], [0, 200]]
    zone = GeofenceZone(polygon_points=pts, label="Restricted Lobby", dwell_threshold_sec=0.08)

    track1 = MockTrack(10, [40, 40, 70, 70])
    zone.update_intrusions([track1])

    # Immediately after entry: no loitering yet
    alerts_early = zone.check_loitering([track1])
    assert len(alerts_early) == 0

    # Wait past threshold
    time.sleep(0.1)
    alerts_late = zone.check_loitering([track1])
    assert len(alerts_late) == 1
    assert alerts_late[0]["track_id"] == 10
    assert alerts_late[0]["threshold_sec"] == 0.08
    assert zone.loitering_incidents == 1

    # Subsequent check should not duplicate alert for same continuous dwell
    alerts_dedup = zone.check_loitering([track1])
    assert len(alerts_dedup) == 0


def test_geofence_exit_resets_dwell():
    pts = [[0, 0], [100, 0], [100, 100], [0, 100]]
    zone = GeofenceZone(polygon_points=pts, label="Corridor")

    inside_track = MockTrack(5, [20, 20, 40, 40])
    zone.update_intrusions([inside_track])
    assert 5 in zone.track_entry_times

    # Move track completely outside (300, 300)
    outside_track = MockTrack(5, [300, 300, 340, 340])
    zone.update_intrusions([outside_track])
    assert 5 not in zone.active_intrusions
    assert 5 not in zone.track_entry_times
    assert zone.get_dwell_time(5) == 0.0


def test_geofence_occupancy_breach():
    pts = [[0, 0], [500, 0], [500, 500], [0, 500]]
    zone = GeofenceZone(polygon_points=pts, label="Server Room", max_occupancy=2)

    t1 = MockTrack(1, [10, 10, 30, 30])
    t2 = MockTrack(2, [40, 40, 60, 60])
    zone.update_intrusions([t1, t2])
    assert zone.check_occupancy_breach() is None  # Exactly at capacity

    # Add 3rd track -> breach
    t3 = MockTrack(3, [70, 70, 90, 90])
    zone.update_intrusions([t1, t2, t3])
    breach = zone.check_occupancy_breach()
    assert breach is not None
    assert breach["current_occupancy"] == 3
    assert breach["max_capacity"] == 2
    assert breach["breach_count"] == 1


def test_velocity_estimator_kinematics():
    # Movement purely East (dx = +100, dy = 0) in 1 second
    p0 = (100, 100)
    p1 = (200, 100)
    vx, vy = VelocityEstimator.calculate_velocity(p0, p1, dt_sec=1.0)
    assert vx == 100.0
    assert vy == 0.0

    speed = VelocityEstimator.calculate_speed(vx, vy)
    assert speed == 100.0

    deg, heading = VelocityEstimator.calculate_heading(p0, p1)
    assert deg == 90.0
    assert heading == "E"

    # Movement purely North (upwards in image coordinates dy = -100)
    p2 = (100, 0)
    deg_n, heading_n = VelocityEstimator.calculate_heading(p0, p2)
    assert deg_n == 0.0
    assert heading_n == "N"

    # Stationary
    deg_stat, heading_stat = VelocityEstimator.calculate_heading(p0, p0)
    assert heading_stat == "STATIONARY"


def test_velocity_estimator_anomaly_detection():
    baseline_speed = 30.0  # e.g., walking
    running_speed = 90.0   # 3x baseline

    assert VelocityEstimator.detect_anomaly(running_speed, baseline_speed, threshold_factor=2.5) is True
    assert VelocityEstimator.detect_anomaly(40.0, baseline_speed, threshold_factor=2.5) is False
    assert VelocityEstimator.detect_anomaly(50.0, baseline_speed=0.0) is False


def test_api_analytics_endpoints(client):
    res_dwell = client.get("/api/analytics/dwell")
    assert res_dwell.status_code == 200
    data_dwell = res_dwell.json()
    assert "dwell_analytics" in data_dwell

    res_occ = client.get("/api/analytics/occupancy")
    assert res_occ.status_code == 200
    data_occ = res_occ.json()
    assert "occupancy_analytics" in data_occ
