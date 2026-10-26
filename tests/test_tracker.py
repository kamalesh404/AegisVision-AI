"""
Unit tests for CentroidIoUTracker and TrackedObject.
"""
from core.tracker import CentroidIoUTracker, calculate_iou

def test_iou_calculation():
    boxA = [0, 0, 10, 10]
    boxB = [0, 0, 10, 10]
    assert calculate_iou(boxA, boxB) == 1.0

    boxC = [10, 10, 20, 20]
    assert calculate_iou(boxA, boxC) == 0.0

    boxD = [0, 0, 10, 5]
    assert 0.4 < calculate_iou(boxA, boxD) < 0.6

def test_tracker_creation_and_update():
    tracker = CentroidIoUTracker(max_distance=50, max_missed=5)
    
    # Frame 1: One object
    detections_f1 = [
        {"bbox": [100, 100, 150, 150], "cls_id": 0, "cls_name": "person", "conf": 0.9}
    ]
    tracks = tracker.update(detections_f1)
    assert len(tracks) == 1
    t1 = tracks[0]
    assert t1.track_id == 1
    assert t1.cls_name == "person"
    assert t1.centroid == (125, 125)

    # Frame 2: Object moves slightly
    detections_f2 = [
        {"bbox": [105, 105, 155, 155], "cls_id": 0, "cls_name": "person", "conf": 0.92}
    ]
    tracks = tracker.update(detections_f2)
    assert len(tracks) == 1
    assert tracks[0].track_id == 1  # ID persists!
    assert len(tracks[0].history) == 2
    assert tracks[0].centroid == (130, 130)

def test_tracker_missed_and_cleanup():
    tracker = CentroidIoUTracker(max_distance=50, max_missed=2)
    tracker.update([{"bbox": [50, 50, 80, 80], "cls_id": 2, "cls_name": "car", "conf": 0.8}])
    assert tracker.active_count() == 1

    # Miss 1
    tracker.update([])
    assert tracker.active_count() == 1

    # Miss 2
    tracker.update([])
    assert tracker.active_count() == 1

    # Miss 3: Should be pruned (exceeds max_missed)
    tracker.update([])
    assert tracker.active_count() == 0
