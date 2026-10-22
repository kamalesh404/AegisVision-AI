"""
Integration tests for FastAPI REST API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from app import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_api_status(client):
    r = client.get("/api/status")
    assert r.status_code == 200
    data = r.json()
    assert "mode" in data
    assert "fps" in data
    assert "track_count" in data
    assert "model" in data

def test_api_modes_list(client):
    r = client.get("/api/modes")
    assert r.status_code == 200
    modes = r.json()["modes"]
    assert len(modes) >= 4
    keys = [m["key"] for m in modes]
    assert "traffic" in keys
    assert "safety" in keys

def test_api_mode_switching(client):
    for m in ["safety", "traffic", "retail", "security"]:
        r = client.post("/api/mode", json={"mode": m})
        assert r.status_code == 200
        assert r.json()["mode"] == m

def test_api_config_update(client):
    r = client.post("/api/config", json={"show_trails": False, "show_heatmap": True})
    assert r.status_code == 200
    cfg = r.json()["config"]
    assert cfg["show_trails"] is False
    assert cfg["show_heatmap"] is True

def test_api_custom_zones(client):
    # Add custom zone
    poly = [[50, 50], [250, 50], [250, 250], [50, 250]]
    r_add = client.post("/api/zones", json={"label": "PyTest Zone", "points": poly})
    assert r_add.status_code == 200
    zone_id = r_add.json()["zone_id"]

    # List zones
    r_list = client.get("/api/zones")
    assert r_list.status_code == 200
    zone_ids = [z["id"] for z in r_list.json()["zones"]]
    assert zone_id in zone_ids

    # Delete custom zone
    r_del = client.delete(f"/api/zones/{zone_id}")
    assert r_del.status_code == 200

def test_api_custom_tripwires(client):
    # Add custom tripwire
    r_add = client.post("/api/tripwires", json={"label": "PyTest Line", "pt1": [0, 100], "pt2": [500, 100]})
    assert r_add.status_code == 200
    line_id = r_add.json()["line_id"]

    # List tripwires
    r_list = client.get("/api/tripwires")
    assert r_list.status_code == 200
    tw_ids = [tw["id"] for tw in r_list.json()["tripwires"]]
    assert line_id in tw_ids

    # Delete tripwire
    r_del = client.delete(f"/api/tripwires/{line_id}")
    assert r_del.status_code == 200

def test_api_snapshot(client):
    r = client.get("/api/snapshot")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert len(r.content) > 500
