def test_legacy_server_has_no_game_routes(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from apps.playground import server

    def fake_load(resident):
        resident.phase = "ready"

    monkeypatch.setattr(server, "ROOT", tmp_path)
    monkeypatch.setattr(server.Resident, "load", fake_load)
    app = server.create_playground(tmp_path / "checkpoint", "cpu", tmp_path / "cache")
    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000)) as client:
        headers = {"X-QEV-Playground": "1"}
        assert client.get("/", headers=headers).status_code == 200
        assert client.get("/api/status", headers=headers).json()["phase"] == "ready"
        assert client.get("/2048", headers=headers).status_code == 404
        assert client.post("/api/2048/games", json={}, headers=headers).status_code in (404, 405)
