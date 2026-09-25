from test_images import _process


def test_non_admin_forbidden(client, auth):
    assert client.get("/api/admin/stats", headers=auth).status_code == 403


def test_reports(client, auth, admin_auth):
    _process(client, auth)
    _process(client, auth, model="lama")  # fails: not installed

    s = client.get("/api/admin/stats", headers=admin_auth).get_json()
    assert s["total_users"] == 2 and s["total_images"] == 2
    assert s["status_counts"]["done"] == 1 and s["status_counts"]["failed"] == 1
    assert s["active_today"] >= 1 and s["total_storage_bytes"] > 0

    logs = client.get("/api/admin/logs", headers=admin_auth).get_json()
    assert logs["total"] == 2 and logs["items"][0]["username"] == "alice"

    errors = client.get("/api/admin/logs?status=failed", headers=admin_auth).get_json()
    assert errors["total"] == 1 and errors["items"][0]["error_message"]

    rep = client.get("/api/admin/reports/processing-time", headers=admin_auth).get_json()
    assert rep["per_model"][0]["model"] == "opencv" and rep["failed"] == 1
    assert len(rep["per_day"]) == 1


def test_manage_users(client, auth, admin_auth):
    users = client.get("/api/admin/users", headers=admin_auth).get_json()["items"]
    alice = next(u for u in users if u["email"] == "alice@example.com")

    r = client.patch(f"/api/admin/users/{alice['user_id']}", json={"is_active": False},
                     headers=admin_auth)
    assert r.status_code == 200
    # deactivated user's token stops working and login is refused
    assert client.get("/api/auth/me", headers=auth).status_code == 401
    r = client.post("/api/auth/login", json={"email": "alice@example.com", "password": "secret123"})
    assert r.status_code == 403

    assert client.delete(f"/api/admin/users/{alice['user_id']}", headers=admin_auth).status_code == 200
    users = client.get("/api/admin/users", headers=admin_auth).get_json()["items"]
    assert len(users) == 1


def test_admin_cannot_demote_self(client, admin_auth):
    me = client.get("/api/auth/me", headers=admin_auth).get_json()["user"]
    r = client.patch(f"/api/admin/users/{me['user_id']}", json={"is_admin": False},
                     headers=admin_auth)
    assert r.status_code == 400
