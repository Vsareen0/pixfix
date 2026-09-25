from conftest import register


def test_register_login_me(client):
    r = register(client)
    assert r.status_code == 201
    body = r.get_json()
    assert body["user"]["email"] == "alice@example.com"
    assert "password" not in body["user"]

    r = client.post("/api/auth/login", json={"email": "ALICE@example.com", "password": "secret123"})
    assert r.status_code == 200
    token = r.get_json()["token"]

    r = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200 and r.get_json()["user"]["username"] == "alice"


def test_password_is_hashed(app, client):
    from app.models import User
    register(client)
    with app.app_context():
        u = User.query.first()
        assert u.password != "secret123" and u.password.startswith("$2")


def test_duplicate_and_validation(client):
    register(client)
    assert register(client).status_code == 409
    r = register(client, email="bad", password="short", username="x")
    assert r.status_code == 400
    assert set(r.get_json()["fields"]) == {"email", "password", "username"}


def test_wrong_password(client):
    register(client)
    r = client.post("/api/auth/login", json={"email": "alice@example.com", "password": "nope12345"})
    assert r.status_code == 401


def test_protected_requires_token(client):
    assert client.get("/api/images").status_code == 401
    assert client.get("/api/auth/me", headers={"Authorization": "Bearer garbage"}).status_code == 401
