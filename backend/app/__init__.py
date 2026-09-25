"""PixFix Flask application factory (Application Tier)."""
import logging

import click
from flask import Flask, jsonify
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from .config import Config
from .extensions import bcrypt, cors, db, jwt


def create_app(config_object=Config) -> Flask:
    app = Flask(__name__)
    app.config.from_object(config_object)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    for sub in ("originals", "masks", "results"):
        (app.config["STORAGE_DIR"] / sub).mkdir(parents=True, exist_ok=True)

    db.init_app(app)
    jwt.init_app(app)
    bcrypt.init_app(app)
    cors.init_app(app, resources={r"/api/*": {"origins": app.config["CORS_ORIGINS"]}})

    from .auth.routes import bp as auth_bp
    from .images.routes import bp as images_bp
    from .admin.routes import bp as admin_bp

    app.register_blueprint(auth_bp, url_prefix="/api/auth")
    app.register_blueprint(images_bp, url_prefix="/api/images")
    app.register_blueprint(admin_bp, url_prefix="/api/admin")

    _register_jwt_callbacks()
    _register_error_handlers(app)
    _register_cli(app)

    @app.get("/api/health")
    def health():
        from .ai.engine import get_engine
        engine = get_engine(app)
        return jsonify(status="ok", ai_backends=engine.available_backends(),
                       default_backend=engine.default_backend, device=str(engine.device))

    with app.app_context():
        from . import models  # noqa: F401  (register models)
        db.create_all()

    return app


def _register_jwt_callbacks():
    from .models import User

    @jwt.user_lookup_loader
    def _load_user(_header, data):
        user = db.session.get(User, int(data["sub"]))
        return user if user and user.is_active else None

    @jwt.user_lookup_error_loader
    def _lookup_error(_header, _data):
        return jsonify(error="User not found or deactivated"), 401

    @jwt.unauthorized_loader
    def _missing(reason):
        return jsonify(error=f"Authentication required: {reason}"), 401

    @jwt.invalid_token_loader
    def _invalid(reason):
        return jsonify(error=f"Invalid token: {reason}"), 401

    @jwt.expired_token_loader
    def _expired(_h, _d):
        return jsonify(error="Session expired, please log in again"), 401


def _register_error_handlers(app: Flask):
    @app.errorhandler(RequestEntityTooLarge)
    def _too_large(_e):
        mb = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
        return jsonify(error=f"File too large (max {mb} MB)"), 413

    @app.errorhandler(HTTPException)
    def _http(e):
        return jsonify(error=e.description), e.code

    @app.errorhandler(Exception)
    def _unhandled(e):
        app.logger.exception("Unhandled error")
        return jsonify(error="Internal server error"), 500


def _register_cli(app: Flask):
    from .models import User

    @app.cli.command("create-admin")
    @click.option("--email", prompt=True)
    @click.option("--username", prompt=True)
    @click.option("--password", prompt=True, hide_input=True, confirmation_prompt=True)
    def create_admin(email, username, password):
        """Create (or promote) an administrator account."""
        user = User.query.filter_by(email=email.lower()).first()
        if user:
            user.is_admin = True
            click.echo(f"Promoted existing user {email} to admin.")
        else:
            user = User(email=email.lower(), username=username, is_admin=True)
            user.set_password(password)
            db.session.add(user)
            click.echo(f"Created admin {email}.")
        db.session.commit()
