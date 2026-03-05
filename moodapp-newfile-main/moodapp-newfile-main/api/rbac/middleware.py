from functools import wraps
from flask import request, jsonify, g
import jwt

from .db import SessionLocal
from .models import User
from .security import decode_token
from .services import is_system_locked


def _get_token():
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth.split(" ", 1)[1].strip()
    return ""


def _load_user():
    token = _get_token()
    if not token:
        return None
    try:
        payload = decode_token(token)
    except jwt.InvalidTokenError:
        return None
    user_id = payload.get("sub")
    if not user_id:
        return None
    with SessionLocal() as db:
        user = db.query(User).filter(User.id == user_id).first()
        return user


def auth_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        # JWT verification gate for protected endpoints.
        user = _load_user()
        if not user:
            return jsonify({"error": "Unauthorized"}), 401
        g.current_user = user
        return fn(*args, **kwargs)

    return wrapper


def require_roles(*roles):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            # Role gate (RBAC) + emergency system lock enforcement.
            user = _load_user()
            if not user:
                return jsonify({"error": "Unauthorized"}), 401
            g.current_user = user
            user_roles = {r.name for r in user.roles}
            if not user_roles.intersection(set(roles)):
                return jsonify({"error": "Forbidden"}), 403
            if is_system_locked() and "superadmin" not in user_roles:
                return jsonify({"error": "System locked"}), 423
            return fn(*args, **kwargs)

        return wrapper

    return decorator


def require_permissions(*perms):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            user = _load_user()
            if not user:
                return jsonify({"error": "Unauthorized"}), 401
            g.current_user = user
            user_perms = set()
            for role in user.roles:
                user_perms.update({p.name for p in role.permissions})
            if not set(perms).issubset(user_perms):
                return jsonify({"error": "Forbidden"}), 403
            if is_system_locked() and "superadmin" not in {r.name for r in user.roles}:
                return jsonify({"error": "System locked"}), 423
            return fn(*args, **kwargs)

        return wrapper

    return decorator
