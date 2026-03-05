import base64
import hashlib
from datetime import datetime, timedelta

import bcrypt
import jwt
from cryptography.fernet import Fernet

from .config import JWT_SECRET, JWT_ALG, JWT_EXPIRES_MIN, DATA_ENC_KEY

# Fernet is used to encrypt sensitive fields at rest (PII like emails/names).
_fernet = None


def get_fernet():
    global _fernet
    if _fernet:
        return _fernet

    key = DATA_ENC_KEY
    if not key:
        # Generate ephemeral key for dev only (data will be unreadable after restart).
        key = Fernet.generate_key().decode("utf-8")
    try:
        _fernet = Fernet(key.encode("utf-8"))
    except Exception:
        # Fallback: if key invalid, derive a deterministic key from JWT secret (dev only).
        derived = base64.urlsafe_b64encode(hashlib.sha256(JWT_SECRET.encode("utf-8")).digest())
        _fernet = Fernet(derived)
    return _fernet


def encrypt_value(value):
    if value is None:
        return ""
    f = get_fernet()
    return f.encrypt(str(value).encode("utf-8")).decode("utf-8")


def decrypt_value(value):
    if not value:
        return ""
    f = get_fernet()
    try:
        return f.decrypt(value.encode("utf-8")).decode("utf-8")
    except Exception:
        return ""


def hash_email(email):
    normalized = (email or "").strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def hash_password(password):
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password, hashed):
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


def create_access_token(user_id, roles, extra=None):
    # JWT access token used for stateless auth; keep expiry short.
    now = datetime.utcnow()
    payload = {
        "sub": user_id,
        "roles": roles,
        "iat": now,
        "exp": now + timedelta(minutes=JWT_EXPIRES_MIN),
    }
    if extra:
        payload.update(extra)
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


def create_temp_token(user_id, roles, minutes=10, purpose="2fa"):
    now = datetime.utcnow()
    payload = {
        "sub": user_id,
        "roles": roles,
        "purpose": purpose,
        "iat": now,
        "exp": now + timedelta(minutes=minutes),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


def decode_token(token):
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
