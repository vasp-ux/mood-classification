import json
import uuid
from datetime import datetime, timedelta

from flask import Blueprint, jsonify, request, g
from sqlalchemy import delete
import pyotp

from .middleware import auth_required, require_permissions, require_roles
from .security import create_access_token, create_temp_token, decode_token, decrypt_value, encrypt_value, hash_email
from .services import (
    authenticate_user,
    serialize_user,
    list_users,
    list_audit_logs,
    list_activity_logs,
    get_user_counts,
    get_mood_distribution,
    get_setting,
    set_setting,
    log_audit,
    update_user_status,
    reset_user_password,
    delete_inactive,
    get_flags,
    update_flag_status,
    flag_content,
    get_user_by_email,
    assign_roles,
    create_user,
    get_user_by_anon_id,
    update_role_permissions,
)
from .db import SessionLocal
from .models import User, UserRole, SystemSetting, MoodEntry, AIFlag, AuditLog, ActivityLog

rbac_bp = Blueprint("rbac", __name__)


def _client_ip():
    return request.headers.get("X-Forwarded-For", request.remote_addr or "")


@rbac_bp.post("/auth/login")
def login():
    payload = request.get_json(silent=True) or {}
    email = str(payload.get("email", "")).strip().lower()
    password = str(payload.get("password", "")).strip()
    user = authenticate_user(email, password)
    if not user:
        return jsonify({"error": "Invalid credentials"}), 401
    roles = [r.name for r in user.roles]
    enforce_2fa = get_setting("enforce_2fa", "false") == "true"
    is_admin = any(r in ("admin", "superadmin") for r in roles)
    if enforce_2fa and is_admin:
        temp = create_temp_token(user.id, roles, minutes=10, purpose="2fa")
        setup_required = not bool(user.twofa_secret_enc)
        return jsonify(
            {
                "2fa_required": True,
                "setup_required": setup_required,
                "temp_token": temp,
                "roles": roles,
                "enforce_2fa": True,
            }
        ), 200

    token = create_access_token(user.id, roles)
    log_audit(user.id, roles[0] if roles else "", "login", target=email, ip=_client_ip())
    return jsonify(
        {
            "token": token,
            "roles": roles,
            "enforce_2fa": enforce_2fa,
        }
    ), 200


@rbac_bp.get("/auth/me")
@auth_required
def me():
    user = g.current_user
    return jsonify(serialize_user(user, include_email=True)), 200


@rbac_bp.post("/auth/2fa/setup")
def setup_2fa():
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return jsonify({"error": "Missing token"}), 401
    token = auth.split(" ", 1)[1].strip()
    try:
        payload = decode_token(token)
    except Exception:
        return jsonify({"error": "Invalid token"}), 401
    if payload.get("purpose") != "2fa":
        return jsonify({"error": "Invalid token"}), 401
    user_id = payload.get("sub")
    if not user_id:
        return jsonify({"error": "Invalid token"}), 401

    with SessionLocal() as db:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return jsonify({"error": "User not found"}), 404
        secret = pyotp.random_base32()
        user.twofa_secret_enc = encrypt_value(secret)
        db.commit()
        email = decrypt_value(user.email_enc)
        uri = pyotp.totp.TOTP(secret).provisioning_uri(email, issuer_name="MoodSense")
    return jsonify({"secret": secret, "otpauth_url": uri}), 200


@rbac_bp.post("/auth/2fa/verify")
def verify_2fa():
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return jsonify({"error": "Missing token"}), 401
    token = auth.split(" ", 1)[1].strip()
    try:
        payload = decode_token(token)
    except Exception:
        return jsonify({"error": "Invalid token"}), 401
    if payload.get("purpose") != "2fa":
        return jsonify({"error": "Invalid token"}), 401

    user_id = payload.get("sub")
    code = str((request.get_json(silent=True) or {}).get("code", "")).strip()
    if not user_id or not code:
        return jsonify({"error": "Missing code"}), 400

    with SessionLocal() as db:
        user = db.query(User).filter(User.id == user_id).first()
        if not user or not user.twofa_secret_enc:
            return jsonify({"error": "2FA not configured"}), 400
        secret = decrypt_value(user.twofa_secret_enc)
        if not pyotp.TOTP(secret).verify(code, valid_window=1):
            return jsonify({"error": "Invalid code"}), 401
        roles = [r.name for r in user.roles]
        access = create_access_token(user.id, roles)
        log_audit(user.id, roles[0] if roles else "", "login_2fa", target=user.id, ip=_client_ip())
        return jsonify({"token": access, "roles": roles}), 200


# -------------------- Moderator --------------------
@rbac_bp.get("/moderator/flags")
@require_permissions("flags:view")
def moderator_flags():
    flags = get_flags(status="open")
    payload = [
        {
            "id": f.id,
            "anon_id": f.anon_id,
            "flag_type": f.flag_type,
            "severity": f.severity,
            "status": f.status,
            "created_at": f.created_at.isoformat(),
            "content_snippet": f.content_snippet or "",
        }
        for f in flags
    ]
    return jsonify(payload), 200


@rbac_bp.post("/moderator/warn")
@require_permissions("users:warn")
def moderator_warn():
    payload = request.get_json(silent=True) or {}
    anon_id = str(payload.get("anon_id", "")).strip()
    note = str(payload.get("note", "")).strip()
    with SessionLocal() as db:
        user = get_user_by_anon_id(db, anon_id)
        if not user:
            return jsonify({"error": "User not found"}), 404
        user.status = "warned"
        db.commit()
    log_audit(g.current_user.id, "moderator", "warn_user", target=anon_id, detail=note, ip=_client_ip())
    return jsonify({"ok": True}), 200


@rbac_bp.post("/moderator/suspend")
@require_permissions("users:suspend")
def moderator_suspend():
    payload = request.get_json(silent=True) or {}
    anon_id = str(payload.get("anon_id", "")).strip()
    minutes = int(payload.get("minutes", 60))
    with SessionLocal() as db:
        user = get_user_by_anon_id(db, anon_id)
        if not user:
            return jsonify({"error": "User not found"}), 404
        user.status = "suspended"
        user.suspended_until = datetime.utcnow() + timedelta(minutes=minutes)
        db.commit()
    log_audit(g.current_user.id, "moderator", "suspend_user", target=anon_id, detail=str(minutes), ip=_client_ip())
    return jsonify({"ok": True}), 200


@rbac_bp.post("/moderator/escalate")
@require_permissions("cases:escalate")
def moderator_escalate():
    payload = request.get_json(silent=True) or {}
    flag_id = int(payload.get("flag_id", 0))
    note = str(payload.get("note", "")).strip()
    flag = update_flag_status(flag_id, "escalated", reviewer_id=g.current_user.id, note=note)
    if not flag:
        return jsonify({"error": "Flag not found"}), 404
    log_audit(g.current_user.id, "moderator", "escalate_flag", target=str(flag_id), detail=note, ip=_client_ip())
    return jsonify({"ok": True}), 200


# -------------------- Admin --------------------
@rbac_bp.get("/admin/dashboard")
@require_permissions("dashboard:view")
def admin_dashboard():
    counts = get_user_counts()
    dist = get_mood_distribution()
    model_acc = get_setting("model_accuracy", "n/a")
    ai_threshold = get_setting("ai_threshold", "0.7")
    llm_mode = get_setting("llm_mode", "llm")
    return jsonify(
        {
            "user_count": counts,
            "mood_distribution": dist,
            "model_accuracy": model_acc,
            "ai_threshold": ai_threshold,
            "llm_mode": llm_mode,
        }
    ), 200


@rbac_bp.get("/admin/users")
@require_permissions("users:manage")
def admin_users():
    try:
        limit = int(request.args.get("limit", 200))
    except (TypeError, ValueError):
        limit = 200
    users = [serialize_user(u, include_email=True) for u in list_users(limit=limit)]
    return jsonify(users), 200


@rbac_bp.post("/admin/users/block")
@require_permissions("users:manage")
def admin_block():
    payload = request.get_json(silent=True) or {}
    email = payload.get("email")
    user_id = payload.get("user_id")
    status = payload.get("status", "blocked")
    user = update_user_status(user_id=user_id, email=email, status=status)
    if not user:
        return jsonify({"error": "User not found"}), 404
    log_audit(g.current_user.id, "admin", "block_user", target=user.id, detail=status, ip=_client_ip())
    return jsonify({"ok": True}), 200


@rbac_bp.post("/admin/users/reset-password")
@require_permissions("users:manage")
def admin_reset_password():
    payload = request.get_json(silent=True) or {}
    email = payload.get("email")
    user_id = payload.get("user_id")
    temp = reset_user_password(user_id=user_id, email=email)
    if not temp:
        return jsonify({"error": "User not found"}), 404
    log_audit(g.current_user.id, "admin", "reset_password", target=str(user_id or email), ip=_client_ip())
    return jsonify({"temp_password": temp}), 200


@rbac_bp.post("/admin/users/delete-inactive")
@require_permissions("users:manage")
def admin_delete_inactive():
    payload = request.get_json(silent=True) or {}
    days = int(payload.get("days", 90))
    removed = delete_inactive(days=days)
    log_audit(g.current_user.id, "admin", "delete_inactive", detail=str(days), ip=_client_ip())
    return jsonify({"removed": removed}), 200


@rbac_bp.post("/admin/settings")
@require_permissions("settings:write")
def admin_settings():
    payload = request.get_json(silent=True) or {}
    for key in ["ai_threshold", "llm_mode", "model_accuracy"]:
        if key in payload:
            set_setting(key, payload.get(key), updated_by=g.current_user.id)
    log_audit(g.current_user.id, "admin", "update_settings", detail=json.dumps(payload), ip=_client_ip())
    return jsonify({"ok": True}), 200


@rbac_bp.get("/admin/logs")
@require_permissions("logs:view")
def admin_logs():
    logs = list_audit_logs(limit=500)
    payload = [
        {
            "action": l.action,
            "actor_id": l.actor_id,
            "actor_role": l.actor_role,
            "target": l.target,
            "detail": l.detail,
            "created_at": l.created_at.isoformat(),
        }
        for l in logs
    ]
    return jsonify(payload), 200


@rbac_bp.get("/admin/activity")
@require_permissions("logs:view")
def admin_activity():
    rows = list_activity_logs(limit=500)
    payload = [
        {
            "anon_id": a.anon_id,
            "event_type": a.event_type,
            "mood": a.mood,
            "confidence": a.confidence,
            "detail": a.detail,
            "created_at": a.created_at.isoformat() if a.created_at else "",
        }
        for a in rows
    ]
    return jsonify(payload), 200


# -------------------- Super Admin --------------------
@rbac_bp.post("/superadmin/users")
@require_roles("superadmin")
def superadmin_create_user():
    payload = request.get_json(silent=True) or {}
    email = str(payload.get("email", "")).strip().lower()
    password = str(payload.get("password", "")).strip()
    roles = payload.get("roles", ["admin"])
    name = payload.get("name", "")
    with SessionLocal() as db:
        user = create_user(db, email, password, name=name, provider="local", roles=roles)
        if not user:
            return jsonify({"error": "Could not create user"}), 400
        log_audit(g.current_user.id, "superadmin", "create_user", target=user.id, ip=_client_ip())
    return jsonify({"ok": True}), 200


@rbac_bp.post("/superadmin/roles/assign")
@require_roles("superadmin")
def superadmin_assign_role():
    payload = request.get_json(silent=True) or {}
    email = str(payload.get("email", "")).strip().lower()
    roles = payload.get("roles", [])
    with SessionLocal() as db:
        user = get_user_by_email(db, email)
        if not user:
            return jsonify({"error": "User not found"}), 404
        assign_roles(db, user, roles)
        log_audit(g.current_user.id, "superadmin", "assign_roles", target=user.id, detail=",".join(roles), ip=_client_ip())
    return jsonify({"ok": True}), 200


@rbac_bp.post("/superadmin/roles/permissions")
@require_roles("superadmin")
def superadmin_update_role_permissions():
    payload = request.get_json(silent=True) or {}
    role = str(payload.get("role", "")).strip()
    perms = payload.get("permissions", [])
    if not role:
        return jsonify({"error": "Missing role"}), 400
    ok = update_role_permissions(role, perms)
    if not ok:
        return jsonify({"error": "Role not found"}), 404
    log_audit(g.current_user.id, "superadmin", "update_role_permissions", target=role, detail=",".join(perms), ip=_client_ip())
    return jsonify({"ok": True}), 200


@rbac_bp.post("/superadmin/lock")
@require_roles("superadmin")
def superadmin_lock():
    payload = request.get_json(silent=True) or {}
    lock = bool(payload.get("lock", False))
    set_setting("system_lock", "true" if lock else "false", updated_by=g.current_user.id)
    log_audit(g.current_user.id, "superadmin", "system_lock", detail=str(lock), ip=_client_ip())
    return jsonify({"ok": True}), 200


@rbac_bp.post("/superadmin/2fa")
@require_roles("superadmin")
def superadmin_2fa():
    payload = request.get_json(silent=True) or {}
    enforce = bool(payload.get("enforce", False))
    set_setting("enforce_2fa", "true" if enforce else "false", updated_by=g.current_user.id)
    log_audit(g.current_user.id, "superadmin", "enforce_2fa", detail=str(enforce), ip=_client_ip())
    return jsonify({"ok": True}), 200


@rbac_bp.get("/superadmin/backup")
@require_roles("superadmin")
def superadmin_backup():
    with SessionLocal() as db:
        payload = {
            "users": [
                {
                    "id": u.id,
                    "anon_id": u.anon_id,
                    "email": decrypt_value(u.email_enc),
                    "name": decrypt_value(u.name_enc),
                    "email_hash": u.email_hash,
                    "password_hash": u.password_hash,
                    "provider": u.provider,
                    "status": u.status,
                    "suspended_until": u.suspended_until.isoformat() if u.suspended_until else "",
                    "created_at": u.created_at.isoformat() if u.created_at else "",
                    "last_login": u.last_login.isoformat() if u.last_login else "",
                    "consent_diary": bool(u.consent_diary),
                    "roles": [r.name for r in u.roles],
                }
                for u in db.query(User).all()
            ],
            "settings": [{"key": s.key, "value": s.value} for s in db.query(SystemSetting).all()],
            "mood_entries": [
                {
                    "anon_id": m.anon_id,
                    "mood": m.mood,
                    "confidence": m.confidence,
                    "severity": m.severity,
                    "source": m.source,
                    "created_at": m.created_at.isoformat(),
                }
                for m in db.query(MoodEntry).all()
            ],
            "flags": [
                {
                    "anon_id": f.anon_id,
                    "flag_type": f.flag_type,
                    "severity": f.severity,
                    "status": f.status,
                    "content_snippet": f.content_snippet,
                    "created_at": f.created_at.isoformat(),
                }
                for f in db.query(AIFlag).all()
            ],
            "audit_logs": [
                {
                    "action": a.action,
                    "actor_id": a.actor_id,
                    "actor_role": a.actor_role,
                    "target": a.target,
                    "detail": a.detail,
                    "created_at": a.created_at.isoformat(),
                }
                for a in db.query(AuditLog).all()
            ],
            "activity_logs": [
                {
                    "anon_id": a.anon_id,
                    "event_type": a.event_type,
                    "mood": a.mood,
                    "confidence": a.confidence,
                    "detail": a.detail,
                    "created_at": a.created_at.isoformat(),
                }
                for a in db.query(ActivityLog).all()
            ],
        }
    encrypted = encrypt_value(json.dumps(payload))
    log_audit(g.current_user.id, "superadmin", "backup", ip=_client_ip())
    return jsonify({"backup": encrypted}), 200


@rbac_bp.post("/superadmin/restore")
@require_roles("superadmin")
def superadmin_restore():
    payload = request.get_json(silent=True) or {}
    backup = payload.get("backup", "")
    if not backup:
        return jsonify({"error": "Missing backup payload"}), 400
    try:
        data = json.loads(decrypt_value(backup))
    except Exception:
        return jsonify({"error": "Invalid backup"}), 400

    with SessionLocal() as db:
        db.execute(delete(ActivityLog))
        db.execute(delete(AuditLog))
        db.execute(delete(AIFlag))
        db.execute(delete(MoodEntry))
        db.execute(delete(SystemSetting))
        db.execute(delete(UserRole))
        db.execute(delete(User))
        db.commit()

        for s in data.get("settings", []):
            db.add(SystemSetting(key=s.get("key", ""), value=s.get("value", "")))

        # Restore users
        for u in data.get("users", []):
            email = u.get("email", "")
            user = User(
                id=u.get("id") or None,
                anon_id=u.get("anon_id") or str(uuid.uuid4()),
                email_hash=u.get("email_hash") or hash_email(email),
                email_enc=encrypt_value(email),
                name_enc=encrypt_value(u.get("name", "")),
                password_hash=u.get("password_hash"),
                provider=u.get("provider", "local"),
                status=u.get("status", "active"),
                created_at=datetime.fromisoformat(u.get("created_at")) if u.get("created_at") else datetime.utcnow(),
                last_login=datetime.fromisoformat(u.get("last_login")) if u.get("last_login") else datetime.utcnow(),
                consent_diary=bool(u.get("consent_diary")),
            )
            db.add(user)
        db.commit()

        # Restore mood entries, flags, logs, activity
        for m in data.get("mood_entries", []):
            db.add(
                MoodEntry(
                    anon_id=m.get("anon_id", ""),
                    mood=m.get("mood", "neutral"),
                    confidence=float(m.get("confidence", 0.0)),
                    severity=m.get("severity", "low"),
                    source=m.get("source", "unknown"),
                    created_at=datetime.fromisoformat(m.get("created_at")) if m.get("created_at") else datetime.utcnow(),
                )
            )
        for f in data.get("flags", []):
            db.add(
                AIFlag(
                    anon_id=f.get("anon_id", ""),
                    flag_type=f.get("flag_type", "high_risk"),
                    severity=f.get("severity", "medium"),
                    status=f.get("status", "open"),
                    content_snippet=f.get("content_snippet", ""),
                    created_at=datetime.fromisoformat(f.get("created_at")) if f.get("created_at") else datetime.utcnow(),
                )
            )
        for a in data.get("audit_logs", []):
            db.add(
                AuditLog(
                    action=a.get("action", ""),
                    actor_id=a.get("actor_id"),
                    actor_role=a.get("actor_role"),
                    target=a.get("target"),
                    detail=a.get("detail"),
                    created_at=datetime.fromisoformat(a.get("created_at")) if a.get("created_at") else datetime.utcnow(),
                )
            )
        for a in data.get("activity_logs", []):
            db.add(
                ActivityLog(
                    anon_id=a.get("anon_id"),
                    event_type=a.get("event_type", ""),
                    mood=a.get("mood", ""),
                    confidence=float(a.get("confidence", 0.0)),
                    detail=a.get("detail", ""),
                    created_at=datetime.fromisoformat(a.get("created_at")) if a.get("created_at") else datetime.utcnow(),
                )
            )
        db.commit()

    log_audit(g.current_user.id, "superadmin", "restore", ip=_client_ip())
    return jsonify({"ok": True}), 200
