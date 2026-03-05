import json
import uuid
from datetime import datetime, timedelta

from sqlalchemy import select, delete

from .config import SUPERADMIN_EMAIL, SUPERADMIN_PASSWORD, SYSTEM_LOCK_DEFAULT
from .db import engine, SessionLocal
from .models import (
    User,
    Role,
    Permission,
    UserRole,
    RolePermission,
    MoodEntry,
    AIFlag,
    AuditLog,
    ActivityLog,
    SystemSetting,
)
from .security import (
    hash_email,
    hash_password,
    verify_password,
    encrypt_value,
    decrypt_value,
)


PERMISSIONS = {
    "flags:view": "View flagged cases (anonymized)",
    "flags:review": "Review AI-flagged content",
    "users:warn": "Warn users",
    "users:suspend": "Suspend users",
    "cases:escalate": "Escalate severe cases",
    "dashboard:view": "View system dashboard",
    "users:manage": "Manage users",
    "settings:write": "Edit system settings",
    "logs:view": "View system logs",
    "roles:assign": "Assign roles and permissions",
    "keys:manage": "Manage API keys",
    "backup:manage": "Backup/restore database",
    "system:lock": "Lock system during emergency",
    "db:full": "Full database access",
}

ROLE_PERMS = {
    "user": [],
    "moderator": [
        "flags:view",
        "flags:review",
        "users:warn",
        "users:suspend",
        "cases:escalate",
    ],
    "admin": [
        "dashboard:view",
        "users:manage",
        "settings:write",
        "logs:view",
        "flags:view",
    ],
    "superadmin": list(PERMISSIONS.keys()),
}


def init_db():
    from .db import Base

    Base.metadata.create_all(engine)
    ensure_user_columns()
    seed_permissions_and_roles()
    seed_system_settings()
    bootstrap_superadmin()


def ensure_user_columns():
    from sqlalchemy import text
    with engine.connect() as conn:
        try:
            res = conn.execute(text("PRAGMA table_info(users)")).fetchall()
        except Exception:
            return
        cols = {row[1] for row in res}
        if "twofa_secret_enc" not in cols:
            conn.execute(text("ALTER TABLE users ADD COLUMN twofa_secret_enc TEXT"))
        conn.commit()


def seed_permissions_and_roles():
    with SessionLocal() as db:
        existing_perms = {p.name for p in db.execute(select(Permission)).scalars().all()}
        for name, desc in PERMISSIONS.items():
            if name not in existing_perms:
                db.add(Permission(name=name, description=desc))
        db.commit()

        roles = {r.name: r for r in db.execute(select(Role)).scalars().all()}
        for role_name in ROLE_PERMS.keys():
            if role_name not in roles:
                db.add(Role(name=role_name, description=f"{role_name} role"))
        db.commit()

        # Attach permissions to roles
        roles = {r.name: r for r in db.execute(select(Role)).scalars().all()}
        perms = {p.name: p for p in db.execute(select(Permission)).scalars().all()}

        for role_name, perm_list in ROLE_PERMS.items():
            role = roles.get(role_name)
            if not role:
                continue
            current = {p.name for p in role.permissions}
            for perm_name in perm_list:
                perm = perms.get(perm_name)
                if perm and perm.name not in current:
                    role.permissions.append(perm)
        db.commit()


def seed_system_settings():
    with SessionLocal() as db:
        existing = {s.key for s in db.execute(select(SystemSetting)).scalars().all()}
        defaults = {
            "ai_threshold": "0.7",
            "llm_mode": "llm",
            "system_lock": "true" if SYSTEM_LOCK_DEFAULT else "false",
            "enforce_2fa": "false",
            "model_accuracy": "n/a",
        }
        for key, val in defaults.items():
            if key not in existing:
                db.add(SystemSetting(key=key, value=val))
        db.commit()


def bootstrap_superadmin():
    if not SUPERADMIN_EMAIL or not SUPERADMIN_PASSWORD:
        return
    with SessionLocal() as db:
        if get_user_by_email(db, SUPERADMIN_EMAIL):
            return
        user = create_user(
            db,
            email=SUPERADMIN_EMAIL,
            password=SUPERADMIN_PASSWORD,
            name="Super Admin",
            provider="local",
            roles=["superadmin"],
        )
        if user:
            db.commit()


def create_user(db, email, password, name="", provider="local", roles=None):
    if not email or not password:
        return None
    email_h = hash_email(email)
    if get_user_by_email(db, email):
        return None
    anon_id = str(uuid.uuid4())
    user = User(
        email_hash=email_h,
        email_enc=encrypt_value(email),
        name_enc=encrypt_value(name),
        password_hash=hash_password(password),
        provider=provider,
        status="active",
        created_at=datetime.utcnow(),
        last_login=datetime.utcnow(),
        anon_id=anon_id,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    if roles:
        assign_roles(db, user, roles)
    return user


def get_user_by_email(db, email):
    email_h = hash_email(email)
    return db.execute(select(User).where(User.email_hash == email_h)).scalars().first()


def get_user_by_id(db, user_id):
    return db.execute(select(User).where(User.id == user_id)).scalars().first()


def get_user_by_anon_id(db, anon_id):
    return db.execute(select(User).where(User.anon_id == anon_id)).scalars().first()


def assign_roles(db, user, roles):
    role_rows = db.execute(select(Role).where(Role.name.in_(roles))).scalars().all()
    for role in role_rows:
        if role not in user.roles:
            user.roles.append(role)
    db.commit()
    db.refresh(user)
    return user


def update_role_permissions(role_name, perm_names):
    with SessionLocal() as db:
        role = db.execute(select(Role).where(Role.name == role_name)).scalars().first()
        if not role:
            return False
        perms = db.execute(select(Permission).where(Permission.name.in_(perm_names))).scalars().all()
        role.permissions = perms
        db.commit()
        return True


def serialize_user(user, include_email=True):
    data = {
        "id": user.id,
        "anon_id": user.anon_id,
        "status": user.status,
        "provider": user.provider,
        "created_at": user.created_at.isoformat() if user.created_at else "",
        "last_login": user.last_login.isoformat() if user.last_login else "",
        "roles": [r.name for r in user.roles],
        "consent_diary": bool(user.consent_diary),
        "twofa_enabled": bool(user.twofa_secret_enc),
    }
    if include_email:
        data["email"] = decrypt_value(user.email_enc)
        data["name"] = decrypt_value(user.name_enc)
    return data


def authenticate_user(email, password):
    with SessionLocal() as db:
        user = get_user_by_email(db, email)
        if not user or not user.password_hash:
            return None
        if user.status in ("blocked", "deleted"):
            return None
        if user.suspended_until and user.suspended_until > datetime.utcnow():
            return None
        if not verify_password(password, user.password_hash):
            return None
        user.last_login = datetime.utcnow()
        db.commit()
        db.refresh(user)
        return user


def upsert_user_from_auth(email, name="", provider="firebase"):
    if not email:
        return None
    with SessionLocal() as db:
        user = get_user_by_email(db, email)
        if user:
            user.last_login = datetime.utcnow()
            if name:
                user.name_enc = encrypt_value(name)
            if provider:
                user.provider = provider
            db.commit()
            db.refresh(user)
            return user

        anon_id = str(uuid.uuid4())
        user = User(
            email_hash=hash_email(email),
            email_enc=encrypt_value(email),
            name_enc=encrypt_value(name),
            password_hash=None,
            provider=provider,
            status="active",
            created_at=datetime.utcnow(),
            last_login=datetime.utcnow(),
            anon_id=anon_id,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        assign_roles(db, user, ["user"])
        return user


def get_setting(key, default=None):
    with SessionLocal() as db:
        row = db.execute(select(SystemSetting).where(SystemSetting.key == key)).scalars().first()
        return row.value if row else default


def set_setting(key, value, updated_by=None):
    with SessionLocal() as db:
        row = db.execute(select(SystemSetting).where(SystemSetting.key == key)).scalars().first()
        if row:
            row.value = str(value)
            row.updated_at = datetime.utcnow()
            row.updated_by = updated_by
        else:
            db.add(SystemSetting(key=key, value=str(value), updated_by=updated_by))
        db.commit()


def is_system_locked():
    return get_setting("system_lock", "false") == "true"


def record_mood_entry(anon_id, mood, confidence, severity, source="unknown"):
    # Store mood events using anonymized IDs to protect identity.
    if not anon_id:
        return
    with SessionLocal() as db:
        db.add(
            MoodEntry(
                anon_id=anon_id,
                mood=mood or "neutral",
                confidence=float(confidence or 0.0),
                severity=severity or "low",
                source=source or "unknown",
                created_at=datetime.utcnow(),
            )
        )
        db.commit()


def record_activity(anon_id, event_type, mood="", confidence=0.0, detail=""):
    with SessionLocal() as db:
        db.add(
            ActivityLog(
                anon_id=anon_id,
                event_type=event_type,
                mood=mood or "",
                confidence=float(confidence or 0.0),
                detail=detail or "",
                created_at=datetime.utcnow(),
            )
        )
        db.commit()


def flag_content(anon_id, flag_type="high_risk", severity="medium", snippet=""):
    with SessionLocal() as db:
        db.add(
            AIFlag(
                anon_id=anon_id or "",
                flag_type=flag_type,
                severity=severity,
                status="open",
                content_snippet=snippet[:160],
                created_at=datetime.utcnow(),
            )
        )
        db.commit()


def log_audit(actor_id, actor_role, action, target="", detail="", ip=""):
    with SessionLocal() as db:
        db.add(
            AuditLog(
                actor_id=actor_id,
                actor_role=actor_role,
                action=action,
                target=target,
                detail=detail,
                ip=ip,
                created_at=datetime.utcnow(),
            )
        )
        db.commit()


def get_mood_distribution():
    with SessionLocal() as db:
        rows = db.execute(select(MoodEntry)).scalars().all()
    dist = {}
    for row in rows:
        dist[row.mood] = dist.get(row.mood, 0) + 1
    return dist


def get_user_counts():
    with SessionLocal() as db:
        total = db.execute(select(User)).scalars().all()
    active = [u for u in total if u.status == "active"]
    return {"total": len(total), "active": len(active)}


def list_users(limit=500):
    with SessionLocal() as db:
        users = db.execute(select(User)).scalars().all()
    users.sort(key=lambda u: u.created_at or datetime.utcnow(), reverse=True)
    if limit:
        users = users[: int(limit)]
    return users


def list_audit_logs(limit=500):
    with SessionLocal() as db:
        rows = db.execute(select(AuditLog)).scalars().all()
    rows.sort(key=lambda r: r.created_at or datetime.utcnow(), reverse=True)
    if limit:
        rows = rows[: int(limit)]
    return rows


def list_activity_logs(limit=500):
    with SessionLocal() as db:
        rows = db.execute(select(ActivityLog)).scalars().all()
    rows.sort(key=lambda r: r.created_at or datetime.utcnow(), reverse=True)
    if limit:
        rows = rows[: int(limit)]
    return rows


def get_flags(status="open"):
    with SessionLocal() as db:
        q = select(AIFlag)
        if status:
            q = q.where(AIFlag.status == status)
        return db.execute(q).scalars().all()


def update_flag_status(flag_id, status, reviewer_id=None, note=""):
    with SessionLocal() as db:
        flag = db.execute(select(AIFlag).where(AIFlag.id == flag_id)).scalars().first()
        if not flag:
            return None
        flag.status = status
        flag.reviewed_by = reviewer_id
        flag.review_note = note or ""
        db.commit()
        db.refresh(flag)
        return flag


def update_user_status(user_id=None, email=None, status="active", suspend_minutes=None):
    with SessionLocal() as db:
        user = get_user_by_email(db, email) if email else get_user_by_id(db, user_id)
        if not user:
            return None
        user.status = status
        if suspend_minutes:
            user.suspended_until = datetime.utcnow() + timedelta(minutes=int(suspend_minutes))
        db.commit()
        db.refresh(user)
        return user


def reset_user_password(user_id=None, email=None):
    with SessionLocal() as db:
        user = get_user_by_email(db, email) if email else get_user_by_id(db, user_id)
        if not user:
            return None
        temp_password = uuid.uuid4().hex[:10]
        user.password_hash = hash_password(temp_password)
        db.commit()
        return temp_password


def delete_inactive(days=90):
    cutoff = datetime.utcnow() - timedelta(days=days)
    with SessionLocal() as db:
        users = db.execute(select(User)).scalars().all()
        removed = 0
        for user in users:
            if user.last_login and user.last_login < cutoff:
                user.status = "deleted"
                removed += 1
        db.commit()
        return removed
