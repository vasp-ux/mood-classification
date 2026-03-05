import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Text, Boolean, Float, ForeignKey, Integer
from sqlalchemy.orm import relationship
from .db import Base


def _uuid():
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"
    id = Column(String(36), primary_key=True, default=_uuid)
    email_hash = Column(String(64), unique=True, nullable=False, index=True)
    email_enc = Column(Text, nullable=False)
    name_enc = Column(Text, nullable=True)
    password_hash = Column(Text, nullable=True)
    provider = Column(String(32), default="local")
    status = Column(String(20), default="active")
    suspended_until = Column(DateTime, nullable=True)
    twofa_secret_enc = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, default=datetime.utcnow)
    anon_id = Column(String(36), unique=True, index=True)
    consent_diary = Column(Boolean, default=False)

    roles = relationship("Role", secondary="user_roles", back_populates="users")


class Role(Base):
    __tablename__ = "roles"
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True, nullable=False)
    description = Column(Text, nullable=True)

    users = relationship("User", secondary="user_roles", back_populates="roles")
    permissions = relationship("Permission", secondary="role_permissions", back_populates="roles")


class Permission(Base):
    __tablename__ = "permissions"
    id = Column(Integer, primary_key=True)
    name = Column(String(80), unique=True, nullable=False)
    description = Column(Text, nullable=True)

    roles = relationship("Role", secondary="role_permissions", back_populates="permissions")


class UserRole(Base):
    __tablename__ = "user_roles"
    user_id = Column(String(36), ForeignKey("users.id"), primary_key=True)
    role_id = Column(Integer, ForeignKey("roles.id"), primary_key=True)


class RolePermission(Base):
    __tablename__ = "role_permissions"
    role_id = Column(Integer, ForeignKey("roles.id"), primary_key=True)
    permission_id = Column(Integer, ForeignKey("permissions.id"), primary_key=True)


class MoodEntry(Base):
    __tablename__ = "mood_entries"
    id = Column(Integer, primary_key=True)
    anon_id = Column(String(36), index=True, nullable=False)
    mood = Column(String(40), nullable=False)
    confidence = Column(Float, default=0.0)
    severity = Column(String(20), default="low")
    source = Column(String(40), default="unknown")
    created_at = Column(DateTime, default=datetime.utcnow)


class AIFlag(Base):
    __tablename__ = "ai_flags"
    id = Column(Integer, primary_key=True)
    anon_id = Column(String(36), index=True, nullable=False)
    flag_type = Column(String(50), default="high_risk")
    severity = Column(String(20), default="medium")
    status = Column(String(20), default="open")
    content_snippet = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    reviewed_by = Column(String(36), nullable=True)
    review_note = Column(Text, nullable=True)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(Integer, primary_key=True)
    actor_id = Column(String(36), nullable=True)
    actor_role = Column(String(50), nullable=True)
    action = Column(String(80), nullable=False)
    target = Column(String(120), nullable=True)
    detail = Column(Text, nullable=True)
    ip = Column(String(60), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class ActivityLog(Base):
    __tablename__ = "activity_logs"
    id = Column(Integer, primary_key=True)
    anon_id = Column(String(36), index=True, nullable=True)
    event_type = Column(String(60), nullable=False)
    mood = Column(String(40), nullable=True)
    confidence = Column(Float, default=0.0)
    detail = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class SystemSetting(Base):
    __tablename__ = "system_settings"
    id = Column(Integer, primary_key=True)
    key = Column(String(80), unique=True, nullable=False)
    value = Column(Text, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow)
    updated_by = Column(String(36), nullable=True)
