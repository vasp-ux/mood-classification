import os
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=True)

RBAC_DB_URL = os.getenv("RBAC_DB_URL", f"sqlite:///{os.path.join(PROJECT_ROOT, 'data', 'rbac.db')}")
JWT_SECRET = os.getenv("JWT_SECRET", "change-me")
JWT_ALG = os.getenv("JWT_ALG", "HS256")
JWT_EXPIRES_MIN = int(os.getenv("JWT_EXPIRES_MIN", "120"))
DATA_ENC_KEY = os.getenv("DATA_ENC_KEY", "").strip()

SUPERADMIN_EMAIL = os.getenv("SUPERADMIN_EMAIL", "").strip().lower()
SUPERADMIN_PASSWORD = os.getenv("SUPERADMIN_PASSWORD", "").strip()

SYSTEM_LOCK_DEFAULT = os.getenv("SYSTEM_LOCK_DEFAULT", "false").strip().lower() == "true"
