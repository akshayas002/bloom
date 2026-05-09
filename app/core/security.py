"""
Password hashing with bcrypt.
Replaces the original insecure SHA-256 implementation.
bcrypt is deliberately slow — brute-force attacks are computationally infeasible.
"""
import bcrypt


def hash_password(password: str) -> str:
    """Hash a plaintext password. Returns a bcrypt hash string."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(stored: str, password: str) -> bool:
    """Verify a plaintext password against a stored bcrypt hash."""
    try:
        return bcrypt.checkpw(password.encode("utf-8"), stored.encode("utf-8"))
    except Exception:
        return False
