"""
User accounts and authentication.

NOTE: password hashing is plain unsalted SHA-256, ported as-is from the
original dashboard.py. Kept identical here so existing user rows keep
working. If you want to harden this later (bcrypt/passlib with per-user
salt), it needs a one-time migration for existing accounts — flag it and
I'll do that as a separate step rather than silently breaking logins.
"""

import hashlib
import datetime
import sqlite3

from src.db.connection import get_db


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def create_user(username: str, password: str) -> tuple[bool, str]:
    if len(username.strip()) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?,?,?)",
            (
                username.strip().lower(),
                hash_password(password),
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        conn.commit()
        return True, "Account created successfully!"
    except sqlite3.IntegrityError:
        return False, "Username already taken. Please choose another."
    finally:
        conn.close()


def verify_user(username: str, password: str) -> dict | None:
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM users WHERE username=? AND password_hash=?",
        (username.strip().lower(), hash_password(password)),
    ).fetchone()
    conn.close()
    return dict(row) if row else None
