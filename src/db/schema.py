"""
Table definitions. init_db() is idempotent (CREATE TABLE IF NOT EXISTS),
so it's safe to call on every app startup.
"""

from src.db.connection import get_db


def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT    UNIQUE NOT NULL,
            password_hash TEXT    NOT NULL,
            created_at    TEXT    NOT NULL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id       INTEGER NOT NULL,
            timestamp     TEXT    NOT NULL,
            day_label     TEXT,
            stress_score  REAL,
            stress_level  TEXT,
            sleep         REAL,
            study         REAL,
            screen        REAL,
            anxiety       INTEGER,
            exercise      INTEGER,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS goals (
            user_id       INTEGER PRIMARY KEY,
            goal_sleep    REAL    DEFAULT 8.0,
            goal_study    REAL    DEFAULT 8.0,
            goal_exercise INTEGER DEFAULT 4,
            goal_screen   REAL    DEFAULT 4.0,
            updated_at    TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    conn.close()


def init_planner_table():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS planner_tasks (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            subject     TEXT    NOT NULL,
            topic       TEXT,
            deadline    TEXT    NOT NULL,
            priority    TEXT    DEFAULT 'Medium',
            duration_h  REAL    DEFAULT 1.0,
            completed   INTEGER DEFAULT 0,
            created_at  TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    conn.close()


def init_all():
    """Call once at app startup — creates every table if missing."""
    init_db()
    init_planner_table()
