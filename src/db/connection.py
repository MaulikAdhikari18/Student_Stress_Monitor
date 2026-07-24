"""Single place that knows how to open a connection to the SQLite database."""

import sqlite3
from src.config import DB_PATH


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn
