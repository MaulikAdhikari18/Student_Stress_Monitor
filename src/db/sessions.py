"""Read/write for the `sessions` table — one row per logged stress entry."""

import datetime
import pandas as pd

from src.db.connection import get_db


def save_session(user_id, stress_score, stress_level,
                  sleep, study, screen, anxiety, exercise,
                  entry_date=None):
    if entry_date is None:
        entry_date = datetime.date.today()
    ts = datetime.datetime.combine(entry_date, datetime.time(12, 0))
    timestamp = ts.strftime("%Y-%m-%d %H:%M")
    day_label = ts.strftime("%a %d %b")

    conn = get_db()
    conn.execute("""
        INSERT INTO sessions
            (user_id,timestamp,day_label,stress_score,stress_level,
             sleep,study,screen,anxiety,exercise)
        VALUES (?,?,?,?,?,?,?,?,?,?)
    """, (
        user_id, timestamp, day_label,
        stress_score, stress_level,
        sleep, study, screen, anxiety, exercise
    ))
    conn.commit()
    conn.close()


def load_sessions(user_id) -> pd.DataFrame:
    conn = get_db()
    df = pd.read_sql_query(
        "SELECT * FROM sessions WHERE user_id=? ORDER BY id ASC",
        conn, params=(user_id,)
    )
    conn.close()
    return df
