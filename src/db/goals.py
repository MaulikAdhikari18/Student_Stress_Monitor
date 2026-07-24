"""Read/write for the `goals` table — one row per user."""

import datetime

from src.db.connection import get_db

DEFAULT_GOALS = {
    "goal_sleep": 8.0,
    "goal_study": 8.0,
    "goal_exercise": 4,
    "goal_screen": 4.0,
}


def load_goals(user_id) -> dict:
    conn = get_db()
    row = conn.execute("SELECT * FROM goals WHERE user_id=?", (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else dict(DEFAULT_GOALS)


def save_goals_db(user_id, goal_sleep, goal_study, goal_exercise, goal_screen):
    conn = get_db()
    conn.execute("""
        INSERT INTO goals (user_id,goal_sleep,goal_study,goal_exercise,goal_screen,updated_at)
        VALUES (?,?,?,?,?,?)
        ON CONFLICT(user_id) DO UPDATE SET
            goal_sleep=excluded.goal_sleep, goal_study=excluded.goal_study,
            goal_exercise=excluded.goal_exercise, goal_screen=excluded.goal_screen,
            updated_at=excluded.updated_at
    """, (user_id, goal_sleep, goal_study, goal_exercise, goal_screen,
          datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
    conn.commit()
    conn.close()
