"""Read/write for the `planner_tasks` table."""

import datetime
import pandas as pd

from src.db.connection import get_db


def add_task(user_id, subject, topic, deadline, priority, duration_h):
    conn = get_db()
    conn.execute("""
        INSERT INTO planner_tasks
            (user_id,subject,topic,deadline,priority,duration_h,completed,created_at)
        VALUES (?,?,?,?,?,?,0,?)
    """, (user_id, subject, topic, deadline, priority, duration_h,
          datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
    conn.commit()
    conn.close()


def load_tasks(user_id) -> pd.DataFrame:
    conn = get_db()
    df = pd.read_sql_query(
        "SELECT * FROM planner_tasks WHERE user_id=? ORDER BY deadline ASC",
        conn, params=(user_id,)
    )
    conn.close()
    return df


def toggle_task(task_id, current_state):
    conn = get_db()
    conn.execute("UPDATE planner_tasks SET completed=? WHERE id=?",
                 (0 if current_state else 1, task_id))
    conn.commit()
    conn.close()


def delete_task(task_id):
    conn = get_db()
    conn.execute("DELETE FROM planner_tasks WHERE id=?", (task_id,))
    conn.commit()
    conn.close()
