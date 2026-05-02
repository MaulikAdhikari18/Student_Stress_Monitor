import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os
import pickle
import sqlite3
import hashlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Student Stress Monitor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* ── Global ── */
    .main-title {
        font-size:2rem; font-weight:700;
        background:linear-gradient(135deg,#534AB7,#D4537E);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    }
    .auth-title { font-size:1.4rem; font-weight:700; color:#534AB7; margin-bottom:0.2rem; }
    .auth-sub   { font-size:0.88rem; color:#888; margin-bottom:1.4rem; }
    .section-header {
        font-size:1.05rem; font-weight:700; color:#AFA9EC;
        letter-spacing:0.04em; text-transform:uppercase;
        margin:1.4rem 0 0.8rem; display:flex; align-items:center; gap:8px;
    }
    .section-header::after {
        content:''; flex:1; height:1px;
        background:linear-gradient(90deg,rgba(175,169,236,0.3),transparent);
    }

    /* ── Stress result box ── */
    .stress-box { padding:1.2rem 1.5rem; border-radius:16px; border-left:6px solid; margin-bottom:1rem; }
    .box-low      { background:rgba(99,153,34,0.13);  border-color:#639922; }
    .box-moderate { background:rgba(186,117,23,0.13); border-color:#BA7517; }
    .box-high     { background:rgba(153,60,29,0.13);  border-color:#993C1D; }
    .box-critical { background:rgba(163,45,45,0.13);  border-color:#A32D2D; }

    /* ── Snapshot stat cards ── */
    .snap-grid {
        display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:1rem;
    }
    .snap-card {
        background:rgba(255,255,255,0.04);
        border:1px solid rgba(255,255,255,0.09);
        border-radius:14px; padding:1rem 1.1rem;
        display:flex; flex-direction:column; gap:4px;
    }
    .snap-label { font-size:0.75rem; font-weight:600; color:#888; text-transform:uppercase; letter-spacing:0.06em; }
    .snap-value { font-size:1.55rem; font-weight:800; color:#fff; line-height:1.1; }
    .snap-sub   { font-size:0.78rem; color:#666; }

    /* ── Chart cards ── */
    .chart-card {
        background:rgba(255,255,255,0.03);
        border:1px solid rgba(255,255,255,0.08);
        border-radius:16px; padding:1.1rem 1.2rem 0.6rem;
        margin-bottom:1rem;
    }
    .chart-title {
        font-size:0.82rem; font-weight:700; color:#AFA9EC;
        text-transform:uppercase; letter-spacing:0.07em; margin-bottom:0.5rem;
    }

    /* ── History stat summary cards ── */
    .hist-stat-grid {
        display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-bottom:1.2rem;
    }
    .hist-stat-card {
        background:rgba(83,74,183,0.12);
        border:1px solid rgba(175,169,236,0.18);
        border-radius:14px; padding:1rem 1.1rem;
        text-align:center;
    }
    .hist-stat-val   { font-size:1.8rem; font-weight:800; color:#AFA9EC; line-height:1; }
    .hist-stat-label { font-size:0.75rem; color:#888; margin-top:4px; text-transform:uppercase; letter-spacing:0.05em; }

    /* ── Recommendation cards (fixed height grid) ── */
    .rec-grid {
        display:grid;
        grid-template-columns:repeat(4,1fr);
        gap:12px;
        margin-bottom:1rem;
    }
    .rec-card {
        border-radius:16px;
        padding:1.1rem 1.1rem 1rem;
        display:flex; flex-direction:column; gap:6px;
        min-height:210px;
        position:relative; overflow:hidden;
        border-top:3px solid transparent;
    }
    .rec-critical { background:rgba(163,45,45,0.16); border-top-color:#E24B4A; }
    .rec-high     { background:rgba(186,117,23,0.16); border-top-color:#FAC775; }
    .rec-moderate { background:rgba(83,74,183,0.14);  border-top-color:#7F77DD; }
    .rec-positive { background:rgba(99,153,34,0.14);  border-top-color:#97C459; }
    .rec-icon     { font-size:1.6rem; line-height:1; }
    .rec-badge {
        font-size:0.65rem; font-weight:700; padding:2px 8px; border-radius:20px;
        text-transform:uppercase; letter-spacing:0.06em; width:fit-content;
    }
    .badge-critical { background:rgba(226,75,74,0.25); color:#F09595; }
    .badge-high     { background:rgba(250,199,117,0.25); color:#FAC775; }
    .badge-moderate { background:rgba(127,119,221,0.25); color:#AFA9EC; }
    .badge-positive { background:rgba(151,196,89,0.25); color:#C0DD97; }
    .rec-title  { font-size:0.88rem; font-weight:700; line-height:1.3; }
    .rec-body   { font-size:0.8rem; opacity:0.78; line-height:1.5; flex:1; }
    .rec-action {
        font-size:0.75rem; font-weight:600; padding:5px 9px;
        border-radius:7px; background:rgba(255,255,255,0.07);
        margin-top:auto; display:block; line-height:1.4;
    }
    .rec-bar-track { height:3px; border-radius:2px; background:rgba(255,255,255,0.07); margin-top:6px; overflow:hidden; }

    /* ── Goal cards ── */
    .goal-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-bottom:1rem; }
    .goal-card {
        background:rgba(255,255,255,0.04);
        border:1px solid rgba(255,255,255,0.09);
        border-radius:14px; padding:1rem 1.1rem;
    }
    .goal-title { font-size:0.75rem; font-weight:600; opacity:0.65; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:6px; }
    .goal-value { font-size:1.5rem; font-weight:800; line-height:1.1; }
    .goal-sub   { font-size:0.78rem; opacity:0.55; margin-top:2px; }
    .streak-badge {
        display:inline-block; padding:2px 9px; border-radius:20px;
        font-size:0.72rem; font-weight:600;
        background:rgba(99,153,34,0.2); color:#97C459; margin-left:6px;
    }
    .streak-zero { background:rgba(128,128,128,0.12); color:#666; }

    /* ── User chip ── */
    .user-chip {
        display:inline-block; background:rgba(83,74,183,0.2); color:#AFA9EC;
        border-radius:20px; padding:3px 12px; font-size:0.85rem; font-weight:600;
    }

    /* ── Planner day cards ── */
    .day-card {
        border:1px solid rgba(255,255,255,0.09);
        border-radius:14px; padding:0.9rem 1rem; margin-bottom:0.6rem;
        background:rgba(255,255,255,0.03);
    }
    .day-header {
        font-size:0.82rem; font-weight:700; color:#AFA9EC;
        margin-bottom:0.5rem; letter-spacing:0.04em; text-transform:uppercase;
    }
    .task-row {
        display:flex; align-items:center; gap:8px;
        padding:5px 0; border-bottom:1px solid rgba(255,255,255,0.05);
        font-size:0.86rem;
    }
    .task-row:last-child { border-bottom:none; }
    .pri-high   { background:rgba(163,45,45,0.2);  color:#F09595; border-radius:4px; padding:1px 7px; font-size:0.72rem; font-weight:700; }
    .pri-medium { background:rgba(186,117,23,0.2); color:#FAC775; border-radius:4px; padding:1px 7px; font-size:0.72rem; font-weight:700; }
    .pri-low    { background:rgba(99,153,34,0.2);  color:#C0DD97; border-radius:4px; padding:1px 7px; font-size:0.72rem; font-weight:700; }

    /* ── Break/insight boxes ── */
    .break-box {
        background:rgba(83,74,183,0.11); border-left:4px solid #7F77DD;
        border-radius:0 12px 12px 0; padding:0.9rem 1.1rem; margin-bottom:1rem;
    }
    .break-stat { font-size:1.9rem; font-weight:800; color:#AFA9EC; display:inline-block; margin-right:0.4rem; }
    .insight-box {
        background:linear-gradient(135deg,rgba(83,74,183,0.18),rgba(212,83,126,0.12));
        border:1px solid rgba(175,169,236,0.25);
        border-radius:14px; padding:1.1rem 1.3rem; margin-bottom:1rem;
    }
    .insight-title { font-size:0.95rem; font-weight:700; color:#AFA9EC; margin-bottom:0.3rem; }
    .insight-body  { font-size:0.86rem; opacity:0.83; line-height:1.6; }

    /* ── Timer ── */
    .timer-container {
        background:linear-gradient(135deg,rgba(83,74,183,0.16),rgba(212,83,126,0.10));
        border:1px solid rgba(175,169,236,0.25);
        border-radius:20px; padding:1.8rem 1.5rem; text-align:center; margin-bottom:1rem;
    }
    .timer-display {
        font-size:4.2rem; font-weight:800; font-family:'Courier New',monospace;
        background:linear-gradient(135deg,#AFA9EC,#D4537E);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
        letter-spacing:0.05em; line-height:1;
    }
    .timer-label {
        font-size:0.8rem; font-weight:600; color:#AFA9EC;
        text-transform:uppercase; letter-spacing:0.1em; margin-top:0.4rem; opacity:0.8;
    }
    .timer-phase-study { border-top:3px solid #639922; }
    .timer-phase-break { border-top:3px solid #534AB7; }
    .session-log-row {
        display:flex; align-items:center; gap:10px; padding:6px 10px;
        border-radius:8px; background:rgba(255,255,255,0.03); margin-bottom:3px; font-size:0.82rem;
    }

    /* ── Factor analysis cards ── */
    .factor-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-bottom:1rem; }
    .factor-card {
        background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08);
        border-radius:14px; padding:0.9rem 1rem;
    }
    .factor-name  { font-size:0.75rem; font-weight:600; opacity:0.65; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:6px; }
    .factor-score { font-size:1.6rem; font-weight:800; line-height:1; margin-bottom:6px; }
    .factor-bar-track { height:5px; border-radius:3px; background:rgba(255,255,255,0.08); overflow:hidden; }
    .factor-bar-fill  { height:100%; border-radius:3px; }

    /* ── tip-box kept for compatibility ── */
    .tip-box {
        background:rgba(83,74,183,0.10); border-left:4px solid #7F77DD;
        border-radius:0 8px 8px 0; padding:0.75rem 1rem; margin-bottom:0.5rem; font-size:0.9rem;
    }
    .tip-box strong { color:#AFA9EC; }
</style>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# DATABASE LAYER
# ═════════════════════════════════════════════════════════════════════════════

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH  = os.path.join(DATA_DIR, 'stress_monitor.db')


def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


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


init_db()

# ── Auth helpers ──────────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def create_user(username: str, password: str):
    if len(username.strip()) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?,?,?)",
            (username.strip().lower(), hash_password(password),
             datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        conn.commit()
        return True, "Account created successfully!"
    except sqlite3.IntegrityError:
        return False, "Username already taken. Please choose another."
    finally:
        conn.close()


def verify_user(username: str, password: str):
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM users WHERE username=? AND password_hash=?",
        (username.strip().lower(), hash_password(password))
    ).fetchone()
    conn.close()
    return dict(row) if row else None


# ── Session data helpers ──────────────────────────────────────────────────────

def save_session(user_id, stress_score, stress_level,
                 sleep, study, screen, anxiety, exercise):
    conn = get_db()
    conn.execute("""
        INSERT INTO sessions
            (user_id,timestamp,day_label,stress_score,stress_level,
             sleep,study,screen,anxiety,exercise)
        VALUES (?,?,?,?,?,?,?,?,?,?)
    """, (
        user_id,
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        datetime.datetime.now().strftime("%a %d %b"),
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


# ── Goals helpers ─────────────────────────────────────────────────────────────

def load_goals(user_id) -> dict:
    conn = get_db()
    row = conn.execute("SELECT * FROM goals WHERE user_id=?", (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else {"goal_sleep":8.0,"goal_study":8.0,
                                   "goal_exercise":4,"goal_screen":4.0}


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


# ── Planner DB helpers ───────────────────────────────────────────────────────

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

init_planner_table()


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


def generate_weekly_schedule(tasks_df, stress_score, daily_study_limit):
    """
    Distribute pending tasks across the next 7 days based on:
    - deadline urgency
    - task priority
    - daily study hour cap (adjusted down if stress is high)
    Returns a dict {date_str: [task_rows]}
    """
    today = datetime.date.today()
    days  = [(today + datetime.timedelta(days=i)) for i in range(7)]

    # Stress-adjusted daily cap
    if stress_score >= 75:
        cap = max(1.5, daily_study_limit * 0.5)
    elif stress_score >= 55:
        cap = max(2.0, daily_study_limit * 0.7)
    else:
        cap = daily_study_limit

    pending = tasks_df[tasks_df['completed'] == 0].copy()
    if pending.empty:
        return {d.strftime("%a %d %b"): [] for d in days}, cap

    # Priority weight for sorting
    pri_weight = {'High': 0, 'Medium': 1, 'Low': 2}
    pending['pri_w'] = pending['priority'].map(pri_weight).fillna(1)
    pending['deadline_dt'] = pd.to_datetime(pending['deadline'], errors='coerce')
    pending = pending.sort_values(['deadline_dt', 'pri_w'])

    schedule = {d.strftime("%a %d %b"): [] for d in days}
    daily_used = {d.strftime("%a %d %b"): 0.0 for d in days}

    for _, task in pending.iterrows():
        try:
            dl = pd.to_datetime(task['deadline']).date()
        except Exception:
            dl = today + datetime.timedelta(days=6)

        assigned = False
        for day in days:
            day_str = day.strftime("%a %d %b")
            if day > dl:
                break
            if daily_used[day_str] + task['duration_h'] <= cap:
                schedule[day_str].append(task)
                daily_used[day_str] += task['duration_h']
                assigned = True
                break

        # If couldn't fit before deadline, place on earliest available day
        if not assigned:
            for day in days:
                day_str = day.strftime("%a %d %b")
                if daily_used[day_str] < cap:
                    schedule[day_str].append(task)
                    daily_used[day_str] += task['duration_h']
                    break

    return schedule, cap


def get_break_schedule(stress_score):
    """Return recommended study block & break duration based on stress."""
    if stress_score >= 75:
        return 20, 10, "🔴 Critical stress — short blocks, frequent breaks"
    elif stress_score >= 55:
        return 25, 8,  "🟠 High stress — Pomodoro 25/8 recommended"
    elif stress_score >= 30:
        return 35, 7,  "🟡 Moderate stress — 35 min focus, 7 min break"
    else:
        return 50, 10, "🟢 Low stress — deep work 50/10 recommended"


# ═════════════════════════════════════════════════════════════════════════════
# AUTH PAGE
# ═════════════════════════════════════════════════════════════════════════════

def show_auth_page():
    st.markdown("""
    <div style="text-align:center;padding:2.5rem 0 1rem;">
        <div style="font-size:3rem;">🧠</div>
        <div class="main-title" style="text-align:center;display:block;font-size:2.2rem;">
            Student Stress Monitor
        </div>
        <div style="color:#888;font-size:0.95rem;margin-top:0.4rem;">
            AI-powered stress tracking • personalized tips • goal setting
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_m, col_r = st.columns([1, 1.1, 1])
    with col_m:
        mode = st.radio("", ["🔑 Login", "✨ Create account"],
                        horizontal=True, label_visibility="collapsed")
        st.markdown("")

        if mode == "🔑 Login":
            with st.form("login_form"):
                st.markdown('<div class="auth-title">Welcome back 👋</div>'
                            '<div class="auth-sub">Sign in to your account to continue</div>',
                            unsafe_allow_html=True)
                username = st.text_input("Username", placeholder="enter your username")
                password = st.text_input("Password", type="password",
                                         placeholder="enter your password")
                submitted = st.form_submit_button("Sign in →",
                                                  use_container_width=True,
                                                  type="primary")
                if submitted:
                    if not username or not password:
                        st.error("Please fill in both fields.")
                    else:
                        user = verify_user(username, password)
                        if user:
                            st.session_state["user"] = user
                            st.rerun()
                        else:
                            st.error("❌ Incorrect username or password.")

        else:
            with st.form("signup_form"):
                st.markdown('<div class="auth-title">Create your account ✨</div>'
                            '<div class="auth-sub">Start tracking your stress today — free & private</div>',
                            unsafe_allow_html=True)
                new_username = st.text_input("Choose a username",
                                             placeholder="at least 3 characters")
                new_password = st.text_input("Choose a password", type="password",
                                             placeholder="at least 6 characters")
                confirm_pw   = st.text_input("Confirm password", type="password",
                                             placeholder="repeat your password")
                submitted = st.form_submit_button("Create account →",
                                                  use_container_width=True,
                                                  type="primary")
                if submitted:
                    if not new_username or not new_password or not confirm_pw:
                        st.error("Please fill in all fields.")
                    elif new_password != confirm_pw:
                        st.error("❌ Passwords don't match.")
                    else:
                        ok, msg = create_user(new_username, new_password)
                        if ok:
                            st.success(f"✅ {msg} You can now sign in.")
                        else:
                            st.error(f"❌ {msg}")

    st.markdown("""
    <div style="text-align:center;color:#ccc;font-size:0.78rem;margin-top:2.5rem;">
        🔒 Passwords are hashed with SHA-256 and never stored in plain text
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# CONSTANTS & ML MODEL
# ═════════════════════════════════════════════════════════════════════════════

LABELS     = ['Low','Moderate','High','Critical']
COLORS     = {'Low':'#639922','Moderate':'#BA7517','High':'#993C1D','Critical':'#A32D2D'}
EMOJIS     = {'Low':'😊','Moderate':'😐','High':'😟','Critical':'😰'}
MODELS_DIR = os.path.join(BASE_DIR, 'models')


@st.cache_resource
def load_model():
    mp = os.path.join(MODELS_DIR,'model.pkl')
    sp = os.path.join(MODELS_DIR,'scaler.pkl')
    tp = os.path.join(MODELS_DIR,'meta.pkl')
    if not os.path.exists(mp): return None,None,None
    with open(mp,'rb') as f: model  = pickle.load(f)
    with open(sp,'rb') as f: scaler = pickle.load(f)
    with open(tp,'rb') as f: meta   = pickle.load(f)
    return model, scaler, meta


def compute_streaks(hdf, g_sleep, g_study, g_exercise, g_screen):
    streaks = {"sleep":0,"study":0,"exercise":0,"screen":0}
    if hdf.empty: return streaks
    for col,goal_val,direction,key in [
        ("sleep",g_sleep,"gte","sleep"),("study",g_study,"lte","study"),
        ("exercise",g_exercise,"gte","exercise"),("screen",g_screen,"lte","screen"),
    ]:
        if col not in hdf.columns: continue
        vals = pd.to_numeric(hdf[col],errors='coerce').dropna().tolist()
        s = 0
        for v in reversed(vals):
            if (v>=goal_val if direction=="gte" else v<=goal_val): s+=1
            else: break
        streaks[key] = s
    return streaks


def week_progress(hdf, g_sleep, g_study, g_exercise, g_screen):
    pct = {"sleep":0,"study":0,"exercise":0,"screen":0}
    if hdf.empty: return pct
    recent = hdf.tail(7)
    for col,goal_val,direction,key in [
        ("sleep",g_sleep,"gte","sleep"),("study",g_study,"lte","study"),
        ("exercise",g_exercise,"gte","exercise"),("screen",g_screen,"lte","screen"),
    ]:
        if col not in recent.columns: continue
        vals = pd.to_numeric(recent[col],errors='coerce').dropna()
        if len(vals)==0: continue
        met = (vals>=goal_val) if direction=="gte" else (vals<=goal_val)
        pct[key] = int(met.sum()/len(vals)*100)
    return pct


# ═════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═════════════════════════════════════════════════════════════════════════════

def show_main_app(user: dict):
    model, scaler, meta = load_model()
    MODEL_READY = model is not None
    user_id  = user["id"]
    username = user["username"]

    # ── Header ───────────────────────────────────────────────────────────────
    h1, h2 = st.columns([5,1])
    with h1:
        st.markdown('<div class="main-title">🧠 Student Stress Monitor</div>',
                    unsafe_allow_html=True)
        st.caption("AI-powered stress prediction • personalized tips • trend tracking")
    with h2:
        st.markdown(f'<div style="text-align:right;padding-top:0.5rem;">'
                    f'<span class="user-chip">👤 {username}</span></div>',
                    unsafe_allow_html=True)
        if st.button("Sign out", use_container_width=True):
            del st.session_state["user"]
            st.rerun()

    history_df  = load_sessions(user_id)
    saved_goals = load_goals(user_id)

    if MODEL_READY:
        acc = meta.get('accuracy',0)
        c1,c2,c3 = st.columns(3)
        c1.metric("ML Model",       meta.get('best_model','Loaded'))
        c2.metric("Model Accuracy", f"{acc*100:.1f}%")
        c3.metric("Sessions logged",str(len(history_df)))
    else:
        st.warning("⚠️ ML model not found. Run `python src/train_model.py` to enable AI predictions.")

    st.divider()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"### 👤 {username}")
        st.markdown("---")
        st.header("📥 Enter Today's Data")

        st.markdown("#### 📚 Academic")
        study       = st.slider("Study hours / day",           0.0,16.0,6.0,0.5)
        assignments = st.slider("Assignments pending",          0,  15,  3)
        exam        = st.slider("Exam pressure (1–10)",         1,  10,  5)
        performance = st.slider("Academic performance (1–10)",  1,  10,  7)

        st.markdown("#### 🏃 Lifestyle")
        sleep    = st.slider("Sleep hours / night",             2.0,12.0,7.0,0.5)
        exercise = st.slider("Exercise days / week",            0,  7,   3)
        social   = st.slider("Social interactions / week",      0,  20,  5)
        screen   = st.slider("Screen time hours / day",         0.0,16.0,4.0,0.5)

        st.markdown("#### 🧠 Mental & Social")
        anxiety  = st.slider("Anxiety level (1–10)",            1,  10,  4)
        finance  = st.slider("Financial stress (1–10)",         1,  10,  3)
        family   = st.slider("Family support (1–10)",           1,  10,  7)
        peer     = st.slider("Peer pressure (1–10)",            1,  10,  4)
        extra    = st.selectbox("Extracurricular activities",[0,1,2],
                    format_func=lambda x:['None','1–2 activities','3+ activities'][x])
        rel      = st.selectbox("Relationship situation",[0,1,2],
                    format_func=lambda x:['Single / N/A','Stable relationship','Relationship issues'][x])

        st.divider()
        save_btn = st.button("💾 Save Today's Entry",
                             use_container_width=True, type="primary")

    # ── Stress computation ─────────────────────────────────────────────────────
    raw = (
        max(0,study-8)*3.5 + assignments*2.5 + (exam-1)*5.0
        + max(0,7-sleep)*4.0 + max(0,5-exercise)*2.0
        + max(0,8-social)*1.5 + max(0,screen-4)*2.0
        + (anxiety-1)*4.5 + (finance-1)*3.0
        - (family-1)*2.5 - (performance-1)*2.0 + (peer-1)*2.5
        + (5 if extra==0 else 0) + (8 if rel==2 else 0)
    )
    stress_score = int(np.clip(raw,0,100))

    if MODEL_READY:
        inp = np.array([[study,assignments,exam,performance,
                         sleep,exercise,social,screen,
                         anxiety,finance,family,peer,extra,rel]])
        inp_sc     = scaler.transform(inp)
        pred_class = int(model.predict(inp_sc)[0])
        pred_proba = model.predict_proba(inp_sc)[0]
        level_name = LABELS[pred_class]
    else:
        pred_proba = None
        if stress_score>74:   level_name='Critical'
        elif stress_score>54: level_name='High'
        elif stress_score>29: level_name='Moderate'
        else:                 level_name='Low'

    level_color = COLORS[level_name]
    level_emoji = EMOJIS[level_name]

    # ── Save entry ────────────────────────────────────────────────────────────
    if save_btn:
        save_session(user_id, stress_score, level_name,
                     sleep, study, screen, anxiety, exercise)
        st.success("✅ Entry saved!")
        st.rerun()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
        "📊 Stress Result","🔍 Factor Analysis",
        "💡 Management Tips","📈 My History","🎯 My Goals","📅 Study Planner"
    ])

    # ══════════════════════════════════════════════════════════
    # TAB 1 — Stress Result
    # ══════════════════════════════════════════════════════════
    with tab1:
        col_left,col_right = st.columns([1.4,1])
        with col_left:
            st.markdown(f"""
            <div class="stress-box box-{level_name.lower()}">
                <h2 style="margin:0;color:{level_color};">{level_emoji} {level_name} Stress</h2>
                <p style="margin:0.4rem 0 0;font-size:1.05rem;">
                    Stress score: <strong>{stress_score} / 100</strong>
                </p>
            </div>""", unsafe_allow_html=True)

            if MODEL_READY and pred_proba is not None:
                st.markdown("#### Confidence across levels")
                for lbl,p in zip(LABELS,pred_proba):
                    ca,cb = st.columns([3,1])
                    ca.progress(float(p),text=lbl)
                    cb.write(f"**{p*100:.1f}%**")
            else:
                st.markdown("#### Score breakdown")
                st.write(f"- Sleep deficit: `{max(0,round(7-sleep,1))}h below target`")
                st.write(f"- Study overload: `{max(0,round(study-8,1))}h above 8h`")
                st.write(f"- Screen excess: `{max(0,round(screen-4,1))}h above 4h`")

        with col_right:
            if MODEL_READY and pred_proba is not None:
                fig,ax = plt.subplots(figsize=(4,4))
                ax.pie(pred_proba, labels=LABELS,
                       colors=['#639922','#EF9F27','#D85A30','#E24B4A'],
                       autopct='%1.1f%%', startangle=140,
                       wedgeprops={'linewidth':1,'edgecolor':'white'})
                ax.set_title('Probability Distribution',fontsize=11,pad=8)
                st.pyplot(fig,use_container_width=True); plt.close()
            else:
                fig,ax = plt.subplots(figsize=(4,1.5))
                ax.barh(['Stress'],[stress_score],color=level_color,height=0.5)
                ax.barh(['Stress'],[100-stress_score],left=[stress_score],
                        color='#e8e8e8',height=0.5)
                ax.set_xlim(0,100); ax.axis('off')
                ax.set_title(f'Score: {stress_score}',fontsize=12)
                st.pyplot(fig,use_container_width=True); plt.close()

        st.divider()
        st.markdown("#### Quick health snapshot")
        m1,m2,m3,m4 = st.columns(4)
        sleep_status = "Optimal ✅" if sleep>=8 else f"-{8-sleep:.1f}h ⚠️"
        study_load   = ["Light","Moderate","Heavy","Extreme"][min(3,int(study//4))]
        recovery     = int(((exercise/7)*0.4+(sleep/10)*0.4+(social/20)*0.2)*100)
        burnout      = min(100,int(stress_score*0.6+max(0,study-8)*4+max(0,10-sleep)*3))
        m1.metric("Sleep status",  sleep_status)
        m2.metric("Study load",    study_load)
        m3.metric("Recovery score",f"{recovery}%")
        m4.metric("Burnout risk",  f"{burnout}/100")

    # ══════════════════════════════════════════════════════════
    # TAB 2 — Factor Analysis
    # ══════════════════════════════════════════════════════════
    with tab2:
        factor_scores = {
            "Academic load":    min(100,int(study/16*50+assignments/15*30+exam/10*20)),
            "Sleep deficit":    min(100,int(max(0,8-sleep)/6*100)),
            "Anxiety":          int(anxiety/10*100),
            "Financial strain": int(finance/10*100),
            "Social isolation": min(100,int(max(0,10-family)/9*70+max(0,8-social)/8*30)),
            "Peer pressure":    int(peer/10*100),
            "Screen overuse":   min(100,int(max(0,screen-4)/12*100)),
            "Exercise deficit": min(100,int(max(0,5-exercise)/5*100)),
        }
        sorted_f = sorted(factor_scores.items(), key=lambda x: x[1], reverse=True)

        factor_icons = {
            "Academic load":"📚","Sleep deficit":"😴","Anxiety":"🧘",
            "Financial strain":"💰","Social isolation":"👥","Peer pressure":"🤝",
            "Screen overuse":"📱","Exercise deficit":"🏃"
        }

        # ── 4-per-row factor cards ─────────────────────────────
        st.markdown('<div class="section-header">Stress Factor Breakdown</div>', unsafe_allow_html=True)
        cards_html = '<div class="factor-grid">'
        for name, val in sorted_f:
            bar_color = ('#E24B4A' if val >= 75 else '#FAC775' if val >= 55
                         else '#7F77DD' if val >= 30 else '#97C459')
            score_color = ('#F09595' if val >= 75 else '#FAC775' if val >= 55
                           else '#AFA9EC' if val >= 30 else '#C0DD97')
            icon = factor_icons.get(name, '📊')
            cards_html += f"""
            <div class="factor-card">
                <div class="factor-name">{icon} {name}</div>
                <div class="factor-score" style="color:{score_color};">{val}</div>
                <div class="factor-bar-track">
                    <div class="factor-bar-fill" style="width:{val}%;background:{bar_color};"></div>
                </div>
                <div style="font-size:0.72rem;color:#666;margin-top:4px;">
                    {'Critical' if val>=75 else 'High' if val>=55 else 'Moderate' if val>=30 else 'Low'}
                </div>
            </div>"""
        cards_html += '</div>'
        st.markdown(cards_html, unsafe_allow_html=True)

        # ── Plotly radar of factors ────────────────────────────
        st.markdown('<div class="section-header">Factor Radar</div>', unsafe_allow_html=True)
        f_names = [f[0] for f in sorted_f]
        f_vals  = [f[1] for f in sorted_f]
        fig_fa = go.Figure()
        fig_fa.add_trace(go.Scatterpolar(
            r=f_vals + [f_vals[0]], theta=f_names + [f_names[0]],
            fill='toself', fillcolor='rgba(83,74,183,0.18)',
            line=dict(color='#AFA9EC', width=2), name='Your Score'
        ))
        fig_fa.add_trace(go.Scatterpolar(
            r=[50]*len(f_names) + [50], theta=f_names + [f_names[0]],
            fill='toself', fillcolor='rgba(226,75,74,0.05)',
            line=dict(color='#E24B4A', width=1.2, dash='dot'), name='Warning (50)'
        ))
        fig_fa.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0,100],
                                tickfont=dict(size=9, color='#888'),
                                gridcolor='rgba(255,255,255,0.08)'),
                angularaxis=dict(tickfont=dict(size=10, color='#ccc'),
                                 gridcolor='rgba(255,255,255,0.1)'),
                bgcolor='rgba(0,0,0,0)'
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(font=dict(color='#ccc'), bgcolor='rgba(0,0,0,0)',
                        orientation='h', y=-0.1),
            margin=dict(t=30, b=50, l=60, r=60), height=400
        )
        col_r, col_l = st.columns([3, 2])
        with col_r:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.plotly_chart(fig_fa, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col_l:
            st.markdown('<div class="section-header">Today\'s Inputs</div>', unsafe_allow_html=True)
            snap=[("📚 Study hrs/day",f"{study}h"),("😴 Sleep hrs",f"{sleep}h"),
                  ("🧘 Anxiety",f"{anxiety}/10"),("🏃 Exercise",f"{exercise} days"),
                  ("📝 Assignments",str(assignments)),("💰 Financial stress",f"{finance}/10"),
                  ("❤️ Family support",f"{family}/10"),("📱 Screen time",f"{screen}h")]
            snap_html = '<div class="snap-grid">'
            for label, val in snap:
                snap_html += f'<div class="snap-card"><div class="snap-label">{label}</div><div class="snap-value">{val}</div></div>'
            snap_html += '</div>'
            st.markdown(snap_html, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # TAB 3 — Management Tips (Advanced)
    # ══════════════════════════════════════════════════════════
    with tab3:

        # ── Smart insight summary ──────────────────────────────
        top_issue = ""
        issue_score = 0
        if sleep < 6 and (7 - sleep) * 15 > issue_score:
            issue_score = int((7 - sleep) * 15); top_issue = "sleep deprivation"
        if anxiety > 7 and anxiety * 9 > issue_score:
            issue_score = int(anxiety * 9); top_issue = "high anxiety"
        if study > 10 and (study - 8) * 8 > issue_score:
            issue_score = int((study - 8) * 8); top_issue = "study overload"
        if not top_issue:
            top_issue = "a generally balanced profile"

        recovery_days = (
            "2–3 days" if stress_score < 35
            else "4–5 days" if stress_score < 60
            else "7–10 days"
        )
        insight_text = (
            f"Your stress score of **{stress_score}/100** places you in the **{level_name}** zone. "
            f"The primary driver appears to be **{top_issue}**. "
            f"With consistent corrective action, meaningful improvement typically takes **{recovery_days}**. "
            f"Focus on the Critical and High-priority recommendations below first."
        ) if stress_score > 20 else (
            f"Your stress indicators look well-balanced (score: **{stress_score}/100**). "
            f"Maintain your current routines and log daily to catch any early drift."
        )

        st.markdown(f"""
        <div class="insight-box">
            <div class="insight-title">🧠 AI Stress Insight</div>
            <div class="insight-body">{insight_text}</div>
        </div>""", unsafe_allow_html=True)

        # ── Severity-graded recommendations ───────────────────
        CRITICAL, HIGH, MODERATE, POSITIVE = "critical", "high", "moderate", "positive"

        recs = []

        # Sleep
        if sleep < 5:
            recs.append((CRITICAL, "😴", "Severe Sleep Deficit",
                f"Only {sleep}h sleep — this is a medical concern. Cognitive function drops 30%+ below 5h.",
                "Go to bed in the next 2 hours. No exceptions tonight.",
                int((7 - sleep) / 5 * 100)))
        elif sleep < 7:
            recs.append((HIGH, "😴", "Sleep Below Threshold",
                f"You're getting {sleep}h vs the recommended 7–9h. This raises cortisol and impairs memory consolidation.",
                "Set a hard lights-out alarm. Try 4-7-8 breathing (inhale 4s, hold 7s, exhale 8s).",
                int((7 - sleep) / 3 * 70)))
        elif sleep >= 8:
            recs.append((POSITIVE, "😴", "Great Sleep",
                f"Excellent — {sleep}h of sleep supports memory, mood, and immune function.",
                "Keep your consistent sleep schedule.", 0))

        # Study load
        if study > 12:
            recs.append((CRITICAL, "📚", "Dangerous Study Load",
                f"{study}h/day is unsustainable and counterproductive. Retention collapses after 6–7h of quality study.",
                "Cut to max 6h today. Use Pomodoro 25/5. Schedule mandatory end-time.",
                min(100, int((study - 8) * 12))))
        elif study > 8:
            recs.append((HIGH, "📚", "Study Overload Risk",
                f"{study}h/day is above the effective threshold. Quality matters more than quantity.",
                "Cap at 8h. Use active recall and spaced repetition for higher retention.",
                min(100, int((study - 8) * 8))))

        # Exercise
        if exercise < 2:
            recs.append((HIGH, "🏃", "Critical Exercise Deficit",
                "0–1 exercise days/week significantly raises stress hormones. Exercise is one of the strongest natural anxiolytics.",
                "Start with a 20-min walk today. You don't need a gym — just movement.",
                80))
        elif exercise < 3:
            recs.append((MODERATE, "🏃", "Increase Physical Activity",
                f"{exercise} exercise days/week is below the recommended 3–5. Even light activity reduces cortisol by ~26%.",
                "Add 2 more days this week. Try a 15-min YouTube workout.",
                50))
        elif exercise >= 5:
            recs.append((POSITIVE, "🏃", "Active & Resilient",
                f"{exercise} exercise days/week — excellent. Exercise is your best stress buffer.",
                "Maintain this habit. Consider adding yoga or stretching for recovery.", 0))

        # Anxiety
        if anxiety >= 8:
            recs.append((CRITICAL, "🧘", "High Anxiety — Immediate Action Needed",
                f"Anxiety at {anxiety}/10 is clinically significant. This is affecting your cognition and sleep.",
                "Try box breathing NOW: inhale 4s → hold 4s → exhale 4s → hold 4s. Repeat 5×.",
                int(anxiety * 10)))
        elif anxiety >= 6:
            recs.append((HIGH, "🧘", "Elevated Anxiety",
                f"Anxiety at {anxiety}/10 is interfering with focus. Cognitive load increases sharply above 6/10.",
                "10-min daily mindfulness practice. Apps: Headspace, Insight Timer (free tier).",
                int(anxiety * 8)))

        # Screen time
        if screen > 8:
            recs.append((HIGH, "📱", "Excessive Screen Time",
                f"{screen}h/day of screens raises cortisol and disrupts melatonin production, directly worsening sleep.",
                "Set app time limits. Use grayscale mode after 9 PM to reduce dopamine spikes.",
                min(100, int((screen - 4) * 10))))
        elif screen > 5:
            recs.append((MODERATE, "📱", "Moderate Screen Overuse",
                f"{screen}h/day is above the 4h guideline. Blue light affects sleep quality.",
                "Use blue-light glasses or Night Shift mode. No screens 30 min before bed.",
                min(100, int((screen - 4) * 7))))

        # Financial stress
        if finance >= 8:
            recs.append((HIGH, "💰", "High Financial Stress",
                "Financial stress is one of the top predictors of academic dropout and mental health issues.",
                "Contact your institution's student welfare office today. Emergency funds may be available.",
                int(finance * 9)))
        elif finance >= 6:
            recs.append((MODERATE, "💰", "Financial Pressure",
                "Moderate financial stress is draining background mental resources.",
                "Track expenses for 1 week. Identify one non-essential cost to cut or defer.",
                int(finance * 6)))

        # Social
        if social < 3:
            recs.append((HIGH, "👥", "Social Isolation Risk",
                "Low social interaction is linked to depression and reduced stress resilience.",
                "Schedule one 20-min call or meet-up this week. Join a study group or club.",
                75))
        elif social < 5:
            recs.append((MODERATE, "👥", "Limited Social Connection",
                "Moderate social contact — aim to increase meaningful interactions.",
                "Even brief positive exchanges count. Say hi to a classmate daily.",
                45))

        # Assignments
        if assignments >= 10:
            recs.append((CRITICAL, "📝", "Task Overload",
                f"{assignments} pending assignments creates decision paralysis and chronic low-grade panic.",
                "Eisenhower matrix: list tasks → sort by urgent+important → do top 1 NOW.",
                min(100, assignments * 7)))
        elif assignments >= 6:
            recs.append((MODERATE, "📝", "Heavy Task Queue",
                f"{assignments} pending items. Unfinished tasks occupy working memory (Zeigarnik effect).",
                "Write every task down — externalising it frees cognitive load immediately.",
                min(100, assignments * 5)))

        # Peer pressure
        if peer >= 8:
            recs.append((HIGH, "🤝", "Severe Peer Pressure",
                f"Peer pressure at {peer}/10 is draining energy and distorting your decisions.",
                "Practice assertive phrases: 'I'm not able to commit to that right now.' Limit time with draining people.",
                int(peer * 9)))

        # Family support
        if family <= 3:
            recs.append((HIGH, "❤️", "Low Support Network",
                "Low family support increases psychological vulnerability significantly.",
                "Campus counselors and peer mentors provide structured support — reach out today.",
                70))

        # Default positive
        if not recs or all(r[0] == POSITIVE for r in recs):
            recs.append((POSITIVE, "🌟", "Strong Wellbeing Profile",
                "Your indicators are well-balanced. You're in the top tier for student wellbeing.",
                "Do weekly check-ins to detect drift early. Share what's working with peers.", 0))

        # Sort: critical → high → moderate → positive
        order = {CRITICAL: 0, HIGH: 1, MODERATE: 2, POSITIVE: 3}
        recs.sort(key=lambda r: order[r[0]])

        badge_labels = {CRITICAL: "Critical", HIGH: "High Priority",
                        MODERATE: "Moderate", POSITIVE: "Positive"}

        active_count = len([r for r in recs if r[0] != POSITIVE])
        st.markdown(f'<div class="section-header">💡 {active_count} Active Recommendations</div>',
                    unsafe_allow_html=True)

        bar_colors = {"critical":"#E24B4A","high":"#FAC775","moderate":"#7F77DD","positive":"#97C459"}

        # Render in rows of 4
        for row_start in range(0, len(recs), 4):
            row_recs = recs[row_start:row_start+4]
            cols = st.columns(len(row_recs))
            for col, (severity, icon, title, body_text, action, score) in zip(cols, row_recs):
                bar_color = bar_colors[severity]
                bar_w = score if severity != POSITIVE else 100
                with col:
                    st.markdown(f"""
                    <div class="rec-card rec-{severity}">
                        <div class="rec-icon">{icon}</div>
                        <span class="rec-badge badge-{severity}">{badge_labels[severity]}</span>
                        <div class="rec-title">{title}</div>
                        <div class="rec-body">{body_text}</div>
                        <span class="rec-action">⚡ {action}</span>
                        <div class="rec-bar-track">
                            <div style="width:{bar_w}%;height:100%;background:{bar_color};border-radius:2px;"></div>
                        </div>
                    </div>""", unsafe_allow_html=True)

        st.divider()

        # ── Stress Radar Chart ─────────────────────────────────
        st.markdown("#### 📡 Wellness Radar")
        radar_cats = ['Sleep', 'Study Balance', 'Exercise', 'Social', 'Low Anxiety', 'Low Screen']
        norm_sleep    = min(100, int(sleep / 9 * 100))
        norm_study    = max(0, 100 - int(max(0, study - 6) / 10 * 100))
        norm_exercise = min(100, int(exercise / 7 * 100))
        norm_social   = min(100, int(social / 15 * 100))
        norm_anxiety  = max(0, 100 - int((anxiety - 1) / 9 * 100))
        norm_screen   = max(0, 100 - int(max(0, screen - 3) / 13 * 100))
        radar_vals    = [norm_sleep, norm_study, norm_exercise,
                         norm_social, norm_anxiety, norm_screen]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_vals + [radar_vals[0]],
            theta=radar_cats + [radar_cats[0]],
            fill='toself',
            fillcolor='rgba(83,74,183,0.2)',
            line=dict(color='#AFA9EC', width=2),
            name='Your Profile'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=[80]*len(radar_cats) + [80],
            theta=radar_cats + [radar_cats[0]],
            fill='toself',
            fillcolor='rgba(99,153,34,0.06)',
            line=dict(color='#639922', width=1.5, dash='dot'),
            name='Target Zone'
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9),
                                gridcolor='rgba(255,255,255,0.1)'),
                angularaxis=dict(gridcolor='rgba(255,255,255,0.15)'),
                bgcolor='rgba(0,0,0,0)'
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=True,
            legend=dict(font=dict(color='#ccc')),
            margin=dict(t=30, b=30, l=30, r=30),
            height=380
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        st.divider()
        st.markdown("#### 🗓️ 7-day recovery plan")
        plan_items = [
            ("Day 1", "Set a consistent sleep time and stick to it all week."),
            ("Day 2", "Write all pending tasks down. Cross off one small thing today."),
            ("Day 3", "Go for a 20-min walk — no phone, no earphones."),
            ("Day 4", "Call or message one friend or family member you trust."),
            ("Day 5", "Do one 25-min Pomodoro study block. Note your focus level."),
            ("Day 6", "Spend 10 min on box breathing or a simple meditation."),
            ("Day 7", "Review the week: what helped? Plan to repeat those habits."),
        ]
        p1, p2 = st.columns(2)
        for i, (day, action) in enumerate(plan_items):
            with (p1 if i % 2 == 0 else p2):
                st.checkbox(f"**{day}** — {action}", key=f"plan_{day}")

    # ══════════════════════════════════════════════════════════
    # TAB 4 — History
    # ══════════════════════════════════════════════════════════
    with tab4:
        st.markdown('<div class="section-header">📊 Stress History</div>', unsafe_allow_html=True)
        if history_df.empty:
            st.info("No history yet. Fill in today's data and hit 💾 Save.")
        else:
            hdf = history_df.copy()
            for col in ['stress_score','sleep','study','screen','anxiety','exercise']:
                hdf[col] = pd.to_numeric(hdf[col], errors='coerce')

            x_labels  = hdf['day_label'].tolist()
            avg_score  = hdf['stress_score'].mean()
            trend      = hdf['stress_score'].iloc[-1] - hdf['stress_score'].iloc[-2] if len(hdf) > 1 else 0
            trend_str  = f"↓ {abs(trend):.0f}" if trend < 0 else (f"↑ {trend:.0f}" if trend > 0 else "→ 0")

            # ── 4-col stat summary cards ───────────────────────
            st.markdown(f"""
            <div class="hist-stat-grid">
                <div class="hist-stat-card">
                    <div class="hist-stat-val">{len(hdf)}</div>
                    <div class="hist-stat-label">Sessions Logged</div>
                </div>
                <div class="hist-stat-card">
                    <div class="hist-stat-val">{avg_score:.0f}</div>
                    <div class="hist-stat-label">Avg Stress Score</div>
                </div>
                <div class="hist-stat-card">
                    <div class="hist-stat-val">{hdf['sleep'].mean():.1f}h</div>
                    <div class="hist-stat-label">Avg Sleep</div>
                </div>
                <div class="hist-stat-card">
                    <div class="hist-stat-val">{trend_str}</div>
                    <div class="hist-stat-label">Last Session Trend</div>
                </div>
            </div>""", unsafe_allow_html=True)

            level_color_map = {'Low':'#639922','Moderate':'#EF9F27',
                               'High':'#D85A30','Critical':'#E24B4A'}
            marker_colors = [level_color_map.get(l,'#AFA9EC')
                             for l in hdf.get('stress_level',['Low']*len(hdf))]

            # ── Row 1: two charts side by side ─────────────────
            r1c1, r1c2 = st.columns(2)

            with r1c1:
                st.markdown('<div class="chart-card"><div class="chart-title">📈 Stress Score Over Time</div>', unsafe_allow_html=True)
                fig_stress = go.Figure()
                fig_stress.add_trace(go.Scatter(
                    x=x_labels, y=hdf['stress_score'],
                    mode='lines+markers',
                    line=dict(color='#AFA9EC', width=2.5, shape='spline'),
                    marker=dict(color=marker_colors, size=9,
                                line=dict(width=1.5, color='rgba(0,0,0,0.3)')),
                    fill='tozeroy', fillcolor='rgba(83,74,183,0.10)',
                    hovertemplate='<b>%{x}</b><br>Score: %{y}<extra></extra>'
                ))
                for thresh, clr, lbl in [(30,'#639922','Low'),(55,'#BA7517','High'),(75,'#A32D2D','Critical')]:
                    fig_stress.add_hline(y=thresh, line_dash='dot', line_color=clr, opacity=0.45,
                                         annotation_text=lbl, annotation_position='right',
                                         annotation_font_color=clr, annotation_font_size=10)
                fig_stress.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(range=[0,105], gridcolor='rgba(255,255,255,0.06)',
                               tickfont=dict(color='#777', size=10)),
                    xaxis=dict(gridcolor='rgba(255,255,255,0.04)',
                               tickfont=dict(color='#777', size=10)),
                    margin=dict(t=10, b=10, l=10, r=60), height=250, showlegend=False
                )
                st.plotly_chart(fig_stress, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with r1c2:
                st.markdown('<div class="chart-card"><div class="chart-title">🛏 Sleep vs Study Hours</div>', unsafe_allow_html=True)
                fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
                fig_dual.add_trace(go.Bar(
                    x=x_labels, y=hdf['sleep'], name='Sleep (hrs)',
                    marker_color='rgba(83,74,183,0.6)', marker_line_width=0,
                    hovertemplate='Sleep: %{y}h<extra></extra>'
                ), secondary_y=False)
                fig_dual.add_trace(go.Scatter(
                    x=x_labels, y=hdf['study'], mode='lines+markers', name='Study (hrs)',
                    line=dict(color='#D4537E', width=2.5, shape='spline'),
                    marker=dict(size=7, color='#D4537E'),
                    hovertemplate='Study: %{y}h<extra></extra>'
                ), secondary_y=True)
                fig_dual.add_hline(y=7, line_dash='dot', line_color='#AFA9EC', opacity=0.35,
                                   annotation_text='7h target', annotation_font_color='#AFA9EC',
                                   annotation_font_size=9)
                fig_dual.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    legend=dict(font=dict(color='#aaa', size=10), bgcolor='rgba(0,0,0,0)',
                                orientation='h', y=1.12),
                    margin=dict(t=10, b=10, l=10, r=50), height=250,
                    yaxis=dict(gridcolor='rgba(255,255,255,0.06)',
                               tickfont=dict(color='#777', size=10)),
                    yaxis2=dict(tickfont=dict(color='#D4537E', size=10))
                )
                st.plotly_chart(fig_dual, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # ── Row 2: lifestyle | donut + box ─────────────────
            r2c1, r2c2 = st.columns(2)

            with r2c1:
                st.markdown('<div class="chart-card"><div class="chart-title">📉 Lifestyle Trends</div>', unsafe_allow_html=True)
                if all(c in hdf.columns for c in ['screen','anxiety','exercise']):
                    fig_multi = go.Figure()
                    for col_, color_, label_ in [
                        ('screen',  '#FAC775','Screen (hrs)'),
                        ('anxiety', '#F09595','Anxiety (/10)'),
                        ('exercise','#97C459','Exercise (days/wk)')
                    ]:
                        fig_multi.add_trace(go.Scatter(
                            x=x_labels, y=hdf[col_], mode='lines+markers',
                            name=label_, line=dict(color=color_, width=2, shape='spline'),
                            marker=dict(size=6, color=color_),
                            hovertemplate=f'{label_}: %{{y}}<extra></extra>'
                        ))
                    fig_multi.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        legend=dict(font=dict(color='#aaa', size=10), bgcolor='rgba(0,0,0,0)',
                                    orientation='h', y=1.12),
                        margin=dict(t=10, b=10, l=10, r=10), height=250,
                        yaxis=dict(gridcolor='rgba(255,255,255,0.06)',
                                   tickfont=dict(color='#777', size=10)),
                        xaxis=dict(gridcolor='rgba(255,255,255,0.04)',
                                   tickfont=dict(color='#777', size=10))
                    )
                    st.plotly_chart(fig_multi, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with r2c2:
                if 'stress_level' in hdf.columns:
                    dc1, dc2 = st.columns(2)
                    with dc1:
                        st.markdown('<div class="chart-card"><div class="chart-title">🍩 Distribution</div>', unsafe_allow_html=True)
                        level_counts = hdf['stress_level'].value_counts()
                        fig_donut = go.Figure(go.Pie(
                            labels=level_counts.index, values=level_counts.values,
                            hole=0.55,
                            marker_colors=[COLORS.get(l,'#888') for l in level_counts.index],
                            textfont=dict(size=11),
                            hovertemplate='%{label}: %{value} (%{percent})<extra></extra>'
                        ))
                        fig_donut.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            legend=dict(font=dict(color='#aaa', size=9), bgcolor='rgba(0,0,0,0)',
                                        orientation='h', y=-0.2),
                            margin=dict(t=10, b=35, l=10, r=10), height=250,
                            annotations=[dict(text=f"{len(hdf)}<br>sessions",
                                              font=dict(size=11, color='#AFA9EC'), showarrow=False)]
                        )
                        st.plotly_chart(fig_donut, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                    with dc2:
                        st.markdown('<div class="chart-card"><div class="chart-title">📦 Score Spread</div>', unsafe_allow_html=True)
                        BOX_FILL = {'Low':'rgba(99,153,34,0.25)','Moderate':'rgba(186,117,23,0.25)',
                                    'High':'rgba(153,60,29,0.25)','Critical':'rgba(163,45,45,0.25)'}
                        fig_box = go.Figure()
                        for lvl in ['Low','Moderate','High','Critical']:
                            lvl_data = hdf[hdf['stress_level']==lvl]['stress_score'].dropna()
                            if not lvl_data.empty:
                                fig_box.add_trace(go.Box(
                                    y=lvl_data, name=lvl,
                                    marker_color=COLORS.get(lvl,'#888'),
                                    line_color=COLORS.get(lvl,'#888'),
                                    fillcolor=BOX_FILL.get(lvl,'rgba(128,128,128,0.25)'),
                                    boxmean=True
                                ))
                        fig_box.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            yaxis=dict(range=[0,105], gridcolor='rgba(255,255,255,0.06)',
                                       tickfont=dict(color='#777', size=10)),
                            xaxis=dict(tickfont=dict(color='#aaa', size=10)),
                            showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=250
                        )
                        st.plotly_chart(fig_box, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)

            with st.expander("📋 View raw data"):
                st.dataframe(hdf.drop(columns=['id','user_id'], errors='ignore'),
                             use_container_width=True)
                csv = hdf.to_csv(index=False).encode()
                st.download_button("⬇️ Download CSV", csv,
                                   f"{username}_stress_history.csv", "text/csv")

    # ══════════════════════════════════════════════════════════
    # TAB 5 — Goals
    # ══════════════════════════════════════════════════════════
    with tab5:
        st.markdown("#### 🎯 Set your weekly wellness goals")
        st.caption("Goals are saved per account and tracked against every session you log.")

        with st.form("goals_form"):
            st.markdown("##### Adjust your targets")
            gc1,gc2 = st.columns(2)
            with gc1:
                g_sleep    = st.slider("😴 Sleep target (hrs/night, min)",
                                       4.0,10.0,float(saved_goals.get("goal_sleep",8.0)),0.5)
                g_exercise = st.slider("🏃 Exercise target (days/week, min)",
                                       1,7,int(saved_goals.get("goal_exercise",4)))
            with gc2:
                g_study    = st.slider("📚 Study limit (hrs/day, max)",
                                       2.0,14.0,float(saved_goals.get("goal_study",8.0)),0.5)
                g_screen   = st.slider("📱 Screen time limit (hrs/day, max)",
                                       1.0,12.0,float(saved_goals.get("goal_screen",4.0)),0.5)
            save_goals_btn = st.form_submit_button("💾 Save Goals",use_container_width=True)

        if save_goals_btn:
            save_goals_db(user_id,g_sleep,g_study,g_exercise,g_screen)
            st.success("✅ Goals saved!")
            st.rerun()
        else:
            g_sleep    = float(saved_goals.get("goal_sleep",   8.0))
            g_study    = float(saved_goals.get("goal_study",   8.0))
            g_exercise = int(saved_goals.get("goal_exercise",  4))
            g_screen   = float(saved_goals.get("goal_screen",  4.0))

        st.divider()
        st.markdown("##### How does today compare?")
        col_a,col_b = st.columns(2)
        for i,(label,actual,target,direction,u_a,u_t) in enumerate([
            ("😴 Sleep",    sleep,          g_sleep,           "gte","hrs tonight",   "hrs target"),
            ("📚 Study",    study,          g_study,           "lte","hrs today",     "hrs max"),
            ("🏃 Exercise", float(exercise),float(g_exercise), "gte","days this week","days target"),
            ("📱 Screen",   screen,         g_screen,          "lte","hrs today",     "hrs limit"),
        ]):
            met  = (actual>=target) if direction=="gte" else (actual<=target)
            icon = "✅" if met else "❌"
            status_txt = "Goal met!" if met else ("Need more" if direction=="gte" else "Too much")
            delta_str = (f"+{round(actual-target,1)}" if actual>=target
                         else str(round(actual-target,1)))
            with (col_a if i%2==0 else col_b):
                st.markdown(f"""
                <div class="goal-card">
                    <div class="goal-title">{icon} {label}</div>
                    <div style="font-size:1.4rem;font-weight:700;
                                color:{'#639922' if met else '#993C1D'};">
                        {actual}
                        <span style="font-size:0.85rem;font-weight:400;color:#888;">{u_a}</span>
                    </div>
                    <div style="font-size:0.82rem;color:#888;margin:2px 0 8px;">
                        Target: {target} {u_t} &nbsp;|&nbsp; {status_txt} ({delta_str})
                    </div>
                </div>""", unsafe_allow_html=True)

        st.divider()
        st.markdown("##### Weekly progress & streaks")
        hdf_g    = history_df.copy() if not history_df.empty else pd.DataFrame()
        streaks  = compute_streaks(hdf_g,g_sleep,g_study,g_exercise,g_screen)
        progress = week_progress(hdf_g,  g_sleep,g_study,g_exercise,g_screen)

        for label,key,rule in [
            ("😴 Sleep",   "sleep",   f"≥ {g_sleep}h/night"),
            ("📚 Study",   "study",   f"≤ {g_study}h/day"),
            ("🏃 Exercise","exercise",f"≥ {g_exercise} days/week"),
            ("📱 Screen",  "screen",  f"≤ {g_screen}h/day"),
        ]:
            pct   = progress[key]; streak = streaks[key]
            bar_c = "#639922" if pct>=70 else "#BA7517" if pct>=40 else "#E24B4A"
            s_cls = "streak-badge" if streak>0 else "streak-badge streak-zero"
            s_txt = f"🔥 {streak}-day streak" if streak>0 else "No streak yet"
            st.markdown(f"""
            <div class="goal-card">
                <div style="display:flex;align-items:center;
                            justify-content:space-between;margin-bottom:6px;">
                    <span class="goal-title" style="margin:0;">{label} &nbsp;
                        <span style="font-weight:400;color:#888;font-size:0.8rem;">({rule})</span>
                    </span>
                    <span class="{s_cls}">{s_txt}</span>
                </div>
                <div style="background:#eee;border-radius:8px;height:12px;overflow:hidden;">
                    <div style="width:{pct}%;height:100%;background:{bar_c};
                                border-radius:8px;"></div>
                </div>
                <div style="font-size:0.8rem;color:#888;margin-top:4px;">
                    {pct}% of last 7 sessions goal was met {"✅" if pct==100 else ""}
                </div>
            </div>""", unsafe_allow_html=True)

        st.divider()
        st.markdown("##### Overall goal score")
        overall = int(sum(progress.values())/len(progress))
        o_color = "#639922" if overall>=70 else "#BA7517" if overall>=40 else "#E24B4A"
        o_label = "Excellent 🌟" if overall>=80 else "Good 👍" if overall>=60 else "Needs work 💪"
        _,oc2,_ = st.columns([1,2,1])
        with oc2:
            st.markdown(f"""
            <div style="text-align:center;padding:1.5rem;background:#fafafa;
                        border-radius:16px;border:0.5px solid #e0e0e0;">
                <div style="font-size:3rem;font-weight:700;color:{o_color};">{overall}%</div>
                <div style="font-size:1rem;color:#555;margin-top:4px;">{o_label}</div>
                <div style="font-size:0.82rem;color:#888;margin-top:4px;">
                    Based on your last 7 logged sessions
                </div>
            </div>""", unsafe_allow_html=True)

        if hdf_g.empty:
            st.info("💡 Start logging daily sessions to see your streaks and progress fill up!")

    # ══════════════════════════════════════════════════════════
    # TAB 6 — Study Planner
    # ══════════════════════════════════════════════════════════
    with tab6:
        tasks_df = load_tasks(user_id)

        st.markdown("#### 📅 Study Planner")
        st.caption("Add subjects and deadlines — the planner builds your week automatically, "
                   "adjusting daily hours based on your current stress level.")

        # ── Break recommendation (stress-aware) ───────────────
        block_min, break_min, break_label = get_break_schedule(stress_score)
        st.markdown(f"""
        <div class="break-box">
            <div style="font-size:0.78rem;font-weight:600;color:#AFA9EC;
                        text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px;">
                ⏱ Recommended study rhythm for today
            </div>
            <div>
                <span class="break-stat">{block_min}m</span>
                <span style="font-size:0.9rem;color:inherit;opacity:0.7;">study block</span>
                &nbsp;→&nbsp;
                <span class="break-stat">{break_min}m</span>
                <span style="font-size:0.9rem;color:inherit;opacity:0.7;">break</span>
            </div>
            <div style="font-size:0.85rem;opacity:0.7;margin-top:4px;">{break_label}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Study Timer ────────────────────────────────────────
        st.markdown("#### ⏱ Study Timer")

        # Timer state initialisation
        if 'timer_running'    not in st.session_state: st.session_state['timer_running']    = False
        if 'timer_phase'      not in st.session_state: st.session_state['timer_phase']      = 'study'
        if 'timer_remaining'  not in st.session_state: st.session_state['timer_remaining']  = block_min * 60
        if 'timer_start_time' not in st.session_state: st.session_state['timer_start_time'] = None
        if 'timer_elapsed'    not in st.session_state: st.session_state['timer_elapsed']    = 0
        if 'sessions_done'    not in st.session_state: st.session_state['sessions_done']    = 0
        if 'timer_log'        not in st.session_state: st.session_state['timer_log']        = []
        if 'custom_study_min' not in st.session_state: st.session_state['custom_study_min'] = block_min
        if 'custom_break_min' not in st.session_state: st.session_state['custom_break_min'] = break_min

        t_cfg1, t_cfg2, t_cfg3 = st.columns([2, 2, 2])
        with t_cfg1:
            timer_mode = st.selectbox("Timer Mode",
                ["🍅 Pomodoro (stress-adjusted)", "⚙️ Custom"],
                key="timer_mode_sel")
        with t_cfg2:
            if "Custom" in timer_mode:
                c_study = st.number_input("Study block (min)", 5, 120,
                                          st.session_state['custom_study_min'], 5,
                                          key="custom_study_inp")
                st.session_state['custom_study_min'] = c_study
            else:
                st.markdown(f"""
                <div style="padding:0.6rem 0.8rem;background:rgba(83,74,183,0.15);
                            border-radius:8px;font-size:0.88rem;margin-top:1.6rem;">
                    📋 Using <strong>{block_min}m/{break_min}m</strong>
                    based on your stress level
                </div>""", unsafe_allow_html=True)
        with t_cfg3:
            if "Custom" in timer_mode:
                c_break = st.number_input("Break (min)", 1, 30,
                                          st.session_state['custom_break_min'], 1,
                                          key="custom_break_inp")
                st.session_state['custom_break_min'] = c_break

        active_study_min = (st.session_state['custom_study_min']
                            if "Custom" in timer_mode else block_min)
        active_break_min = (st.session_state['custom_break_min']
                            if "Custom" in timer_mode else break_min)

        # Recalculate remaining if not running and phase changed
        if not st.session_state['timer_running']:
            if st.session_state['timer_phase'] == 'study':
                st.session_state['timer_remaining'] = (
                    active_study_min * 60 - st.session_state['timer_elapsed'])
            else:
                st.session_state['timer_remaining'] = (
                    active_break_min * 60 - st.session_state['timer_elapsed'])

        # Compute live remaining time
        if st.session_state['timer_running'] and st.session_state['timer_start_time']:
            elapsed_now = int(
                (datetime.datetime.now() - st.session_state['timer_start_time']).total_seconds()
            ) + st.session_state['timer_elapsed']
            total_phase = (active_study_min if st.session_state['timer_phase'] == 'study'
                           else active_break_min) * 60
            remaining = max(0, total_phase - elapsed_now)
        else:
            remaining = st.session_state['timer_remaining']
            elapsed_now = st.session_state['timer_elapsed']

        mins, secs = divmod(int(remaining), 60)
        phase_cls  = "timer-phase-study" if st.session_state['timer_phase'] == 'study' else "timer-phase-break"
        phase_icon = "📖" if st.session_state['timer_phase'] == 'study' else "☕"
        phase_lbl  = "Study Block" if st.session_state['timer_phase'] == 'study' else "Break Time"

        total_secs = (active_study_min if st.session_state['timer_phase'] == 'study'
                      else active_break_min) * 60
        pct_done   = max(0, min(100, int((1 - remaining / max(1, total_secs)) * 100)))

        st.markdown(f"""
        <div class="timer-container {phase_cls}">
            <div class="timer-label">{phase_icon} {phase_lbl} — Session #{st.session_state['sessions_done'] + 1}</div>
            <div class="timer-display">{mins:02d}:{secs:02d}</div>
            <div style="margin-top:1rem;background:rgba(255,255,255,0.1);
                        border-radius:8px;height:8px;overflow:hidden;">
                <div style="width:{pct_done}%;height:100%;
                            background:{'#639922' if st.session_state['timer_phase']=='study' else '#534AB7'};
                            border-radius:8px;transition:width 0.5s ease;"></div>
            </div>
            <div style="font-size:0.8rem;color:#aaa;margin-top:6px;">{pct_done}% complete</div>
        </div>""", unsafe_allow_html=True)

        # Timer controls
        btn1, btn2, btn3, btn4 = st.columns(4)
        with btn1:
            if st.button("▶ Start" if not st.session_state['timer_running'] else "⏸ Pause",
                         use_container_width=True, type="primary"):
                if not st.session_state['timer_running']:
                    st.session_state['timer_running']    = True
                    st.session_state['timer_start_time'] = datetime.datetime.now()
                else:
                    st.session_state['timer_running']  = False
                    st.session_state['timer_elapsed']  = elapsed_now
                    st.session_state['timer_remaining'] = remaining
                st.rerun()

        with btn2:
            if st.button("⏭ Skip Phase", use_container_width=True):
                # Log completed phase
                now_str = datetime.datetime.now().strftime("%H:%M")
                if st.session_state['timer_phase'] == 'study':
                    st.session_state['sessions_done'] += 1
                    st.session_state['timer_log'].append(
                        f"✅ {now_str} — Study block #{st.session_state['sessions_done']} "
                        f"({active_study_min}m)")
                    st.session_state['timer_phase'] = 'break'
                    next_secs = active_break_min * 60
                else:
                    st.session_state['timer_log'].append(
                        f"☕ {now_str} — Break complete")
                    st.session_state['timer_phase'] = 'study'
                    next_secs = active_study_min * 60
                st.session_state['timer_running']    = False
                st.session_state['timer_remaining']  = next_secs
                st.session_state['timer_elapsed']    = 0
                st.session_state['timer_start_time'] = None
                st.rerun()

        with btn3:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state['timer_running']    = False
                st.session_state['timer_phase']      = 'study'
                st.session_state['timer_remaining']  = active_study_min * 60
                st.session_state['timer_elapsed']    = 0
                st.session_state['timer_start_time'] = None
                st.rerun()

        with btn4:
            total_study_done = st.session_state['sessions_done'] * active_study_min
            st.markdown(f"""
            <div style="text-align:center;padding:0.4rem;background:rgba(83,74,183,0.15);
                        border-radius:8px;font-size:0.82rem;">
                🔥 <strong>{st.session_state['sessions_done']}</strong> sessions<br>
                <span style="color:#AFA9EC;">{total_study_done}m studied</span>
            </div>""", unsafe_allow_html=True)

        # Auto-refresh while running
        if st.session_state['timer_running']:
            import time
            time.sleep(1)
            st.rerun()

        # Session log
        if st.session_state['timer_log']:
            with st.expander(f"📋 Session log ({len(st.session_state['timer_log'])} entries)",
                             expanded=False):
                for entry in reversed(st.session_state['timer_log']):
                    st.markdown(f'<div class="session-log-row">⏺ {entry}</div>',
                                unsafe_allow_html=True)
                if st.button("🗑 Clear log"):
                    st.session_state['timer_log'] = []
                    st.rerun()

        st.divider()
        with st.expander("➕ Add a new task", expanded=tasks_df.empty):
            with st.form("add_task_form", clear_on_submit=True):
                fc1, fc2 = st.columns(2)
                with fc1:
                    t_subject  = st.text_input("Subject / Course",
                                               placeholder="e.g. Mathematics")
                    t_topic    = st.text_input("Topic (optional)",
                                               placeholder="e.g. Integration by parts")
                    t_deadline = st.date_input("Deadline",
                                               value=datetime.date.today() +
                                               datetime.timedelta(days=3),
                                               min_value=datetime.date.today())
                with fc2:
                    t_priority = st.selectbox("Priority", ["High","Medium","Low"])
                    t_duration = st.slider("Estimated hours needed", 0.5, 8.0, 1.5, 0.5)
                    st.markdown("")
                    st.markdown("")
                    submitted = st.form_submit_button("Add task →",
                                                      use_container_width=True,
                                                      type="primary")
                if submitted:
                    if not t_subject.strip():
                        st.error("Please enter a subject name.")
                    else:
                        add_task(user_id, t_subject.strip(), t_topic.strip(),
                                 t_deadline.strftime("%Y-%m-%d"),
                                 t_priority, t_duration)
                        st.success(f"✅ Task added: {t_subject}")
                        st.rerun()

        if tasks_df.empty:
            st.info("No tasks yet — add your first task above to generate your study plan.")
        else:
            st.divider()

            # ── Settings row ──────────────────────────────────
            pc1, pc2, pc3 = st.columns([2,2,2])
            with pc1:
                daily_limit = st.slider("Max study hours per day",
                                        1.0, 12.0,
                                        float(saved_goals.get("goal_study", 8.0)),
                                        0.5,
                                        help="Stress level may reduce this automatically")
            with pc2:
                show_done = st.toggle("Show completed tasks", value=False)
            with pc3:
                st.metric("Total tasks",   str(len(tasks_df)))
                st.metric("Pending",       str(len(tasks_df[tasks_df['completed']==0])))

            st.divider()

            # ── Priority task list ─────────────────────────────
            st.markdown("#### 🔢 Priority task list")
            st.caption("Sorted by deadline then priority — tackle from the top.")

            pending_tasks = tasks_df[tasks_df['completed']==0].copy()
            done_tasks    = tasks_df[tasks_df['completed']==1].copy()

            pri_weight = {'High':0,'Medium':1,'Low':2}
            pending_tasks['pri_w'] = pending_tasks['priority'].map(pri_weight).fillna(1)
            pending_tasks['deadline_dt'] = pd.to_datetime(
                pending_tasks['deadline'], errors='coerce')
            pending_tasks = pending_tasks.sort_values(['deadline_dt','pri_w'])

            def days_left_str(deadline_str):
                try:
                    dl = datetime.date.fromisoformat(str(deadline_str)[:10])
                    diff = (dl - datetime.date.today()).days
                    if diff < 0:   return "⚠️ Overdue"
                    if diff == 0:  return "🔥 Due today"
                    if diff == 1:  return "⏰ Due tomorrow"
                    return f"📅 {diff} days left"
                except Exception:
                    return ""

            display_tasks = pd.concat([pending_tasks,
                                       done_tasks]) if show_done else pending_tasks

            for _, row in display_tasks.iterrows():
                task_id   = int(row['id'])
                is_done   = int(row['completed']) == 1
                pri_cls   = f"pri-{row['priority'].lower()}"
                dl_str    = days_left_str(row['deadline'])
                topic_str = f" — {row['topic']}" if row['topic'] else ""
                dur_str   = f"{row['duration_h']}h"

                col_chk, col_info, col_del = st.columns([0.5, 8, 0.8])
                with col_chk:
                    if st.button("✅" if is_done else "⬜",
                                 key=f"chk_{task_id}",
                                 help="Toggle complete"):
                        toggle_task(task_id, is_done)
                        st.rerun()
                with col_info:
                    done_style = "opacity:0.45;text-decoration:line-through;" if is_done else ""
                    st.markdown(f"""
                    <div class="task-row" style="{done_style}">
                        <span class="{pri_cls}">{row['priority']}</span>
                        <span style="font-weight:600;">{row['subject']}</span>
                        <span style="opacity:0.65;">{topic_str}</span>
                        <span style="margin-left:auto;opacity:0.55;font-size:0.82rem;">
                            {dur_str} &nbsp;|&nbsp; {dl_str}
                        </span>
                    </div>""", unsafe_allow_html=True)
                with col_del:
                    if st.button("🗑", key=f"del_{task_id}", help="Delete task"):
                        delete_task(task_id)
                        st.rerun()

            st.divider()

            # ── Weekly schedule ────────────────────────────────
            st.markdown("#### 🗓 Your 7-day study schedule")

            pending_only = tasks_df[tasks_df['completed']==0].copy()
            schedule, effective_cap = generate_weekly_schedule(
                pending_only, stress_score, daily_limit)

            if stress_score >= 55:
                st.info(f"⚠️ Your stress score is **{stress_score}/100** — daily study cap "
                        f"has been reduced to **{effective_cap:.1f}h/day** to protect your wellbeing.")

            DAYS_PER_ROW = 4
            day_items = list(schedule.items())

            for row_start in range(0, 7, DAYS_PER_ROW):
                row_days = day_items[row_start:row_start+DAYS_PER_ROW]
                cols = st.columns(len(row_days))
                for col, (day_str, day_tasks) in zip(cols, row_days):
                    with col:
                        total_h = sum(t['duration_h'] for t in day_tasks)
                        load_color = ("#A32D2D" if total_h >= effective_cap * 0.9
                                      else "#BA7517" if total_h >= effective_cap * 0.6
                                      else "#639922")
                        st.markdown(f"""
                        <div class="day-card">
                            <div class="day-header">{day_str}</div>
                            <div style="font-size:0.78rem;color:{load_color};
                                        margin-bottom:8px;font-weight:600;">
                                {total_h:.1f}h / {effective_cap:.1f}h
                            </div>""", unsafe_allow_html=True)

                        if day_tasks:
                            for t in day_tasks:
                                pri_cls = f"pri-{t['priority'].lower()}"
                                st.markdown(f"""
                                <div style="font-size:0.82rem;padding:3px 0;
                                            border-bottom:0.5px solid rgba(255,255,255,0.06);">
                                    <span class="{pri_cls}">{t['priority'][0]}</span>
                                    &nbsp;<strong>{t['subject']}</strong>
                                    <span style="opacity:0.55;"> {t['duration_h']}h</span>
                                </div>""", unsafe_allow_html=True)
                        else:
                            st.markdown(
                                '<div style="font-size:0.82rem;opacity:0.4;'
                                'padding:4px 0;">Rest day 🌿</div>',
                                unsafe_allow_html=True)

                        st.markdown("</div>", unsafe_allow_html=True)

            st.divider()

            # ── Exam countdowns ────────────────────────────────
            st.markdown("#### ⏳ Upcoming deadlines")
            today = datetime.date.today()
            upcoming = tasks_df[tasks_df['completed']==0].copy()
            upcoming['deadline_dt'] = pd.to_datetime(
                upcoming['deadline'], errors='coerce')
            upcoming = upcoming.dropna(subset=['deadline_dt'])
            upcoming['days_left'] = upcoming['deadline_dt'].apply(
                lambda x: (x.date()-today).days)
            upcoming = upcoming.sort_values('days_left').head(6)

            if upcoming.empty:
                st.success("🎉 No upcoming deadlines — you're all caught up!")
            else:
                dcols = st.columns(min(3, len(upcoming)))
                for i, (_, row) in enumerate(upcoming.iterrows()):
                    dl = int(row['days_left'])
                    color = ("#A32D2D" if dl<=1 else
                             "#BA7517" if dl<=3 else
                             "#639922")
                    label = ("⚠️ Overdue" if dl<0 else
                             "🔥 Today"   if dl==0 else
                             f"{dl}d left")
                    with dcols[i % 3]:
                        st.markdown(f"""
                        <div class="day-card" style="text-align:center;">
                            <div style="font-size:2rem;font-weight:700;color:{color};">
                                {label}
                            </div>
                            <div style="font-weight:600;margin-top:4px;">
                                {row['subject']}
                            </div>
                            <div style="font-size:0.8rem;opacity:0.55;margin-top:2px;">
                                {row.get('topic','') or ''}
                            </div>
                            <div style="font-size:0.78rem;opacity:0.45;margin-top:4px;">
                                Due: {str(row['deadline'])[:10]}
                            </div>
                        </div>""", unsafe_allow_html=True)

    st.divider()
    st.caption("🧠 Student Stress Monitor | Built with Streamlit & scikit-learn | For educational purposes only.")


# ═════════════════════════════════════════════════════════════════════════════
# ROUTER
# ═════════════════════════════════════════════════════════════════════════════

if "user" not in st.session_state or st.session_state["user"] is None:
    show_auth_page()
else:
    show_main_app(st.session_state["user"])