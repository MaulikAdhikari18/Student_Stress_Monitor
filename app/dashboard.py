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
    .main-title {
        font-size:2rem; font-weight:700;
        background:linear-gradient(135deg,#534AB7,#D4537E);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    }
    .auth-title { font-size:1.4rem; font-weight:700; color:#534AB7; margin-bottom:0.2rem; }
    .auth-sub   { font-size:0.88rem; color:#888; margin-bottom:1.4rem; }
    .stress-box { padding:1rem 1.4rem; border-radius:12px; border-left:6px solid; margin-bottom:1rem; }
    .box-low      { background:rgba(99,153,34,0.15);  border-color:#639922; color:inherit; }
    .box-moderate { background:rgba(186,117,23,0.15); border-color:#BA7517; color:inherit; }
    .box-high     { background:rgba(153,60,29,0.15);  border-color:#993C1D; color:inherit; }
    .box-critical { background:rgba(163,45,45,0.15);  border-color:#A32D2D; color:inherit; }
    .tip-box {
        background:rgba(83,74,183,0.12);
        border-left:4px solid #7F77DD;
        border-radius:0 8px 8px 0;
        padding:0.75rem 1rem;
        margin-bottom:0.5rem;
        font-size:0.92rem;
        color:inherit;
    }
    .tip-box strong { color:#AFA9EC; }
    .goal-card {
        background:rgba(255,255,255,0.05);
        border:0.5px solid rgba(255,255,255,0.12);
        border-radius:12px; padding:1rem 1.2rem; margin-bottom:0.6rem;
        color:inherit;
    }
    .goal-title { font-size:0.85rem; font-weight:600; color:inherit; opacity:0.75; margin-bottom:6px; }
    .streak-badge {
        display:inline-block; padding:2px 10px; border-radius:20px;
        font-size:0.78rem; font-weight:600;
        background:rgba(99,153,34,0.2); color:#97C459; margin-left:8px;
    }
    .streak-zero { background:rgba(128,128,128,0.15); color:#888; }
    .user-chip {
        display:inline-block; background:rgba(83,74,183,0.2); color:#AFA9EC;
        border-radius:20px; padding:3px 12px; font-size:0.85rem; font-weight:600;
    }
    .day-card {
        border:0.5px solid rgba(255,255,255,0.12);
        border-radius:12px; padding:0.8rem 1rem; margin-bottom:0.6rem;
        background:rgba(255,255,255,0.04);
    }
    .day-header {
        font-size:0.85rem; font-weight:600; color:#AFA9EC;
        margin-bottom:0.5rem; letter-spacing:0.03em;
    }
    .task-row {
        display:flex; align-items:center; gap:8px;
        padding:5px 0; border-bottom:0.5px solid rgba(255,255,255,0.06);
        font-size:0.88rem;
    }
    .task-row:last-child { border-bottom:none; }
    .pri-high   { background:rgba(163,45,45,0.2);   color:#F09595; border-radius:4px; padding:1px 7px; font-size:0.75rem; font-weight:600; }
    .pri-medium { background:rgba(186,117,23,0.2);  color:#FAC775; border-radius:4px; padding:1px 7px; font-size:0.75rem; font-weight:600; }
    .pri-low    { background:rgba(99,153,34,0.2);   color:#C0DD97; border-radius:4px; padding:1px 7px; font-size:0.75rem; font-weight:600; }
    .break-box {
        background:rgba(83,74,183,0.12); border-left:4px solid #7F77DD;
        border-radius:0 10px 10px 0; padding:0.9rem 1.1rem; margin-bottom:1rem;
    }
    .break-stat { font-size:2rem; font-weight:700; color:#AFA9EC; display:inline-block; margin-right:0.5rem; }
    /* ── Timer styles ── */
    .timer-container {
        background: linear-gradient(135deg, rgba(83,74,183,0.18), rgba(212,83,126,0.12));
        border: 1px solid rgba(175,169,236,0.3);
        border-radius: 20px;
        padding: 2rem 1.5rem;
        text-align: center;
        margin-bottom: 1.2rem;
    }
    .timer-display {
        font-size: 4.5rem;
        font-weight: 800;
        font-family: 'Courier New', monospace;
        background: linear-gradient(135deg, #AFA9EC, #D4537E);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 0.05em;
        line-height: 1;
    }
    .timer-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #AFA9EC;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 0.4rem;
        opacity: 0.8;
    }
    .timer-phase-study { border-top: 3px solid #639922; }
    .timer-phase-break { border-top: 3px solid #534AB7; }
    .session-log-row {
        display: flex; align-items: center; gap: 10px;
        padding: 6px 10px; border-radius: 8px;
        background: rgba(255,255,255,0.04);
        margin-bottom: 4px; font-size: 0.84rem;
    }
    /* ── Advanced recommendation styles ── */
    .rec-card {
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.75rem;
        position: relative;
        overflow: hidden;
    }
    .rec-critical { background: rgba(163,45,45,0.18); border-left: 5px solid #E24B4A; }
    .rec-high     { background: rgba(186,117,23,0.18); border-left: 5px solid #FAC775; }
    .rec-moderate { background: rgba(83,74,183,0.15);  border-left: 5px solid #7F77DD; }
    .rec-positive { background: rgba(99,153,34,0.15);  border-left: 5px solid #97C459; }
    .rec-header   { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }
    .rec-badge {
        font-size: 0.7rem; font-weight: 700; padding: 2px 8px; border-radius: 20px;
        text-transform: uppercase; letter-spacing: 0.05em;
    }
    .badge-critical { background: rgba(226,75,74,0.3); color: #F09595; }
    .badge-high     { background: rgba(250,199,117,0.3); color: #FAC775; }
    .badge-moderate { background: rgba(127,119,221,0.3); color: #AFA9EC; }
    .badge-positive { background: rgba(151,196,89,0.3); color: #C0DD97; }
    .rec-title  { font-size: 0.95rem; font-weight: 700; }
    .rec-body   { font-size: 0.87rem; opacity: 0.85; line-height: 1.55; margin: 0; }
    .rec-action {
        margin-top: 8px; padding: 5px 10px; border-radius: 6px;
        font-size: 0.8rem; font-weight: 600;
        background: rgba(255,255,255,0.08);
        display: inline-block; opacity: 0.9;
    }
    .rec-score-bar {
        height: 4px; border-radius: 2px; margin-top: 10px;
        background: rgba(255,255,255,0.08);
        overflow: hidden;
    }
    .insight-box {
        background: linear-gradient(135deg, rgba(83,74,183,0.2), rgba(212,83,126,0.15));
        border: 1px solid rgba(175,169,236,0.3);
        border-radius: 14px; padding: 1.2rem 1.4rem; margin-bottom: 1rem;
    }
    .insight-title { font-size: 1rem; font-weight: 700; color: #AFA9EC; margin-bottom: 0.4rem; }
    .insight-body  { font-size: 0.88rem; opacity: 0.85; line-height: 1.6; }
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
                 sleep, study, screen, anxiety, exercise,
                 entry_date=None):
    if entry_date is None:
        entry_date = datetime.date.today()
    ts        = datetime.datetime.combine(entry_date, datetime.time(12, 0))
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
    if stress_score >= 90:
        cap = max(1.5, daily_study_limit * 0.5)
    elif stress_score >= 75:
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
    if stress_score >= 90:
        return 20, 10, "🔴 Critical stress — short blocks, frequent breaks"
    elif stress_score >= 75:
        return 25, 8,  "🟠 High stress — Pomodoro 25/8 recommended"
    elif stress_score >= 30:
        return 35, 7,  "🟡 Moderate stress — 35 min focus, 7 min break"
    else:
        return 50, 10, "🟢 Low stress — deep work 50/10 recommended"


# ═════════════════════════════════════════════════════════════════════════════
# AUTH PAGE
# ═════════════════════════════════════════════════════════════════════════════

def show_landing_page():
    """Full-screen hero landing page shown before login."""
    st.markdown("""
    <style>
    /* Hide streamlit default chrome on landing */
    #MainMenu, footer, header { visibility: hidden; }

    .landing-hero {
        text-align: center;
        padding: 3rem 1rem 2rem;
    }
    .landing-logo {
        font-size: 5rem;
        line-height: 1;
        margin-bottom: 0.5rem;
    }
    .landing-title {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(135deg, #AFA9EC, #D4537E);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
        margin-bottom: 0.4rem;
    }
    .landing-sub {
        font-size: 1.15rem;
        color: #aaa;
        margin-bottom: 2.5rem;
        max-width: 520px;
        margin-left: auto;
        margin-right: auto;
        line-height: 1.6;
    }
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        max-width: 860px;
        margin: 0 auto 2.5rem;
    }
    .feature-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(175,169,236,0.18);
        border-radius: 16px;
        padding: 1.3rem 1.1rem;
        text-align: left;
    }
    .feature-icon { font-size: 1.8rem; margin-bottom: 0.5rem; }
    .feature-title { font-size: 0.95rem; font-weight: 700; color: #AFA9EC; margin-bottom: 0.3rem; }
    .feature-desc  { font-size: 0.82rem; color: #888; line-height: 1.5; }

    .stat-row {
        display: flex;
        justify-content: center;
        gap: 40px;
        margin-bottom: 2.5rem;
    }
    .stat-item { text-align: center; }
    .stat-val  { font-size: 2rem; font-weight: 800; color: #AFA9EC; line-height: 1; }
    .stat-lbl  { font-size: 0.78rem; color: #666; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.05em; }

    .tech-bar {
        display: flex; justify-content: center; gap: 10px;
        flex-wrap: wrap; margin-bottom: 2rem;
    }
    .tech-pill {
        background: rgba(83,74,183,0.15);
        border: 1px solid rgba(175,169,236,0.25);
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.8rem;
        color: #AFA9EC;
        font-weight: 600;
    }
    .cta-note { font-size: 0.8rem; color: #555; margin-top: 0.5rem; }
    </style>
    """, unsafe_allow_html=True)

    # ── Hero ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class="landing-hero">
        <div class="landing-logo">🧠</div>
        <div class="landing-title">Student Stress Monitor</div>
        <div class="landing-sub">
            An AI-powered daily wellness companion that tracks your stress,
            predicts burnout before it happens, and helps you study smarter.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Stats row ─────────────────────────────────────────────────────
    st.markdown("""
    <div class="stat-row">
        <div class="stat-item">
            <div class="stat-val">14</div>
            <div class="stat-lbl">Daily inputs tracked</div>
        </div>
        <div class="stat-item">
            <div class="stat-val">81.5%</div>
            <div class="stat-lbl">Model accuracy</div>
        </div>
        <div class="stat-item">
            <div class="stat-val">4</div>
            <div class="stat-lbl">Stress levels predicted</div>
        </div>
        <div class="stat-item">
            <div class="stat-val">6</div>
            <div class="stat-lbl">Dashboard pages</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Feature cards ─────────────────────────────────────────────────
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <div class="feature-icon">📅</div>
            <div class="feature-title">Calendar Logging</div>
            <div class="feature-desc">Log any day directly from a colour-coded stress calendar. See your entire month at a glance.</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🤖</div>
            <div class="feature-title">ML Prediction</div>
            <div class="feature-desc">GradBoost classifier predicts Low / Moderate / High / Critical with confidence probabilities.</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">💡</div>
            <div class="feature-title">Smart Recommendations</div>
            <div class="feature-desc">Severity-graded action cards sorted by urgency — Critical issues always appear first.</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Rich Analytics</div>
            <div class="feature-desc">5 interactive Plotly charts — stress trends, sleep vs study, lifestyle heatmap, and more.</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📋</div>
            <div class="feature-title">Weekly & Monthly Summary</div>
            <div class="feature-desc">7-day heatmap, sparklines, best/worst day callouts, and metric pass/fail grid.</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">⏱️</div>
            <div class="feature-title">Study Timer</div>
            <div class="feature-desc">Pomodoro timer with stress-adaptive block lengths and session logging built right in.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tech pills ────────────────────────────────────────────────────
    st.markdown("""
    <div class="tech-bar">
        <span class="tech-pill">🐍 Python</span>
        <span class="tech-pill">⚡ Streamlit</span>
        <span class="tech-pill">🤖 scikit-learn</span>
        <span class="tech-pill">📈 Plotly</span>
        <span class="tech-pill">🗃️ SQLite</span>
        <span class="tech-pill">🐼 pandas</span>
    </div>
    """, unsafe_allow_html=True)

    # ── CTA buttons ───────────────────────────────────────────────────
    _, c1, c2, _ = st.columns([1.5, 1, 1, 1.5])
    with c1:
        if st.button("🚀 Get Started", use_container_width=True, type="primary"):
            st.session_state["show_landing"] = False
            st.rerun()
    with c2:
        if st.button("🔑 Sign In", use_container_width=True):
            st.session_state["show_landing"] = False
            st.rerun()

    st.markdown('<div class="cta-note" style="text-align:center;">'
                '🔒 Free · Private · All data stored locally on your device'
                '</div>', unsafe_allow_html=True)


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



def generate_stress_report(user, history_df, stress_score, level_name,
                            level_color, sleep, study, screen, anxiety,
                            exercise, pred_proba, LABELS, COLORS):
    """Generate a one-page PDF stress report using reportlab."""
    import io, subprocess, sys
    try:
        import reportlab
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "reportlab", "--quiet"])
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.units import mm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                     Table, TableStyle, HRFlowable)
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=18*mm, rightMargin=18*mm,
                            topMargin=14*mm, bottomMargin=14*mm)

    # ── Colour map ─────────────────────────────────────────────────
    LEVEL_HEX = {'Low':'#639922','Moderate':'#BA7517','High':'#993C1D','Critical':'#A32D2D'}
    hex_to_rl = lambda h: rl_colors.HexColor(h)

    lc = hex_to_rl(LEVEL_HEX.get(level_name, '#534AB7'))
    PRIMARY = rl_colors.HexColor('#534AB7')
    ACCENT  = rl_colors.HexColor('#D4537E')
    DARK    = rl_colors.HexColor('#1E1B3A')
    LIGHT   = rl_colors.HexColor('#F4F3FC')
    MUTED   = rl_colors.HexColor('#888888')
    WHITE   = rl_colors.white
    BLACK   = rl_colors.HexColor('#1A1A2E')

    # ── Styles ─────────────────────────────────────────────────────
    def sty(name, **kw):
        return ParagraphStyle(name, **kw)

    S = {
        'title':   sty('title',   fontSize=24, textColor=PRIMARY,    fontName='Helvetica-Bold', alignment=TA_LEFT, spaceAfter=2),
        'sub':     sty('sub',     fontSize=10, textColor=MUTED,       fontName='Helvetica',      alignment=TA_LEFT, spaceAfter=8),
        'h2':      sty('h2',      fontSize=13, textColor=PRIMARY,     fontName='Helvetica-Bold', spaceBefore=10, spaceAfter=4),
        'body':    sty('body',    fontSize=9,  textColor=BLACK,       fontName='Helvetica',      spaceAfter=4, leading=14),
        'badge':   sty('badge',   fontSize=20, textColor=lc,          fontName='Helvetica-Bold', alignment=TA_LEFT, spaceAfter=2),
        'score':   sty('score',   fontSize=11, textColor=MUTED,       fontName='Helvetica',      spaceAfter=6),
        'centre':  sty('centre',  fontSize=9,  textColor=MUTED,       fontName='Helvetica',      alignment=TA_CENTER),
        'rec_title':sty('rt',     fontSize=9,  textColor=BLACK,       fontName='Helvetica-Bold', spaceAfter=1),
        'rec_body': sty('rb',     fontSize=8,  textColor=rl_colors.HexColor('#444444'), fontName='Helvetica', spaceAfter=2, leading=12),
        'footer':  sty('footer',  fontSize=7,  textColor=MUTED,       fontName='Helvetica',      alignment=TA_CENTER),
    }

    import datetime as _dt
    today_str = _dt.date.today().strftime('%A, %d %B %Y')
    username  = user.get('username', 'Student').title()

    story = []

    # ── Header band ────────────────────────────────────────────────
    header_data = [[
        Paragraph(f'<b>Student Stress Monitor</b>', sty('hd', fontSize=14, textColor=WHITE, fontName='Helvetica-Bold')),
        Paragraph(f'Wellness Report · {today_str}', sty('hd2', fontSize=9, textColor=rl_colors.HexColor('#ccccff'), fontName='Helvetica', alignment=TA_RIGHT)),
    ]]
    header_tbl = Table(header_data, colWidths=[105*mm, 65*mm])
    header_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), PRIMARY),
        ('TEXTCOLOR',  (0,0), (-1,-1), WHITE),
        ('ALIGN',      (1,0), (1,0),   'RIGHT'),
        ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING',(0,0),(-1,-1),8),
        ('LEFTPADDING',(0,0), (-1,-1), 10),
        ('RIGHTPADDING',(0,0),(-1,-1), 10),
        ('ROUNDEDCORNERS', [4]),
    ]))
    story.append(header_tbl)
    story.append(Spacer(1, 8*mm))

    # ── Student + stress result row ────────────────────────────────
    emoji_map = {'Low':'Low Stress','Moderate':'Moderate Stress',
                 'High':'High Stress','Critical':'Critical Stress'}
    result_left = [
        [Paragraph(f'Student', sty('sl',fontSize=8,textColor=MUTED,fontName='Helvetica'))],
        [Paragraph(f'<b>{username}</b>', sty('sn',fontSize=13,textColor=BLACK,fontName='Helvetica-Bold'))],
        [Spacer(1, 4)],
        [Paragraph(f'{emoji_map.get(level_name,level_name)}', S['badge'])],
        [Paragraph(f'Stress Score: <b>{stress_score} / 100</b>', S['score'])],
    ]
    # Score bar as a table row
    filled = max(1, int(stress_score * 0.82))
    empty  = 82 - filled
    bar_data = [['']*filled + ['']*empty]
    bar_tbl = Table(bar_data, colWidths=[1*mm]*82, rowHeights=[4*mm])
    bar_style = [
        ('BACKGROUND',(0,0),(filled-1,0), lc),
        ('BACKGROUND',(filled,0),(81,0),  rl_colors.HexColor('#E8E8F0')),
        ('LEFTPADDING',(0,0),(-1,-1),0),
        ('RIGHTPADDING',(0,0),(-1,-1),0),
        ('TOPPADDING',(0,0),(-1,-1),0),
        ('BOTTOMPADDING',(0,0),(-1,-1),0),
    ]
    if filled > 0:
        bar_style.append(('ROUNDEDCORNERS',[2]))
    bar_tbl.setStyle(TableStyle(bar_style))
    result_left.append([bar_tbl])

    # Probability column
    prob_rows = [[Paragraph('<b>ML Confidence</b>', sty('pc',fontSize=8,textColor=MUTED,fontName='Helvetica-Bold'))]]
    LEVEL_COLS = {'Low':'#639922','Moderate':'#BA7517','High':'#993C1D','Critical':'#A32D2D'}
    if pred_proba is not None:
        for i, lbl in enumerate(LABELS):
            pct = float(pred_proba[i]) * 100
            bar_w = max(1, int(pct * 0.5))
            bc = rl_colors.HexColor(LEVEL_COLS.get(lbl,'#888'))
            mini_bar = Table([['']*bar_w + ['']*(50-bar_w)],
                              colWidths=[1*mm]*50, rowHeights=[3*mm])
            mini_bar.setStyle(TableStyle([
                ('BACKGROUND',(0,0),(bar_w-1,0), bc),
                ('BACKGROUND',(bar_w,0),(49,0),  rl_colors.HexColor('#EEEEEE')),
                ('LEFTPADDING',(0,0),(-1,-1),0),('RIGHTPADDING',(0,0),(-1,-1),0),
                ('TOPPADDING',(0,0),(-1,-1),0),('BOTTOMPADDING',(0,0),(-1,-1),0),
            ]))
            prob_rows.append([
                Table([[
                    Paragraph(f'{lbl}', sty(f'pl{i}',fontSize=8,textColor=rl_colors.HexColor(LEVEL_COLS.get(lbl,'#888')),fontName='Helvetica-Bold')),
                    mini_bar,
                    Paragraph(f'{pct:.1f}%', sty(f'pp{i}',fontSize=8,textColor=BLACK,fontName='Helvetica')),
                ]], colWidths=[18*mm, 25*mm, 12*mm])
            ])
    else:
        prob_rows.append([Paragraph('Rule-based score used', S['body'])])

    result_tbl = Table(
        [[Table(result_left, colWidths=[88*mm]), Table(prob_rows, colWidths=[82*mm])]],
        colWidths=[92*mm, 82*mm]
    )
    result_tbl.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,-1), LIGHT),
        ('VALIGN',(0,0),(-1,-1),'TOP'),
        ('TOPPADDING',(0,0),(-1,-1),10),
        ('BOTTOMPADDING',(0,0),(-1,-1),10),
        ('LEFTPADDING',(0,0),(-1,-1),10),
        ('RIGHTPADDING',(0,0),(-1,-1),10),
        ('BOX',(0,0),(-1,-1),1,rl_colors.HexColor('#C5C2E8')),
        ('ROUNDEDCORNERS',[6]),
    ]))
    story.append(result_tbl)
    story.append(Spacer(1, 5*mm))

    # ── Today's inputs grid ────────────────────────────────────────
    story.append(Paragraph("Today's Inputs", S['h2']))
    exer_txt = 'Yes' if int(exercise) >= 1 else 'No'
    input_data = [
        ['Sleep', f'{sleep}h', 'Study', f'{study}h', 'Screen', f'{screen}h'],
        ['Anxiety', f'{anxiety}/10', 'Exercise', exer_txt, 'Score', f'{stress_score}/100'],
    ]
    in_tbl = Table(input_data, colWidths=[22*mm,28*mm,22*mm,28*mm,22*mm,28*mm])
    in_tbl.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(0,-1), PRIMARY),('TEXTCOLOR',(0,0),(0,-1),WHITE),
        ('BACKGROUND',(2,0),(2,-1), PRIMARY),('TEXTCOLOR',(2,0),(2,-1),WHITE),
        ('BACKGROUND',(4,0),(4,-1), PRIMARY),('TEXTCOLOR',(4,0),(4,-1),WHITE),
        ('BACKGROUND',(1,0),(1,-1), LIGHT),
        ('BACKGROUND',(3,0),(3,-1), LIGHT),
        ('BACKGROUND',(5,0),(5,-1), LIGHT),
        ('FONTNAME',(0,0),(-1,-1),'Helvetica'),
        ('FONTSIZE',(0,0),(-1,-1),8),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),
        ('GRID',(0,0),(-1,-1),0.5,rl_colors.HexColor('#C5C2E8')),
        ('ROUNDEDCORNERS',[4]),
    ]))
    story.append(in_tbl)
    story.append(Spacer(1, 5*mm))

    # ── Top recommendations ────────────────────────────────────────
    story.append(Paragraph('Top Recommendations', S['h2']))

    RECS = []
    if float(sleep) < 7:
        RECS.append(('High','Sleep Below Target',
                     f'Only {sleep}h of sleep. Target 7-9h. Set a fixed bedtime tonight.'))
    if float(study) > 8:
        RECS.append(('High','Study Overload',
                     f'{study}h of study today. Sustainable limit is 8h. Schedule a hard stop.'))
    if int(exercise) == 0:
        RECS.append(('Moderate','No Exercise Today',
                     'Exercise is the strongest natural stress reducer. Even a 20-min walk helps.'))
    if float(screen) > 4:
        RECS.append(('Moderate','High Screen Time',
                     f'{screen}h screen time. Limit to 4h. No screens 30 min before bed.'))
    if int(anxiety) >= 7:
        RECS.append(('Critical','High Anxiety',
                     f'Anxiety at {anxiety}/10. Try box breathing: inhale 4s, hold 4s, exhale 4s.'))
    if not RECS:
        RECS.append(('Positive','Great Balance',
                     'All key metrics are within healthy ranges. Keep up the consistent habits.'))

    REC_COLORS = {'Critical':'#A32D2D','High':'#BA7517','Moderate':'#534AB7','Positive':'#639922'}
    for sev, title, desc in RECS[:4]:
        rc = rl_colors.HexColor(REC_COLORS.get(sev,'#534AB7'))
        rec_row = Table([[
            Table([[Paragraph(f'<b>{sev}</b>', sty(f'rs{sev}',fontSize=7,textColor=WHITE,fontName='Helvetica-Bold',alignment=TA_CENTER))]], colWidths=[16*mm], rowHeights=[5*mm]),
            Table([[Paragraph(f'<b>{title}</b>', S['rec_title'])],[Paragraph(desc, S['rec_body'])]], colWidths=[148*mm]),
        ]], colWidths=[18*mm, 150*mm])
        rec_row.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(0,0), rc),
            ('BACKGROUND',(1,0),(1,0), rl_colors.HexColor('#F8F7FF')),
            ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
            ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
            ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),
            ('BOX',(0,0),(-1,-1),0.5,rl_colors.HexColor('#C5C2E8')),
        ]))
        story.append(rec_row)
        story.append(Spacer(1, 2*mm))

    # ── Session history summary ────────────────────────────────────
    if not history_df.empty:
        story.append(Spacer(1, 3*mm))
        story.append(Paragraph('Session History Summary', S['h2']))
        hdf = history_df.copy()
        hdf['stress_score'] = _pd_to_numeric(hdf['stress_score'])
        avg_s = hdf['stress_score'].mean()
        sessions = len(hdf)
        best_s   = hdf['stress_score'].min()
        worst_s  = hdf['stress_score'].max()

        hist_data = [
            ['Metric', 'Value', 'Metric', 'Value'],
            ['Total sessions', str(sessions), 'Avg stress score', f'{avg_s:.0f}'],
            ['Best score (lowest)', f'{best_s:.0f}', 'Worst score (highest)', f'{worst_s:.0f}'],
        ]
        hist_tbl = Table(hist_data, colWidths=[42*mm,46*mm,42*mm,42*mm])
        hist_tbl.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),PRIMARY),('TEXTCOLOR',(0,0),(-1,0),WHITE),
            ('BACKGROUND',(0,1),(0,-1),LIGHT),('BACKGROUND',(2,1),(2,-1),LIGHT),
            ('FONTNAME',(0,0),(-1,-1),'Helvetica'),('FONTSIZE',(0,0),(-1,-1),8),
            ('ALIGN',(0,0),(-1,-1),'CENTER'),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
            ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),
            ('GRID',(0,0),(-1,-1),0.5,rl_colors.HexColor('#C5C2E8')),
        ]))
        story.append(hist_tbl)

    # ── Footer ─────────────────────────────────────────────────────
    story.append(Spacer(1, 6*mm))
    story.append(HRFlowable(width='100%', thickness=0.5, color=rl_colors.HexColor('#C5C2E8')))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        f'Student Stress Monitor · {today_str} · For educational purposes only · Data stored locally',
        S['footer']
    ))

    doc.build(story)
    buf.seek(0)
    return buf.read()


def _pd_to_numeric(series):
    import pandas as pd
    return pd.to_numeric(series, errors='coerce').fillna(0)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═════════════════════════════════════════════════════════════════════════════

def show_main_app(user: dict):
    import calendar as cal_mod
    model, scaler, meta = load_model()
    MODEL_READY = model is not None
    user_id  = user["id"]
    username = user["username"]

    history_df  = load_sessions(user_id)
    saved_goals = load_goals(user_id)

    # ── Page state ────────────────────────────────────────────────────────────
    if "page" not in st.session_state:
        st.session_state["page"] = "dashboard"

    # ── Top navbar ────────────────────────────────────────────────────────────
    nav_pages = [
        ("dashboard", "📊 Dashboard"),
        ("entry",     "✏️ New Entry"),
        ("summary",   "📋 Summary"),
        ("history",   "📈 History"),
        ("goals",     "🎯 Goals"),
        ("planner",   "📅 Planner"),
    ]
    n1,n2,n3,n4,n5,n6,nr = st.columns([1.2,1.1,1,1,1,1,1.1])
    for col,(pg_key,pg_label) in zip([n1,n2,n3,n4,n5,n6], nav_pages):
        with col:
            is_active = st.session_state["page"] == pg_key
            if st.button(pg_label, use_container_width=True,
                         type="primary" if is_active else "secondary",
                         key=f"nav_{pg_key}"):
                st.session_state["page"] = pg_key
                st.rerun()
    with nr:
        st.markdown(
            f'<div style="text-align:right;padding-top:2px;">' +
            f'<span class="user-chip">👤 {username}</span></div>',
            unsafe_allow_html=True)
        if st.button("Sign out", use_container_width=True, key="so_btn"):
            del st.session_state["user"]; st.rerun()

    st.markdown("<hr style='margin:0.5rem 0 1rem;border-color:rgba(255,255,255,0.07);'>",
                unsafe_allow_html=True)

    current_page = st.session_state["page"]

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE: NEW ENTRY
    # ═══════════════════════════════════════════════════════════════════════════
    if current_page == "entry":
        st.markdown('<div class="main-title">✏️ Log a New Entry</div>', unsafe_allow_html=True)
        st.caption("Pick a date on the calendar, fill in your details, and save.")
        st.divider()

        # Calendar data
        hdf_cal = history_df.copy() if not history_df.empty else pd.DataFrame()
        if not hdf_cal.empty:
            hdf_cal["stress_score"] = pd.to_numeric(hdf_cal["stress_score"], errors="coerce")
            hdf_cal["cal_date"]     = pd.to_datetime(hdf_cal["timestamp"]).dt.date
            date_stress = hdf_cal.groupby("cal_date")["stress_score"].mean().to_dict()
            date_level  = hdf_cal.groupby("cal_date")["stress_level"].last().to_dict()
        else:
            date_stress = {}; date_level = {}

        today = datetime.date.today()
        if "cal_year"   not in st.session_state: st.session_state["cal_year"]   = today.year
        if "cal_month"  not in st.session_state: st.session_state["cal_month"]  = today.month
        if "entry_date" not in st.session_state: st.session_state["entry_date"] = today

        cy = st.session_state["cal_year"]
        cm = st.session_state["cal_month"]

        cal_col, form_col = st.columns([1, 1.4])

        # ── Calendar ─────────────────────────────────────────────────────────
        with cal_col:
            st.markdown('<div class="section-header">📅 Select Date</div>', unsafe_allow_html=True)

            # Month navigation
            mn1,mn2,mn3 = st.columns([1,3,1])
            with mn1:
                if st.button("◀", key="cal_prev"):
                    if cm==1: st.session_state["cal_month"]=12; st.session_state["cal_year"]-=1
                    else:     st.session_state["cal_month"]-=1
                    st.rerun()
            with mn2:
                st.markdown(
                    f'<div style="text-align:center;font-weight:700;font-size:1rem;' +
                    f'color:#AFA9EC;padding-top:4px;">{cal_mod.month_name[cm]} {cy}</div>',
                    unsafe_allow_html=True)
            with mn3:
                if st.button("▶", key="cal_next"):
                    if cm==12: st.session_state["cal_month"]=1; st.session_state["cal_year"]+=1
                    else:      st.session_state["cal_month"]+=1
                    st.rerun()

            def stress_bg(score):
                if score is None: return "rgba(255,255,255,0.04)"
                if score>=90:     return "rgba(163,45,45,0.55)"
                if score>=75:     return "rgba(153,60,29,0.50)"
                if score>=30:     return "rgba(186,117,23,0.45)"
                return                   "rgba(99,153,34,0.45)"

            def stress_border(score):
                if score is None: return "rgba(255,255,255,0.1)"
                if score>=90:     return "#E24B4A"
                if score>=75:     return "#D85A30"
                if score>=30:     return "#EF9F27"
                return                   "#639922"

            cal_mod.setfirstweekday(6)
            month_weeks   = cal_mod.monthcalendar(cy, cm)
            selected_date = st.session_state["entry_date"]

            # Build calendar HTML
            cal_html = """
<style>
.ssm-cal{display:grid;grid-template-columns:repeat(7,1fr);gap:5px;margin-top:6px;}
.ssm-ch{text-align:center;font-size:0.68rem;font-weight:700;color:#555;
         padding:3px 0;text-transform:uppercase;letter-spacing:0.04em;}
.ssm-cd{border-radius:10px;padding:5px 3px 4px;text-align:center;
         font-size:0.8rem;font-weight:600;border:2px solid transparent;
         min-height:48px;display:flex;flex-direction:column;
         align-items:center;justify-content:center;gap:2px;
         transition:transform .12s;}
.ssm-cd:hover{transform:scale(1.06);}
.ssm-num{font-size:0.84rem;line-height:1;}
.ssm-sc{font-size:0.6rem;opacity:0.8;line-height:1;}
.ssm-dot{width:5px;height:5px;border-radius:50%;}
</style>
<div class="ssm-cal">"""

            for d in ["Sun","Mon","Tue","Wed","Thu","Fri","Sat"]:
                cal_html += f'<div class="ssm-ch">{d}</div>'

            for week in month_weeks:
                for day in week:
                    if day == 0:
                        cal_html += '<div></div>'; continue
                    d_obj  = datetime.date(cy, cm, day)
                    score  = date_stress.get(d_obj)
                    bg     = stress_bg(score)
                    border = stress_border(score)
                    is_sel = d_obj == selected_date
                    is_tod = d_obj == today
                    is_fut = d_obj > today

                    ring    = "box-shadow:0 0 0 3px #AFA9EC,0 0 0 5px rgba(175,169,236,0.2);" if is_sel else ""
                    op      = "opacity:0.3;pointer-events:none;" if is_fut else ""
                    dash    = "border-style:dashed;" if is_tod and not is_sel else ""
                    sc_html = f'<div class="ssm-sc">{score:.0f}</div>' if score is not None else ""
                    dot_col = border if score is not None else "transparent"
                    dot     = f'<div class="ssm-dot" style="background:{dot_col};"></div>'

                    _score_tip = f" • {score:.0f}" if score is not None else ""
                    cal_html += (
                        '<div class="ssm-cd" ' +
                        f'style="background:{bg};border-color:{border};{ring}{op}{dash}" ' +
                        f'title="{d_obj:%b %d}{_score_tip}">' +
                        f'<div class="ssm-num">{day}</div>{sc_html}{dot}</div>'
                    )

            cal_html += "</div>"
            st.markdown(cal_html, unsafe_allow_html=True)

            # Date picker below calendar
            st.markdown("<div style='margin-top:0.7rem'></div>", unsafe_allow_html=True)
            picked = st.date_input("Pick date", value=selected_date,
                                   max_value=today, key="dpick",
                                   label_visibility="collapsed")
            if picked != st.session_state["entry_date"]:
                st.session_state["entry_date"] = picked
                st.session_state["cal_year"]   = picked.year
                st.session_state["cal_month"]  = picked.month
                st.rerun()

            # Legend
            st.markdown("""
<div style="display:flex;gap:8px;margin-top:0.6rem;flex-wrap:wrap;">
  <span style="font-size:0.7rem;color:#555;display:flex;align-items:center;gap:4px;">
    <span style="width:9px;height:9px;border-radius:3px;background:rgba(99,153,34,0.6);
                 border:1.5px solid #639922;display:inline-block;"></span>Low</span>
  <span style="font-size:0.7rem;color:#555;display:flex;align-items:center;gap:4px;">
    <span style="width:9px;height:9px;border-radius:3px;background:rgba(186,117,23,0.55);
                 border:1.5px solid #EF9F27;display:inline-block;"></span>Moderate</span>
  <span style="font-size:0.7rem;color:#555;display:flex;align-items:center;gap:4px;">
    <span style="width:9px;height:9px;border-radius:3px;background:rgba(153,60,29,0.55);
                 border:1.5px solid #D85A30;display:inline-block;"></span>High</span>
  <span style="font-size:0.7rem;color:#555;display:flex;align-items:center;gap:4px;">
    <span style="width:9px;height:9px;border-radius:3px;background:rgba(163,45,45,0.6);
                 border:1.5px solid #E24B4A;display:inline-block;"></span>Critical</span>
  <span style="font-size:0.7rem;color:#555;display:flex;align-items:center;gap:4px;">
    <span style="width:9px;height:9px;border-radius:3px;border:1.5px dashed #AFA9EC;
                 display:inline-block;"></span>Today</span>
</div>""", unsafe_allow_html=True)

        # ── Entry form ────────────────────────────────────────────────────────
        with form_col:
            sel_date = st.session_state["entry_date"]
            existing_level = date_level.get(sel_date)
            existing_score = date_stress.get(sel_date)

            if existing_level:
                ec = {"Low":"#639922","Moderate":"#EF9F27","High":"#D85A30","Critical":"#E24B4A"}.get(existing_level,"#888")
                st.markdown(
                    f'<div style="background:rgba(255,255,255,0.04);border:1px solid {ec}44;' +
                    f'border-left:4px solid {ec};border-radius:12px;padding:0.8rem 1rem;margin-bottom:0.8rem;">' +
                    f'<div style="font-size:0.72rem;color:#777;text-transform:uppercase;letter-spacing:0.05em;">' +
                    f'Existing entry — {sel_date:%A, %d %b %Y}</div>' +
                    f'<div style="font-size:1.3rem;font-weight:800;color:{ec};">{existing_level} Stress</div>' +
                    f'<div style="font-size:0.82rem;color:#888;">Score: {existing_score:.0f}/100</div></div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div style="background:rgba(83,74,183,0.1);border:1px solid rgba(175,169,236,0.2);' +
                    f'border-radius:12px;padding:0.8rem 1rem;margin-bottom:0.8rem;">' +
                    f'<div style="font-size:0.72rem;color:#777;text-transform:uppercase;letter-spacing:0.05em;">Logging entry for</div>' +
                    f'<div style="font-size:1.1rem;font-weight:700;color:#AFA9EC;">{sel_date:%A, %d %b %Y}</div>' +
                    f'<div style="font-size:0.8rem;color:#666;">No entry yet</div></div>',
                    unsafe_allow_html=True)

            with st.form("entry_form"):
                st.markdown("""
                <style>
                .entry-section {
                    background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08);
                    border-radius:14px; padding:1rem 1.2rem; margin-bottom:0.8rem;
                }
                .entry-section-title {
                    font-size:0.8rem; font-weight:700; color:#AFA9EC;
                    text-transform:uppercase; letter-spacing:0.07em; margin-bottom:0.6rem;
                }
                </style>""", unsafe_allow_html=True)

                # ── Section 1: Academic ────────────────────────────────────
                st.markdown('<div class="entry-section-title">📚 Academic — Today</div>', unsafe_allow_html=True)
                ac1, ac2, ac3, ac4 = st.columns(4)
                with ac1:
                    study = st.number_input("Hours studied today", 0.0, 24.0, 6.0, 0.5, key="e_study",
                                            help="How many hours did you study today?")
                with ac2:
                    assignments = st.slider("Pending assignments", 0, 15, 3, key="e_asgn",
                                            help="Total unfinished tasks right now")
                with ac3:
                    exam = st.slider("Exam pressure today (1–10)", 1, 10, 5, key="e_exam",
                                     help="How much exam-related pressure do you feel today?")
                with ac4:
                    performance = st.slider("Academic performance (1–10)", 1, 10, 7, key="e_perf",
                                            help="How well are you performing academically overall?")

                st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

                # ── Section 2: Lifestyle ───────────────────────────────────
                st.markdown('<div class="entry-section-title">🌙 Lifestyle — Today</div>', unsafe_allow_html=True)
                lc1, lc2, lc3, lc4 = st.columns(4)
                with lc1:
                    sleep = st.number_input("Hours slept last night", 0.0, 24.0, 7.0, 0.5, key="e_sleep",
                                            help="How many hours did you sleep last night?")
                with lc2:
                    exercised_today = st.radio("Exercised today?", ["Yes", "No"],
                                               index=1, key="e_exer_radio", horizontal=True,
                                               help="Any physical activity today (walk, gym, sport, etc.)")
                    exercise = 1 if exercised_today == "Yes" else 0
                with lc3:
                    screen = st.number_input("Screen time today (hrs)", 0.0, 24.0, 4.0, 0.5, key="e_screen",
                                             help="Total hours on phone, laptop, TV, etc. today")
                with lc4:
                    social = st.slider("Meaningful interactions today", 0, 20, 5, key="e_soc",
                                       help="How many meaningful conversations/interactions did you have today?")

                st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

                # ── Section 3: Mental & Emotional ─────────────────────────
                st.markdown('<div class="entry-section-title">🧠 Mental & Emotional — Today</div>', unsafe_allow_html=True)
                mc1, mc2, mc3, mc4 = st.columns(4)
                with mc1:
                    anxiety = st.slider("Anxiety level today (1–10)", 1, 10, 4, key="e_anx",
                                        help="How anxious do you feel today?")
                with mc2:
                    finance = st.slider("Financial stress (1–10)", 1, 10, 3, key="e_fin",
                                        help="How stressed are you about money/finances right now?")
                with mc3:
                    family = st.slider("Family support felt today (1–10)", 1, 10, 7, key="e_fam",
                                       help="How supported do you feel by family today?")
                with mc4:
                    peer = st.slider("Peer pressure today (1–10)", 1, 10, 4, key="e_peer",
                                     help="How much peer pressure did you experience today?")

                st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

                # ── Section 4: Misc ────────────────────────────────────────
                st.markdown('<div class="entry-section-title">✨ Other</div>', unsafe_allow_html=True)
                oc1, oc2 = st.columns(2)
                with oc1:
                    extra = st.selectbox("Extracurricular activities (overall)",
                                         [0, 1, 2],
                                         format_func=lambda x: ["None", "1–2 activities", "3+ activities"][x],
                                         key="e_extra")
                with oc2:
                    rel = st.selectbox("Relationship situation",
                                       [0, 1, 2],
                                       format_func=lambda x: ["Single / N/A", "Stable relationship", "Relationship issues"][x],
                                       key="e_rel")

                st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
                save_btn = st.form_submit_button("💾 Save Entry",
                                                  use_container_width=True, type="primary")

            # ── Compute live stress (exercise: 1 day = ~1/7 weekly) ───────
            # exercise is 0 or 1 (did/didn't exercise today)
            # For the stress formula, scale: 1 exercised day ≈ 5/7 of the weekly target
            exercise_formula = 5 if exercise == 1 else 0  # penalty: max(0, 5 - x)*2
            raw = (
                max(0, study - 8) * 3.5 + assignments * 2.5 + (exam - 1) * 5.0
                + max(0, 7 - sleep) * 4.0 + max(0, 5 - exercise_formula) * 2.0
                + max(0, 8 - social) * 1.5 + max(0, screen - 4) * 2.0
                + (anxiety - 1) * 4.5 + (finance - 1) * 3.0
                - (family - 1) * 2.5 - (performance - 1) * 2.0 + (peer - 1) * 2.5
                + (5 if extra == 0 else 0) + (8 if rel == 2 else 0)
            )
            stress_score = int(np.clip(raw, 0, 100))
            if MODEL_READY:
                inp        = np.array([[study, assignments, exam, performance,
                                        sleep, exercise_formula, social, screen,
                                        anxiety, finance, family, peer, extra, rel]])
                inp_sc     = scaler.transform(inp)
                pred_class = int(model.predict(inp_sc)[0])
                pred_proba = model.predict_proba(inp_sc)[0]
                # Apply rule-based correction — ML is biased 1 class higher
                rule_class = (3 if stress_score >= 90 else 2 if stress_score >= 75
                              else 1 if stress_score >= 30 else 0)
                if pred_class > rule_class:
                    if pred_proba is not None:
                        corrected = pred_proba.copy().astype(float)
                        corrected[rule_class] += corrected[pred_class]
                        corrected[pred_class] = 0.0
                        pred_proba = corrected
                    pred_class = rule_class
                level_name = LABELS[pred_class]
            else:
                pred_proba = None; pred_class = 0
                if   stress_score >= 90: level_name = "Critical"
                elif stress_score >= 75: level_name = "High"
                elif stress_score >= 30: level_name = "Moderate"
                else:                    level_name = "Low"

            level_color = COLORS[level_name]
            level_emoji = EMOJIS[level_name]
            recovery    = int(((exercise * 0.4) + (sleep / 10) * 0.4 + (social / 20) * 0.2) * 100)
            burnout     = min(100, int(stress_score * 0.6 + max(0, study - 8) * 4 + max(0, 10 - sleep) * 3))

            # ── Live result cards ─────────────────────────────────────────
            rc1, rc2 = st.columns(2)
            with rc1:
                st.markdown(
                    f'<div style="background:rgba(255,255,255,0.04);border:1px solid {level_color}55;' +
                    f'border-top:4px solid {level_color};border-radius:14px;padding:1rem 1.1rem;margin-top:0.5rem;">' +
                    f'<div style="font-size:0.72rem;color:#888;text-transform:uppercase;letter-spacing:0.05em;">Live Result</div>' +
                    f'<div style="font-size:1.8rem;font-weight:800;color:{level_color};margin:4px 0 2px;">{level_emoji} {level_name}</div>' +
                    f'<div style="font-size:0.85rem;color:#aaa;margin-bottom:8px;">Score: <strong style="color:#fff">{stress_score}</strong>/100</div>' +
                    f'<div style="background:rgba(255,255,255,0.07);border-radius:5px;height:7px;overflow:hidden;">' +
                    f'<div style="width:{stress_score}%;height:100%;background:{level_color};border-radius:5px;"></div>' +
                    f'</div></div>',
                    unsafe_allow_html=True)
            with rc2:
                exer_txt = "✅ Exercised" if exercise else "❌ Rest day"
                st.markdown(
                    f'<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);' +
                    f'border-radius:14px;padding:1rem 1.1rem;margin-top:0.5rem;">' +
                    f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">' +
                    f'<div><div style="font-size:0.7rem;color:#666;">Recovery</div>' +
                    f'<div style="font-size:1.4rem;font-weight:800;color:#97C459;">{recovery}%</div></div>' +
                    f'<div><div style="font-size:0.7rem;color:#666;">Burnout Risk</div>' +
                    f'<div style="font-size:1.4rem;font-weight:800;color:#F09595;">{burnout}</div></div>' +
                    f'<div><div style="font-size:0.7rem;color:#666;">Sleep</div>' +
                    f'<div style="font-size:1.4rem;font-weight:800;color:#AFA9EC;">{sleep}h</div></div>' +
                    f'<div><div style="font-size:0.7rem;color:#666;">Exercise</div>' +
                    f'<div style="font-size:1rem;font-weight:700;color:#AFA9EC;padding-top:4px;">{exer_txt}</div></div>' +
                    f'</div></div>',
                    unsafe_allow_html=True)

            if save_btn:
                save_session(user_id, stress_score, level_name,
                             sleep, study, screen, anxiety, exercise,
                             entry_date=st.session_state["entry_date"])
                st.success(f"✅ Entry saved for {sel_date:%A, %d %b %Y}!")
                st.rerun()

        return  # entry page ends

    # ═══════════════════════════════════════════════════════════════════════════
    # Shared values for all other pages (from latest session)
    # ═══════════════════════════════════════════════════════════════════════════
    if not history_df.empty:
        latest       = history_df.iloc[-1]
        study        = float(latest.get("study",    6))
        sleep        = float(latest.get("sleep",    7))
        screen       = float(latest.get("screen",   4))
        anxiety      = int(  latest.get("anxiety",  4))
        exercise     = int(  latest.get("exercise", 3))
        assignments  = 3; exam=5; performance=7; social=5; finance=3; family=7; peer=4; extra=0; rel=0
        _raw_score   = pd.to_numeric(latest.get("stress_score", None), errors="coerce")
        stress_score = int(_raw_score) if pd.notna(_raw_score) else 30
        level_name   = str(latest.get("stress_level","Low"))
    else:
        study=6;sleep=7;screen=4;anxiety=4;exercise=3;assignments=3
        exam=5;performance=7;social=5;finance=3;family=7;peer=4;extra=0;rel=0
        stress_score=0; level_name="Low"

    level_color = COLORS.get(level_name,"#639922")
    level_emoji = EMOJIS.get(level_name,"😊")

    if MODEL_READY and not history_df.empty:
        try:
            # exercise stored as 0/1 per day; model trained on 0-7 weekly scale
            exercise_ml = 5 if int(exercise) >= 1 else 0
            inp        = np.array([[study,assignments,exam,performance,
                                    sleep,exercise_ml,social,screen,
                                    anxiety,finance,family,peer,extra,rel]])
            inp_sc     = scaler.transform(inp)
            pred_class = int(model.predict(inp_sc)[0])
            pred_proba = model.predict_proba(inp_sc)[0]
            # Rule-based class from our calibrated thresholds
            rule_class = (3 if stress_score>=90 else 2 if stress_score>=75
                          else 1 if stress_score>=30 else 0)
            # The ML model is systematically biased 1 class higher due to
            # training data distribution. Always use rule_class when it gives
            # a lower (less severe) stress level — rule-based is calibrated to
            # our exact thresholds (0-29 Low, 30-74 Moderate, 75-89 High, 90+ Critical)
            if pred_class > rule_class:
                # Shift probability mass from ML class to rule class
                if pred_proba is not None:
                    corrected = pred_proba.copy().astype(float)
                    corrected[rule_class] += corrected[pred_class]
                    corrected[pred_class] = 0.0
                    pred_proba = corrected
                pred_class = rule_class
            level_name = LABELS[pred_class]
            level_color = COLORS.get(level_name, "#639922")
            level_emoji = EMOJIS.get(level_name, "😊")
        except Exception:
            pred_proba=None; pred_class=0
    else:
        pred_proba=None; pred_class=0

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE: DASHBOARD  (Stress Result + Factor Analysis + Tips)
    # ═══════════════════════════════════════════════════════════════════════════
    if current_page == "dashboard":
        if MODEL_READY:
            acc=meta.get("accuracy",0)
            c1,c2,c3=st.columns(3)
            c1.metric("ML Model",       meta.get("best_model","Loaded"))
            c2.metric("Model Accuracy", f"{acc*100:.1f}%")
            c3.metric("Sessions logged",str(len(history_df)))
        else:
            st.warning("⚠️ ML model not found. Run `python src/train_model.py` to enable AI predictions.")

        if history_df.empty:
            st.info("👋 Welcome! Head to **✏️ New Entry** to log your first stress entry.")
            return

        tab1,tab2,tab3 = st.tabs([
            "📊 Stress Result","🔍 Factor Analysis","💡 Management Tips"
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
                    _proba_vals = [float(p) for p in pred_proba]
                    _pull = [0.05 if i == pred_class else 0 for i in range(4)]
                    fig_prob = go.Figure(go.Pie(
                        labels=LABELS,
                        values=_proba_vals,
                        hole=0.45,
                        pull=_pull,
                        marker=dict(
                            colors=['#639922','#EF9F27','#D85A30','#E24B4A'],
                            line=dict(color='#1a1a2e', width=2)
                        ),
                        textinfo='percent',
                        textposition='inside',
                        insidetextorientation='radial',
                        hovertemplate='<b>%{label}</b><br>%{percent}<extra></extra>',
                        sort=False,
                    ))
                    fig_prob.update_layout(
                        title=dict(text='Probability Distribution',
                                   font=dict(size=13, color='#ccc'),
                                   x=0.5, xanchor='center'),
                        paper_bgcolor='rgba(0,0,0,0)',
                        legend=dict(
                            orientation='v', x=1.02, y=0.5,
                            font=dict(color='#ccc', size=11),
                            bgcolor='rgba(0,0,0,0)'
                        ),
                        margin=dict(t=40, b=10, l=10, r=80),
                        height=300,
                        annotations=[dict(
                            text=f"{_proba_vals[pred_class]*100:.0f}%",
                            x=0.5, y=0.5, showarrow=False,
                            font=dict(size=18, color=COLORS.get(LABELS[pred_class],'#fff'), family='Arial'),
                        )]
                    )
                    st.plotly_chart(fig_prob, use_container_width=True)
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

            # ── PDF Report Download ───────────────────────────────────
            st.divider()
            st.markdown('<div class="section-header">📄 Export Report</div>',
                        unsafe_allow_html=True)
            rpt_col1, rpt_col2 = st.columns([2, 1])
            with rpt_col1:
                st.markdown(
                    '<div style="font-size:0.88rem;color:#888;padding-top:0.4rem;">'
                    'Generate a one-page PDF summary of your current stress level, '
                    'inputs, ML confidence, and top recommendations.</div>',
                    unsafe_allow_html=True)
            with rpt_col2:
                import datetime as _dt2
                if st.button("📄 Generate PDF Report", use_container_width=True, type="primary"):
                    with st.spinner("Generating report…"):
                        try:
                            pdf_bytes = generate_stress_report(
                                user, history_df, stress_score, level_name,
                                level_color, sleep, study, screen, anxiety,
                                exercise, pred_proba, LABELS, COLORS
                            )
                            st.session_state["pdf_bytes"] = pdf_bytes
                            st.session_state["pdf_fn"] = (
                                f"stress_report_{user['username']}_{_dt2.date.today()}.pdf"
                            )
                        except Exception as e:
                            st.error(f"PDF generation failed: {e}")
                if "pdf_bytes" in st.session_state and st.session_state["pdf_bytes"]:
                    st.download_button(
                        label="⬇️ Download PDF Report",
                        data=st.session_state["pdf_bytes"],
                        file_name=st.session_state.get("pdf_fn", "stress_report.pdf"),
                        mime="application/pdf",
                        use_container_width=True,
                    )

        # ══════════════════════════════════════════════════════════
        # TAB 2 — Factor Analysis
        # ══════════════════════════════════════════════════════════
        with tab2:
            st.markdown("#### Which factors are driving your stress?")
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
            sorted_f = sorted(factor_scores.items(),key=lambda x:x[1],reverse=True)
            fig,ax = plt.subplots(figsize=(8,5))
            names=[f[0] for f in sorted_f]; vals=[f[1] for f in sorted_f]
            clrs=['#A32D2D' if v>=75 else '#993C1D' if v>=55
                  else '#BA7517' if v>=30 else '#639922' for v in vals]
            bars = ax.barh(names,vals,color=clrs,edgecolor='white',linewidth=0.5)
            for bar,v in zip(bars,vals):
                ax.text(v+2,bar.get_y()+bar.get_height()/2,str(v),va='center',fontsize=10)
            ax.axvline(55,ls='--',lw=1,color='#BA7517',alpha=0.6,label='High threshold')
            ax.axvline(75,ls='--',lw=1,color='#A32D2D',alpha=0.6,label='Critical threshold')
            ax.set_xlim(0,115); ax.set_xlabel("Stress contribution score")
            ax.set_title("Stress Factor Breakdown",fontsize=13); ax.legend(fontsize=9)
            plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()

            st.markdown("#### Your inputs at a glance")
            sc = st.columns(4)
            snap=[("Study hrs/day",f"{study}h"),("Sleep hrs/night",f"{sleep}h"),
                  ("Anxiety",f"{anxiety}/10"),("Exercise days",str(exercise)),
                  ("Assignments",str(assignments)),("Financial stress",f"{finance}/10"),
                  ("Family support",f"{family}/10"),("Screen time",f"{screen}h")]
            for i,(k,v) in enumerate(snap): sc[i%4].metric(k,v)

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

            st.markdown(f"#### 💡 {len([r for r in recs if r[0] != POSITIVE])} active recommendations")

            cols_r = st.columns(2)
            for idx, (severity, icon, title, body, action, score) in enumerate(recs):
                with cols_r[idx % 2]:
                    bar_color = {"critical":"#E24B4A","high":"#FAC775",
                                 "moderate":"#7F77DD","positive":"#97C459"}[severity]
                    bar_width = score if severity != POSITIVE else 100
                    st.markdown(f"""
                    <div class="rec-card rec-{severity}">
                        <div class="rec-header">
                            <span style="font-size:1.4rem;">{icon}</span>
                            <div>
                                <span class="rec-badge badge-{severity}">{badge_labels[severity]}</span>
                                <div class="rec-title">{title}</div>
                            </div>
                        </div>
                        <p class="rec-body">{body}</p>
                        <div class="rec-action">⚡ {action}</div>
                        <div class="rec-score-bar">
                            <div style="width:{bar_width}%;height:100%;background:{bar_color};
                                        border-radius:2px;"></div>
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



    # PAGE: history
    if current_page == "history":

        st.markdown(f'<div class="main-title">📈 Stress History</div>', unsafe_allow_html=True)
        st.caption(f"All logged sessions for {username}")

        if history_df.empty:
            st.info("No history yet. Head to ✏️ New Entry to log your first session.")
        else:
            hdf = history_df.copy()
            for _c in ['stress_score','sleep','study','screen','anxiety','exercise']:
                hdf[_c] = pd.to_numeric(hdf[_c], errors='coerce')

            x_labels = hdf['day_label'].tolist()
            level_color_map = {'Low':'#639922','Moderate':'#EF9F27',
                               'High':'#D85A30','Critical':'#E24B4A'}

            # ── Helper: small number stat card ────────────────
            def num_card(label, value, sub="", color="#AFA9EC", border_color=None):
                bc = border_color or color
                return (
                    f'<div style="background:rgba(255,255,255,0.04);border:1px solid {bc}44;' +
                    f'border-top:3px solid {bc};border-radius:12px;padding:0.8rem 1rem;">' +
                    f'<div style="font-size:0.72rem;color:#666;text-transform:uppercase;' +
                    f'letter-spacing:0.05em;margin-bottom:4px;">{label}</div>' +
                    f'<div style="font-size:1.6rem;font-weight:800;color:{color};line-height:1;">{value}</div>' +
                    f'<div style="font-size:0.75rem;color:#666;margin-top:3px;">{sub}</div>' +
                    f'</div>'
                )

            # ════════════════════════════════════════════════════
            # SECTION 1 — Stress Score Over Time
            # ════════════════════════════════════════════════════
            st.markdown('<div class="section-header">📈 Stress Score Over Time</div>',
                        unsafe_allow_html=True)

            # Numbers row
            avg_stress  = hdf['stress_score'].mean()
            min_stress  = hdf['stress_score'].min()
            max_stress  = hdf['stress_score'].max()
            last_stress = hdf['stress_score'].iloc[-1]
            trend_val   = hdf['stress_score'].iloc[-1] - hdf['stress_score'].iloc[-2] if len(hdf)>1 else 0
            trend_str   = f"↓ {abs(trend_val):.0f} vs prev" if trend_val < 0 else (f"↑ {trend_val:.0f} vs prev" if trend_val > 0 else "→ No change")
            trend_color = "#97C459" if trend_val < 0 else ("#F09595" if trend_val > 0 else "#888")
            dom_level   = hdf['stress_level'].mode()[0] if 'stress_level' in hdf.columns else "—"
            dom_color   = level_color_map.get(dom_level, "#888")

            nc1,nc2,nc3,nc4,nc5 = st.columns(5)
            for col, lbl, val, sub, clr in [
                (nc1, "Sessions",      str(len(hdf)),          "total logged",       "#AFA9EC"),
                (nc2, "Avg Score",     f"{avg_stress:.0f}",     "out of 100",         "#AFA9EC"),
                (nc3, "Latest Score",  f"{last_stress:.0f}",    trend_str,             trend_color),
                (nc4, "Best Score",    f"{min_stress:.0f}",     "lowest stress",      "#97C459"),
                (nc5, "Worst Score",   f"{max_stress:.0f}",     "highest stress",     "#F09595"),
            ]:
                col.markdown(num_card(lbl, val, sub, clr), unsafe_allow_html=True)

            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

            # Chart
            marker_colors = [level_color_map.get(l, '#AFA9EC')
                             for l in hdf.get('stress_level', ['Low'] * len(hdf))]
            fig_stress = go.Figure()
            fig_stress.add_trace(go.Scatter(
                x=x_labels, y=hdf['stress_score'],
                mode='lines+markers+text',
                text=hdf['stress_score'].apply(lambda v: f"{v:.0f}"),
                textposition='top center',
                textfont=dict(size=9, color='#ccc'),
                line=dict(color='#AFA9EC', width=2.5, shape='spline'),
                marker=dict(color=marker_colors, size=9, line=dict(width=1.5, color='rgba(0,0,0,0.3)')),
                fill='tozeroy', fillcolor='rgba(83,74,183,0.10)',
                hovertemplate='<b>%{x}</b><br>Score: %{y}<extra></extra>'
            ))
            for threshold, color, label in [(30,'#639922','Low'),(55,'#BA7517','High'),(75,'#A32D2D','Critical')]:
                fig_stress.add_hline(y=threshold, line_dash='dot', line_color=color, opacity=0.45,
                                     annotation_text=label, annotation_position='right',
                                     annotation_font_color=color, annotation_font_size=10)
            fig_stress.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(range=[0,115], gridcolor='rgba(255,255,255,0.06)', tickfont=dict(color='#777',size=10)),
                xaxis=dict(gridcolor='rgba(255,255,255,0.04)', tickfont=dict(color='#777',size=10)),
                margin=dict(t=20, b=10, l=10, r=60), height=270, showlegend=False
            )
            st.plotly_chart(fig_stress, use_container_width=True)
            st.divider()

            # ════════════════════════════════════════════════════
            # SECTION 2 — Sleep & Study Hours
            # ════════════════════════════════════════════════════
            st.markdown('<div class="section-header">🛏 Sleep & Study Hours</div>',
                        unsafe_allow_html=True)

            avg_sleep   = hdf['sleep'].mean()
            min_sleep   = hdf['sleep'].min()
            avg_study   = hdf['study'].mean()
            max_study   = hdf['study'].max()
            nights_ok   = int((hdf['sleep'] >= 7).sum())
            days_ok_st  = int((hdf['study'] <= 8).sum())

            sc1,sc2,sc3,sc4,sc5,sc6 = st.columns(6)
            for col, lbl, val, sub, clr in [
                (sc1, "Avg Sleep",       f"{avg_sleep:.1f}h",   "per night",          "#AFA9EC"),
                (sc2, "Lowest Sleep",    f"{min_sleep:.1f}h",   "worst night",        "#F09595"),
                (sc3, "Nights ≥7h",      f"{nights_ok}",        f"of {len(hdf)} logged","#97C459"),
                (sc4, "Avg Study",       f"{avg_study:.1f}h",   "per day",            "#AFA9EC"),
                (sc5, "Max Study",       f"{max_study:.1f}h",   "heaviest day",       "#FAC775"),
                (sc6, "Days ≤8h Study",  f"{days_ok_st}",       f"of {len(hdf)} logged","#97C459"),
            ]:
                col.markdown(num_card(lbl, val, sub, clr), unsafe_allow_html=True)

            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

            fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
            fig_dual.add_trace(go.Bar(
                x=x_labels, y=hdf['sleep'], name='Sleep (hrs)',
                marker_color='rgba(83,74,183,0.6)', marker_line_width=0,
                text=hdf['sleep'].apply(lambda v: f"{v:.1f}"), textposition='outside',
                textfont=dict(size=9, color='#AFA9EC'),
                hovertemplate='Sleep: %{y}h<extra></extra>'
            ), secondary_y=False)
            fig_dual.add_trace(go.Scatter(
                x=x_labels, y=hdf['study'], mode='lines+markers+text', name='Study (hrs)',
                text=hdf['study'].apply(lambda v: f"{v:.1f}"), textposition='top center',
                textfont=dict(size=9, color='#D4537E'),
                line=dict(color='#D4537E', width=2.5, shape='spline'),
                marker=dict(size=7, color='#D4537E'),
                hovertemplate='Study: %{y}h<extra></extra>'
            ), secondary_y=True)
            fig_dual.add_hline(y=7, line_dash='dot', line_color='#AFA9EC', opacity=0.35,
                               annotation_text='7h target', annotation_font_color='#AFA9EC',
                               annotation_font_size=9)
            fig_dual.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(font=dict(color='#aaa',size=10), bgcolor='rgba(0,0,0,0)',
                            orientation='h', y=1.1),
                margin=dict(t=30, b=10, l=10, r=50), height=270,
                yaxis=dict(gridcolor='rgba(255,255,255,0.06)', tickfont=dict(color='#777',size=10)),
                yaxis2=dict(tickfont=dict(color='#D4537E',size=10))
            )
            st.plotly_chart(fig_dual, use_container_width=True)
            st.divider()

            # ════════════════════════════════════════════════════
            # SECTION 3 — Lifestyle Trends
            # ════════════════════════════════════════════════════
            if all(c in hdf.columns for c in ['screen','anxiety','exercise']):
                st.markdown('<div class="section-header">📉 Lifestyle Trends</div>',
                            unsafe_allow_html=True)

                avg_screen  = hdf['screen'].mean()
                avg_anxiety = hdf['anxiety'].mean()
                ex_days     = int((hdf['exercise'].clip(0,1) > 0).sum())
                hi_anxiety  = int((hdf['anxiety'] >= 7).sum())
                hi_screen   = int((hdf['screen'] > 4).sum())

                lc1,lc2,lc3,lc4,lc5 = st.columns(5)
                for col, lbl, val, sub, clr in [
                    (lc1, "Avg Screen Time",  f"{avg_screen:.1f}h",  "per day",           "#FAC775" if avg_screen>4 else "#97C459"),
                    (lc2, "Days Screen >4h",  f"{hi_screen}",        f"of {len(hdf)} days","#FAC775"),
                    (lc3, "Avg Anxiety",      f"{avg_anxiety:.1f}",  "out of 10",          "#F09595" if avg_anxiety>=6 else "#97C459"),
                    (lc4, "Days Anxiety ≥7",  f"{hi_anxiety}",       "high-anxiety days",  "#F09595"),
                    (lc5, "Exercise Days",    f"{ex_days}",          f"of {len(hdf)} logged","#97C459" if ex_days/max(1,len(hdf))>=0.5 else "#F09595"),
                ]:
                    col.markdown(num_card(lbl, val, sub, clr), unsafe_allow_html=True)

                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

                fig_multi = go.Figure()
                for col_, color_, label_ in [
                    ('screen',  '#FAC775','Screen (hrs)'),
                    ('anxiety', '#F09595','Anxiety (/10)'),
                    ('exercise','#97C459','Exercise (0/1)'),
                ]:
                    if col_ in hdf.columns:
                        fig_multi.add_trace(go.Scatter(
                            x=x_labels, y=hdf[col_], mode='lines+markers+text',
                            text=hdf[col_].apply(lambda v: f"{v:.0f}"),
                            textposition='top center', textfont=dict(size=8, color=color_),
                            name=label_,
                            line=dict(color=color_, width=2, shape='spline'),
                            marker=dict(size=6, color=color_),
                            hovertemplate=f'{label_}: %{{y}}<extra></extra>'
                        ))
                fig_multi.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    legend=dict(font=dict(color='#aaa',size=10), bgcolor='rgba(0,0,0,0)',
                                orientation='h', y=1.1),
                    margin=dict(t=30, b=10, l=10, r=10), height=270,
                    yaxis=dict(gridcolor='rgba(255,255,255,0.06)', tickfont=dict(color='#777',size=10)),
                    xaxis=dict(gridcolor='rgba(255,255,255,0.04)', tickfont=dict(color='#777',size=10))
                )
                st.plotly_chart(fig_multi, use_container_width=True)
                st.divider()

            # ════════════════════════════════════════════════════
            # SECTION 4 — Distribution & Spread
            # ════════════════════════════════════════════════════
            if 'stress_level' in hdf.columns:
                st.markdown('<div class="section-header">🍩 Stress Level Distribution & Spread</div>',
                            unsafe_allow_html=True)

                level_counts = hdf['stress_level'].value_counts()

                # Number cards — one per stress level
                dcols = st.columns(4)
                for i, lvl in enumerate(['Low','Moderate','High','Critical']):
                    cnt  = int(level_counts.get(lvl, 0))
                    pct  = cnt / len(hdf) * 100
                    clr  = COLORS.get(lvl, '#888')
                    lvl_data = hdf[hdf['stress_level']==lvl]['stress_score'].dropna()
                    avg_lvl  = f"{lvl_data.mean():.0f}" if not lvl_data.empty else "—"
                    dcols[i].markdown(
                        num_card(f"{lvl} Sessions", f"{cnt}", f"{pct:.0f}% · avg score {avg_lvl}", clr, clr),
                        unsafe_allow_html=True)

                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

                d_pie, d_box = st.columns(2)
                with d_pie:
                    fig_donut = go.Figure(go.Pie(
                        labels=level_counts.index, values=level_counts.values, hole=0.52,
                        marker_colors=[COLORS.get(l,'#888') for l in level_counts.index],
                        textinfo='percent+value', textfont=dict(size=11),
                        hovertemplate='%{label}: %{value} sessions (%{percent})<extra></extra>'
                    ))
                    fig_donut.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        legend=dict(font=dict(color='#ccc',size=10), bgcolor='rgba(0,0,0,0)'),
                        margin=dict(t=10, b=10, l=10, r=10), height=280,
                        annotations=[dict(text=f"{len(hdf)}<br>sessions",
                                          font=dict(size=13, color='#AFA9EC'), showarrow=False)]
                    )
                    st.plotly_chart(fig_donut, use_container_width=True)

                with d_box:
                    fig_box = go.Figure()
                    BOX_FILL = {'Low':'rgba(99,153,34,0.25)','Moderate':'rgba(186,117,23,0.25)',
                                'High':'rgba(153,60,29,0.25)','Critical':'rgba(163,45,45,0.25)'}
                    for lvl in ['Low','Moderate','High','Critical']:
                        lvl_data = hdf[hdf['stress_level']==lvl]['stress_score'].dropna()
                        if not lvl_data.empty:
                            fig_box.add_trace(go.Box(
                                y=lvl_data, name=lvl,
                                marker_color=COLORS.get(lvl,'#888'),
                                line_color=COLORS.get(lvl,'#888'),
                                fillcolor=BOX_FILL.get(lvl,'rgba(128,128,128,0.25)'),
                                boxmean=True,
                                boxpoints='all', jitter=0.4, pointpos=-1.6,
                                hovertemplate='Score: %{y}<extra></extra>'
                            ))
                    fig_box.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        yaxis=dict(range=[0,105], gridcolor='rgba(255,255,255,0.06)',
                                   tickfont=dict(color='#777',size=10)),
                        xaxis=dict(tickfont=dict(color='#aaa',size=10)),
                        showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=280
                    )
                    st.plotly_chart(fig_box, use_container_width=True)

            # ── Raw data expander ──────────────────────────────
            with st.expander("📋 View raw data"):
                st.dataframe(hdf.drop(columns=['id','user_id'], errors='ignore'),
                             use_container_width=True)
                csv = hdf.to_csv(index=False).encode()
                st.download_button("⬇️ Download CSV", csv,
                                   f"{username}_stress_history.csv", "text/csv")



    # PAGE: goals
    if current_page == "goals":

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



    # PAGE: planner
    if current_page == "planner":

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

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE: SUMMARY  (weekly + monthly breakdown)
    # ═══════════════════════════════════════════════════════════════════════════
    if current_page == "summary":
        st.markdown('<div class="main-title">📋 Wellness Summary</div>', unsafe_allow_html=True)
        st.caption("Weekly and monthly breakdown of your stress, sleep, study, and lifestyle habits.")
        st.divider()

        if history_df.empty:
            st.info("No data yet — log entries on the ✏️ New Entry page to see your summary.")
        else:
            hdf = history_df.copy()
            hdf["stress_score"] = pd.to_numeric(hdf["stress_score"], errors="coerce")
            hdf["sleep"]        = pd.to_numeric(hdf["sleep"],        errors="coerce")
            hdf["study"]        = pd.to_numeric(hdf["study"],        errors="coerce")
            hdf["screen"]       = pd.to_numeric(hdf["screen"],       errors="coerce")
            hdf["anxiety"]      = pd.to_numeric(hdf["anxiety"],      errors="coerce")
            hdf["exercise"]     = pd.to_numeric(hdf["exercise"],     errors="coerce")
            hdf["entry_dt"]     = pd.to_datetime(hdf["timestamp"], errors="coerce")
            hdf["date"]         = hdf["entry_dt"].dt.date
            hdf["week"]         = hdf["entry_dt"].dt.isocalendar().week.astype(int)
            hdf["month"]        = hdf["entry_dt"].dt.month
            hdf["year"]         = hdf["entry_dt"].dt.year
            hdf["year_month"]   = hdf["entry_dt"].dt.to_period("M").astype(str)

            today      = datetime.date.today()
            week_start = today - datetime.timedelta(days=today.weekday())   # Monday
            week_end   = week_start + datetime.timedelta(days=6)
            month_start= today.replace(day=1)

            # Keep only the LAST entry per date (most recent log for that day)
            hdf_dedup = hdf.sort_values("entry_dt").groupby("date", as_index=False).last()
            week_df  = hdf_dedup[hdf_dedup["date"].apply(lambda d: week_start  <= d <= today)]
            month_df = hdf_dedup[hdf_dedup["date"].apply(lambda d: month_start <= d <= today)]

            LEVEL_COLOR = {"Low":"#639922","Moderate":"#EF9F27","High":"#D85A30","Critical":"#E24B4A"}

            def summary_card(label, value, sub, color="#AFA9EC"):
                return (
                    f'<div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.09);' +
                    f'border-radius:14px;padding:0.9rem 1.1rem;">' +
                    f'<div style="font-size:0.72rem;color:#666;text-transform:uppercase;letter-spacing:0.05em;">{label}</div>' +
                    f'<div style="font-size:1.7rem;font-weight:800;color:{color};line-height:1.1;margin:4px 0 2px;">{value}</div>' +
                    f'<div style="font-size:0.78rem;color:#666;">{sub}</div>' +
                    f'</div>'
                )

            def level_badge(level):
                c = LEVEL_COLOR.get(level, "#888")
                return f'<span style="background:{c}33;color:{c};border-radius:20px;padding:2px 10px;font-size:0.75rem;font-weight:700;">{level}</span>'

            # ── WEEK SUMMARY ─────────────────────────────────────────────
            st.markdown('<div class="section-header">📅 This Week</div>', unsafe_allow_html=True)
            st.caption(f"{week_start.strftime('%d %b')} – {min(week_end, today).strftime('%d %b %Y')} · {len(week_df)} entries")

            if week_df.empty:
                st.info("No entries this week yet.")
            else:
                # 4-col stat cards
                wc1,wc2,wc3,wc4 = st.columns(4)
                avg_stress_w = week_df["stress_score"].mean()
                avg_sleep_w  = week_df["sleep"].mean()
                avg_study_w  = week_df["study"].mean()
                # Deduplicate: one exercise flag per calendar date (take max)
                _wex            = week_df.groupby("date")["exercise"].max()
                days_logged_w   = len(_wex)                           # unique days with any entry
                exercise_days_w = int((_wex.clip(0, 1) > 0).sum())   # days where exercise=1
                dominant_w   = week_df["stress_level"].mode()[0] if "stress_level" in week_df.columns else "—"
                dc_w         = LEVEL_COLOR.get(dominant_w, "#888")

                cards_w = [
                    ("Avg Stress Score", f"{avg_stress_w:.0f}", f"Dominant: {dominant_w}", dc_w),
                    ("Avg Sleep/Night",  f"{avg_sleep_w:.1f}h", "Target: 7–9h", "#AFA9EC"),
                    ("Avg Study/Day",    f"{avg_study_w:.1f}h", "Recommended: ≤8h", "#AFA9EC"),
                    ("Exercise Days",
                     f"{exercise_days_w}/{days_logged_w}",
                     f"{exercise_days_w} of {days_logged_w} logged days",
                     "#97C459" if days_logged_w > 0 and exercise_days_w / days_logged_w >= 0.5
                     else "#F09595"),
                ]
                for col, (label, val, sub, clr) in zip([wc1,wc2,wc3,wc4], cards_w):
                    col.markdown(summary_card(label, val, sub, clr), unsafe_allow_html=True)

                # Day-by-day table
                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
                st.markdown('<div class="section-header">Day-by-Day Breakdown</div>', unsafe_allow_html=True)

                day_cols = ["date","stress_score","stress_level","sleep","study","screen","anxiety","exercise"]
                wshow = week_df[[c for c in day_cols if c in week_df.columns]].copy()
                wshow = wshow.sort_values("date")
                wshow["date"] = wshow["date"].apply(lambda d: d.strftime("%a %d %b"))
                wshow.columns = [c.replace("_"," ").title() for c in wshow.columns]
                st.dataframe(wshow.reset_index(drop=True), use_container_width=True, hide_index=True)

                # Weekly heatmap bar
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                st.markdown('<div class="section-header">Stress Heatmap — This Week</div>', unsafe_allow_html=True)

                # Build 7-day heatmap
                heat_html = '<div style="display:grid;grid-template-columns:repeat(7,1fr);gap:6px;">' 
                for i in range(7):
                    d = week_start + datetime.timedelta(days=i)
                    day_entry = week_df[week_df["date"] == d]
                    if not day_entry.empty:
                        sc  = day_entry["stress_score"].values[0]
                        lv  = day_entry["stress_level"].values[0] if "stress_level" in day_entry.columns else "Low"
                        bg  = LEVEL_COLOR.get(lv, "#888") + "44"
                        brd = LEVEL_COLOR.get(lv, "#888")
                        sc_txt = f'<div style="font-size:1.1rem;font-weight:800;color:{brd};">{sc:.0f}</div>'
                        lv_txt = f'<div style="font-size:0.65rem;color:{brd};font-weight:600;">{lv}</div>'
                    else:
                        bg = "rgba(255,255,255,0.03)"; brd = "rgba(255,255,255,0.08)"
                        sc_txt = '<div style="font-size:0.75rem;color:#444;">—</div>'
                        lv_txt = '<div style="font-size:0.65rem;color:#444;">No entry</div>'

                    is_today = (d == today)
                    ring = "box-shadow:0 0 0 2px #AFA9EC;" if is_today else ""
                    heat_html += (
                        f'<div style="background:{bg};border:1.5px solid {brd};border-radius:12px;' +
                        f'padding:0.7rem 0.4rem;text-align:center;{ring}">' +
                        f'<div style="font-size:0.7rem;color:#888;font-weight:600;">{d.strftime("%a")}</div>' +
                        f'<div style="font-size:0.75rem;color:#666;">{d.strftime("%d")}</div>' +
                        sc_txt + lv_txt +
                        f'</div>'
                    )
                heat_html += '</div>'
                st.markdown(heat_html, unsafe_allow_html=True)

                # Mini sparkline
                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
                wk_sorted = week_df.sort_values("entry_dt")
                if len(wk_sorted) > 1:
                    wfig = go.Figure()
                    wfig.add_trace(go.Scatter(
                        x=wk_sorted["date"].apply(lambda d: d.strftime("%a %d")),
                        y=wk_sorted["stress_score"],
                        mode="lines+markers+text",
                        text=wk_sorted["stress_score"].apply(lambda v: f"{v:.0f}"),
                        textposition="top center",
                        line=dict(color="#AFA9EC", width=2.5, shape="spline"),
                        marker=dict(size=9, color=[LEVEL_COLOR.get(l,"#888") for l in wk_sorted.get("stress_level", ["Low"]*len(wk_sorted))],
                                    line=dict(width=1.5, color="rgba(0,0,0,0.3)")),
                        fill="tozeroy", fillcolor="rgba(83,74,183,0.09)"
                    ))
                    wfig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        yaxis=dict(range=[0,105], gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#777",size=10)),
                        xaxis=dict(gridcolor="rgba(255,255,255,0.04)", tickfont=dict(color="#777",size=10)),
                        margin=dict(t=20,b=10,l=10,r=10), height=200, showlegend=False
                    )
                    st.plotly_chart(wfig, use_container_width=True)

                # Weekly insights
                st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
                insights = []
                if avg_sleep_w < 7:
                    insights.append(("⚠️", f"Average sleep this week is {avg_sleep_w:.1f}h — below the 7h minimum.", "#FAC775"))
                if avg_stress_w >= 55:
                    insights.append(("🔴", f"Weekly average stress is {avg_stress_w:.0f} — in the High range. Consider adjusting workload.", "#F09595"))
                if exercise_days_w == 0:
                    insights.append(("🏃", "No exercise logged this week. Even a 20-min walk helps reduce cortisol.", "#97C459"))
                if avg_study_w > 9:
                    insights.append(("📚", f"Averaging {avg_study_w:.1f}h of study/day — above sustainable levels. Plan recovery blocks.", "#FAC775"))
                if not insights:
                    insights.append(("✅", "Good week overall! All key metrics are within healthy ranges.", "#97C459"))

                for icon, msg, clr in insights:
                    st.markdown(
                        f'<div style="display:flex;align-items:flex-start;gap:10px;padding:0.6rem 0.9rem;' +
                        f'background:{clr}11;border-left:3px solid {clr};border-radius:0 8px 8px 0;margin-bottom:6px;">' +
                        f'<span style="font-size:1rem;">{icon}</span>' +
                        f'<span style="font-size:0.85rem;color:#ccc;">{msg}</span></div>',
                        unsafe_allow_html=True)

            st.divider()

            # ── MONTH SUMMARY ────────────────────────────────────────────
            import calendar as cal_mod
            st.markdown('<div class="section-header">🗓️ This Month</div>', unsafe_allow_html=True)
            st.caption(f"{today.strftime('%B %Y')} · {len(month_df)} entries logged")

            if month_df.empty:
                st.info("No entries this month yet.")
            else:
                import calendar as _cal
                # days_in_month = unique dates logged this month (not calendar days)
                avg_stress_m  = month_df["stress_score"].mean()
                avg_sleep_m   = month_df["sleep"].mean()
                avg_study_m   = month_df["study"].mean()
                # Deduplicate: one exercise flag per calendar date (take max)
                _mex            = month_df.groupby("date")["exercise"].max()
                days_in_month   = len(_mex)                           # unique logged days this month
                exercise_days_m = int((_mex.clip(0, 1) > 0).sum())   # days where exercise=1
                avg_screen_m  = month_df["screen"].mean() if "screen" in month_df.columns else 0
                avg_anxiety_m = month_df["anxiety"].mean() if "anxiety" in month_df.columns else 0
                dominant_m    = month_df["stress_level"].mode()[0] if "stress_level" in month_df.columns else "—"
                dc_m          = LEVEL_COLOR.get(dominant_m, "#888")
                best_day      = month_df.loc[month_df["stress_score"].idxmin()]
                worst_day     = month_df.loc[month_df["stress_score"].idxmax()]

                # 4 stat cards
                mc1,mc2,mc3,mc4 = st.columns(4)
                month_cards = [
                    ("Monthly Avg Stress", f"{avg_stress_m:.0f}", f"Dominant: {dominant_m}", dc_m),
                    ("Avg Sleep/Night",    f"{avg_sleep_m:.1f}h", f"Target: 7–9h", "#AFA9EC"),
                    ("Exercise Days",
                     f"{exercise_days_m}/{days_in_month}",
                     f"{exercise_days_m} of {days_in_month} logged days",
                     "#97C459" if days_in_month > 0 and exercise_days_m / days_in_month >= 0.5
                     else "#FAC775"),
                    ("Avg Screen Time",    f"{avg_screen_m:.1f}h",f"Target: ≤4h/day", "#AFA9EC" if avg_screen_m <= 4 else "#F09595"),
                ]
                for col, (label, val, sub, clr) in zip([mc1,mc2,mc3,mc4], month_cards):
                    col.markdown(summary_card(label, val, sub, clr), unsafe_allow_html=True)

                st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

                # Best / Worst day callouts
                bd_col, wd_col = st.columns(2)
                best_date_str  = pd.to_datetime(best_day["timestamp"]).strftime("%A, %d %b")
                worst_date_str = pd.to_datetime(worst_day["timestamp"]).strftime("%A, %d %b")
                with bd_col:
                    st.markdown(
                        f'<div style="background:rgba(99,153,34,0.12);border:1px solid #63992244;' +
                        f'border-left:4px solid #639922;border-radius:0 12px 12px 0;padding:0.8rem 1rem;">' +
                        f'<div style="font-size:0.72rem;color:#888;text-transform:uppercase;letter-spacing:0.05em;">🌿 Best Day This Month</div>' +
                        f'<div style="font-size:1.1rem;font-weight:700;color:#97C459;">{best_date_str}</div>' +
                        f'<div style="font-size:0.85rem;color:#aaa;">Stress score: {best_day["stress_score"]:.0f} · {best_day.get("stress_level","Low")}</div>' +
                        f'</div>',
                        unsafe_allow_html=True)
                with wd_col:
                    worst_color = LEVEL_COLOR.get(str(worst_day.get("stress_level","High")), "#E24B4A")
                    st.markdown(
                        f'<div style="background:{worst_color}18;border:1px solid {worst_color}44;' +
                        f'border-left:4px solid {worst_color};border-radius:0 12px 12px 0;padding:0.8rem 1rem;">' +
                        f'<div style="font-size:0.72rem;color:#888;text-transform:uppercase;letter-spacing:0.05em;">🔴 Hardest Day This Month</div>' +
                        f'<div style="font-size:1.1rem;font-weight:700;color:{worst_color};">{worst_date_str}</div>' +
                        f'<div style="font-size:0.85rem;color:#aaa;">Stress score: {worst_day["stress_score"]:.0f} · {worst_day.get("stress_level","High")}</div>' +
                        f'</div>',
                        unsafe_allow_html=True)

                st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

                # Monthly trend chart — two lines: stress + sleep
                m_sorted = month_df.sort_values("entry_dt")
                if len(m_sorted) > 1:
                    st.markdown('<div class="section-header">Monthly Trend</div>', unsafe_allow_html=True)
                    mfig = make_subplots(specs=[[{"secondary_y": True}]])
                    mfig.add_trace(go.Scatter(
                        x=m_sorted["date"].apply(lambda d: d.strftime("%d %b")),
                        y=m_sorted["stress_score"],
                        mode="lines+markers", name="Stress Score",
                        line=dict(color="#AFA9EC", width=2.5, shape="spline"),
                        marker=dict(size=7, color=[LEVEL_COLOR.get(l,"#888") for l in m_sorted.get("stress_level",["Low"]*len(m_sorted))]),
                        fill="tozeroy", fillcolor="rgba(83,74,183,0.08)"
                    ), secondary_y=False)
                    mfig.add_trace(go.Scatter(
                        x=m_sorted["date"].apply(lambda d: d.strftime("%d %b")),
                        y=m_sorted["sleep"],
                        mode="lines+markers", name="Sleep (hrs)",
                        line=dict(color="#97C459", width=2, shape="spline", dash="dot"),
                        marker=dict(size=6, color="#97C459")
                    ), secondary_y=True)
                    mfig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        legend=dict(font=dict(color="#aaa", size=10), bgcolor="rgba(0,0,0,0)",
                                    orientation="h", y=1.1),
                        margin=dict(t=30,b=10,l=10,r=50), height=260,
                        yaxis=dict(range=[0,105], gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#777",size=10)),
                        yaxis2=dict(title="Sleep hrs", tickfont=dict(color="#97C459",size=10))
                    )
                    st.plotly_chart(mfig, use_container_width=True)

                # Stress level distribution this month
                if "stress_level" in month_df.columns:
                    dist_col, metric_col = st.columns([1, 1.4])
                    with dist_col:
                        st.markdown('<div class="section-header">Level Distribution</div>', unsafe_allow_html=True)
                        lc_m = month_df["stress_level"].value_counts()
                        fig_dm = go.Figure(go.Pie(
                            labels=lc_m.index, values=lc_m.values, hole=0.55,
                            marker_colors=[LEVEL_COLOR.get(l,"#888") for l in lc_m.index],
                            textfont=dict(size=11),
                            hovertemplate='%{label}: %{value} days (%{percent})<extra></extra>'
                        ))
                        fig_dm.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            legend=dict(font=dict(color="#aaa",size=9), bgcolor="rgba(0,0,0,0)",
                                        orientation="h", y=-0.2),
                            margin=dict(t=10,b=35,l=10,r=10), height=240,
                            annotations=[dict(text=f"{len(month_df)}<br>entries",
                                              font=dict(size=11,color="#AFA9EC"), showarrow=False)]
                        )
                        st.plotly_chart(fig_dm, use_container_width=True)

                    with metric_col:
                        st.markdown('<div class="section-header">Monthly Averages</div>', unsafe_allow_html=True)
                        metrics_data = [
                            ("😴 Sleep",         f"{avg_sleep_m:.1f}h/night", avg_sleep_m >= 7,   "Target ≥7h"),
                            ("📚 Study",          f"{avg_study_m:.1f}h/day",  avg_study_m <= 8,   "Target ≤8h"),
                            ("📱 Screen",         f"{avg_screen_m:.1f}h/day", avg_screen_m <= 4,  "Target ≤4h"),
                            ("🧘 Anxiety",        f"{avg_anxiety_m:.1f}/10",  avg_anxiety_m <= 5, "Target ≤5"),
                            ("🏃 Exercise days",
                             f"{exercise_days_m}/{days_in_month}",
                             days_in_month > 0 and exercise_days_m / days_in_month >= 0.5,
                             f"Target ≥50% of logged days"),
                            ("💯 Stress avg",     f"{avg_stress_m:.0f}/100",  avg_stress_m < 30,  "Target <30"),
                        ]
                        rows_html = ""
                        for icon_label, value, good, target in metrics_data:
                            clr   = "#97C459" if good else "#F09595"
                            tick  = "✅" if good else "❌"
                            rows_html += (
                                f'<div style="display:flex;align-items:center;justify-content:space-between;' +
                                f'padding:7px 10px;border-radius:9px;background:rgba(255,255,255,0.03);' +
                                f'border:1px solid rgba(255,255,255,0.07);margin-bottom:5px;">' +
                                f'<div style="font-size:0.85rem;">{icon_label}</div>' +
                                f'<div style="font-size:0.85rem;font-weight:700;color:{clr};">{value}</div>' +
                                f'<div style="font-size:0.75rem;color:#555;">{target}</div>' +
                                f'<div style="font-size:0.9rem;">{tick}</div>' +
                                f'</div>'
                            )
                        st.markdown(rows_html, unsafe_allow_html=True)

                # Month insights
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                m_insights = []
                low_days  = len(month_df[month_df["stress_level"]=="Low"])   if "stress_level" in month_df.columns else 0
                crit_days = len(month_df[month_df["stress_level"]=="Critical"]) if "stress_level" in month_df.columns else 0
                if crit_days > 0:
                    m_insights.append(("🚨", f"{crit_days} Critical stress day(s) this month. Review what happened on those days.", "#E24B4A"))
                if low_days >= len(month_df) // 2:
                    m_insights.append(("🌿", f"{low_days} Low-stress days this month — great resilience!", "#97C459"))
                if avg_sleep_m < 6.5:
                    m_insights.append(("💤", f"Monthly average sleep is only {avg_sleep_m:.1f}h — chronic sleep deficit detected.", "#FAC775"))
                ex_rate_m = exercise_days_m / days_in_month if days_in_month > 0 else 0
                if ex_rate_m < 0.5 and days_in_month >= 3:
                    m_insights.append(("🏃", f"Exercised on {exercise_days_m} of {days_in_month} logged days ({ex_rate_m*100:.0f}%). Try to hit at least 50% of your logged days.", "#FAC775"))
                if not m_insights:
                    m_insights.append(("🎉", "Excellent month! Your wellness indicators are consistently healthy.", "#97C459"))

                for icon, msg, clr in m_insights:
                    st.markdown(
                        f'<div style="display:flex;align-items:flex-start;gap:10px;padding:0.6rem 0.9rem;' +
                        f'background:{clr}11;border-left:3px solid {clr};border-radius:0 8px 8px 0;margin-bottom:6px;">' +
                        f'<span style="font-size:1rem;">{icon}</span>' +
                        f'<span style="font-size:0.85rem;color:#ccc;">{msg}</span></div>',
                        unsafe_allow_html=True)

        return  # summary page ends





# ═════════════════════════════════════════════════════════════════════════════
# ROUTER
# ═════════════════════════════════════════════════════════════════════════════

# Show landing page on very first visit
if "show_landing" not in st.session_state:
    st.session_state["show_landing"] = True

if st.session_state.get("show_landing", True):
    show_landing_page()
elif "user" not in st.session_state or st.session_state["user"] is None:
    show_auth_page()
else:
    show_main_app(st.session_state["user"])