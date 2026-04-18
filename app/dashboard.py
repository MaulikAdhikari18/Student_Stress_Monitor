import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os
import pickle
import sqlite3
import hashlib
import matplotlib.pyplot as plt

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
    # TAB 3 — Management Tips
    # ══════════════════════════════════════════════════════════
    with tab3:
        st.markdown("#### Personalized recommendations")
        tips=[]
        if sleep<7:       tips.append(("😴 Sleep hygiene",    f"You're getting {sleep}h — below the 7–9h ideal. Set a consistent bedtime, cut caffeine after 3 PM, go screen-free 30 min before bed."))
        if study>8:       tips.append(("📚 Study smarter",    f"Studying {study}h/day risks burnout. Try Pomodoro (25 min focus, 5 min break). Schedule a hard stop time each day."))
        if exercise<3:    tips.append(("🏃 Get moving",       "Less than 3 exercise days/week amplifies stress. Even a 20-min walk daily reduces cortisol by ~26%."))
        if anxiety>6:     tips.append(("🧘 Manage anxiety",   f"Anxiety at {anxiety}/10 is significant. Try box breathing: inhale 4s → hold 4s → exhale 4s → hold 4s."))
        if social<4:      tips.append(("👥 Social connection","Low social interaction increases stress. Even one meaningful conversation daily helps."))
        if screen>6:      tips.append(("📱 Digital detox",    f"{screen}h/day screens disrupts sleep. Try app timers and a 'no-phone hour' before bed."))
        if finance>7:     tips.append(("💰 Financial stress", "Talk to your institution's welfare office — scholarships, emergency funds or part-time work may be available."))
        if assignments>8: tips.append(("📝 Task overload",    f"{assignments} pending items is stressful. Use the Eisenhower matrix and tackle one high-priority item each morning."))
        if family<4:      tips.append(("❤️ Build support",   "Low family support increases vulnerability. Campus counselors and peer mentors can help."))
        if peer>7:        tips.append(("🤝 Peer pressure",    "High peer pressure drains energy. Practice assertive communication and spend time with people who motivate you."))
        if not tips:      tips.append(("🌟 You're doing great!","Your indicators look balanced. Keep your routines and do weekly check-ins to catch early changes."))

        for cat,tip in tips:
            st.markdown(f'<div class="tip-box"><strong>{cat}</strong><br>{tip}</div>',
                        unsafe_allow_html=True)

        st.divider()
        st.markdown("#### 🗓️ 7-day recovery plan")
        for day,action in [
            ("Day 1","Set a consistent sleep time and stick to it all week."),
            ("Day 2","Write all pending tasks down. Cross off one small thing today."),
            ("Day 3","Go for a 20-min walk — no phone, no earphones."),
            ("Day 4","Call or message one friend or family member you trust."),
            ("Day 5","Do one 25-min Pomodoro study block. Note your focus level."),
            ("Day 6","Spend 10 min on box breathing or a simple meditation."),
            ("Day 7","Review the week: what helped? Plan to repeat those habits."),
        ]:
            st.checkbox(f"**{day}** — {action}", key=f"plan_{day}")

    # ══════════════════════════════════════════════════════════
    # TAB 4 — History
    # ══════════════════════════════════════════════════════════
    with tab4:
        st.markdown(f"#### Your stress history — {username}")
        if history_df.empty:
            st.info("No history yet. Fill in today's data and hit 💾 Save.")
        else:
            hdf = history_df.copy()
            hdf['stress_score'] = pd.to_numeric(hdf['stress_score'],errors='coerce')
            hdf['sleep']        = pd.to_numeric(hdf['sleep'],       errors='coerce')

            s1,s2,s3,s4 = st.columns(4)
            s1.metric("Sessions logged",  str(len(hdf)))
            s2.metric("Avg stress score", f"{hdf['stress_score'].mean():.0f}")
            s3.metric("Avg sleep",        f"{hdf['sleep'].mean():.1f}h")
            s4.metric("Last level",       str(hdf['stress_level'].iloc[-1]) if 'stress_level' in hdf.columns else "—")

            st.subheader("📈 Stress Score Over Time")
            st.line_chart(hdf.set_index('day_label')[['stress_score']].rename(
                columns={'stress_score':'Stress'}))

            st.subheader("📊 Sleep Hours Per Session")
            st.bar_chart(hdf.set_index('day_label')[['sleep']].rename(columns={'sleep':'Sleep'}))

            if 'study' in hdf.columns and 'screen' in hdf.columns:
                st.subheader("📉 Study & Screen Time Trends")
                trend_cols=[c for c in ['study','screen','anxiety','exercise'] if c in hdf.columns]
                st.line_chart(hdf.set_index('day_label')[trend_cols])

            if 'stress_level' in hdf.columns:
                st.subheader("🥧 Stress Level Distribution")
                level_counts = hdf['stress_level'].value_counts()
                fig2,ax2 = plt.subplots(figsize=(5,4))
                clrs2=[COLORS.get(l,'#888') for l in level_counts.index]
                ax2.pie(level_counts.values,labels=level_counts.index,colors=clrs2,
                        autopct='%1.0f%%',startangle=140,
                        wedgeprops={'linewidth':1,'edgecolor':'white'})
                ax2.set_title('Sessions by stress level',fontsize=12)
                plt.tight_layout(); st.pyplot(fig2,use_container_width=True); plt.close()

            with st.expander("📋 View raw data"):
                st.dataframe(hdf.drop(columns=['id','user_id'],errors='ignore'),
                             use_container_width=True)
                csv = hdf.to_csv(index=False).encode()
                st.download_button("⬇️ Download CSV",csv,
                                   f"{username}_stress_history.csv","text/csv")

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

        # ── Add task form ──────────────────────────────────────
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