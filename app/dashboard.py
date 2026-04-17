import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ─── Page Config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Student Stress Monitor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-title {
        font-size: 2rem; font-weight: 700;
        background: linear-gradient(135deg, #534AB7, #D4537E);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .stress-box {
        padding: 1rem 1.4rem; border-radius: 12px;
        border-left: 6px solid; margin-bottom: 1rem;
    }
    .box-low      { background:#EAF3DE; border-color:#639922; }
    .box-moderate { background:#FAEEDA; border-color:#BA7517; }
    .box-high     { background:#FAECE7; border-color:#993C1D; }
    .box-critical { background:#FCEBEB; border-color:#A32D2D; }
    .tip-box {
        background:#f4f2ff; border-left:4px solid #534AB7;
        border-radius:0 8px 8px 0; padding:0.7rem 1rem;
        margin-bottom:0.5rem; font-size:0.92rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Constants ───────────────────────────────────────────────────────────────

LABELS  = ['Low', 'Moderate', 'High', 'Critical']
COLORS  = {'Low':'#639922','Moderate':'#BA7517','High':'#993C1D','Critical':'#A32D2D'}
EMOJIS  = {'Low':'😊','Moderate':'😐','High':'😟','Critical':'😰'}
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

FEATURES = [
    'study_hours', 'assignments_pending', 'exam_pressure',
    'academic_performance', 'sleep_hours', 'exercise_days_per_week',
    'social_interactions_per_week', 'screen_time_hours',
    'anxiety_level', 'financial_stress', 'family_support',
    'peer_pressure', 'extracurricular_activities', 'relationship_issues',
]

# ─── Load ML Model ───────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    mp = os.path.join(MODELS_DIR, 'model.pkl')
    sp = os.path.join(MODELS_DIR, 'scaler.pkl')
    tp = os.path.join(MODELS_DIR, 'meta.pkl')
    if not os.path.exists(mp):
        return None, None, None
    with open(mp, 'rb') as f: model  = pickle.load(f)
    with open(sp, 'rb') as f: scaler = pickle.load(f)
    with open(tp, 'rb') as f: meta   = pickle.load(f)
    return model, scaler, meta

model, scaler, meta = load_model()
MODEL_READY = model is not None

# ─── Load / Init history CSV (your original data store) ──────────────────────

file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'stress_data.csv')
os.makedirs(os.path.dirname(file_path), exist_ok=True)

HISTORY_COLS = ["Timestamp", "Day", "StressScore", "StressLevel",
                "Sleep", "Study", "Screen", "Anxiety", "Exercise"]
if not os.path.exists(file_path):
    pd.DataFrame(columns=HISTORY_COLS).to_csv(file_path, index=False)

history_df = pd.read_csv(file_path)
for col in HISTORY_COLS:
    if col not in history_df.columns:
        history_df[col] = np.nan

# ─── Header ──────────────────────────────────────────────────────────────────

st.markdown('<div class="main-title">🧠 Student Stress Monitor Dashboard</div>', unsafe_allow_html=True)
st.caption("AI-powered stress prediction • personalized tips • trend tracking")

if MODEL_READY:
    acc = meta.get('accuracy', 0)
    c1, c2, c3 = st.columns(3)
    c1.metric("ML Model", meta.get('best_model', 'Loaded'))
    c2.metric("Model Accuracy", f"{acc*100:.1f}%")
    c3.metric("Sessions logged", str(len(history_df)))
else:
    st.warning("⚠️ ML model not found. Run `python src/train_model.py` to enable AI predictions. Basic scoring is active.")

st.divider()

# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("📥 Enter Today's Data")

    st.markdown("#### 📚 Academic")
    study        = st.slider("Study hours / day",       0.0, 16.0, 6.0, 0.5)
    assignments  = st.slider("Assignments pending",     0,   15,   3)
    exam         = st.slider("Exam pressure (1–10)",    1,   10,   5)
    performance  = st.slider("Academic performance (1–10)", 1, 10, 7)

    st.markdown("#### 🏃 Lifestyle")
    sleep        = st.slider("Sleep hours / night",     2.0, 12.0, 7.0, 0.5)
    exercise     = st.slider("Exercise days / week",    0,   7,    3)
    social       = st.slider("Social interactions / week", 0, 20,  5)
    screen       = st.slider("Screen time hours / day", 0.0, 16.0, 4.0, 0.5)

    st.markdown("#### 🧠 Mental & Social")
    anxiety      = st.slider("Anxiety level (1–10)",    1,   10,   4)
    finance      = st.slider("Financial stress (1–10)", 1,   10,   3)
    family       = st.slider("Family support (1–10)",   1,   10,   7)
    peer         = st.slider("Peer pressure (1–10)",    1,   10,   4)
    extra        = st.selectbox("Extracurricular activities", [0,1,2],
                                format_func=lambda x: ['None','1–2 activities','3+ activities'][x])
    rel          = st.selectbox("Relationship situation", [0,1,2],
                                format_func=lambda x: ['Single / N/A','Stable relationship','Relationship issues'][x])

    st.divider()
    save_btn = st.button("💾 Save Today's Entry", use_container_width=True, type="primary")

# ─── Stress Computation ───────────────────────────────────────────────────────

# Always compute a rule-based score (your original logic, extended)
raw = (
    max(0, study - 8) * 3.5
    + assignments * 2.5
    + (exam - 1) * 5.0
    + max(0, 7 - sleep) * 4.0
    + max(0, 5 - exercise) * 2.0
    + max(0, 8 - social) * 1.5
    + max(0, screen - 4) * 2.0
    + (anxiety - 1) * 4.5
    + (finance - 1) * 3.0
    - (family - 1) * 2.5
    - (performance - 1) * 2.0
    + (peer - 1) * 2.5
    + (5 if extra == 0 else 0)
    + (8 if rel == 2 else 0)
)
stress_score = int(np.clip(raw, 0, 100))

# ML prediction (if model loaded)
if MODEL_READY:
    inp = np.array([[study, assignments, exam, performance,
                     sleep, exercise, social, screen,
                     anxiety, finance, family, peer, extra, rel]])
    inp_sc     = scaler.transform(inp)
    pred_class = int(model.predict(inp_sc)[0])
    pred_proba = model.predict_proba(inp_sc)[0]
    level_name = LABELS[pred_class]
else:
    # Fallback to your original 3-tier rule
    pred_proba = None
    if stress_score > 74:   level_name = 'Critical'
    elif stress_score > 54: level_name = 'High'
    elif stress_score > 29: level_name = 'Moderate'
    else:                   level_name = 'Low'

level_color = COLORS[level_name]
level_emoji = EMOJIS[level_name]

# ─── Save entry ──────────────────────────────────────────────────────────────

if save_btn:
    new_row = {
        "Timestamp":   datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Day":         datetime.datetime.now().strftime("%a %d %b"),
        "StressScore": stress_score,
        "StressLevel": level_name,
        "Sleep":       sleep,
        "Study":       study,
        "Screen":      screen,
        "Anxiety":     anxiety,
        "Exercise":    exercise,
    }
    history_df = pd.concat([history_df, pd.DataFrame([new_row])], ignore_index=True)
    history_df.to_csv(file_path, index=False)
    st.success("✅ Data Saved Successfully!")
    st.rerun()

# ─── Main Tabs ────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Stress Result", "🔍 Factor Analysis", "💡 Management Tips", "📈 My History"])

# ══════════════════════════════════════════════════════════════
# TAB 1 — Stress Result
# ══════════════════════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([1.4, 1])

    with col_left:
        box_css = level_name.lower()
        st.markdown(f"""
        <div class="stress-box box-{box_css}">
            <h2 style="margin:0;color:{level_color};">{level_emoji} {level_name} Stress</h2>
            <p style="margin:0.4rem 0 0;font-size:1.05rem;">
                Stress score: <strong>{stress_score} / 100</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

        if MODEL_READY and pred_proba is not None:
            st.markdown("#### Confidence across levels")
            for lbl, p in zip(LABELS, pred_proba):
                ca, cb = st.columns([3, 1])
                ca.progress(float(p), text=lbl)
                cb.write(f"**{p*100:.1f}%**")
        else:
            st.markdown("#### Score breakdown")
            st.write(f"- Sleep deficit: `{max(0, round(7-sleep,1))}h below target`")
            st.write(f"- Study overload: `{max(0, round(study-8,1))}h above 8h`")
            st.write(f"- Screen excess: `{max(0, round(screen-4,1))}h above 4h`")

    with col_right:
        if MODEL_READY and pred_proba is not None:
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.pie(pred_proba,
                   labels=LABELS,
                   colors=['#639922','#EF9F27','#D85A30','#E24B4A'],
                   autopct='%1.1f%%', startangle=140,
                   wedgeprops={'linewidth':1,'edgecolor':'white'})
            ax.set_title('Probability Distribution', fontsize=11, pad=8)
            st.pyplot(fig, use_container_width=True)
            plt.close()
        else:
            # Simple gauge using bar
            fig, ax = plt.subplots(figsize=(4, 1.5))
            ax.barh(['Stress'], [stress_score], color=level_color, height=0.5)
            ax.barh(['Stress'], [100 - stress_score], left=[stress_score],
                    color='#e8e8e8', height=0.5)
            ax.set_xlim(0, 100)
            ax.set_xlabel('Score / 100')
            ax.set_title(f'Score: {stress_score}', fontsize=12)
            ax.axis('off')
            st.pyplot(fig, use_container_width=True)
            plt.close()

    st.divider()
    st.markdown("#### Quick health snapshot")
    m1, m2, m3, m4 = st.columns(4)
    sleep_status = "Optimal ✅" if sleep >= 8 else f"-{8-sleep:.1f}h ⚠️"
    study_load   = ["Light","Moderate","Heavy","Extreme"][
                    min(3, int(study // 4))] if study <= 16 else "Extreme"
    recovery     = int(((exercise/7)*0.4 + (sleep/10)*0.4 + (social/20)*0.2) * 100)
    burnout      = min(100, int(stress_score*0.6 + max(0,study-8)*4 + max(0,10-sleep)*3))
    m1.metric("Sleep status",    sleep_status)
    m2.metric("Study load",      study_load)
    m3.metric("Recovery score",  f"{recovery}%")
    m4.metric("Burnout risk",    f"{burnout}/100")

# ══════════════════════════════════════════════════════════════
# TAB 2 — Factor Analysis
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### Which factors are driving your stress?")

    factor_scores = {
        "Academic load":    min(100, int(study/16*50 + assignments/15*30 + exam/10*20)),
        "Sleep deficit":    min(100, int(max(0, 8-sleep)/6*100)),
        "Anxiety":          int(anxiety/10*100),
        "Financial strain": int(finance/10*100),
        "Social isolation": min(100, int(max(0,10-family)/9*70 + max(0,8-social)/8*30)),
        "Peer pressure":    int(peer/10*100),
        "Screen overuse":   min(100, int(max(0,screen-4)/12*100)),
        "Exercise deficit": min(100, int(max(0,5-exercise)/5*100)),
    }
    sorted_f = sorted(factor_scores.items(), key=lambda x: x[1], reverse=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    names  = [f[0] for f in sorted_f]
    vals   = [f[1] for f in sorted_f]
    clrs   = ['#A32D2D' if v>=75 else '#993C1D' if v>=55
              else '#BA7517' if v>=30 else '#639922' for v in vals]
    bars = ax.barh(names, vals, color=clrs, edgecolor='white', linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(v+2, bar.get_y()+bar.get_height()/2, str(v), va='center', fontsize=10)
    ax.axvline(55, ls='--', lw=1, color='#BA7517', alpha=0.6, label='High threshold')
    ax.axvline(75, ls='--', lw=1, color='#A32D2D', alpha=0.6, label='Critical threshold')
    ax.set_xlim(0, 115)
    ax.set_xlabel("Stress contribution score")
    ax.set_title("Stress Factor Breakdown", fontsize=13)
    ax.legend(fontsize=9)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("#### Your inputs at a glance")
    snap_cols = st.columns(4)
    snap = [
        ("Study hrs/day", f"{study}h"),
        ("Sleep hrs/night", f"{sleep}h"),
        ("Anxiety", f"{anxiety}/10"),
        ("Exercise days", str(exercise)),
        ("Assignments", str(assignments)),
        ("Financial stress", f"{finance}/10"),
        ("Family support", f"{family}/10"),
        ("Screen time", f"{screen}h"),
    ]
    for i, (k, v) in enumerate(snap):
        snap_cols[i % 4].metric(k, v)

# ══════════════════════════════════════════════════════════════
# TAB 3 — Management Tips
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("#### Personalized recommendations based on your inputs")

    tips = []
    if sleep < 7:
        tips.append(("😴 Sleep hygiene",
            f"You're getting {sleep}h — below the 7–9h ideal. Set a consistent bedtime, "
            "cut caffeine after 3 PM, and go screen-free 30 min before bed."))
    if study > 8:
        tips.append(("📚 Study smarter",
            f"Studying {study}h/day risks burnout. Try the Pomodoro method "
            "(25 min focus, 5 min break). Schedule a hard stop time each day."))
    if exercise < 3:
        tips.append(("🏃 Get moving",
            "Less than 3 days of exercise/week amplifies stress. Even a 20-min "
            "walk daily reduces cortisol by ~26%. Schedule it like a class."))
    if anxiety > 6:
        tips.append(("🧘 Manage anxiety",
            f"Anxiety at {anxiety}/10 is significant. Try box breathing: "
            "inhale 4s → hold 4s → exhale 4s → hold 4s. Repeat 4 times."))
    if social < 4:
        tips.append(("👥 Social connection",
            "Low social interaction increases stress. Even one meaningful conversation "
            "daily helps. Consider a study group or campus club."))
    if screen > 6:
        tips.append(("📱 Digital detox",
            f"{screen}h/day of screens disrupts sleep and focus. Try app timers "
            "and a 'no-phone hour' before bed."))
    if finance > 7:
        tips.append(("💰 Financial pressure",
            "Talk to your institution's student welfare office — scholarships, "
            "emergency funds, or part-time work may be available."))
    if assignments > 8:
        tips.append(("📝 Task overload",
            f"{assignments} pending assignments is stressful. Use the Eisenhower matrix "
            "(urgent × important) and tackle one high-priority item each morning."))
    if family < 4:
        tips.append(("❤️ Build your support net",
            "Low family support increases vulnerability. Campus counselors, "
            "peer mentors, and student wellbeing groups can help fill that gap."))
    if peer > 7:
        tips.append(("🤝 Peer pressure",
            "High peer pressure drains energy. Practice saying no assertively. "
            "Spend time with people who motivate rather than pressure you."))
    if not tips:
        tips.append(("🌟 You're doing great!",
            "Your stress indicators look balanced. Keep your current routines "
            "and do weekly check-ins to catch any early changes."))

    for cat, tip in tips:
        st.markdown(f'<div class="tip-box"><strong>{cat}</strong><br>{tip}</div>',
                    unsafe_allow_html=True)

    st.divider()
    st.markdown("#### 🗓️ 7-day recovery plan")
    plan = [
        ("Day 1", "Set a consistent sleep time and stick to it all week."),
        ("Day 2", "Write all pending tasks down. Cross off one small thing today."),
        ("Day 3", "Go for a 20-min walk — no phone, no earphones."),
        ("Day 4", "Call or message one friend or family member you trust."),
        ("Day 5", "Do one 25-min Pomodoro study block. Note your focus level."),
        ("Day 6", "Spend 10 min on box breathing or a simple meditation."),
        ("Day 7", "Review the week: what helped? Plan to repeat those habits."),
    ]
    for day, action in plan:
        st.checkbox(f"**{day}** — {action}", key=f"plan_{day}")

# ══════════════════════════════════════════════════════════════
# TAB 4 — History (your original charts, enhanced)
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown("#### Your stress history")

    if history_df.empty or history_df['StressScore'].dropna().empty:
        st.info("No history yet. Fill in today's data and hit 💾 Save.")
    else:
        hdf = history_df.dropna(subset=['StressScore']).copy()
        hdf['StressScore'] = pd.to_numeric(hdf['StressScore'], errors='coerce')
        hdf['Sleep']       = pd.to_numeric(hdf['Sleep'],       errors='coerce')

        # Summary metrics
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Sessions logged",  str(len(hdf)))
        s2.metric("Avg stress score", f"{hdf['StressScore'].mean():.0f}")
        s3.metric("Avg sleep",        f"{hdf['Sleep'].mean():.1f}h")
        s4.metric("Last level",       str(hdf['StressLevel'].iloc[-1]) if 'StressLevel' in hdf.columns else "—")

        # Line chart — stress over time (your original, now with enhanced data)
        st.subheader("📈 Stress Score Over Time")
        chart_df = hdf.set_index('Day')[['StressScore']].rename(columns={'StressScore':'Stress'})
        st.line_chart(chart_df)

        # Bar chart — sleep (your original)
        st.subheader("📊 Sleep Hours Per Session")
        st.bar_chart(hdf.set_index('Day')[['Sleep']])

        # New: multi-feature trend
        if 'Study' in hdf.columns and 'Screen' in hdf.columns:
            st.subheader("📉 Study & Screen Time Trends")
            trend_cols = [c for c in ['Study','Screen','Anxiety','Exercise'] if c in hdf.columns]
            st.line_chart(hdf.set_index('Day')[trend_cols])

        # New: stress level distribution
        if 'StressLevel' in hdf.columns:
            st.subheader("🥧 Stress Level Distribution")
            level_counts = hdf['StressLevel'].value_counts()
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            clrs = [COLORS.get(l, '#888') for l in level_counts.index]
            ax2.pie(level_counts.values, labels=level_counts.index,
                    colors=clrs, autopct='%1.0f%%', startangle=140,
                    wedgeprops={'linewidth':1,'edgecolor':'white'})
            ax2.set_title('Sessions by stress level', fontsize=12)
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close()

        # Raw data table
        with st.expander("📋 View raw data"):
            st.dataframe(hdf, use_container_width=True)
            csv = hdf.to_csv(index=False).encode()
            st.download_button("⬇️ Download CSV", csv,
                               "stress_history.csv", "text/csv")

# ─── Footer ──────────────────────────────────────────────────────────────────
st.divider()
st.caption("🧠 Student Stress Monitor | Built with Streamlit & scikit-learn | For educational purposes only.")