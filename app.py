"""
Student Stress Monitor
Streamlit App — Main entry point
Run: streamlit run app.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# ─── Page config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Student Stress Monitor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 700;
        background: linear-gradient(135deg, #534AB7, #D4537E);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header { color: #666; font-size: 1rem; margin-bottom: 1.5rem; }
    .stress-card {
        padding: 1.2rem 1.5rem; border-radius: 12px;
        border-left: 6px solid; margin-bottom: 1rem;
    }
    .stress-low    { background: #EAF3DE; border-color: #639922; }
    .stress-moderate { background: #FAEEDA; border-color: #BA7517; }
    .stress-high   { background: #FAECE7; border-color: #993C1D; }
    .stress-critical { background: #FCEBEB; border-color: #A32D2D; }
    .metric-box {
        background: #f8f8f8; border-radius: 10px;
        padding: 0.8rem 1rem; text-align: center;
    }
    .tip-box {
        background: #f4f2ff; border-left: 4px solid #534AB7;
        border-radius: 0 8px 8px 0; padding: 0.8rem 1rem;
        margin-bottom: 0.6rem; font-size: 0.92rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Load model ─────────────────────────────────────────────────────────────

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

@st.cache_resource
def load_model():
    with open(os.path.join(MODELS_DIR, 'model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODELS_DIR, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    return model, scaler, meta

try:
    model, scaler, meta = load_model()
    MODEL_LOADED = True
except FileNotFoundError:
    MODEL_LOADED = False

LABELS  = ['Low', 'Moderate', 'High', 'Critical']
COLORS  = {'Low': '#639922', 'Moderate': '#BA7517', 'High': '#993C1D', 'Critical': '#A32D2D'}
EMOJIS  = {'Low': '😊', 'Moderate': '😐', 'High': '😟', 'Critical': '😰'}

FEATURES = [
    'study_hours', 'assignments_pending', 'exam_pressure',
    'academic_performance', 'sleep_hours', 'exercise_days_per_week',
    'social_interactions_per_week', 'screen_time_hours',
    'anxiety_level', 'financial_stress', 'family_support',
    'peer_pressure', 'extracurricular_activities', 'relationship_issues',
]

# ─── Sidebar inputs ─────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📋 Enter Your Details")

    st.markdown("#### 📚 Academic")
    study_hours   = st.slider("Study hours / day", 0.0, 16.0, 6.0, 0.5)
    assignments   = st.slider("Assignments pending", 0, 15, 3)
    exam_pressure = st.slider("Exam pressure (1–10)", 1, 10, 5)
    performance   = st.slider("Academic performance (1–10)", 1, 10, 7)

    st.markdown("#### 🏃 Lifestyle")
    sleep_hours   = st.slider("Sleep hours / night", 2.0, 12.0, 7.0, 0.5)
    exercise      = st.slider("Exercise days / week", 0, 7, 3)
    social        = st.slider("Social interactions / week", 0, 20, 5)
    screen_time   = st.slider("Screen time hours / day", 0.0, 16.0, 4.0, 0.5)

    st.markdown("#### 🧠 Mental & Social")
    anxiety       = st.slider("Anxiety level (1–10)", 1, 10, 4)
    finance       = st.slider("Financial stress (1–10)", 1, 10, 3)
    family        = st.slider("Family support (1–10)", 1, 10, 7)
    peer          = st.slider("Peer pressure (1–10)", 1, 10, 4)
    extra         = st.selectbox("Extracurricular activities",
                                 [0, 1, 2],
                                 format_func=lambda x: ['None','1–2 activities','3+ activities'][x])
    rel           = st.selectbox("Relationship situation",
                                 [0, 1, 2],
                                 format_func=lambda x: ['Single / not applicable',
                                                         'Stable relationship',
                                                         'Relationship issues'][x])

    predict_btn = st.button("🔍 Analyze My Stress", use_container_width=True, type="primary")

# ─── Main area ──────────────────────────────────────────────────────────────

st.markdown('<div class="main-header">🧠 Student Stress Monitor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered stress analysis & personalized management recommendations</div>', unsafe_allow_html=True)

if not MODEL_LOADED:
    st.error("⚠️ Model not found. Please run: `python src/train_model.py` first.")
    st.stop()

acc = meta.get('accuracy', 0)
best = meta.get('best_model', 'ML Model')
col1, col2, col3 = st.columns(3)
col1.metric("Model", best)
col2.metric("Accuracy", f"{acc*100:.1f}%")
col3.metric("Features", str(len(FEATURES)))

st.divider()

# ─── Prediction ─────────────────────────────────────────────────────────────

input_data = np.array([[
    study_hours, assignments, exam_pressure, performance,
    sleep_hours, exercise, social, screen_time,
    anxiety, finance, family, peer, extra, rel,
]])
input_scaled = scaler.transform(input_data)

pred_class = int(model.predict(input_scaled)[0])
pred_proba = model.predict_proba(input_scaled)[0]
level_name  = LABELS[pred_class]
level_color = COLORS[level_name]
level_emoji = EMOJIS[level_name]

# Compute interpretable score
raw_score = (
    max(0, study_hours - 8) * 3.5
    + assignments * 2.5
    + (exam_pressure - 1) * 5.0
    + max(0, 7 - sleep_hours) * 4.0
    + max(0, 5 - exercise) * 2.0
    + max(0, 8 - social) * 1.5
    + max(0, screen_time - 4) * 2.0
    + (anxiety - 1) * 4.5
    + (finance - 1) * 3.0
    - (family - 1) * 2.5
    - (performance - 1) * 2.0
    + (peer - 1) * 2.5
    + (5 if extra == 0 else 0)
    + (8 if rel == 2 else 0)
)
score = int(np.clip(raw_score, 0, 100))

if predict_btn or True:   # show live by default
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Stress Result", "🔍 Factor Analysis", "💡 Management Tips", "📈 Model Insights"])

    # ── Tab 1: Result ──
    with tab1:
        c1, c2 = st.columns([1.4, 1])
        with c1:
            cls_css = level_name.lower()
            st.markdown(f"""
            <div class="stress-card stress-{cls_css}">
                <h2 style="margin:0;color:{level_color};">{level_emoji} {level_name} Stress</h2>
                <p style="font-size:1.05rem;margin:0.4rem 0 0;">Stress score: <strong>{score}/100</strong></p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### Confidence across levels")
            for i, (lbl, p) in enumerate(zip(LABELS, pred_proba)):
                col_a, col_b = st.columns([3, 1])
                col_a.progress(float(p), text=lbl)
                col_b.write(f"**{p*100:.1f}%**")

        with c2:
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.pie(pred_proba, labels=LABELS,
                   colors=['#639922','#EF9F27','#D85A30','#E24B4A'],
                   autopct='%1.1f%%', startangle=140,
                   wedgeprops={'linewidth': 1, 'edgecolor': 'white'})
            ax.set_title('Stress Level Probabilities', fontsize=12, pad=10)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        st.markdown("#### Quick health metrics")
        m1, m2, m3, m4 = st.columns(4)
        sleep_status = "Optimal" if sleep_hours >= 8 else f"-{8-sleep_hours:.1f}h deficit"
        study_load   = "Light" if study_hours <= 4 else "Moderate" if study_hours <= 7 else "Heavy" if study_hours <= 10 else "Extreme"
        recovery     = int(((exercise/7)*0.4 + (sleep_hours/10)*0.4 + (social/20)*0.2) * 100)
        burnout_risk = min(100, int((score * 0.6) + (max(0, study_hours-8)*4) + (max(0,10-sleep_hours)*3)))
        m1.metric("Sleep status", sleep_status)
        m2.metric("Study load", study_load)
        m3.metric("Recovery score", f"{recovery}%")
        m4.metric("Burnout risk", f"{burnout_risk}/100")

    # ── Tab 2: Factor Analysis ──
    with tab2:
        st.markdown("#### Which factors are driving your stress?")
        factor_scores = {
            "Academic load":    min(100, int(study_hours/16*50 + assignments/15*30 + exam_pressure/10*20)),
            "Sleep deficit":    min(100, int(max(0, 8-sleep_hours)/6*100)),
            "Anxiety":          int(anxiety/10*100),
            "Financial strain": int(finance/10*100),
            "Social isolation": min(100, int(max(0, 10-family)/9*80 + max(0,8-social)/8*20)),
            "Peer pressure":    int(peer/10*100),
            "Screen overuse":   min(100, int(max(0, screen_time-4)/12*100)),
            "Exercise deficit": min(100, int(max(0, 5-exercise)/5*100)),
        }
        sorted_factors = sorted(factor_scores.items(), key=lambda x: x[1], reverse=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        names  = [f[0] for f in sorted_factors]
        vals   = [f[1] for f in sorted_factors]
        clrs   = ['#A32D2D' if v>=75 else '#993C1D' if v>=55 else '#BA7517' if v>=30 else '#639922' for v in vals]
        bars = ax.barh(names, vals, color=clrs, edgecolor='white', linewidth=0.5)
        ax.set_xlim(0, 110)
        for bar, v in zip(bars, vals):
            ax.text(v+2, bar.get_y()+bar.get_height()/2, f"{v}", va='center', fontsize=10)
        ax.axvline(55, ls='--', lw=1, color='#BA7517', alpha=0.6, label='Moderate threshold')
        ax.axvline(75, ls='--', lw=1, color='#A32D2D', alpha=0.6, label='High threshold')
        ax.legend(fontsize=9)
        ax.set_xlabel("Stress contribution score")
        ax.set_title("Stress Factor Breakdown", fontsize=13, pad=10)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown("#### Your profile snapshot")
        snap = {
            "Study hours/day": f"{study_hours}h",
            "Sleep hours/night": f"{sleep_hours}h",
            "Anxiety level": f"{anxiety}/10",
            "Exercise days/week": str(exercise),
            "Assignments pending": str(assignments),
            "Financial stress": f"{finance}/10",
            "Family support": f"{family}/10",
            "Screen time/day": f"{screen_time}h",
        }
        cols = st.columns(4)
        for i, (k, v) in enumerate(snap.items()):
            cols[i % 4].metric(k, v)

    # ── Tab 3: Tips ──
    with tab3:
        st.markdown("#### Personalized stress management recommendations")

        tips = []
        if sleep_hours < 7:
            tips.append(("😴 Sleep hygiene", f"You're getting {sleep_hours}h — below the 7–9h ideal. Set a consistent bedtime, cut caffeine after 3 PM, and go screen-free 30 min before bed. Even one extra hour can halve cortisol levels."))
        if study_hours > 8:
            tips.append(("📚 Study smarter", f"Studying {study_hours}h/day can cause diminishing returns. Use the Pomodoro method (25 min on, 5 min off). Schedule a proper end time — your brain consolidates learning during rest."))
        if exercise < 3:
            tips.append(("🏃 Get moving", "Less than 3 days of exercise/week is a major stress amplifier. Even a 20-min brisk walk daily reduces cortisol by 26%. Try scheduling it like a class — non-negotiable."))
        if anxiety > 6:
            tips.append(("🧘 Manage anxiety", f"Your anxiety is at {anxiety}/10. Try box breathing: inhale 4s, hold 4s, exhale 4s, hold 4s. Apps like Headspace, Calm, or free YouTube meditations can help significantly."))
        if social < 4:
            tips.append(("👥 Social connection", "Social isolation intensifies stress significantly. Even one meaningful conversation daily helps. Consider joining a study group, club, or campus support circle."))
        if screen_time > 6:
            tips.append(("📱 Digital detox", f"{screen_time}h/day of screen time disrupts both sleep and focus. Try app timers, a 'no-phone hour' before bed, and batched social media checks (e.g., twice a day)."))
        if finance > 7:
            tips.append(("💰 Financial pressure", "High financial stress is common among students. Speak to your institution's student welfare office about emergency funds, scholarships, or part-time work options you may not know about."))
        if assignments > 8:
            tips.append(("📝 Task management", f"{assignments} pending assignments is a lot. Use the Eisenhower matrix: sort by urgency × importance. Tackle one urgent item per morning before checking messages."))
        if family < 4:
            tips.append(("❤️ Build your support net", "Low family support increases vulnerability to stress. Campus counselors, peer mentors, and student wellbeing groups can meaningfully fill that gap."))
        if extra == 0 and score > 40:
            tips.append(("🎨 Structured downtime", "Structured non-academic activities — sport, art, volunteering — create necessary mental distance from academic pressure and build long-term resilience."))
        if peer > 7:
            tips.append(("🤝 Peer pressure", "High peer pressure drains energy. Practice assertive communication — it's OK to say no. Spend time with people who motivate rather than pressure you."))
        if not tips:
            tips.append(("🌟 Great balance!", "Your indicators look healthy! Keep your current routines and do a weekly check-in to catch any early changes. Prevention is always easier than recovery."))

        for cat, tip in tips:
            st.markdown(f"""<div class="tip-box"><strong>{cat}</strong><br>{tip}</div>""",
                        unsafe_allow_html=True)

        st.divider()
        st.markdown("#### 🗓️ 7-day recovery plan")
        plan = [
            ("Day 1", "Set a consistent sleep time tonight and stick to it all week."),
            ("Day 2", "Write down all pending tasks. Cross off one small thing today."),
            ("Day 3", "Go for a 20-minute walk. No phone, no earphones. Just walk."),
            ("Day 4", "Call or message one friend or family member you trust."),
            ("Day 5", "Try one 25-min Pomodoro study block. Rate how focused you felt."),
            ("Day 6", "Take 10 minutes to practice box breathing or simple meditation."),
            ("Day 7", "Review the week: what helped? Plan to repeat those habits."),
        ]
        for day, action in plan:
            st.checkbox(f"**{day}** — {action}", key=day)

    # ── Tab 4: Model Insights ──
    with tab4:
        st.markdown("#### Model performance overview")

        img_col1, img_col2 = st.columns(2)
        cm_path = os.path.join(MODELS_DIR, 'confusion_matrix.png')
        fi_path = os.path.join(MODELS_DIR, 'feature_importance.png')

        if os.path.exists(cm_path):
            with img_col1:
                st.image(cm_path, caption="Confusion Matrix", use_container_width=True)
        if os.path.exists(fi_path):
            with img_col2:
                st.image(fi_path, caption="Feature Importance", use_container_width=True)

        st.markdown("#### Dataset & model summary")
        data_path = os.path.join(os.path.dirname(__file__), 'data', 'student_stress_data.csv')
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            st.dataframe(df.describe().round(2), use_container_width=True)

            fig2, axes = plt.subplots(1, 2, figsize=(10, 4))
            level_counts = df['stress_level'].map({0:'Low',1:'Moderate',2:'High',3:'Critical'}).value_counts()
            axes[0].bar(level_counts.index, level_counts.values,
                        color=['#639922','#EF9F27','#D85A30','#E24B4A'])
            axes[0].set_title('Stress Level Distribution', fontsize=12)
            axes[0].set_ylabel('Count')

            df[['anxiety_level','study_hours','sleep_hours','exam_pressure']].hist(
                ax=axes[1], bins=15, color='#534AB7', edgecolor='white', alpha=0.8)
            axes[1].set_title('Key Feature Distributions', fontsize=12)
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close()

# ─── Footer ─────────────────────────────────────────────────────────────────
st.divider()
st.caption("🧠 Student Stress Monitor — Built with Streamlit & scikit-learn | For educational purposes only.")
