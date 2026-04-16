import streamlit as st
import pandas as pd
import datetime
import os

st.title("🎓 Student Stress Monitor Dashboard")

# -------------------- LOAD DATA --------------------
file_path = os.path.join("data", "stress_data.csv")

# ✅ Create file if not exists
if not os.path.exists(file_path):
    df = pd.DataFrame(columns=["Day", "Stress", "Sleep"])
    df.to_csv(file_path, index=False)

df = pd.read_csv(file_path)

# -------------------- SIDEBAR INPUT --------------------
st.sidebar.header("📥 Enter Today's Data")

sleep = st.sidebar.slider("Sleep Hours", 0, 10, 5)
study = st.sidebar.slider("Study Hours", 0, 12, 6)
screen = st.sidebar.slider("Screen Time", 0, 12, 5)

# -------------------- STRESS CALCULATION --------------------
stress_score = (10 - sleep) + study + screen

if stress_score > 15:
    level = "HIGH"
elif stress_score > 10:
    level = "MEDIUM"
else:
    level = "LOW"

# -------------------- DISPLAY --------------------
st.subheader("🎯 Predicted Stress Level")
st.write(f"**{level}** (Score: {stress_score})")

# -------------------- SAVE BUTTON --------------------
if st.sidebar.button("💾 Save Data"):
    new_data = {
        "Day": datetime.datetime.now().strftime("%a"),
        "Stress": stress_score,
        "Sleep": sleep
    }

    df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
    df.to_csv(file_path, index=False)

    st.success("✅ Data Saved Successfully!")
    st.rerun()   # ✅ refresh dashboard immediately

# -------------------- CHARTS --------------------
st.subheader("📈 Stress Over Time")

if not df.empty:
    st.line_chart(df.set_index("Day")["Stress"])
else:
    st.warning("No data available yet.")

st.subheader("📊 Sleep Hours")

if not df.empty:
    st.bar_chart(df.set_index("Day")["Sleep"])

# -------------------- RECOMMENDATION --------------------
st.subheader("💡 Recommendation")

if level == "HIGH":
    st.warning("⚠️ High stress detected! Take a break, reduce screen time, and relax.")
elif level == "MEDIUM":
    st.info("🙂 Moderate stress. Try balancing study and rest.")
else:
    st.success("✅ You're doing well!")

# -------------------- FOOTER --------------------
st.markdown("---")
st.caption("Built using Streamlit | Student Stress Monitor")