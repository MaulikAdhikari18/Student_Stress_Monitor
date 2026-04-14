import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("🎓 Student Stress Monitor Dashboard")

# Sample data (we will replace later)
data = {
    "Day": ["Mon", "Tue", "Wed", "Thu", "Fri"],
    "Stress": [3, 5, 6, 4, 7],
    "Sleep": [6, 5, 4, 7, 3]
}

df = pd.DataFrame(data)

# Line Chart (Stress)
st.subheader("📈 Stress Over Time")
st.line_chart(df.set_index("Day")["Stress"])

# Bar Chart (Sleep)
st.subheader("📊 Sleep Hours")
st.bar_chart(df.set_index("Day")["Sleep"])

# Stats
st.subheader("📋 Current Status")
st.write("Stress Level: HIGH")
st.write("Average Sleep: 5 hrs")
st.write("Recommendation: Take a break 😌")

st.sidebar.header("Enter Today's Data")

sleep = st.sidebar.slider("Sleep Hours", 0, 10, 5)
study = st.sidebar.slider("Study Hours", 0, 12, 6)
screen = st.sidebar.slider("Screen Time", 0, 12, 5)

# Simple stress logic (temporary AI)
stress_score = (10 - sleep) + study + screen

if stress_score > 15:
    level = "HIGH"
elif stress_score > 10:
    level = "MEDIUM"
else:
    level = "LOW"

st.subheader("🎯 Predicted Stress Level")
st.write(level)

st.subheader("💡 Recommendation")

if level == "HIGH":
    st.warning("⚠️ Take a break, reduce screen time, and relax")
elif level == "MEDIUM":
    st.info("🙂 Try balancing study and rest")
else:
    st.success("✅ You're doing well! Keep it up")