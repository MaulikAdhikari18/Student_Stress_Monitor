"""Login / create-account page, shown after the landing page."""

import streamlit as st

from src.db.users import create_user, verify_user


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
                confirm_pw = st.text_input("Confirm password", type="password",
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
