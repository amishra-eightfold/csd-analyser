"""
Simple test for authentication display
"""
import streamlit as st
import auth

st.set_page_config(page_title="Auth Test", layout="wide")

st.title("Authentication Test")

# Test authentication
if auth.handle_auth():
    st.success("✅ Authentication successful!")
    user_info = auth.get_current_user()
    if user_info:
        st.write(f"Welcome, {user_info.get('name', 'User')}")
        st.write(f"Email: {user_info.get('email', 'N/A')}")
        
        if st.button("Logout"):
            auth.logout()
else:
    st.info("Authentication page should be displayed above.") 