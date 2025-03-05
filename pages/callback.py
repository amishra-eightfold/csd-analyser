import streamlit as st

# This is a minimal callback page that just redirects to the main page
# It exists only to prevent the "Page not found" error

# The auth.py handle_auth function will process the code parameter
# We just need to redirect to the main page

# Redirect to the main page
st.switch_page("app.py")