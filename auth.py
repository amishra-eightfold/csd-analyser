import streamlit as st
import os
import requests
import json
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from dotenv import load_dotenv
import time
import sys

# Load environment variables
load_dotenv()

# Debug information
print("==== OAuth Configuration ====")
print(f"CLIENT_ID: {os.getenv('GOOGLE_CLIENT_ID', 'Not set')}")
print(f"CLIENT_SECRET: {os.getenv('GOOGLE_CLIENT_SECRET', 'Not set')[:6]}...")
print(f"REDIRECT_URI: {os.getenv('REDIRECT_URI', 'Not set')}")
print(f"ALLOWED_DOMAIN: {os.getenv('ALLOWED_DOMAIN', 'Not set')}")
print(f"Python version: {sys.version}")
print("============================")

# Get the current directory for the redirect.html file
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REDIRECT_HTML_PATH = os.path.join(CURRENT_DIR, "redirect.html")

# Determine if running on Streamlit Cloud
IS_STREAMLIT_CLOUD = os.environ.get('STREAMLIT_RUNTIME_IS_CLOUD', False)

# Google OAuth configuration
CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "902153275427-cig5li22d32gvgmvnucmffcgdp3dfl9b.apps.googleusercontent.com")
CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "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")

# Set the appropriate redirect URI based on environment
if IS_STREAMLIT_CLOUD:
    # For Streamlit Cloud
    DEFAULT_REDIRECT_URI = "https://csd-analyser.streamlit.app/callback"
else:
    # For local development
    DEFAULT_REDIRECT_URI = "http://localhost:8501/callback"

REDIRECT_URI = os.getenv("REDIRECT_URI", DEFAULT_REDIRECT_URI)
ALLOWED_DOMAIN = os.getenv("ALLOWED_DOMAIN", "eightfold.ai")

# OAuth scopes
SCOPES = [
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "openid"
]

def init_auth_session_state():
    """Initialize session state variables for authentication"""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user_info" not in st.session_state:
        st.session_state.user_info = None
    if "auth_code" not in st.session_state:
        st.session_state.auth_code = None
    if "credentials" not in st.session_state:
        st.session_state.credentials = None
    if "auth_url" not in st.session_state:
        st.session_state.auth_url = None

def create_flow():
    """Create OAuth flow instance to manage the OAuth 2.0 Authorization Grant Flow"""
    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [REDIRECT_URI]
            }
        },
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )
    return flow

def get_auth_url():
    """Get the authorization URL to redirect the user to"""
    flow = create_flow()
    auth_url, _ = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true',
        prompt='consent'
    )
    print(f"Generated Auth URL: {auth_url}")
    return auth_url

def exchange_code(code):
    """Exchange authorization code for tokens"""
    flow = create_flow()
    flow.fetch_token(code=code)
    return flow.credentials

def get_user_info(credentials):
    """Get user info from Google API"""
    response = requests.get(
        'https://www.googleapis.com/oauth2/v2/userinfo',
        headers={'Authorization': f'Bearer {credentials.token}'}
    )
    return response.json()

def verify_domain(email):
    """Verify if the user's email domain is allowed"""
    domain = email.split('@')[-1]
    return domain.lower() == ALLOWED_DOMAIN.lower()

def display_login_page():
    """Display the login page"""
    st.title("CSD Analyzer")
    st.subheader("Login Required")
    
    # Generate auth URL if not already in session state
    if st.session_state.auth_url is None:
        st.session_state.auth_url = get_auth_url()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 20px; border-radius: 10px; border: 1px solid #ddd; margin-bottom: 20px;">
            <p>Please sign in with your Eightfold.ai Google account to access the CSD Analyzer.</p>
            <p style="font-size: 0.8em; color: #666;">Only @eightfold.ai email addresses are authorized.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Check if redirect.html exists
        if os.path.exists(REDIRECT_HTML_PATH):
            with open(REDIRECT_HTML_PATH, 'r') as f:
                redirect_html = f.read()
                
            # Replace placeholders in the HTML template
            redirect_html = redirect_html.replace('__AUTH_URL__', st.session_state.auth_url)
            
            # Create a unique HTML component to trigger the redirect
            html_component = f"""
            <div style="text-align: center; margin: 30px 0;">
                <a href="{st.session_state.auth_url}" target="_blank" style="
                    background-color: #4285F4;
                    color: white;
                    padding: 12px 24px;
                    text-decoration: none;
                    border-radius: 4px;
                    font-weight: bold;
                    display: inline-block;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.25);
                ">
                    Sign in with Google
                </a>
            </div>
            """
            st.markdown(html_component, unsafe_allow_html=True)
        else:
            print(f"Redirect HTML template not found at {REDIRECT_HTML_PATH}")
            # Fallback to direct link
            st.markdown(
                f"""
                <div style="text-align: center; margin: 30px 0;">
                    <a href="{st.session_state.auth_url}" target="_blank" style="
                        background-color: #4285F4;
                        color: white;
                        padding: 12px 24px;
                        text-decoration: none;
                        border-radius: 4px;
                        font-weight: bold;
                        display: inline-block;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.25);
                    ">
                        Sign in with Google
                    </a>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Show the URL as text for easy copying
        st.markdown("### Direct Authentication URL")
        st.markdown("If the button doesn't work, copy and paste this URL into your browser:")
        st.code(st.session_state.auth_url, language=None)

def handle_auth():
    """Handle authentication flow"""
    # Initialize authentication session state
    init_auth_session_state()
    
    # Check for callback with authorization code
    query_params = st.query_params
    
    if "code" in query_params:
        # Handle the OAuth callback
        code = query_params["code"]
        print(f"Received auth code: {code[:10]}...")
        st.session_state.auth_code = code
        
        # Exchange code for credentials
        try:
            credentials = exchange_code(code)
            st.session_state.credentials = {
                "token": credentials.token,
                "refresh_token": credentials.refresh_token,
                "token_uri": credentials.token_uri,
                "client_id": credentials.client_id,
                "client_secret": credentials.client_secret,
                "scopes": credentials.scopes
            }
            
            # Get user info
            user_info = get_user_info(credentials)
            print(f"User info received: {user_info.get('email', 'Unknown')}")
            st.session_state.user_info = user_info
            
            # Verify domain
            if verify_domain(user_info.get("email", "")):
                st.session_state.authenticated = True
                print(f"Authentication successful for {user_info.get('email')}")
                # Clear the URL parameters to avoid reusing the authorization code
                st.query_params.clear()
                return True
            else:
                st.error(f"Access denied. Only @{ALLOWED_DOMAIN} email addresses are allowed.")
                st.session_state.authenticated = False
                time.sleep(3)  # Give user time to read the message
                # Clear session and reload
                for key in list(st.session_state.keys()):
                    if key.startswith("auth_") or key in ["authenticated", "user_info", "credentials"]:
                        del st.session_state[key]
                st.query_params.clear()
                st.rerun()
                
        except Exception as e:
            st.error(f"Authentication error: {str(e)}")
            st.session_state.authenticated = False
    
    # Display login page if not authenticated
    if not st.session_state.authenticated:
        display_login_page()
        return False
    
    return True

def logout():
    """Log out the current user"""
    # Clear authentication-related session state
    for key in list(st.session_state.keys()):
        if key.startswith("auth_") or key in ["authenticated", "user_info", "credentials"]:
            del st.session_state[key]
    
    # Rerun the app to show login page
    st.rerun()

def get_current_user():
    """Get the current authenticated user info"""
    return st.session_state.user_info if st.session_state.authenticated else None 