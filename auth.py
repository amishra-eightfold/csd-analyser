"""
Google OAuth authentication module for CSD Analyzer.

This module handles Google OAuth 2.0 authentication flow, ensuring only users
with authorized domains can access the application.
"""

import streamlit as st
import os
import requests
import json
import sys
import urllib.parse
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from typing import Dict, Any, Optional, Tuple
from config.logging_config import get_logger

# Initialize logger
logger = get_logger('auth')

# Debug information
print("==== OAuth Configuration ====")
print(f"CLIENT_ID: {st.secrets.get('GOOGLE_CLIENT_ID', 'Not set')}")
print(f"CLIENT_SECRET: {st.secrets.get('GOOGLE_CLIENT_SECRET', 'Not set')[:6]}..." if st.secrets.get('GOOGLE_CLIENT_SECRET') else "CLIENT_SECRET: Not set")
print(f"REDIRECT_URI: {st.secrets.get('REDIRECT_URI', 'Not set')}")
print(f"ALLOWED_DOMAIN: {st.secrets.get('ALLOWED_DOMAIN', 'Not set')}")
print(f"Python version: {sys.version}")

# Check if secrets exist and are accessible
try:
    secrets_available = hasattr(st, 'secrets') and st.secrets is not None
    print(f"Streamlit secrets available: {secrets_available}")
    if secrets_available:
        print(f"Secrets keys: {list(st.secrets.keys()) if hasattr(st.secrets, 'keys') else 'No keys method'}")
except Exception as e:
    print(f"Error accessing Streamlit secrets: {e}")

print("============================")

# Get the current directory for the HTML files
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGIN_HTML_PATH = os.path.join(CURRENT_DIR, "login.html")
REDIRECT_HTML_PATH = os.path.join(CURRENT_DIR, "redirect.html")

# Google OAuth configuration with better error handling
try:
    CLIENT_ID = st.secrets.get("GOOGLE_CLIENT_ID")
    CLIENT_SECRET = st.secrets.get("GOOGLE_CLIENT_SECRET")
    print(f"Retrieved CLIENT_ID: {'Yes' if CLIENT_ID else 'No'}")
    print(f"Retrieved CLIENT_SECRET: {'Yes' if CLIENT_SECRET else 'No'}")
except Exception as e:
    print(f"Error retrieving OAuth credentials: {e}")
    CLIENT_ID = None
    CLIENT_SECRET = None

# Validate that credentials are provided
if not CLIENT_ID or not CLIENT_SECRET:
    error_msg = f"Google OAuth credentials not configured. CLIENT_ID: {'✓' if CLIENT_ID else '✗'}, CLIENT_SECRET: {'✓' if CLIENT_SECRET else '✗'}"
    print(f"CREDENTIAL ERROR: {error_msg}")
    st.error("Google OAuth credentials not configured. Please set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in Streamlit secrets.")
    st.error(f"Debug: CLIENT_ID exists: {'Yes' if CLIENT_ID else 'No'}, CLIENT_SECRET exists: {'Yes' if CLIENT_SECRET else 'No'}")
    st.stop()

# Determine if we're running on Streamlit Cloud or locally
IS_CLOUD = (
    os.environ.get('STREAMLIT_SHARING', '') == 'true' or 
    os.environ.get('STREAMLIT_CLOUD', '') == 'true' or
    os.environ.get('HOSTNAME', '').startswith('streamlit') or
    'streamlit.app' in os.environ.get('STREAMLIT_SERVER_BASE_URL_PATH', '') or
    'streamlit.app' in os.environ.get('EXTERNAL_URL', '')
)
print(f"Running on Streamlit Cloud: {IS_CLOUD}")
print(f"Environment variables check:")
print(f"  STREAMLIT_SHARING: {os.environ.get('STREAMLIT_SHARING', 'Not set')}")
print(f"  STREAMLIT_CLOUD: {os.environ.get('STREAMLIT_CLOUD', 'Not set')}")
print(f"  HOSTNAME: {os.environ.get('HOSTNAME', 'Not set')}")
print(f"  STREAMLIT_SERVER_BASE_URL_PATH: {os.environ.get('STREAMLIT_SERVER_BASE_URL_PATH', 'Not set')}")
print(f"  EXTERNAL_URL: {os.environ.get('EXTERNAL_URL', 'Not set')}")

# Set the redirect URI based on the environment
if IS_CLOUD:
    DEFAULT_REDIRECT_URI = "https://ef-csd-analyser.streamlit.app/callback"
else:
    DEFAULT_REDIRECT_URI = "http://localhost:8501/callback"

print(f"Default redirect URI: {DEFAULT_REDIRECT_URI}")

# Get REDIRECT_URI from secrets with additional debugging
try:
    REDIRECT_URI = st.secrets.get("REDIRECT_URI", DEFAULT_REDIRECT_URI)
    print(f"Retrieved REDIRECT_URI from secrets: {REDIRECT_URI}")
    print(f"REDIRECT_URI type: {type(REDIRECT_URI)}")
    print(f"REDIRECT_URI is None: {REDIRECT_URI is None}")
    print(f"REDIRECT_URI is empty: {REDIRECT_URI == '' if REDIRECT_URI else 'REDIRECT_URI is None'}")
except Exception as e:
    print(f"Error retrieving REDIRECT_URI from secrets: {e}")
    REDIRECT_URI = DEFAULT_REDIRECT_URI
    print(f"Using default REDIRECT_URI: {REDIRECT_URI}")

print(f"Final REDIRECT_URI: {REDIRECT_URI}")

try:
    ALLOWED_DOMAIN = st.secrets.get("ALLOWED_DOMAIN", "eightfold.ai")
    print(f"Retrieved ALLOWED_DOMAIN: {ALLOWED_DOMAIN}")
except Exception as e:
    print(f"Error retrieving ALLOWED_DOMAIN: {e}")
    ALLOWED_DOMAIN = "eightfold.ai"

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
    
    # Force authentication to be False when running on Streamlit Cloud
    # This ensures the login page is always shown initially
    if IS_CLOUD and "force_auth_check" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.force_auth_check = True
        print("Forcing authentication check on Streamlit Cloud")

def create_flow():
    """Create OAuth flow instance to manage the OAuth 2.0 Authorization Grant Flow"""
    try:
        print(f"Creating OAuth flow with CLIENT_ID: {'[PRESENT]' if CLIENT_ID else '[MISSING]'}")
        print(f"Creating OAuth flow with CLIENT_SECRET: {'[PRESENT]' if CLIENT_SECRET else '[MISSING]'}")
        print(f"Creating OAuth flow with REDIRECT_URI: {REDIRECT_URI}")
        print(f"REDIRECT_URI type: {type(REDIRECT_URI)}")
        print(f"REDIRECT_URI length: {len(REDIRECT_URI) if REDIRECT_URI else 0}")
        
        if not CLIENT_ID:
            logger.error("CLIENT_ID is missing - cannot create OAuth flow")
            print("ERROR: CLIENT_ID is None or empty")
            return None
            
        if not CLIENT_SECRET:
            logger.error("CLIENT_SECRET is missing - cannot create OAuth flow")
            print("ERROR: CLIENT_SECRET is None or empty")
            return None
            
        if not REDIRECT_URI:
            logger.error("REDIRECT_URI is missing - cannot create OAuth flow")
            print("ERROR: REDIRECT_URI is None or empty")
            return None
            
        client_config = {
            "web": {
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [REDIRECT_URI]
            }
        }
        print(f"Client config created successfully")
        print(f"Client config redirect_uris: {client_config['web']['redirect_uris']}")
        
        flow = Flow.from_client_config(
            client_config,
            scopes=SCOPES,
            redirect_uri=REDIRECT_URI
        )
        print("OAuth flow created successfully")
        print(f"Flow redirect_uri: {flow.redirect_uri}")
        return flow
    except Exception as e:
        logger.error(f"Error creating OAuth flow: {str(e)}")
        print(f"ERROR creating OAuth flow: {str(e)}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return None

def get_auth_url():
    """Get the authorization URL to redirect the user to"""
    try:
        print("🔄 Starting auth URL generation...")
        flow = create_flow()
        if flow is None:
            print("❌ OAuth flow creation returned None")
            return None
        
        print("✅ OAuth flow created successfully, generating auth URL...")
        auth_url, state = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
            prompt='consent'
        )
        print(f"✅ Auth URL generated successfully")
        print(f"Generated Auth URL: {auth_url}")
        print(f"State: {state}")
        logger.info(f"Generated authentication URL for OAuth flow")
        return auth_url
    except Exception as e:
        print(f"❌ ERROR generating auth URL: {str(e)}")
        logger.error(f"Error generating auth URL: {str(e)}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return None

def display_login_page():
    """Display the beautiful login page directly in Streamlit"""
    
    # Add immediate debugging output
    st.write("🔍 **DEBUG INFO:**")
    st.write(f"- IS_CLOUD: {IS_CLOUD}")
    st.write(f"- CLIENT_ID exists: {'✅ Yes' if CLIENT_ID else '❌ No'}")
    st.write(f"- CLIENT_SECRET exists: {'✅ Yes' if CLIENT_SECRET else '❌ No'}")
    st.write(f"- REDIRECT_URI: {REDIRECT_URI}")
    
    # Check if login.html exists
    if not os.path.exists(LOGIN_HTML_PATH):
        st.error(f"Login HTML file not found at: {LOGIN_HTML_PATH}")
        st.info("Please ensure the login.html file exists in the project directory.")
        return
    
    # Get auth URL with debugging
    st.write("🔄 **Generating Auth URL...**")
    auth_url = get_auth_url()
    
    if not auth_url:
        st.error("❌ **Unable to generate authentication URL. Please check the OAuth configuration.**")
        st.write("**Debug: OAuth flow creation failed**")
        return
    else:
        st.success("✅ **Auth URL generated successfully**")
        st.write(f"**Auth URL length**: {len(auth_url)} characters")
        st.write(f"**Auth URL preview**: `{auth_url[:100]}...`")
    
    # Read the login.html file
    try:
        with open(LOGIN_HTML_PATH, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Inject the real auth URL into the HTML by replacing the JavaScript line
        original_line = 'const urlParams = new URLSearchParams(window.location.search);'
        encoded_auth_url = urllib.parse.quote(auth_url, safe=':/?#[]@!$&\'()*+,;=')
        replacement_line = f'const urlParams = new URLSearchParams("auth_url={encoded_auth_url}");'
        
        html_content = html_content.replace(original_line, replacement_line)
        
        # Display the beautiful login page directly in Streamlit
        st.components.v1.html(html_content, height=800, scrolling=True)
        
    except Exception as e:
        st.error(f"Error loading login page: {str(e)}")
        logger.error(f"Error loading login page: {str(e)}")
        
        # Fallback to direct auth URL
        st.markdown("**Fallback - Please click the link below to authenticate:**")
        st.markdown(f"[🔗 **Sign in with Google**]({auth_url})")

def handle_auth_code(code: str) -> bool:
    """Handle the authorization code and complete authentication"""
    try:
        # Exchange code for credentials
        credentials = exchange_code(code)
        if not credentials:
            return False
        
        # Get user info
        user_info = get_user_info(credentials)
        if not user_info:
            return False
        
        # Verify domain
        email = user_info.get('email', '')
        if not verify_domain(email):
            st.error(f"Access denied. Only @{ALLOWED_DOMAIN} email addresses are allowed.")
            return False
        
        # Store in session state
        st.session_state.authenticated = True
        st.session_state.user_info = user_info
        st.session_state.credentials = credentials
        st.session_state.auth_code = code
        
        logger.info(f"User {email} authenticated successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error handling auth code: {str(e)}")
        return False

def exchange_code(code: str):
    """Exchange authorization code for tokens"""
    try:
        flow = create_flow()
        if flow is None:
            return None
            
        flow.fetch_token(code=code)
        logger.info("Successfully exchanged authorization code for tokens")
        return flow.credentials
    except Exception as e:
        logger.error(f"Error exchanging authorization code: {str(e)}")
        return None

def get_user_info(credentials) -> Dict[str, Any]:
    """Get user info from Google API"""
    try:
        response = requests.get(
            'https://www.googleapis.com/oauth2/v2/userinfo',
            headers={'Authorization': f'Bearer {credentials.token}'},
            timeout=10
        )
        
        if response.status_code == 200:
            user_info = response.json()
            logger.info(f"Retrieved user info for {user_info.get('email', 'unknown')}")
            return user_info
        else:
            logger.error(f"Failed to get user info: HTTP {response.status_code}")
            return {}
    except Exception as e:
        logger.error(f"Error getting user info: {str(e)}")
        return {}

def verify_domain(email: str) -> bool:
    """Verify if the user's email domain is allowed"""
    if not email:
        return False
    
    domain = email.split('@')[-1]
    is_allowed = domain.lower() == ALLOWED_DOMAIN.lower()
    
    if is_allowed:
        logger.info(f"Domain verification successful for {email}")
    else:
        logger.warning(f"Domain verification failed for {email}. Expected: {ALLOWED_DOMAIN}")
    
    return is_allowed

def handle_callback():
    """Handle the OAuth callback from URL parameters"""
    try:
        # Check URL parameters for authorization code
        query_params = st.query_params
        
        if 'code' in query_params:
            code = query_params['code']
            logger.info("Authorization code received via callback")
            
            if handle_auth_code(code):
                # Clear query parameters
                st.query_params.clear()
                st.success("Authentication successful!")
                st.rerun()
            else:
                st.error("Authentication failed.")
                
        elif 'error' in query_params:
            error = query_params['error']
            logger.warning(f"OAuth error received: {error}")
            st.error(f"Authentication error: {error}")
            
    except Exception as e:
        logger.error(f"Error handling callback: {str(e)}")
        st.error("Error processing authentication callback.")

def handle_auth():
    """Main authentication handler"""
    init_auth_session_state()
    
    # Handle callback first
    handle_callback()
    
    # Check if user is authenticated
    if st.session_state.authenticated and st.session_state.user_info:
        logger.info(f"User {st.session_state.user_info.get('email')} is authenticated")
        return True
    
    # Show login page
    display_login_page()
    return False

def logout():
    """Logout the current user"""
    try:
        if st.session_state.authenticated:
            email = st.session_state.user_info.get('email', 'unknown') if st.session_state.user_info else 'unknown'
            logger.info(f"User {email} logged out")
        
        # Clear all authentication-related session state
        st.session_state.authenticated = False
        st.session_state.user_info = None
        st.session_state.auth_code = None
        st.session_state.credentials = None
        st.session_state.auth_url = None
        
        # Force page refresh
        st.rerun()
        
    except Exception as e:
        logger.error(f"Error during logout: {str(e)}")

def get_current_user() -> Optional[Dict[str, Any]]:
    """Get the current authenticated user info"""
    if st.session_state.authenticated and st.session_state.user_info:
        return st.session_state.user_info
    return None

def is_authenticated() -> bool:
    """Check if user is authenticated"""
    return st.session_state.get('authenticated', False)

def require_auth():
    """Decorator/function to require authentication"""
    if not handle_auth():
        st.stop() 