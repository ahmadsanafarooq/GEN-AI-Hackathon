import streamlit as st
import os, json, datetime, hashlib
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from gtts import gTTS
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
import altair as alt
import speech_recognition as sr
from transformers import pipeline
import torch
import pickle
import re
import pandas as pd 

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

CRISIS_KEYWORDS = ["suicide", "kill myself", "end it all", "worthless", "can't go on", "hurt myself", "self harm", "want to disappear", "no reason to live"]

# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False
if "transcribed_text" not in st.session_state:
    st.session_state.transcribed_text = ""

# Admin configuration
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME")  # Set in HF Spaces secrets
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")  # Set in HF Spaces secrets

# --- UI/UX Modifications (Applied globally) ---
def set_background_and_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600;700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@300;400;700&display=swap');
        .stApp {
            background: linear-gradient(135deg, #e0e7ff 0%, #c6e2ff 50%, #b0c4de 100%); /* Light blueish gradient */
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            font-family: 'Montserrat', sans-serif;
            color: #333; /* Default text color to dark for light background */
        }
        
        h1, h2, h3, h4, h5, h6, .stMarkdown, label {
            font-family: 'Merriweather', serif;
            color: #1a237e; /* Darker blue for headings */
        }
        
        /* Ensure text area label is dark */
        .stTextArea > label {
            color: #333 !important;
        }
        .stButton>button {
            background-image: linear-gradient(to right, #6a11cb 0%, #2575fc 100%); /* Blue-purple gradient */
            color: white;
            border-radius: 8px;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            font-weight: bold;
            transition: all 0.2s ease-in-out;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
        }
        
        /* Secondary button style for "Explore Opportunities" */
        .secondary-button > button {
            background-color: white;
            color: #6a11cb;
            border: 1px solid #6a11cb;
            box-shadow: none;
        }
        .secondary-button > button:hover {
            background-color: #f0f4f8; /* Light hover */
            color: #2575fc;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .stTextInput>div>div>input, .stTextArea>div>div>textarea, .stSelectbox>div>div>div {
            border-radius: 8px;
            border: 1px solid #b3e5fc; /* Light blue border */
            padding: 10px;
            background-color: rgba(255, 255, 255, 0.95); /* Nearly opaque white */
            color: #333; /* Input text color dark */
        }
        /* Adjusted stSuccess and stInfo for white text on darker transparent black background */
        .stSuccess { 
            border-left: 5px solid #28a745; 
            background-color: rgba(0, 0, 0, 0.6); /* Transparent black */
            color: white; /* Make text white */
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 10px;
        }
        .stInfo { 
            border-left: 5px solid #17a2b8; 
            background-color: rgba(0, 0, 0, 0.6); /* Transparent black */
            color: white; /* Make text white */
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 10px;
        }
        /* Keeping warning/error as original light background for now, as black transparent might not suit warnings well */
        .stWarning { 
            border-left: 5px solid #ffc107; 
            background-color: rgba(255, 255, 255, 0.9); 
            color: #333; 
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 10px;
        }
        .stError { 
            border-left: 5px solid #dc3545; 
            background-color: rgba(255, 255, 255, 0.9); 
            color: #333; 
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 10px;
        }
        /* Custom container for content with blur background - used for auth form */
        .auth-content-container {
            background-color: rgba(255, 255, 255, 0.8); /* Slightly transparent white */
            backdrop-filter: blur(8px); /* Blur effect */
            border-radius: 15px;
            padding: 30px;
            margin: 20px auto;
            max-width: 450px; /* Slimmer for auth forms */
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
            text-align: center; /* Center content within */
        }
        
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); /* Adjusted for more flexibility */
            gap: 20px;
            margin-top: 30px;
            margin-bottom: 30px;
            max-width: 900px; /* Max width for the grid itself */
            margin-left: auto;
            margin-right: auto;
        }
        .feature-item {
            background-color: rgba(255, 255, 255, 0.9);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08);
            transition: transform 0.2s ease-in-out;
            border: 1px solid #e0e0e0;
        }
        .feature-item:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 15px rgba(0, 0, 0, 0.12);
        }
        .feature-item .icon-img {
            height: 60px; /* Adjust icon size as needed */
            margin-bottom: 10px;
            display: block; /* Ensure icon is on its own line */
            margin-left: auto;
            margin-right: auto;
        }
        .feature-item h3 {
            font-size: 1.2em;
            color: #1a237e;
            margin-bottom: 8px;
        }
        .feature-item p {
            font-size: 0.9em;
            color: #555;
            line-height: 1.5;
        }
        /* Main app content container */
        .main-app-content-container {
            background-color: rgba(255, 255, 255, 0.8); /* Slightly transparent white */
            backdrop-filter: blur(8px); /* Blur effect */
            border-radius: 15px;
            padding: 30px;
            margin: 20px auto;
            max-width: 800px; /* Wider for main app content */
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
            color: #333; /* Dark text inside */
        }
        /* Ensure headers within main app container are dark */
        .main-app-content-container h1, .main-app-content-container h2, 
        .main-app-content-container h3, .main-app-content-container h4, 
        .main-app-content-container h5, .main-app-content-container h6,
        .main-app-content-container .stMarkdown, .main-app-content-container label {
            color: #1a237e; /* Darker blue for headers in main app */
        }
        /* Specifically target text within main-app-content-container to be dark */
        .main-app-content-container p, .main-app-content-container li, .main-app-content-container div {
            color: #333;
        }
        /* Override specific Streamlit elements that don't pick up general styles for main app */
        .main-app-content-container .st-emotion-cache-1jmve6n, /* st.subheader */
        .main-app-content-container .st-emotion-cache-1gcs47q, /* st.text or similar */
        .main-app-content-container .st-emotion-cache-10q7f27, /* st.info text */
        .main-app-content-container .st-emotion-cache-1j0qsvo { /* more specific text */
             color: #333 !important;
        }
        /* Navbar Styling */
        .navbar {
            display: flex;
            justify-content: center; /* Center the logo and title */
            align-items: center;
            padding: 15px 40px;
            background-color: rgba(255, 255, 255, 0.8); /* Slightly transparent white */
            backdrop-filter: blur(5px);
            border-bottom: 1px solid rgba(0, 0, 0, 0.05);
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
            border-radius: 10px;
            margin-bottom: 20px;
            /* position: relative; Removed absolute positioning as logout button is gone */
        }
        .navbar-logo-container {
            display: flex;
            align-items: center;
            justify-content: center; /* Center the logo and text */
            flex-grow: 1; /* Allows it to take available space for centering */
        }
        .navbar-logo-text {
            font-family: 'Merriweather', serif;
            font-size: 24px;
            font-weight: bold;
            color: #1a237e;
            margin-left: 10px; /* Space between logo image and text */
        }
        .navbar-logo-img {
            height: 30px; /* Size of your logo image */
            vertical-align: middle;
        }
        .navbar-links { /* This block might not be needed if links are removed */
            display: flex;
            gap: 30px;
        }
        .navbar-link { /* This block might not be needed if links are removed */
            font-family: 'Montserrat', sans-serif;
            font-size: 16px;
            color: #3f51b5; /* Medium blue */
            text-decoration: none;
            padding: 5px 10px;
            transition: color 0.2s ease-in-out;
            cursor: pointer;
        }
        .navbar-link:hover { /* This block might not be needed if links are removed */
            color: #6a11cb; /* Purple hover */
        }
        .navbar-button { /* This block might not be needed if links are removed */
            background-image: linear-gradient(to right, #6a11cb 0%, #2575fc 100%);
            color: white;
            border-radius: 8px;
            padding: 8px 15px;
            font-size: 14px;
            font-weight: bold;
            text-decoration: none;
            transition: all 0.2s ease-in-out;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            cursor: pointer;
        }
        .navbar-button:hover { /* This block might not be needed if links are removed */
            transform: translateY(-1px);
            box_shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        }
        /* AI Powered Educational Platform banner style */
        .ai-powered-banner {
            display: inline-flex; /* Use inline-flex to shrink-wrap content */
            align-items: center;
            background-color: rgba(255, 255, 255, 0.7); /* Light, semi-transparent */
            border-radius: 20px;
            padding: 8px 15px;
            margin-bottom: 20px;
            font-size: 14px;
            font-weight: 600;
            color: #3f51b5; /* Blue text */
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            text-align: center; /* Ensure text is centered if container grows */
            margin-left: auto; /* Center horizontally */
            margin-right: auto; /* Center horizontally */
        }
        .centered-container {
            display: flex;
            justify-content: center;
            width: 100%;
        }
        /* Metrics section styles (50K+ Students Helped, etc.) */
        .metrics-container {
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
            margin-top: 40px;
            gap: 20px;
            /* border: 1px solid red; REMOVED for debugging */ 
        }
        .metric-item {
            text-align: center;
            background-color: rgba(255, 255, 255, 0.7);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.05);
            flex: 1; /* Allow items to grow/shrink */
            min-width: 180px; /* Minimum width before wrapping */
        }
        .metric-item h2 {
            font-size: 32px;
            color: #6a11cb; /* Purple */
            margin-bottom: 5px;
        }
        .metric-item p {
            font-size: 16px;
            color: #555;
        }
        .metric-item img { /* Style for icon images if used */
            height: 40px;
            margin-bottom: 10px;
        }
        /* Center plots */
        .st-emotion-cache-1pxazr6 { /* Specific Streamlit container for pyplot */
            display: flex;
            justify-content: center;
        }
        /* Override specific Streamlit elements that don't pick up general styles */
        .st-emotion-cache-1jmve6n { /* This class is often for st.subheader */
            color: #1a237e !important; /* Darker blue */
        }
        .st-emotion-cache-1gcs47q { /* This class can be for specific text elements */
             color: #333 !important;
        }
        .st-emotion-cache-10q7f27 { /* Example for st.info text */
             color: #333 !important;
        }
        /* Hide the top grey bar */
        header.st-emotion-cache-1gh8zsi {
            display: none !important;
        }
        div.st-emotion-cache-fis6y8 { 
            padding-top: 0 !important;
        }
        div.st-emotion-cache-z5inrg { 
            display: none !important;
        }
        /* Adjust overall container width */
        .st-emotion-cache-fg4lbf { 
            max-width: 1000px !important; /* Adjusted for wider layout */
            padding-left: 0 !important; 
            padding-right: 0 !important;
        }
        .block-container { 
            padding-left: 1rem;
            padding-right: 1rem;
            color: #333; /* Default text color within block-container */
        }
        /* Specific style for button alignment */
        /* Updated .button-row to stack and add spacing */
        .button-row {
            display: flex;
            flex-direction: column; /* Stack buttons vertically */
            align-items: center; /* Center buttons horizontally */
            gap: 20px; /* Space between buttons */
            margin-top: 20px;
            width: 100%; /* Ensure row takes full width to center its content */
        }
        .button-row > div { /* Target Streamlit column divs within button-row if still used */
            width: 100%; /* Make columns take full width to center buttons */
            display: flex;
            justify-content: center; /* Center the buttons inside their columns */
        }
        /* Custom footer style */
        .app-footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #f0f2f6; /* Streamlit's default background color */
            color: #31333F; /* Streamlit's default text color */
            text-align: center;
            padding: 10px;
            font-size: 14px;
            border-top: 1px solid #d3d3d3;
            z-index: 1000; /* Ensure footer is on top */
        }
        /* Adjust app padding to prevent content from being hidden by footer */
        .stApp {
            padding-bottom: 50px; /* Approx height of footer */
        }
        /* History entry styling */
        .history-entry {
            background-color: rgba(255, 255, 255, 0.9);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border: 1px solid #e0e0e0;
        }
        .history-entry h4 {
            color: #2575fc; /* Blue for history entry title */
            margin-top: 0;
            margin-bottom: 5px;
        }
        .history-entry p {
            color: #444;
            font-size: 0.95em;
        }
        .history-entry strong {
            color: #1a237e;
        }
        /* New style for output text color from DilBot */
        .dilbot-response-text {
            color: black !important; /* Force black text */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# User management functions
def hash_password(password):
    """Hash password using SHA-256 with salt"""
    salt = "dilbot_secure_salt_2024"  # You can change this
    return hashlib.sha256((password + salt).encode()).hexdigest()

def get_secure_users_path():
    """Get path to users file in a hidden directory"""
    secure_dir = ".secure_data"
    os.makedirs(secure_dir, exist_ok=True)
    return os.path.join(secure_dir, "users_encrypted.json")

def load_users():
    """Load users from secure file"""
    users_path = get_secure_users_path()
    if os.path.exists(users_path):
        try:
            with open(users_path, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_users(users):
    """Save users to secure file"""
    users_path = get_secure_users_path()
    with open(users_path, "w") as f:
        json.dump(users, f, indent=4)

def create_user_directory(username):
    """Create user-specific directory structure"""
    user_dir = f"users/{username}"
    os.makedirs(user_dir, exist_ok=True)
    return user_dir

def get_user_file_path(username, filename):
    """Get path to user-specific file"""
    user_dir = f"users/{username}"
    return os.path.join(user_dir, filename)

def signup(username, password, email):
    """Register new user"""
    users = load_users()
    if username in users:
        return False, "Username already exists"
    
    email_pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
    if not re.match(email_pattern, email):
         return False, "Invalid email format"
        
    users[username] = {
        "password": hash_password(password),
        "email": email,
        "created_at": str(datetime.datetime.now())
    }
    save_users(users)
    create_user_directory(username)
    return True, "Account created successfully!"

def login(username, password):
    """Authenticate user or admin"""
    # Check if admin login
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        return True, "Admin login successful!", True
    
    # Regular user login
    users = load_users()
    if username not in users:
        return False, "User not found.Please signup.", False
    
    if users[username]["password"] == hash_password(password):
        return True, "Login successful!", False
    return False, "Incorrect password", False

# Emotion detection
@st.cache_resource
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=1,
        device=-1
    )

def detect_emotion(text):
    emotion_pipeline = load_emotion_model()
    prediction = emotion_pipeline(text)[0][0]
    return prediction['label'].lower(), prediction['score']


# Authentication UI
def show_auth_page():
    set_background_and_styles()
    st.markdown(
        """
        <div class="navbar">
            <div class="navbar-logo-container">
                <img src="https://e7.pngegg.com/pngimages/498/917/png-clipart-computer-icons-desktop-chatbot-icon-blue-angle-thumbnail.png" class="navbar-logo-img">
                <div class="navbar-logo-text">DilBot</div>
            </div>
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown("<div class='centered-container'><div class='ai-powered-banner'><span>✨</span>AI-Powered Emotional Companion</div></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <h1 style='text-align: center; font-size: 3.5em; color: #1a237e;'>Transform Your Emotional Journey</h1>
        <p style='text-align: center; font-size: 1.2em; color: #555;'>
            DilBot empowers individuals with AI-driven insights for emotional well-being, personal growth, and self-discovery worldwide.
        </p>
        """, unsafe_allow_html=True
    )

    st.markdown('<div class="button-row">', unsafe_allow_html=True)
    
    # Buttons placed directly in a single column layout
    if st.button("Paid Plan", key="paid_plan_btn", use_container_width=True):
        st.info("Paid version coming soon") # Text for paid plan button
    st.markdown('<div class="secondary-button" style="margin-top: 20px;">', unsafe_allow_html=True) # Added margin for spacing
    if st.button("Login Below FOR FREE", key="login_free_btn", use_container_width=True):
        st.info("Slide down below") # Text for free login button
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True) # Close button-row


    # --- Section for icons and descriptions ---
    st.markdown("<div class='feature-grid'>", unsafe_allow_html=True)
    
    # First Image: Users vast amount satisfied (https://cdn-icons-png.flaticon.com/512/33/33308.png)
    st.markdown(f"""
        <div class="feature-item">
            <img src="https://cdn-icons-png.flaticon.com/512/33/33308.png" class="icon-img">
            <h3>Users Vastly Satisfied</h3>
            <p>Our community experiences significant positive emotional shifts.</p>
        </div>
    """, unsafe_allow_html=True)

    # Second Image: Successful User's Emotion's results a happy lifestyle (https://cdn-icons-png.flaticon.com/512/10809/10809501.png)
    st.markdown(f"""
        <div class="feature-item">
            <img src="https://cdn-icons-png.flaticon.com/512/10809/10809501.png" class="icon-img">
            <h3>Happy Lifestyle</h3>
            <p>Empowering users to achieve emotional well-being and a happier life.</p>
        </div>
    """, unsafe_allow_html=True)

    # Third Image: 10,000+ Daily Users (https://www.clipartmax.com/png/middle/225-2254363_checklist-comments-form-approved-icon.png)
    st.markdown(f"""
        <div class="feature-item">
            <img src="https://www.clipartmax.com/png/middle/225-2254363_checklist-comments-form-appr.png" class="icon-img">
            <h3>10,000+ Daily Users</h3>
            <p>Join our growing community finding support and growth daily.</p>
        </div>
    """, unsafe_allow_html=True)

    # Re-adding the original Empathetic Chatbot feature from previous response for completeness
    st.markdown("""
        <div class="feature-item">
            <span class="icon-img"></span> <h3>Empathetic Chatbot</h3>
            <p>Connect with an AI that listens and understands your feelings.</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    # --- End of icon and description section ---


    st.markdown("<div class='auth-content-container'>", unsafe_allow_html=True) # Container for login/signup

    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        st.subheader("Login to Your Account( Admin will use own creditionals )")
        login_username = st.text_input("Username", key="login_user")
        login_password = st.text_input("Password", type="password", key="login_pass")
        
        if st.button("Login", key="login_btn"):
            if login_username and login_password:
                success, message, is_admin = login(login_username, login_password)
                if success:
                    st.session_state.authenticated = True
                    st.session_state.username = login_username
                    st.session_state.is_admin = is_admin
                    st.session_state.page = "main_app" if not is_admin else "admin_dashboard" # Direct to correct page
                    st.rerun()
                else:
                    st.error(message)
            else:
                st.warning("Please fill in all fields")
    
    with tab2:
        st.subheader("Create New Account")
        signup_username = st.text_input("Choose Username", key="signup_user")
        signup_email = st.text_input("Email Address", key="signup_email")
        signup_password = st.text_input("Choose Password", type="password", key="signup_pass")
        signup_confirm = st.text_input("Confirm Password", type="password", key="signup_confirm")
        
        if st.button("Create Account", key="signup_btn"):
            if all([signup_username, signup_email, signup_password, signup_confirm]):
                if signup_password != signup_confirm:
                    st.error("Passwords don't match!")
                elif len(signup_password) < 6:
                    st.error("Password must be at least 6 characters long!")
                else:
                    success, message = signup(signup_username, signup_password, signup_email)
                    if success:
                        st.success(message)
                        st.info("You can now login with your credentials!")
                    else:
                        st.error(message)
            else:
                st.warning("Please fill in all fields")

    st.markdown("</div>", unsafe_allow_html=True)
    
# Main app functions
def build_user_vectorstore(username, quotes):
    """Build and save user-specific vectorstore"""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(quotes, embedding=embeddings)
    
    # Save vectorstore for user
    vectorstore_path = get_user_file_path(username, "vectorstore")
    vectorstore.save_local(vectorstore_path)
    return vectorstore

def load_user_vectorstore(username):
    """Load user-specific vectorstore"""
    vectorstore_path = get_user_file_path(username, "vectorstore")
    if os.path.exists(vectorstore_path):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
    return None

def save_user_journal(username, user_input, emotion, score, response):
    """Save journal entry for specific user"""
    journal_path = get_user_file_path(username, "journal.json")
    entry = {
        "date": str(datetime.date.today()),
        "timestamp": str(datetime.datetime.now()),
        "user_input": user_input,
        "emotion": emotion,
        "confidence": round(score * 100, 2),
        "response": response
    }
    
    journal = []
    if os.path.exists(journal_path):
        with open(journal_path, "r") as f:
            journal = json.load(f)
    
    journal.append(entry)
    with open(journal_path, "w") as f:
        json.dump(journal, f, indent=4)

def load_user_journal(username):
    """Load journal for specific user"""
    journal_path = get_user_file_path(username, "journal.json")
    if os.path.exists(journal_path):
        with open(journal_path, "r") as f:
            return json.load(f)
    return []

def get_admin_stats():
    """Get comprehensive admin statistics"""
    users = load_users()
    stats = {
        "total_users": len(users),
        "users_today": 0,
        "users_this_week": 0,
        "total_conversations": 0,
        "active_users": 0,
        "user_details": []
    }
    
    today = datetime.date.today()
    week_ago = today - datetime.timedelta(days=7)
    
    for username, user_data in users.items():
        created_date = datetime.datetime.fromisoformat(user_data["created_at"]).date()
        
        # Count registrations
        if created_date == today:
            stats["users_today"] += 1
        if created_date >= week_ago:
            stats["users_this_week"] += 1
        
        # Get user journal stats
        journal_data = load_user_journal(username)
        conversation_count = len(journal_data)
        stats["total_conversations"] += conversation_count
        
        if conversation_count > 0:
            stats["active_users"] += 1
            last_activity = journal_data[-1]["date"] if journal_data else "Never"
        else:
            last_activity = "Never"
        
        # Get emotion breakdown
        emotions = [entry['emotion'] for entry in journal_data]
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        most_common_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else "None"
        
        stats["user_details"].append({
            "username": username,
            "email": user_data["email"],
            "joined": created_date.strftime("%Y-%m-%d"),
            "conversations": conversation_count,
            "last_activity": last_activity,
            "most_common_emotion": most_common_emotion.capitalize(),
            "emotions_breakdown": emotion_counts
        })
    
    return stats

def log_admin_activity(action, details=""):
    """Log admin activities"""
    admin_log_path = "data/admin_log.json"  # Persistent and visible
    log_entry = {
        "timestamp": str(datetime.datetime.now()),
        "action": action,
        "details": details,
        "admin": st.session_state.username
    }
    
    admin_log = []
    if os.path.exists(admin_log_path):
        with open(admin_log_path, "r") as f:
            admin_log = json.load(f)
    
    admin_log.append(log_entry)
    
    # Keep only last 100 entries
    admin_log = admin_log[-100:]
    
    os.makedirs("data", exist_ok=True)
    with open(admin_log_path, "w") as f:
        json.dump(admin_log, f, indent=4)

def get_admin_logs():
    """Get admin activity logs"""
    admin_log_path = "data/admin_log.json"
    if os.path.exists(admin_log_path):
        with open(admin_log_path, "r") as f:
            return json.load(f)
    return []

def is_crisis(text):
    """Check for crisis keywords"""
    return any(phrase in text.lower() for phrase in CRISIS_KEYWORDS)

def show_admin_dashboard():
    """Admin dashboard for monitoring users and app usage"""
    st.set_page_config(page_title="DilBot Admin Dashboard", page_icon="👑", layout="wide")

    # --- ENHANCED CUSTOM CSS FOR ADMIN DASHBOARD (Consistent with main app) ---
    st.markdown("""
    <style>
    /* Global Styles & Background (Consistent with main app) */
    .stApp {
        background: linear-gradient(to bottom right, #f8f9fa, #e9ecef);
        color: #343a40;
        font-family: 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        -webkit-font-smoothing: antialiased;
        padding-top: 2rem;
    }
    /* Streamlit's main block container for content centering and width */
    .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    /* Hide Streamlit's header/footer if you want a fully custom layout */
    .stApp > header {
        display: none;
    }
    .stApp > footer {
        display: none;
    }
    /* Header & Titles (Consistent with main app) */
    h1, h2, h3, h4, h5, h6 {
        color: #212529;
        margin-top: 1.8rem;
        margin-bottom: 0.9rem;
        font-weight: 700;
    }
    h1 {
        font-size: 2.8em;
        font-weight: 800;
        color: #5d6dbe;
        letter-spacing: -0.8px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.05);
    }
    .header-tagline { /* Added for general sub-titles */
        font-size: 1.2em;
        color: #6c757d;
        margin-top: -0.5rem;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }
    h2 {
        font-size: 2em;
        font-weight: 700;
        border-bottom: 2px solid #e9ecef;
        padding-bottom: 0.7rem;
        margin-bottom: 2rem;
        color: #343a40;
    }
    h3 {
        font-size: 1.6em;
        font-weight: 600;
        color: #495057;
        margin-top: 2rem;
        margin-bottom: 1.2rem;
    }
    h4 { /* Added for sub-sections */
        font-size: 1.3em;
        font-weight: 600;
        color: #343a40;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    h5 {
        font-size: 1.1em;
        font-weight: 600;
        color: #495057;
        margin-bottom: 1rem;
    }
    /* Metrics (Consistent with main app) */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
        text-align: center;
        margin-bottom: 1.5rem;
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 30px rgba(0,0,0,0.12);
    }
    [data-testid="stMetricLabel"] {
        font-size: 1em;
        color: #6c757d;
        margin-bottom: 8px;
        font-weight: 500;
    }
    [data-testid="stMetricValue"] { /* Overriding the admin-specific green with main app blue */
        font-size: 2.8em;
        font-weight: 800;
        color: #5d6dbe; /* Main app's primary blue */
    }
    /* Buttons (Consistent with main app) */
    .stButton>button {
        background-color: #5d6dbe;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 25px;
        font-size: 1.1em;
        font-weight: bold;
        transition: background-color 0.3s ease, transform 0.2s ease, box-shadow 0.3s ease;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        cursor: pointer;
        letter-spacing: 0.5px;
    }
    .stButton>button:hover {
        background-color: #4a5c9d;
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    }
    .stButton>button:active {
        background-color: #3b4b80;
        transform: translateY(0);
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    }
    /* Logout button specific style */
    [key="admin_logout"] > button {
        background-color: #dc3545;
        box-shadow: 0 5px 15px rgba(220,53,69,0.2);
    }
    [key="admin_logout"] > button:hover {
        background-color: #c82333;
        box-shadow: 0 8px 20px rgba(220,53,69,0.3);
    }
    /* "Back to Main App" button style */
    [key="admin_back_btn"] > button {
        background-color: #6c757d; /* Grey for back button */
        box-shadow: 0 5px 15px rgba(108,117,125,0.2);
    }
    [key="admin_back_btn"] > button:hover {
        background-color: #5a6268;
        box-shadow: 0 8px 20px rgba(108,117,125,0.3);
    }
    /* Reset Data buttons for users */
    .stButton > button[kind="secondary"] { /* Target buttons that are not primary, e.g., Reset/Confirm */
        background-color: #ffc107; /* Warning yellow */
        color: #343a40; /* Dark text on yellow */
        border: none;
    }
    .stButton > button[kind="secondary"]:hover {
        background-color: #e0a800; /* Darker yellow on hover */
        transform: translateY(-2px);
    }
    .stButton > button[key*="confirm_reset_"] {
        background-color: #dc3545; /* Danger red for confirm reset */
        color: white;
    }
    .stButton > button[key*="confirm_reset_"]:hover {
        background-color: #c82333;
    }
    .stButton > button[key*="cancel_reset_"] {
        background-color: #6c757d; /* Grey for cancel */
        color: white;
    }
    .stButton > button[key*="cancel_reset_"]:hover {
        background-color: #5a6268;
    }
    /* Text Inputs and Text Areas (Consistent with main app) */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea, .stNumberInput>div>div>input {
        border-radius: 10px;
        border: 1px solid #ced4da;
        padding: 14px 18px;
        font-size: 1.05em;
        color: #343a40;
        background-color: #000000;
        box-shadow: inset 0 2px 5px rgba(0,0,0,0.03);
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }
    .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus, .stNumberInput>div>div>input:focus {
        border-color: #5d6dbe;
        box-shadow: 0 0 0 0.25rem rgba(93,109,190,0.25);
        outline: none;
    }
    .stTextInput>label, .stTextArea>label, .stNumberInput>label {
        font-weight: 600;
        color: #495057;
        margin-bottom: 0.6rem;
    }
    /* Information, Success, Error, Warning Boxes (Consistent with main app) */
    [data-testid="stAlert"] {
        border-radius: 10px;
        padding: 18px 25px;
        margin-bottom: 1.8rem;
        font-size: 1.05em;
        line-height: 1.6;
        text-align: left;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    [data-testid="stAlert"] .streamlit-warning {
        background-color: #fef7e0; color: #7a5f00; border-left: 6px solid #ffcc00;
    }
    [data-testid="stAlert"] .streamlit-success {
        background-color: #e6ffe6;
        border-left: 6px solid #4CAF50;
        color: black !important;
    }
    [data-testid="stAlert"] .streamlit-success p,
    [data-testid="stAlert"] .streamlit-success span,
    [data-testid="stAlert"] .streamlit-success div,
    [data-testid="stAlert"] .streamlit-success strong {
        color: black !important;
        -webkit-text-fill-color: black !important;
        opacity: 1 !important;
    }
    [data-testid="stAlert"] .streamlit-error {
        background-color: #ffe6e6; color: #8c0a0a; border-left: 6px solid #e74c3c;
    }
    .stInfo { /* Ensure st.info is also styled consistently */
        background-color: #e6f7ff;
        border-left: 6px solid #64b5f6;
        color: black !important;
        border-radius: 10px;
        padding: 18px 25px;
        margin-top: 2rem;
        font-style: italic;
        font-size: 1.05em;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    .stInfo p, .stInfo span, .stInfo strong, .stInfo div {
        color: black !important;
        -webkit-text-fill-color: black !important;
        opacity: 1 !important;
    }
    /* Container for sections (Consistent with main app) */
    .stContainer {
        border-radius: 12px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
        border: 1px solid #e0e0e0;
        padding: 2rem;
        margin-bottom: 2.5rem;
        background-color: #ffffff;
    }
    .stContainer.has-border {
        border: 1px solid #e0e0e0;
    }
    /* Expander styling (Consistent with main app) */
    .streamlit-expanderHeader {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 12px 18px;
        margin-bottom: 0.8rem;
        cursor: pointer;
        transition: background-color 0.2s ease, box-shadow 0.2s ease;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    }
    .streamlit-expanderHeader:hover {
        background-color: #f7f9fb;
        box-shadow: 0 6px 15px rgba(0,0,0,0.08);
    }
    .streamlit-expanderHeader > div > div > p {
        font-weight: 600;
        color: #343a40;
        font-size: 1.05em;
    }
    .streamlit-expanderContent {
        background-color: #f8f9fa;
        border-left: 4px solid #ced4da;
        padding: 15px 20px;
        border-bottom-left-radius: 10px;
        border-bottom-right-radius: 10px;
        margin-top: -10px;
        box-shadow: inset 0 2px 5px rgba(0,0,0,0.03);
        line-height: 1.6;
    }
    /* Ensure text within expander content is black */
    .streamlit-expanderContent p,
    .streamlit-expanderContent span,
    .streamlit-expanderContent div,
    .streamlit-expanderContent strong {
        color: black !important;
        -webkit-text-fill-color: black !important;
        opacity: 1 !important;
    }
    /* Table styling (for dataframes) */
    .stDataFrame {
        border: 1px solid #e9ecef;
        border-radius: 10px;
        overflow: hidden; /* Ensures rounded corners */
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    }
    /* General text color in case Streamlit overrides */
    div[data-testid="stVerticalBlock"] div > p,
    div[data-testid="stHorizontalBlock"] div > p,
    div[data-testid="stText"] p {
        color: #343a40 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    # --- END CUSTOM CSS ---

    # Header
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"<h1>👑 DilBot Admin Dashboard</h1>", unsafe_allow_html=True)
        st.markdown("<p class='header-tagline'>Monitor users, conversations, and app analytics</p>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True) # Spacer
        if st.button("Logout", key="admin_logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.is_admin = False
            st.rerun()
    
    # Log admin access
    log_admin_activity("Dashboard Access", "Viewed admin dashboard")
    
    # Get statistics
    stats = get_admin_stats()
    
    # Overview metrics
    st.markdown("<h2> Overview</h2>", unsafe_allow_html=True)
    with st.container(border=True): # Wrap metrics in a container for styling
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Users", stats["total_users"])
        with col2:
            st.metric("Active Users", stats["active_users"])
        with col3:
            st.metric("New Today", stats["users_today"])
        with col4:
            st.metric("Total Conversations", stats["total_conversations"])
    
    # User registration trend
    st.markdown("<h2> User Registration Trend</h2>", unsafe_allow_html=True)
    with st.container(border=True): # Wrap chart in a container
        if stats["user_details"]:
            # Create registration data
            reg_data = {}
            for user in stats["user_details"]:
                date = user["joined"]
                reg_data[date] = reg_data.get(date, 0) + 1
            
            # Convert to DataFrame for Altair compatibility and sorting
            chart_data_list = [{"date": date, "registrations": count} for date, count in reg_data.items()]
            chart_df = pd.DataFrame(chart_data_list).sort_values("date")

            if not chart_df.empty:
                chart = alt.Chart(chart_df).mark_line(point=True).encode(
                    x=alt.X('date:T', title='Date'),
                    y=alt.Y('registrations:Q', title='New Registrations'),
                    tooltip=[alt.Tooltip('date:T', title='Date'), 'registrations:Q']
                ).properties(
                    title="Daily User Registrations"
                ).interactive() # Make interactive for zoom/pan
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("No registration data available to display trend.")
        else:
            st.info("No user data available to display registration trend.")
    
    # Detailed user table
    st.markdown("<h2>👥 User Details</h2>", unsafe_allow_html=True)
    with st.container(border=True): # Wrap user details section in a container
        # Search and filter
        col1, col2 = st.columns([2, 1])
        with col1:
            search_term = st.text_input(" Search users", placeholder="Search by username or email", label_visibility="visible")
        with col2:
            min_conversations = st.number_input("Min conversations", min_value=0, value=0, label_visibility="visible")
        
        # Filter users
        filtered_users = stats["user_details"]
        if search_term:
            filtered_users = [u for u in filtered_users if
                              search_term.lower() in u["username"].lower() or
                              search_term.lower() in u["email"].lower()]
        
        if min_conversations > 0:
            filtered_users = [u for u in filtered_users if u["conversations"] >= min_conversations]
        
        # Display user table
        if filtered_users:
            for user in filtered_users:
                with st.expander(f"👤 **{user['username']}** ({user['conversations']} conversations)"): # Bold username
                    col1_detail, col2_detail = st.columns(2) # Renamed columns to avoid conflict
                    
                    with col1_detail:
                        st.write(f"**Email:** {user['email']}")
                        st.write(f"**Joined:** {user['joined']}")
                        st.write(f"**Last Activity:** {user['last_activity']}")
                    
                    with col2_detail:
                        st.write(f"**Conversations:** {user['conversations']}")
                        st.write(f"**Most Common Emotion:** {user['most_common_emotion'].capitalize()}") # Capitalize emotion
                        
                        # Show emotion breakdown
                        if user['emotions_breakdown']:
                            st.write("**Emotion Breakdown:**")
                            # Convert to DataFrame for a nicer table display
                            emotions_df = pd.DataFrame([{"Emotion": e.capitalize(), "Count": c} for e, c in user['emotions_breakdown'].items()])
                            st.dataframe(emotions_df, hide_index=True, use_container_width=True)
                        else:
                            st.info("No emotion data for this user.")
                    
                    st.markdown("---") # Separator for actions
                    # Quick actions
                    col1_actions, col2_actions, col3_actions = st.columns(3) # Renamed columns
                    with col1_actions:
                        if st.button(f"View Journal", key=f"view_{user['username']}", use_container_width=True):
                            # Show user's recent conversations
                            user_journal = load_user_journal(user['username'])
                            if user_journal:
                                st.subheader(f"Recent conversations for {user['username']}")
                                for entry in user_journal[-5:]: # Last 5 activities
                                    st.text_area(
                                        f"{entry['date']} - {entry['emotion'].capitalize()} ({round(entry['confidence'] * 100)}/ confidence)",
                                        f"User: {entry['user_input']}\nDilBot: {entry['response']}",
                                        height=100,
                                        disabled=True,
                                        key=f"journal_entry_{user['username']}_{entry['date']}_{hash(entry['user_input'])}" # Unique key for text_area
                                    )
                            else:
                                st.info("No conversations found for this user.")
                    
                    with col2_actions:
                        reset_key = f"reset_{user['username']}"
                        confirm_key = f"confirm_{user['username']}"
                        
                        # Use a persistent state for confirmation
                        if confirm_key not in st.session_state:
                            st.session_state[confirm_key] = False

                        if st.button(f"Reset Data", key=reset_key, use_container_width=True):
                            st.session_state[confirm_key] = True  # Flag to show confirmation
                            st.rerun() # Rerun to display confirmation message/buttons
                        
                        if st.session_state.get(confirm_key, False):
                            st.warning(f"Are you sure you want to reset ALL data for {user['username']}? This action is irreversible.")
                            col_confirm_yes, col_confirm_no = st.columns(2)
                            with col_confirm_yes:
                                if st.button(f"Yes, Reset", key=f"confirm_reset_{user['username']}", use_container_width=True):
                                    # Clear user's journal
                                    journal_path = get_user_file_path(user['username'], "journal.json")
                                    if os.path.exists(journal_path):
                                        os.remove(journal_path)
                                        # Also remove vectorstore if it exists
                                        vectorstore_path = get_user_file_path(user['username'], "faiss_index")
                                        if os.path.exists(vectorstore_path):
                                            import shutil
                                            shutil.rmtree(vectorstore_path) # Remove directory for FAISS
                                        
                                    log_admin_activity("User Data Reset", f"Reset data for {user['username']}")
                                    st.success(f"Data reset for {user['username']}!")
                                    st.session_state[confirm_key] = False  # Reset confirmation flag
                                    st.rerun()
                            with col_confirm_no:
                                if st.button(f"Cancel", key=f"cancel_reset_{user['username']}", use_container_width=True):
                                    st.session_state[confirm_key] = False  # Cancel confirmation
                                    st.info(f"Reset for {user['username']} cancelled.")
                                    st.rerun() # Rerun to hide confirmation buttons

                    #with col3_actions:
                        # Placeholder for other actions like "Deactivate User"
                        #st.button("Deactivate User (WIP)", key=f"deactivate_{user['username']}", disabled=True, use_container_width=True)

        else:
            st.info("No users found matching your criteria.")
    
    # System Analytics
    st.markdown("<h2> System Analytics</h2>", unsafe_allow_html=True)
    with st.container(border=True): # Wrap system analytics in a container
        col1_analytics, col2_analytics = st.columns(2) # Renamed columns
        
        with col1_analytics:
            st.markdown("<h4>Emotion Distribution (All Users)</h4>", unsafe_allow_html=True)
            # Aggregate all emotions
            all_emotions = {}
            for user in stats["user_details"]:
                for emotion, count in user['emotions_breakdown'].items():
                    all_emotions[emotion] = all_emotions.get(emotion, 0) + count
            
            if all_emotions:
                emotion_chart_data = [{"emotion": emotion.capitalize(), "count": count}
                                      for emotion, count in all_emotions.items()]
                
                emotion_chart = alt.Chart(pd.DataFrame(emotion_chart_data)).mark_bar().encode(
                    x=alt.X('emotion:N', title='Emotion', sort='-y'), # Sort by count descending
                    y=alt.Y('count:Q', title='Frequency'),
                    color=alt.Color('emotion:N', legend=None, scale=alt.Scale(
                        range=['#4CAF50', '#FFC107', '#E74C3C', '#3498DB', '#9B59B6', '#1ABC9C', '#FF5733'])), # More colors
                    tooltip=['emotion:N', 'count:Q']
                ).properties(
                    title="Overall Emotion Distribution"
                ).interactive()
                st.altair_chart(emotion_chart, use_container_width=True)
            else:
                st.info("No emotion data available for overall analysis.")
        
        with col2_analytics:
            st.markdown("<h4>User Activity Levels</h4>", unsafe_allow_html=True)
            activity_levels = {"Inactive (0)": 0, "Light (1-5)": 0, "Moderate (6-20)": 0, "Heavy (21+)": 0}
            
            for user in stats["user_details"]:
                conv_count = user["conversations"]
                if conv_count == 0:
                    activity_levels["Inactive (0)"] += 1
                elif conv_count <= 5:
                    activity_levels["Light (1-5)"] += 1
                elif conv_count <= 20:
                    activity_levels["Moderate (6-20)"] += 1
                else:
                    activity_levels["Heavy (21+)"] += 1
            
            activity_data = [{"level": level, "users": count} for level, count in activity_levels.items()]
            
            if activity_data:
                activity_chart = alt.Chart(pd.DataFrame(activity_data)).mark_arc(outerRadius=120, innerRadius=80).encode( # Donut chart
                    theta=alt.Theta('users:Q'),
                    color=alt.Color('level:N', title="Activity Level", scale=alt.Scale(
                        range=['#6c757d', '#64b5f6', '#5d6dbe', '#4a5c9d'])), # Consistent color scheme
                    order=alt.Order('users:Q', sort='descending'),
                    tooltip=['level:N', 'users:Q']
                ).properties(
                    title="User Activity Distribution"
                ).interactive()
                st.altair_chart(activity_chart, use_container_width=True)
            else:
                st.info("No user activity data to display.")
    
    # Admin logs
    st.markdown("<h2> Admin Activity Logs</h2>", unsafe_allow_html=True)
    with st.container(border=True): # Wrap admin logs in a container
        admin_logs = get_admin_logs()
        
        if admin_logs:
            st.subheader("Recent Admin Activities (Last 10)")
            # Display logs in a more readable format, perhaps a table or structured text
            for log_entry in reversed(admin_logs[-10:]): # Last 10 activities, reversed to show newest first
                timestamp = datetime.datetime.fromisoformat(log_entry["timestamp"])
                st.markdown(f"**{timestamp.strftime('%Y-%m-%d %H:%M:%S')}** - **{log_entry['action']}**: `{log_entry['details']}`")
                st.markdown("---") # Small separator
        else:
            st.info("No admin activities logged yet.")
    
    # Data Export and Admin Log Clearing
    st.markdown("<h2>Data Management</h2>", unsafe_allow_html=True)
    with st.container(border=True): # Wrap data export in a container
        col1_export, col2_export = st.columns(2) # Renamed columns
        
        with col1_export:
            st.markdown("<h4>Export Application Data</h4>", unsafe_allow_html=True)
            if st.button("Export All Data (JSON)", key="export_data_btn", use_container_width=True):
                export_data = {
                    "export_timestamp": str(datetime.datetime.now()),
                    "statistics": stats,
                    "admin_logs": admin_logs
                }
                
                st.download_button(
                    label="Download Exported Data",
                    data=json.dumps(export_data, indent=4),
                    file_name=f"dilbot_admin_export_{datetime.date.today().isoformat()}.json",
                    mime="application/json",
                    key="download_export_btn",
                    use_container_width=True
                )
                
                log_admin_activity("Data Export", "Initiated data export")
                st.success("Data export ready for download!")
        
        with col2_export:
            st.markdown("<h4>Clear Admin Activity Logs</h4>", unsafe_allow_html=True)
            # Add a confirmation step for clearing logs
            clear_log_confirm_key = "clear_log_confirm"
            if clear_log_confirm_key not in st.session_state:
                st.session_state[clear_log_confirm_key] = False

            if st.button("Clear Admin Logs", key="clear_admin_logs_btn", use_container_width=True):
                st.session_state[clear_log_confirm_key] = True
                st.rerun()

            if st.session_state.get(clear_log_confirm_key, False):
                st.warning("Are you sure you want to clear ALL admin logs? This action is irreversible.")
                col_clear_yes, col_clear_no = st.columns(2)
                with col_clear_yes:
                    if st.button("Yes, Clear Logs", key="confirm_clear_logs_btn", use_container_width=True):
                        admin_log_path = "data/admin_log.json"
                        if os.path.exists(admin_log_path):
                            os.remove(admin_log_path)
                        st.success("Admin logs cleared successfully!")
                        log_admin_activity("Admin Logs Cleared", "All admin activity logs were cleared")
                        st.session_state[clear_log_confirm_key] = False
                        st.rerun()
                with col_clear_no:
                    if st.button("Cancel", key="cancel_clear_logs_btn", use_container_width=True):
                        st.session_state[clear_log_confirm_key] = False
                        st.info("Clearing admin logs cancelled.")
                        st.rerun()

    st.markdown("---")
    # Back to main app button (aligned left in a container)
    


    st.markdown("<p class='footer-caption'>DilBot Admin Panel | Built by Members of CSG Hackathon Team</p>", unsafe_allow_html=True)
            
def speak(text, username):
    """Generate and play audio response"""
    tts = gTTS(text=text, lang='en')
    audio_path = get_user_file_path(username, "response.mp3")
    tts.save(audio_path)
    st.audio(audio_path, format="audio/mp3")


def transcribe_audio_file(uploaded_audio):
    """Transcribe uploaded audio file"""
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(uploaded_audio) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return text
    except Exception as e:
        return f"Error: {str(e)}"

def show_main_app():
    """Main DilBot application"""
    username = st.session_state.username

    st.set_page_config(page_title="DilBot - Emotional AI", page_icon="🧠", layout="wide")

    
    quote_categories = {
        "Grief": ["Grief is the price we pay for love.", "Tears are the silent language of grief.", "What we have once enjoyed we can never lose; all that we love deeply becomes a part of us."],
        "Motivation": ["Believe in yourself and all that you are.", "Tough times never last, but tough people do.", "The only way to do great work is to love what you do."],
        "Healing": ["Every wound has its own time to heal.", "It's okay to take your time to feel better.", "Healing is not linear, and that's perfectly okay."],
        "Relationships": ["The best relationships are built on trust.", "Love is not about possession but appreciation.", "Healthy relationships require both people to show up authentically."]
    }

    # ---  CUSTOM CSS  ---
    st.markdown("""
    <style>
    /* Global Styles & Background */
    .stApp {
        background: linear-gradient(to bottom right, #f8f9fa, #e9ecef);
        color: #343a40;
        font-family: 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        -webkit-font-smoothing: antialiased;
        padding-top: 2rem;
    }
    /* Streamlit's main block container for content centering and width */
    .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    /* Hide Streamlit's header/footer if you want a fully custom layout */
    .stApp > header {
        display: none;
    }
    .stApp > footer {
        display: none;
    }
    /* Header & Titles */
    h1, h2, h3, h4, h5, h6 {
        color: #212529;
        margin-top: 1.8rem;
        margin-bottom: 0.9rem;
        font-weight: 700;
    }
    h1 {
        font-size: 2.8em;
        font-weight: 800;
        color: #5d6dbe;
        letter-spacing: -0.8px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.05);
    }
    .header-tagline {
        font-size: 1.2em;
        color: #6c757d;
        margin-top: -0.5rem;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }
    h2 {
        font-size: 2em;
        font-weight: 700;
        border-bottom: 2px solid #e9ecef;
        padding-bottom: 0.7rem;
        margin-bottom: 2rem;
        color: #343a40;
    }
    h3 {
        font-size: 1.6em;
        font-weight: 600;
        color: #495057;
        margin-top: 2rem;
        margin-bottom: 1.2rem;
    }
    h5 {
        font-size: 1.1em;
        font-weight: 600;
        color: #495057;
        margin-bottom: 1rem;
    }
    /* Metrics (Total Conversations, Most Common Emotion, Avg. Confidence) */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
        text-align: center;
        margin-bottom: 1.5rem;
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 30px rgba(0,0,0,0.12);
    }
    [data-testid="stMetricLabel"] {
        font-size: 1em;
        color: #6c757d;
        margin-bottom: 8px;
        font-weight: 500;
    }
    [data-testid="stMetricValue"] {
        font-size: 2.8em;
        font-weight: 800;
        color: #5d6dbe;
    }
    /* Buttons */
    .stButton>button {
        background-color: #5d6dbe;
        color: black;
        border: none;
        border-radius: 10px;
        padding: 12px 25px;
        font-size: 1.1em;
        font-weight: bold;
        transition: background-color 0.3s ease, transform 0.2s ease, box-shadow 0.3s ease;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        cursor: pointer;
        letter-spacing: 0.5px;
    }
    .stButton>button:hover {
        background-color: #4a5c9d;
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    }
    .stButton>button:active {
        background-color: #3b4b80;
        transform: translateY(0);
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    }
    /* Logout button specific style */
    [key="logout_btn"] > button {
        background-color: #dc3545;
        box-shadow: 0 5px 15px rgba(220,53,69,0.2);
    }
    [key="logout_btn"] > button:hover {
        background-color: #c82333;
        box-shadow: 0 8px 20px rgba(220,53,69,0.3);
    }
    /* Text Inputs and Text Areas */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        border-radius: 10px;
        border: 1px solid #ced4da;
        padding: 14px 18px;
        font-size: 1.05em;
        color: #343a40;
        background-color: #fcfcfc;
        box-shadow: inset 0 2px 5px rgba(0,0,0,0.03);
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }
    .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
        border-color: #5d6dbe;
        box-shadow: 0 0 0 0.25rem rgba(93,109,190,0.25);
        outline: none;
    }
    .stTextInput>label, .stTextArea>label {
        font-weight: 600;
        color: #495057;
        margin-bottom: 0.6rem;
    }
    /* Selectbox (Dropdown) Text Color Fixes */
    [data-testid="stSelectbox"] > div:first-child > div:first-child {
        border-radius: 10px;
        border: 1px solid #ced4da;
        background-color: #fcfcfc;
        box-shadow: inset 0 2px 5px rgba(0,0,0,0.03);
        display: flex;
        align-items: center;
        min-height: 48px;
        padding: 0 10px;
    }
    .stSelectbox>label {
        font-weight: 600;
        color: #495057;
        margin-bottom: 0.6rem;
    }
    /* Force selected value text to black */
    [data-testid="stSelectbox"] input[type="text"] {
        color: black !important;
        -webkit-text-fill-color: black !important; /* For Webkit browsers */
        opacity: 1 !important;
        flex-grow: 1;
        padding: 12px 5px;
        font-size: 1.05em;
        line-height: 1.2em;
        min-height: 1.2em;
        background-color: transparent !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }
    /* Force dropdown arrow to black */
    [data-testid="stSelectbox"] button {
        background-color: transparent !important;
        border: none !important;
        padding: 0 5px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    [data-testid="stSelectbox"] svg {
        fill: black !important;
        color: black !important;
        font-size: 1.5em !important;
        width: 1em !important;
        height: 1em !important;
    }
    [data-testid="stSelectbox"] svg path {
        fill: black !important;
    }
     /* File Uploader with White Text */
    [data-testid="stFileUploaderDropzone"] {
    border-radius: 12px;
    border: 2px dashed #a0a8b4;
    background-color: #2c3e50; /* Dark background to contrast with white text */
    padding: 25px;
    transition: border-color 0.3s ease, background-color 0.3s ease;
    margin-bottom: 1.5rem;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
    border-color: #5d6dbe;
    background-color: #34495e; /* Slightly lighter on hover */
    }
    /* Force ALL text inside file uploader dropzone to WHITE */
    [data-testid="stFileUploaderDropzone"] *,
    [data-testid="stFileUploaderDropzone"] p,
    [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploaderDropzone"] div,
    [data-testid="stFileUploaderDropzone"] small {
    color: white !important;
    -webkit-text-fill-color: white !important;
    opacity: 1 !important;
    font-weight: normal !important;
    }
    /* Specific targeting for drag and drop text */
    [data-testid="stFileUploaderDropzone"] > div > div > div:nth-child(2) > div:first-child {
    color: white !important;
    }
    [data-testid="stFileUploaderDropzone"] > div > div > div > span {
    color: white !important;
    }
    /* Style the Browse files button with white text */
    [data-testid="stFileUploaderDropzone"] button {
    background-color: #5d6dbe !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    font-weight: bold !important;
    transition: all 0.3s ease !important;
    }
    [data-testid="stFileUploaderDropzone"] button:hover {
    background-color: #4a5c9d !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
     }
    /* File uploader label (outside the dropzone) */
    [data-testid="stFileUploader"] label {
    color: black !important;
    -webkit-text-fill-color: black !important;
    opacity: 1 !important;
    font-weight: 600 !important;
    margin-bottom: 0.5rem !important;
     }
    /* Uploaded file names */
    [data-testid="stFileUploaderFileName"] {
    font-size: 0.95em;
    color: black !important;
    -webkit-text-fill-color: black !important;
    margin-top: 10px;
    word-break: break-all;
    background-color: rgba(255,255,255,0.1) !important;
    padding: 8px 12px !important;
    border-radius: 6px !important;
    }
    [data-testid="stFileUploaderFile"] {
    background-color: rgba(255,255,255,0.1) !important;
    border-radius: 8px;
    padding: 8px 12px;
    margin-top: 10px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    /* Style the cloud upload icon to be white */
    [data-testid="stFileUploaderDropzone"] svg {
    fill: white !important;
    color: white !important;
    }
    
    /* Information, Success, Error, Warning Boxes */
    [data-testid="stAlert"] {
        border-radius: 10px;
        padding: 18px 25px;
        margin-bottom: 1.8rem;
        font-size: 1.05em;
        line-height: 1.6;
        text-align: left;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05));
    }
    /* Warning alert style  */
    [data-testid="stAlert"] .streamlit-warning {
        background-color: #fef7e0; color: #7a5f00; border-left: 6px solid #ffcc00;
    }
    [data-testid="stAlert"] .streamlit-success {
        background-color: #9aff9a;
        border-left: 6px solid #4CAF50;
        color: black !important; 
    }
    /* Aggressive targeting for specific text elements within st.success */
    [data-testid="stAlert"] .streamlit-success p,
    [data-testid="stAlert"] .streamlit-success span,
    [data-testid="stAlert"] .streamlit-success div,
    [data-testid="stAlert"] .streamlit-success strong { /* Added strong for bold text */
        color: black !important;
        -webkit-text-fill-color: black !important;
        opacity: 1 !important;
    }
    /* Error alert style  */
    [data-testid="stAlert"] .streamlit-error {
        background-color: #ffe6e6; color: #8c0a0a; border-left: 6px solid #e74c3c;
    }
    .stInfo {
    background-color: #e6f7ff;
    border-left: 6px solid #64b5f6;
    color: black !important; /* Changed from black to white */
    border-radius: 10px;
    padding: 18px 25px;
    margin-top: 2rem;
    font-style: italic;
    font-size: 1.05em;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    /* Aggressive targeting for specific text elements within st.info */
    .stInfo p,
    .stInfo span,
    .stInfo strong,
    .stInfo div {
    color: white !important;
    -webkit-text-fill-color: white !important;
    opacity: 1 !important;
     }
    /* Chat Message Bubbles  */
    .user-message-container {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 15px;
    }
    .user-message {
        background-color: #e0e6f6;
        color: #343a40;
        border-radius: 18px 18px 5px 18px;
        padding: 12px 18px;
        max-width: 75%;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
        line-height: 1.5;
        text-align: left;
    }
    .bot-message-container {
        display: flex;
        justify-content: flex-start;
        margin-bottom: 15px;
    }
    .bot-message {
        background-color: #ffffff;
        color: #343a40;
        border-radius: 18px 18px 18px 5px;
        padding: 12px 18px;
        max-width: 75%;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
        line-height: 1.5;
        text-align: left;
    }
    .chat-separator {
        border-bottom: 1px dashed #ced4da;
        margin: 25px 0;
    }
    .chat-title {
        font-size: 1.8em;
        font-weight: 700;
        color: #5d6dbe;
        text-align: center;
        margin-bottom: 2rem;
    }
    /* Audio Player (kept original styles) */
    .stAudio {
        margin-top: 1.5rem;
        margin-bottom: 2rem;
    }
    /* Expander styling (for conversations - kept original styles) */
    .streamlit-expanderHeader {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 12px 18px;
        margin-bottom: 0.8rem;
        cursor: pointer;
        transition: background-color 0.2s ease, box-shadow 0.2s ease;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    }
    .streamlit-expanderHeader:hover {
        background-color: #f7f9fb;
        box-shadow: 0 6px 15px rgba(0,0,0,0.08);
    }
    .streamlit-expanderHeader > div > div > p {
        font-weight: 600;
        color: #343a40;
        font-size: 1.05em;
    }
    .streamlit-expanderContent {
        background-color: #f8f9fa;
        border-left: 4px solid #ced4da;
        padding: 15px 20px;
        border-bottom-left-radius: 10px;
        border-bottom-right-radius: 10px;
        margin-top: -10px;
        box-shadow: inset 0 2px 5px rgba(0,0,0,0.03);
        line-height: 1.6;
    }
    /* Horizontal rule (kept original styles) */
    hr {
        border-top: 2px solid #e9ecef;
        margin-top: 3rem;
        margin-bottom: 3rem;
    }
    /* Caption (footer - kept original styles) */
    .footer-caption {
        color: #868e96;
        font-size: 0.9em;
        margin-top: 3rem;
        display: block;
        text-align: center;
        letter-spacing: 0.2px;
    }
    /* Container for sections (kept original styles) */
    .stContainer {
        border-radius: 12px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
        border: 1px solid #e0e0e0;
        padding: 2rem;
        margin-bottom: 2.5rem;
        background-color: #ffffff;
    }
    .stContainer.has-border {
        border: 1px solid #e0e0e0;
    }
    /*confidence color*/
    .black-text {
    color: black !important;
    -webkit-text-fill-color: black !important;
    opacity: 1 !important;
     }
    .stCustomSuccess {
    background-color: #e6ffe6;
    border-left: 6px solid #4CAF50;
    border-radius: 10px;
    padding: 18px 25px;
    margin-bottom: 1.8rem;
    font-size: 1.05em;
    line-height: 1.6;
    text-align: left;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
     }
    .stCustomSuccess .black-text { 
    color: black !important;
    -webkit-text-fill-color: black !important;
    opacity: 1 !important;
      }
      /*for quotes */
      /* Custom CSS for a reusable "info-box" component */
    .custom-info-box {
    background-color: #e6f7ff;
    border-left: 6px solid #64b5f6;
    border-radius: 10px;
    padding: 18px 25px;
    margin-top: 2rem;
    font-size: 1.05em;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    display: flex;
    align-items: center;
     }
    /* Style for the icon within the custom info box */
    .custom-info-box .info-icon {
    font-size: 1.5em;
    margin-right: 10px;
    color: #64b5f6;
    }
/* Styles for the text content within the custom info box */
   .custom-info-box .info-text {
    color: black !important;
    -webkit-text-fill-color: black !important;
    opacity: 1 !important;
    font-style: italic; /* Apply italic here directly */
    margin: 0; /* Remove default paragraph margin */
      }
}
    </style>
    """, unsafe_allow_html=True)

    # --- Streamlit UI Components ---
    # Header with logout button
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"<h1>🧠 DilBot</h1>", unsafe_allow_html=True)
        st.markdown(f"<p class='header-tagline'>Welcome back, {username}! Your personal emotional AI companion</p>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True) # Spacer
        if st.button("Logout", key="logout_btn", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.rerun()

    # Input section wrapped in a styled container
    st.markdown("<h3>Input Your Thoughts</h3>", unsafe_allow_html=True)
    with st.container(border=True): # Streamlit's built-in container with border
        col1, col2 = st.columns(2)
        with col1:
            # Reverting to direct st.selectbox with a visible label
            st.markdown(" **Choose a quote theme:**") # Custom label for the selectbox
            selected_category = st.selectbox("Select Quote Theme", list(quote_categories.keys()), label_visibility="collapsed") # Original label hidden
            # The custom markdown above provides the visual label, while the st.selectbox's internal label is hidden.

        with col2:
            st.markdown("<h5> Custom Quotes & Voice Input</h5>", unsafe_allow_html=True)
            # Ensure file uploader labels are visible
            uploaded_quotes = st.file_uploader("Upload your own quotes (.txt)", type=["txt"], key="quote_uploader")
            uploaded_audio = st.file_uploader("Upload a voice message (.wav)", type=["wav"], key="audio_uploader")

            # Voice transcription button
            if uploaded_audio and st.button(" Transcribe Voice Message", key="transcribe_btn", use_container_width=True):
                with st.spinner("Transcribing your voice..."):
                    transcribed = transcribe_audio_file(uploaded_audio)
                    if transcribed.startswith("Error:"):
                        st.error(transcribed)
                    else:
                        st.session_state.transcribed_text = transcribed
                        st.success(" Voice transcribed successfully!")

    # Handle vectorstore 
    current_quotes = []
    vectorstore = None

    if uploaded_quotes:
        custom_quotes = uploaded_quotes.read().decode("utf-8").splitlines()
        custom_quotes = [quote.strip() for quote in custom_quotes if quote.strip()]
        vectorstore = build_user_vectorstore(username, custom_quotes)
        current_quotes = custom_quotes
        st.success(f" {len(custom_quotes)} custom quotes uploaded and saved!")
    else:
        default_quotes = quote_categories[selected_category]
        vectorstore = load_user_vectorstore(username)
        if vectorstore is None:
            vectorstore = build_user_vectorstore(username, default_quotes)
        current_quotes = default_quotes

    # Input area for user message
    st.markdown("<h3>What's on your mind?</h3>", unsafe_allow_html=True)
    user_input = st.text_area(
        "Share your thoughts, feelings, or experiences...",
        value=st.session_state.transcribed_text,
        height=180, # Increased height for more typing space
        placeholder="Type here or use your transcribed voice message...",

    )
    # Re-adding explicit label for text area if needed, to be safe
    # st.markdown("<label class='stTextArea>label' style='margin-bottom:0.6rem;padding-left:5px;'>Share your thoughts, feelings, or experiences...</label>", unsafe_allow_html=True)


    final_input = user_input.strip() or st.session_state.transcribed_text.strip()

    # Main interaction button
    if st.button("🧠 Talk to DilBot", type="primary", use_container_width=True):
        if not final_input:
            st.warning(" Please enter something to share or upload a voice message.")
        else:
            with st.spinner("DilBot is thinking and feeling..."):
                # Emotion detection 
                emotion, score = detect_emotion(final_input)

                # Get AI response
                prompt_template = PromptTemplate(
                    input_variables=["context", "user_input", "username"],
                    template="""You are DilBot, an empathetic emotional support AI companion for {username}.
Use the following emotional quote context to respond gently, supportively, and personally.
Context quotes:
{context}
User's message:
{user_input}
Respond as DilBot with warmth, empathy, and understanding. Keep it conversational and supportive."""
                )

                # Get similar quotes (ORIGINAL LOGIC - NO CHANGE)
                similar_docs = vectorstore.similarity_search(final_input, k=2)
                context = "\n".join([doc.page_content for doc in similar_docs])

                # Generate response (ORIGINAL LOGIC - NO CHANGE)
                groq_llm = ChatGroq(api_key=GROQ_API_KEY, model="llama3-70b-8192")
                chain = LLMChain(llm=groq_llm, prompt=prompt_template)
                response = chain.run(context=context, user_input=final_input, username=username)

                # Save to user's journal (ORIGINAL LOGIC - NO CHANGE)
                save_user_journal(username, final_input, emotion, score, response)

                # Display results with new chat bubble styling
                st.markdown("<h3 class='chat-title'>DilBot's Conversation:</h3>", unsafe_allow_html=True)
                with st.container(border=True): # Container for the conversation output
                    # User's input presented in a chat bubble
                    st.markdown(f"<div class='user-message-container'><div class='user-message'>You: {final_input}</div></div>", unsafe_allow_html=True)
                    #st.success(f"**Emotion Detected:** {emotion.capitalize()} ({round(score*100)}/ confidence)")
                    st.markdown(
                            f"""
                               <div class="stCustomSuccess">
                                    <p class="black-text">
                    <strong>Emotion Detected:</strong> {emotion.capitalize()} ({round(score*100)}% confidence)</p></div> """,unsafe_allow_html=True
                                )

                    if is_crisis(final_input):
                        st.error(" Crisis detected! Please reach out to a mental health professional immediately. "
                                 "You are not alone. Consider contacting a helpline like the National Suicide Prevention Lifeline (988 in the US) or a local emergency service.")

                    if current_quotes:
                        model = SentenceTransformer("all-MiniLM-L6-v2")
                        quote_embeddings = model.encode(current_quotes, convert_to_tensor=True)
                        user_embedding = model.encode(final_input, convert_to_tensor=True)
                        sims = util.pytorch_cos_sim(user_embedding, quote_embeddings)[0]
                        best_match = sims.argmax().item()
                        selected_quote = current_quotes[best_match]
                        #st.info(f" **Quote for you:** *{selected_quote}*")
                        st.markdown(f"""<div class="custom-info-box"><span class="info-icon">&#x2139;</span><p class="info-text">
                        <strong>Quote for you:</strong> {selected_quote}</p> </div> """, unsafe_allow_html=True)

                    # DilBot's response presented in a chat bubble
                    st.markdown(f"<div class='bot-message-container'><div class='bot-message'>DilBot: {response}</div></div>", unsafe_allow_html=True)

                    speak(response, username)
                    st.session_state.transcribed_text = ""

            # Add a visual separator after each conversation turn (optional)
            st.markdown("<div class='chat-separator'></div>", unsafe_allow_html=True)


    # User's personal dashboard
    st.markdown("---")
    st.header(" Your Personal Dashboard")

    # Load user's journal
    journal_data = load_user_journal(username)

    if journal_data:
        # Statistics
        st.subheader(" Your Emotional Statistics") # Moved statistics to the top of dashboard for prominence
        with st.container(border=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Conversations", len(journal_data))
            with col2:
                emotions = [entry['emotion'] for entry in journal_data]
                most_common = max(set(emotions), key=emotions.count) if emotions else "None"
                st.metric("Most Common Emotion", most_common.capitalize())
            with col3:
                if journal_data:
                    avg_confidence = sum(entry['confidence'] for entry in journal_data) / len(journal_data)
                    st.metric("Avg. Confidence", f"{avg_confidence:.1f}%")
                else:
                    st.metric("Avg. Confidence", "N/A") # Handle case with no journal data for avg. confidence


        # Mood tracker
        st.subheader(" Your Daily Mood Tracker")
        with st.container(border=True): # Wrap chart in a container
            # Prepare data for chart 
            df_data = []
            for entry in journal_data:
                df_data.append({
                "date": entry["date"],
                "emotion": entry["emotion"].capitalize(),
                "confidence": entry["confidence"]
                })
            if df_data:
                df_chart = pd.DataFrame(df_data) # Use pandas DataFrame for better Altair integration

                chart = alt.Chart(df_chart).mark_bar().encode(
                x=alt.X('date:N', title='Date', sort=None), # Sort by date ensures correct order
                y=alt.Y('count():Q', title='Frequency'),
                color=alt.Color('emotion:N', title='Emotion', scale=alt.Scale(range=['#4CAF50', '#FFC107', '#E74C3C', '#3498DB', '#9B59B6', '#1ABC9C'])), # Custom colors
                tooltip=['date:N', 'emotion:N', 'count():Q']
                ).properties(
                    height=350, # Slightly increased height
                    title="Your Emotional Journey Over Time"
                ).interactive() # Make chart interactive for zoom/pan
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("No emotional data to display yet. Start interacting with DilBot!")

        # Recent conversations
        st.subheader("Recent Conversations")
        recent_entries = journal_data[-5:] if len(journal_data) >= 5 else journal_data

        with st.container(border=True): # Wrap recent conversations in a container
            if recent_entries:
                for i, entry in enumerate(reversed(recent_entries)):
                    with st.expander(f"{entry['date']} - {entry['emotion'].capitalize()} ({round(entry['confidence'] * 100)}%)"): # Round confidence for display
                        st.markdown(f"**You said:** {entry['user_input']}")
                        st.markdown(f"**DilBot replied:** {entry['response']}")
            else:
                st.info("No recent conversations yet. Start talking to DilBot!")

    else:
        st.markdown("""   <div style="background-color: #2c3e50;
            border-left: 6px solid #64b5f6;
            color: white;
            border-radius: 10px;
            padding: 18px 25px;
            margin-top: 2rem;
            font-style: italic;
            font-size: 1.05em;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            margin-bottom: 1.5rem;
              ">
             <p style="color: white; margin: 0; font-size: 1.05em;">
            Start your first conversation with DilBot to see your personal dashboard and insights!
                  </p>
              </div>
                   """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<p class='footer-caption'>Built by Members of CSG Hackathon Team | Your data is stored privately and securely</p>", unsafe_allow_html=True)
# Main app logic
def main():
    if not st.session_state.authenticated:
        show_auth_page()
    elif st.session_state.is_admin:
        show_admin_dashboard()
    else:
        show_main_app()

if __name__ == "__main__":
    main()
