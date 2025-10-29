'''"""
Aura — Streamlit Chat App (Login / Signup / Persona / Persistent users & chat logs)
- SQLite persistent storage (aura.db)
- bcrypt password hashing with passlib
- Optional OpenAI (reads OPENAI_API_KEY from .env). If missing, demo mode is used.
- Persona selector (Sage / Analyst / Muse)
- Clean, professional UI and safe error handling
"""

import os
import json
import sqlite3
import re
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any

import streamlit as st # type: ignore
from passlib.context import CryptContext # type: ignore
from dotenv import load_dotenv # type: ignore
from openai import OpenAI as OpenAIClient # type: ignore

# ========== CONFIG ==========
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
DB_PATH = os.getenv("AURA_DB_PATH", "aura_users.db")

# Try to import OpenAI (legacy) or modern client when available
_openai_client = None
if OPENAI_API_KEY:
    try:
        _openai_client = OpenAIClient(api_key=OPENAI_API_KEY)
    except Exception as e:
        st.warning(f"Failed to initialize OpenAI client: {e}. Running in demo mode.")

# ========== SECURITY ==========
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
CHAT_HISTORY_MAX = 40

# ========== DATABASE (SQLite) ==========
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        hashed_password TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS chats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
    );
    """)
    conn.commit()
    conn.close()

def create_user(username: str, password: str) -> Tuple[bool, str]:
    username = username.strip()
    if not username:
        return False, "Username cannot be empty."
    if len(password) < 8: # Increased password length for security
        return False, "Password must be at least 8 characters."
    hashed = pwd_context.hash(password)
    now = datetime.utcnow().isoformat()
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("INSERT INTO users (username, hashed_password, created_at) VALUES (?,?,?)",
                    (username, hashed, now))
        conn.commit()
        conn.close()
        return True, "User created."
    except sqlite3.IntegrityError:
        return False, "Username already exists."
    except Exception as e:
        return False, f"DB error: {e}"

def authenticate_user(username: str, password: str) -> Optional[int]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, hashed_password FROM users WHERE username = ?", (username.strip(),))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    user_id = row["id"]
    hashed = row["hashed_password"]
    if pwd_context.verify(password, hashed):
        return user_id
    return None

def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, username, created_at FROM users WHERE id = ?", (user_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None

def log_chat(user_id: int, role: str, content: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO chats (user_id, role, content, created_at) VALUES (?,?,?,?)",
                (user_id, role, content, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def get_user_chats(user_id: int, limit: int = 200) -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT role, content, created_at FROM chats WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                (user_id, limit))
    rows = cur.fetchall()
    conn.close()
    return [{"role": r["role"], "content": r["content"], "created_at": r["created_at"]} for r in reversed(rows)]

# ========== OPENAI / DEMO ==========
def call_openai_chat(messages: List[Dict[str, str]], model: str = "gpt-4o-mini") -> str:
    if not _openai_client:
        raise RuntimeError("OpenAI not configured")
    
    resp = _openai_client.chat.completions.create(model=model, messages=messages, temperature=0.1, max_tokens=600)
    try:
        return resp.choices[0].message.content
    except Exception:
        return str(resp)

def demo_responder(text: str) -> str:
    t = (text or "").strip().lower()
    if not t:
        return "Please type something—I'm listening."
    if any(t.startswith(g) for g in ["hi","hello","hey"]):
        return "Hello! I'm Aura (demo). Ask me anything — I'll try to help."
    if "time" in t:
        return f"Current UTC time: {datetime.utcnow().isoformat()}"
    if any(k in t for k in ("top","most","best","highest","largest")):
        return "Demo: I can't query a DB here, but a sample SQL you might use is: `SELECT name, price FROM products ORDER BY price DESC LIMIT 10;`"
    return "[Demo] I don't have an API key configured. Rephrase your request or add an OPENAI_API_KEY in .env."

# ========== UI helpers & CSS ==========
def inject_css():
    st.markdown(
        """
        <style>
        .stApp { background: linear-gradient(135deg,#071029 0%, #081229 40%, #051026 100%); color: #e6eef8; }
        .card { background: rgba(255,255,255,0.02); padding: 18px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.03); }
        .brand { font-weight:700; color: #ffd97a; font-size: 20px; }
        .muted { color:#9fb0d9; font-size:12px; }
        .stButton>button { background: linear-gradient(90deg,#5b21b6 0%, #2563eb 100%); color:white; border-radius:8px; padding:8px 12px; }
        .stButton>button:hover { opacity:0.95; }
        .persona-pill { padding:6px 10px; border-radius: 999px; background: rgba(255,255,255,0.04); display:inline-block; margin-right:6px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ========== PAGES ==========
def page_signup():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='brand'>Aura — Create account</div>", unsafe_allow_html=True)
    st.write("")
    with st.form("signup_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        password2 = st.text_input("Confirm password", type="password")
        submitted = st.form_submit_button("Create account")
        if submitted:
            ok, msg = create_user(username, password)
            if ok:
                st.success("Account created. Please log in.")
                st.session_state.page = "login"
                st.rerun()
            else:
                st.error(msg)
    st.markdown("</div>", unsafe_allow_html=True)
    if st.button("← Back to Login"):
        st.session_state.page = "login"
        st.rerun()

def page_login():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='brand'>Aura — Login</div>", unsafe_allow_html=True)
    st.write("")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            user_id = authenticate_user(username, password)
            if user_id:
                st.success(f"Welcome back, {username}!")
                st.session_state.user_id = user_id
                st.session_state.username = username
                st.session_state.logged_in = True
                st.session_state.page = "persona"
                st.rerun()
            else:
                st.error("Invalid credentials.")
    st.markdown("</div>", unsafe_allow_html=True)
    if st.button("Create a new account"):
        st.session_state.page = "signup"
        st.rerun()

def page_persona():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='brand'>Aura — Choose persona</div>", unsafe_allow_html=True)
    st.write("")
    persona = st.radio("Choose persona:", ["Sage", "Analyst", "Muse"])
    st.write("")
    if st.button("Enter chat"):
        st.session_state.persona = persona
        st.session_state.page = "chat"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    if st.button("Log out"):
        st.session_state.clear()
        st.rerun()

SYSTEM_PROMPTS = {
    "Sage": "You are a wise, poetic assistant who answers with metaphor and calm insight.",
    "Analyst": "You are a precise, data-focused assistant who gives structured, analytical answers.",
    "Muse" : "You are a creative, inspiring assistant who responds with creativity and warmth."
}

def page_chat():
    if not st.session_state.get("logged_in"):
        st.error("Please log in first.")
        st.session_state.page = "login"
        st.rerun()
        return

    user_id = st.session_state.get("user_id")
    st.markdown(f"<div class='card'><div class='brand'>Aura — Chat ({st.session_state.get('username')})</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='muted'>Persona: <span class='persona-pill'>{st.session_state.get('persona','Sage')}</span></div><br>", unsafe_allow_html=True)

    if "loaded_from_db" not in st.session_state:
        stored = get_user_chats(user_id, limit=200)
        st.session_state.chat_history = [{"role": r["role"], "content": r["content"]} for r in stored]
        st.session_state.loaded_from_db = True

    for msg in st.session_state.get("chat_history", [])[-CHAT_HISTORY_MAX:]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask Aura (type /help for tips)")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        log_chat(user_id, "user", user_input)

        messages = [{"role":"system", "content": SYSTEM_PROMPTS.get(st.session_state.get("persona","Sage"), "")}]
        for m in st.session_state.chat_history[-12:]:
            messages.append({"role": m["role"], "content": m["content"]})

        assistant_text = None
        if _openai_client:
            try:
                assistant_text = call_openai_chat(messages, model="gpt-4o-mini")
            except Exception as e:
                st.warning(f"OpenAI error: {e} — falling back to demo responder.")
                assistant_text = demo_responder(user_input)
        else:
            assistant_text = demo_responder(user_input)

        st.session_state.chat_history.append({"role":"assistant","content":assistant_text})
        log_chat(user_id, "assistant", assistant_text)

        with st.chat_message("assistant"):
            st.markdown(assistant_text)

    st.write("")
    c1,c2,c3,c4 = st.columns([1,1,1,1])
    with c1:
        if st.button("Export chat (JSON)"):
            payload = {"user": st.session_state.get("username"), "chat": st.session_state.chat_history}
            st.download_button("Download", json.dumps(payload, indent=2), file_name="aura_chat.json", mime="application/json")
    with c2:
        if st.button("Clear session chat"):
            st.session_state.chat_history = []
            st.success("Session chat cleared (database logs retained).")
    with c3:
        if st.button("Logout"):
            st.session_state.clear()
            st.rerun()
    with c4:
        mode = "OpenAI" if _openai_client else "Demo"
        st.markdown(f"<div class='muted'>Mode: {mode}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ========== APP ENTRY ==========
def main():
    init_db()
    inject_css()
    st.sidebar.markdown("## Aura Chat")
    st.sidebar.markdown("Secure login, persona & chat logs (SQLite)")

    if "page" not in st.session_state:
        st.session_state.page = "login"
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Router
    if st.session_state.page == "signup":
        page_signup()
    elif st.session_state.page == "login":
        page_login()
    elif st.session_state.page == "persona":
        page_persona()
    elif st.session_state.page == "chat":
        page_chat()
    else:
        st.session_state.page = "login"
        st.rerun()

if __name__ == "__main__":
    st.set_page_config(page_title="Aura Chat", page_icon="💬", layout="centered")
    main()'''



import streamlit as st # type: ignore
import sqlite3
import json
import os
from passlib.context import CryptContext # type: ignore
from dotenv import load_dotenv # type: ignore
from datetime import datetime



# ========== CONFIG ==========
APP_TITLE = "Aura — AI Dashboard"
DB_PATH = os.getenv("AURA_DB_PATH", "aura_users.db")
load_dotenv()

# ✅ MUST BE FIRST STREAMLIT CALL
st.set_page_config(page_title=APP_TITLE, page_icon="💬", layout="wide")

# Password hashing
from passlib.context import CryptContext # type: ignore
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")



# ========== DATABASE ==========
def init_db():
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # --- Create users table if not exists ---
    c.execute("""CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password TEXT NOT NULL
    )""")

    # --- Add missing columns if needed ---
    # Check existing columns
    c.execute("PRAGMA table_info(users)")
    columns = [col[1] for col in c.fetchall()]

    if "avatar" not in columns:
        c.execute("ALTER TABLE users ADD COLUMN avatar TEXT DEFAULT '🧑'")
    if "role" not in columns:
        c.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'Undefined Hero 🌌'")

    # --- Create chats table if not exists ---
    c.execute("""CREATE TABLE IF NOT EXISTS chats (
        username TEXT,
        timestamp TEXT,
        persona TEXT,
        message TEXT,
        response TEXT
    )""")

    conn.commit()
    conn.close()


def add_user(username, password, avatar, role):

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    hashed = pwd_context.hash(password)
    try:
        c.execute("INSERT INTO users (username, password, avatar, role) VALUES (?, ?, ?, ?)", (username, hashed, avatar, role))

    except sqlite3.IntegrityError:
        conn.close()
        return False
    conn.commit()
    conn.close()
    return True

def validate_user(username, password):
    """Checks credentials. If valid, returns (username, avatar, role)."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT password, avatar, role FROM users WHERE username=?", (username,))
    row = c.fetchone()
    conn.close()
    if row and pwd_context.verify(password, row[0]):
        # Returns (username, avatar, role)
        return (username, row[1], row[2])
    return None

def get_user_data(username):
    """Fetches full profile data for a user."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT username, avatar, role FROM users WHERE username=?", (username,))
    row = c.fetchone()
    conn.close()
    return row # Returns (username, avatar, role)

def update_password(username, new_password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    hashed = pwd_context.hash(new_password)
    c.execute("UPDATE users SET password=? WHERE username=?", (hashed, username))
    conn.commit()
    conn.close()

def log_chat(username, persona, message, response):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO chats (username, timestamp, persona, message, response) VALUES (?,?,?,?,?)",
                (username, datetime.now().isoformat(), persona, message, response)
            )
    except sqlite3.Error as e:
        print(f"Error logging chat: {e}")




def load_chats(username):
    """
    🕹️ Ultra Legend Gaming Pro AI Chat Loader
    Fetches all chats for a given user from the database, 
    sorted from newest to oldest.
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute(
                """
                SELECT timestamp, persona, message, response 
                FROM chats 
                WHERE username=? 
                ORDER BY timestamp DESC
                """, 
                (username,)
            )
            rows = c.fetchall()
        return rows
    except sqlite3.Error as e:
        print(f"🔥 Error loading chats for {username}: {e}")
        return []

def update_profile(username, avatar, role):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE users SET avatar=?, role=? WHERE username=?", (avatar, role, username))
    conn.commit()
    conn.close()

# ========== GEMINI AI INTEGRATION ==========

import google.generativeai as genai # type: ignore
import os
import streamlit as st # type: ignore

# ========== GEMINI ==========
def get_gemini_model():
    from dotenv import load_dotenv  # type: ignore
    import os  # ✅ Needed to access environment variables
    load_dotenv()

    key = os.getenv("GEMINI_API_KEY")  # ✅ Secure key loading

    if not key:
        return None
    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel("models/gemini-2.5-pro")
        return model
    except Exception as e:
        st.warning(f"Gemini init failed: {e}. Using demo mode.")
        return None

def ai_response(message, persona):
    model = get_gemini_model()
    if not model:
        # Demo fallback
        return f"[{persona}] Echo: {message}"
    try:
        prompt = f"You are {persona}. Respond to the user: {message}"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Gemini error: {e}"

# ========== APP SECTIONS ==========
def login_page():
    #from database import update_password # type: ignore

    import streamlit as st # type: ignore
    import sqlite3
    #from database import validate_user # type: ignore

    #from database import DB_PATH, pwd_context  # type: ignore # Add this at the top

    def reset_password(username, new_password):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        hashed = pwd_context.hash(new_password)
        c.execute("UPDATE users SET password=? WHERE username=?", (hashed, username))
        conn.commit()
        conn.close()

    # ===== Pro-style CSS =====
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600&family=Cinzel+Decorative:wght@700&display=swap');

        body { font-family: 'Orbitron', sans-serif; background: radial-gradient(circle at top left, #0f2027, #203a43, #2c5364); color: #e0f7fa; }
        
        .hero-banner { padding: 30px; border-radius: 20px; background: linear-gradient(135deg,#1e3c72,#2a5298,#00eaff); 
                       text-align: center; margin-bottom: 40px; box-shadow: 0 0 40px #00eaff; animation: pulseGlow 3s infinite; }
        @keyframes pulseGlow {0% { box-shadow:0 0 20px #00eaff; } 50% { box-shadow:0 0 40px #00eaff; } 100% { box-shadow:0 0 20px #00eaff; }}
        .hero-banner h1 { font-family: 'Cinzel Decorative', cursive; font-size: 42px; margin-bottom:12px; text-shadow:0 0 25px #00eaff; }
        .hero-banner p { font-size:18px; opacity:0.9; }

        .stTextInput>div>div>input {background: rgba(10,10,10,0.85) !important; color:#00eaff !important; border:2px solid #00eaff !important; border-radius:12px; font-weight:bold; font-size:16px;}
        .stButton>button {background: linear-gradient(90deg,#00c3ff,#0072ff); color:white; font-size:18px; font-weight:bold; border-radius:12px; padding:10px 20px; transition: all 0.3s ease-in-out; box-shadow: 0 0 20px #00c3ff;}
        .stButton>button:hover {background: linear-gradient(90deg,#0072ff,#00c3ff); transform: scale(1.07); box-shadow:0 0 40px #00eaff;}
        </style>
    """, unsafe_allow_html=True)

    # ===== Banner & Logo =====
    st.markdown('<div class="hero-banner"><h1>💠 Aura AI Login Arena</h1><p>Step into the legend — or reset your keys if lost 🔑</p></div>', unsafe_allow_html=True)
    st.image("unnamed.png", width=150)

    # ===== Inputs =====
    user = st.text_input("👤 Username", placeholder="Enter your Sonic ID")
    pwd = st.text_input("🔑 Password", type="password", placeholder="Enter your secret key")
    remember = st.checkbox("💾 Remember Me")
    forgot = st.button("❓ Forgot Password?")

    # ===== Forgot Password Pro Mode =====
    if forgot:
        st.markdown('<hr style="border:1px solid #00eaff;">', unsafe_allow_html=True)
        st.markdown('<h3 style="color:#00eaff;">⚡ Password Reset — Hero Mode ⚡</h3>', unsafe_allow_html=True)
        reset_user = st.text_input("🧬 Enter your Username")
        new_pwd = st.text_input("🔑 New Secret Key", type="password")
        confirm_new_pwd = st.text_input("🔐 Confirm New Key", type="password")
        
        avatars = ["🦸", "🐉", "⚡", "🚀", "🧠", "🔥", "🕶️"]
        avatar = st.selectbox("🎭 Choose Avatar for your profile", avatars)
        role = st.selectbox("🎮 Choose Your Legendary Path", [
            "Student 🧑‍🎓 – Apprentice of Knowledge",
            "Developer 🧑‍💻 – Architect of Realms",
            "Researcher 🧠 – Seeker of Truths",
            "Creator 🎨 – Weaver of Worlds",
            "Other 🌌 – Undefined but Infinite"
        ])
        st.markdown(f"**Avatar Preview:** {avatar} | **Role:** {role.split('–')[0].strip()}", unsafe_allow_html=True)

        if st.button("💥 Reset Password Pro", key="reset_btn"):
            if not reset_user.strip() or not new_pwd.strip() or not confirm_new_pwd.strip():
                st.error("⚠️ All fields must be filled!")
            elif new_pwd != confirm_new_pwd:
                st.error("❌ Keys do not match!")
            else:
                reset_password(reset_user, new_pwd)
                st.success(f"✅ Password reset for **{reset_user}**! Avatar: {avatar}, Role: {role.split('–')[0].strip()}")

    # ===== Login Button =====
    if st.button("⚡ Warp into Aura Realm ⚡", use_container_width=True):
        user_data = validate_user(user, pwd)
        if user_data:
            st.session_state["user"] = user
            st.success(f"✅ Welcome back, **{user_data[0]}**! 🌌 Your Aura AI awaits...")
            st.balloons()
            st.rerun()
        else:
            st.error("❌ Wrong credentials. The Chaos Emeralds reject you!")

def signup_page():
    # [Keep existing imports and CSS]

    # --- Banner ---
    st.markdown("""
        <div class="hero-banner">
            <h1>⚡ Aura AI Sign-Up</h1>
            <p>Join the future. Create your legendary account today 🚀</p>
        </div>
    """, unsafe_allow_html=True)

    # --- Inputs ---
    user = st.text_input("👤 Choose a Username", placeholder="e.g. cyber_warrior99")
    pwd = st.text_input("🔑 Choose a Password", type="password", placeholder="Min 6 chars, mix letters & numbers")
    confirm_pwd = st.text_input("🔐 Confirm Password", type="password", placeholder="Re-enter password")

    # Avatar selection
    avatars = ["🦸", "🐉", "⚡", "🚀", "🧠", "🔥", "🕶️"]
    avatar = st.selectbox("🎭 Choose your Avatar", avatars)

    # --- CORRECTION: Use the full descriptive role string for consistency ---
    full_roles = {
        "Student": "Student 🧑‍🎓 – Apprentice of Knowledge",
        "Developer": "Developer 🧑‍💻 – Architect of Realms",
        "Researcher": "Researcher 🧠 – Seeker of Truths",
        "Creator": "Creator 🎨 – Weaver of Worlds",
        "Other": "Other 🌌 – Undefined but Infinite"
    }
    role_key = st.selectbox("🎮 Choose your Role", list(full_roles.keys()))
    final_role_string = full_roles[role_key]

    # Terms agreement
    agree = st.checkbox("✅ I agree to the Terms & Conditions")

    # --- Password Strength Meter (Keep existing logic) ---
    import re
    if pwd:
        strength = 0
        if len(pwd) >= 6: strength += 25
        if re.search(r"[A-Z]", pwd): strength += 25
        if re.search(r"\d", pwd): strength += 25
        if re.search(r"[@$!%*?&#]", pwd): strength += 25

        st.progress(strength)
        if strength < 50:
            st.error("⚠️ Weak password. Use uppercase, numbers & symbols.")
        elif strength < 100:
            st.warning("💪 Good! Add more variety for extra strength.")
        else:
            st.success("🛡️ Ultra Strong Password!")

    # --- Signup Button ---
    if st.button("✨ Create Account", use_container_width=True):
        if not user.strip() or not pwd.strip() or not confirm_pwd.strip():
            st.error("⚠️ Please fill in all fields.")
        elif pwd != confirm_pwd:
            st.error("❌ Passwords do not match.")
        elif len(pwd) < 6:
            st.error("⚠️ Password must be at least 6 characters long.")
        elif not agree:
            st.error("⚠️ You must agree to the Terms & Conditions.")
        else:
            # --- CORRECTION: Pass the full role string to the database ---
            if add_user(user, pwd, avatar, final_role_string):

                st.success(f"✅ Welcome {avatar} {user}! Role: {role_key}")
                st.balloons()
                st.session_state["user"] = user
                st.session_state["avatar"] = avatar
                st.session_state["role"] = final_role_string # Store the full string
                st.rerun()
            else:
                st.error("⚠️ Username already exists. Try another.")

# --- End of corrected signup_page ---

def home_page():
    import streamlit as st # type: ignore

    # --- Pro Gaming CSS ---
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Cinzel+Decorative:wght@700&display=swap');

        body {
            font-family: 'Orbitron', sans-serif;
            background: radial-gradient(circle at top left, #0f2027, #203a43, #2c5364);
            color: #e0f7fa;
        }

        .header-banner {
            padding: 30px;
            border-radius: 20px;
            background: linear-gradient(135deg,#1e3c72,#2a5298,#00eaff);
            text-align: center;
            margin-bottom: 25px;
            box-shadow: 0 0 50px #00eaff;
            animation: pulseGlow 3s infinite;
        }

        @keyframes pulseGlow {
            0% { box-shadow:0 0 20px #00eaff; }
            50% { box-shadow:0 0 40px #00eaff; }
            100% { box-shadow:0 0 20px #00eaff; }
        }

        .profile-card {
            padding:20px;
            border-radius:15px;
            background:linear-gradient(90deg,#4b6cb7,#182848);
            color:white;
            margin-bottom:20px;
            box-shadow: 0 0 25px #00eaff;
            transition: transform 0.2s;
        }

        .profile-card:hover {
            transform: scale(1.02);
            box-shadow: 0 0 50px #00eaff;
        }

        .stat-card {
            padding:20px;
            border-radius:15px;
            background:linear-gradient(135deg,#0f2027,#203a43,#2c5364);
            color:white;
            text-align:center;
            box-shadow: 0 0 20px #00eaff;
            transition: transform 0.2s;
        }

        .stat-card:hover {
            transform: scale(1.05);
            box-shadow: 0 0 45px #00eaff;
        }

        .btn-primary {
            background: linear-gradient(90deg,#00c3ff,#0072ff);
            color:white;
            font-size:20px;
            font-weight:bold;
            border-radius:14px;
            padding:12px 20px;
            box-shadow:0 0 25px #00c3ff;
            transition: all 0.3s ease-in-out;
        }

        .btn-primary:hover {
            background: linear-gradient(90deg,#0072ff,#00c3ff);
            transform: scale(1.08);
            box-shadow:0 0 50px #00eaff;
        }
        </style>
    """, unsafe_allow_html=True)

    # --- Hero Header ---
    st.markdown(f"""
        <div class="header-banner">
            <h1>🌌 Welcome, {st.session_state['user']} 👾</h1>
            <p>⚡ Aura AI Dashboard — Sonic Ultra Legend Edition ⚡</p>
        </div>
    """, unsafe_allow_html=True)

    # --- Load Chat Stats ---
    chats = load_chats(st.session_state["user"]) if "user" in st.session_state else []
    last_active = chats[0][0].split("T")[0] if chats else "No activity yet"

    # --- Profile Card with Avatar & Role ---
    avatar = st.session_state.get("avatar", "🦸")
    role = st.session_state.get("role", "Undefined Hero 🌌")
    st.markdown(f"""
        <div class="profile-card">
            <h2>{avatar} {st.session_state['user']}</h2>
            <h4>Role: {role.split('–')[0].strip()}</h4>
            <p>Welcome back, legend! Ready to continue your journey into the Aura AI realm?</p>
        </div>
    """, unsafe_allow_html=True)

    # --- Stats Cards ---
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
            <div class="stat-card">
                <h3>💬 Chats Logged</h3>
                <h2>{len(chats)}</h2>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
            <div class="stat-card">
                <h3>📅 Last Active</h3>
                <h2>{last_active}</h2>
            </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
            <div class="stat-card">
                <h3>🛡️ Aura Level</h3>
                <h2>{len(chats)*10 if chats else 0}</h2>
            </div>
        """, unsafe_allow_html=True)

    # --- Quick Navigation ---
    st.markdown("---")
    if st.button("🚀 Jump into Chat Now", key="jump_chat", use_container_width=True):
        st.session_state["nav"] = "Chat"
        st.rerun()


def chat_page():
    import streamlit as st # type: ignore

    # --- Pro Gaming CSS ---
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&display=swap');

        body {
            font-family: 'Orbitron', sans-serif;
            background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #1a2a6c);
            background-size: 400% 400%;
            animation: gradientBG 12s ease infinite;
            color: #e0f7fa;
        }

        @keyframes gradientBG {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }

        .persona-card {
            background: linear-gradient(135deg,#1e3c72,#2a5298,#00eaff);
            border-radius: 15px;
            text-align: center;
            padding: 20px;
            margin: 5px;
            color: white;
            box-shadow: 0 0 20px #00eaff;
            transition: transform 0.3s, box-shadow 0.3s;
        }
        .persona-card:hover {
            transform: scale(1.05);
            box-shadow: 0 0 50px #00eaff;
        }

        .chat-bubble-user {
            text-align: right;
            margin: 5px;
        }
        .chat-bubble-user span {
            background: #4CAF50;
            color: white;
            padding: 8px 12px;
            border-radius: 12px;
            display: inline-block;
        }

        .chat-bubble-ai {
            text-align: left;
            margin: 5px;
        }
        .chat-bubble-ai span {
            background: #1e1e2f;
            color: #00eaff;
            padding: 8px 12px;
            border-radius: 12px;
            display: inline-block;
            box-shadow: 0 0 12px #00eaff;
        }

        .send-btn {
            background: linear-gradient(90deg,#00c3ff,#0072ff);
            color:white;
            font-size:18px;
            font-weight:bold;
            border-radius:12px;
            padding:10px 16px;
            box-shadow:0 0 25px #00c3ff;
            transition: all 0.3s ease-in-out;
        }
        .send-btn:hover {
            background: linear-gradient(90deg,#0072ff,#00c3ff);
            transform: scale(1.08);
            box-shadow:0 0 50px #00eaff;
        }
        </style>
    """, unsafe_allow_html=True)

    # --- Page Header ---
    st.markdown("<h1 style='text-align:center;'>💬 Sonic Ultra Legend Chat Arena</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>⚡ Converse with your AI personas and level up!</p>", unsafe_allow_html=True)

    # --- Persona Selector ---
    st.subheader("Choose Your Persona 🎭")
    personas = ["Sage", "Analyst", "Muse"]
    icons = ["🧙", "📊", "🎨"]
    cols = st.columns(len(personas))

    if "persona" not in st.session_state:
        st.session_state["persona"] = "Sage"

    for i, p in enumerate(personas):
        if cols[i].button(f"{icons[i]} {p}", key=f"persona_{i}"):
            st.session_state["persona"] = p

    persona = st.session_state["persona"]
    st.success(f"✨ Active Persona: {persona}")

    # 🎨 Persona Styles
    persona_styles = {
        "Sage": {"emoji": "🧙", "color": "#8e44ad"},
        "Analyst": {"emoji": "📊", "color": "#2980b9"},
        "Muse": {"emoji": "🎨", "color": "#e67e22"}
    }


    # --- Chat Log ---
    if "chat_log" not in st.session_state:
        st.session_state["chat_log"] = []

    # --- Input Box ---
    msg = st.text_input("💡 Enter your message and press Enter:", key="chat_input")
    if st.button("Send", key="send_btn", use_container_width=True):
        if msg.strip():
            reply = ai_response(msg, persona)  # your AI response function

            # Append to chat log
            st.session_state["chat_log"].append(("You", msg))
            st.session_state["chat_log"].append((persona, reply))
            style = persona_styles.get(persona, {"emoji": "🤖", "color": "#00c3ff"})
            st.markdown(f"""
            <div style="border-radius:12px; padding:12px; background:{style['color']}; color:white; box-shadow:0 0 12px {style['color']}; margin-top:10px;">
                <b>{style['emoji']} {persona}:</b> {reply}
            </div>
            """, unsafe_allow_html=True)


            # Save to DB
            log_chat(st.session_state["user"], persona, msg, reply)

            # Clear input
            st.session_state["chat_input"] = ""
            st.rerun()

    # --- Display Chat Log with Neon Bubbles ---
    st.subheader("🗨️ Conversation")
    for speaker, text in st.session_state["chat_log"]:
        if speaker == "You":
            st.markdown(
                f"<div class='chat-bubble-user'><span>{text}</span></div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='chat-bubble-ai'><span><b>{speaker}:</b> {text}</span></div>",
                unsafe_allow_html=True
            )

def history_page():
    import streamlit as st # type: ignore
    import json

    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&display=swap');

        body {
            font-family: 'Orbitron', sans-serif;
            background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #1a2a6c);
            background-size: 400% 400%;
            animation: gradientBG 12s ease infinite;
            color: #e0f7fa;
        }

        @keyframes gradientBG {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }

        .history-card {
            padding: 12px 16px;
            border-radius: 15px;
            margin-bottom: 12px;
            background: linear-gradient(135deg, #1e3c72, #2a5298, #00eaff);
            box-shadow: 0 0 15px #00eaff;
            color: white;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .history-card:hover {
            transform: scale(1.02);
            box-shadow: 0 0 35px #00eaff;
        }

        .history-time {
            font-size: 12px;
            color: #aaa;
        }

        .history-message {
            margin: 4px 0;
        }
        .user-msg {
            color: #4CAF50;
        }
        .ai-msg {
            color: #ffd700;
        }
        </style>
    """, unsafe_allow_html=True)

    st.header("📜 Chat History — Sonic Ultra Legend Mode")

    chats = load_chats(st.session_state["user"])

    if not chats:
        st.info("No chats found yet. Start a conversation in the Chat page!")
        return

    # Filter by persona
    personas = list(set([p for _, p, _, _ in chats]))
    persona_filter = st.selectbox("Filter by Persona", ["All"] + personas)

    # Display Chats
    for t, p, m, r in chats:
        if persona_filter != "All" and p != persona_filter:
            continue
        st.markdown(
            f"""
            <div class="history-card">
                <p class="history-time">[{t}] — Persona: <b>{p}</b></p>
                <p class="history-message user-msg">🧑 You: {m}</p>
                <p class="history-message ai-msg">🤖 Aura: {r}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    # Export Options
    st.subheader("📂 Export Options")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("💾 Export JSON"):
            fname = f"{st.session_state['user']}_history.json"
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(chats, f, indent=2, ensure_ascii=False)
            st.success(f"Exported {fname}")

    with col2:
        if st.button("📝 Export TXT"):
            fname = f"{st.session_state['user']}_history.txt"
            with open(fname, "w", encoding="utf-8") as f:
                for t, p, m, r in chats:
                    f.write(f"[{t}] {p}\nYou: {m}\nAura: {r}\n\n")
            st.success(f"Exported {fname}")

    with col3:
        try:
            from fpdf import FPDF # type: ignore
            if st.button("📑 Export PDF"):
                fname = f"{st.session_state['user']}_history.pdf"
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                for t, p, m, r in chats:
                    pdf.multi_cell(0, 10, f"[{t}] {p}\nYou: {m}\nAura: {r}\n")
                    pdf.ln()
                pdf.output(fname)
                st.success(f"Exported {fname}")
        except ImportError:
            st.warning("Install `fpdf` to enable PDF export (pip install fpdf).")



def profile_page():
    import streamlit as st # type: ignore
    import sqlite3
    from passlib.context import CryptContext # type: ignore
    import random
    #from database import update_password  # type: ignore # ✅ Add this near your other imports
    #from database import DB_PATH  # type: ignore # ✅ Add this near your other imports

    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&display=swap');

        body {
            font-family: 'Orbitron', sans-serif;
            background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #1a2a6c);
            background-size: 400% 400%;
            animation: gradientBG 12s ease infinite;
            color: #e0f7fa;
        }

        @keyframes gradientBG {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }

        .profile-card {
            padding: 20px;
            border-radius: 15px;
            background: linear-gradient(135deg, #1e3c72, #2a5298, #00eaff);
            box-shadow: 0 0 25px #00eaff;
            margin-bottom: 20px;
        }

        .stat-card {
            padding: 15px;
            border-radius: 12px;
            margin-bottom: 15px;
            background: #1e1e2f;
            color: white;
            box-shadow: 0 0 15px #00eaff;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .stat-card:hover {
            transform: scale(1.02);
            box-shadow: 0 0 35px #00eaff;
        }
        </style>
    """, unsafe_allow_html=True)

    st.header("👤 Profile — Sonic Ultra Legend Mode")

    # --- Profile Card ---
    avatar = st.session_state.get("avatar", "🧑")
    role = st.session_state.get("role", "Undefined Hero 🌌")

    st.subheader("Account Information")
    st.markdown(f"""
        <div class="profile-card">
            <h2>{avatar} {st.session_state['user']}</h2>
            <p>Status: <b>Active</b></p>
            <p>Role: <b>{role}</b></p>
        </div>
    """, unsafe_allow_html=True)


    # --- XP and Badge Display ---
    st.subheader("🏆 Experience & Badges")
    xp = random.randint(120, 999)  # demo XP
    level = xp // 100
    badges = ["🔥 Firestarter", "⚡ Speedster", "🛡️ Guardian", "🎯 Sharpshooter"]
    earned_badges = random.sample(badges, k=random.randint(1, len(badges)))
    st.progress(min(xp / 1000, 1.0))
    st.write(f"Level: **{level}** — XP: **{xp}/1000**")
    st.write("Badges Earned: " + ", ".join(earned_badges))

    # --- Password Change ---
    st.subheader("🔒 Change Password")

    new_pwd = st.text_input("Enter new password", type="password")
    confirm_pwd = st.text_input("Confirm new password", type="password")

    if st.button("Update Password"):
        if not new_pwd.strip():
            st.error("⚠️ Password cannot be empty.")
        elif new_pwd != confirm_pwd:
            st.error("⚠️ Passwords do not match.")
        else:
            update_password(st.session_state["user"], new_pwd)
            st.success("✅ Password updated successfully!")



    # --- Avatar & Bio ---
    st.subheader("✨ Customize Avatar & Bio")
    avatar_choice = st.selectbox("Choose Avatar Theme", ["🐉 Dragon", "⚡ Sonic Ultra", "🦸 Hero", "👽 Alien", "🧙 Wizard"])
    st.markdown(f"**Avatar Preview:** {avatar_choice}", unsafe_allow_html=True)

    bio = st.text_area("Write your bio", placeholder="Tell the world who you are...")
    #from database import update_profile  # type: ignore # ✅ Add this near your imports

    if st.button("💾 Save Bio"):
        update_profile(st.session_state["user"], avatar_choice, bio)
        st.session_state["avatar"] = avatar_choice
        st.session_state["role"] = bio
        st.success("✅ Profile updated successfully!")


    # --- Danger Zone ---
    st.markdown("---")
    st.subheader("🚨 Danger Zone")
    if st.button("❌ Delete Account"):
        st.warning("⚠️ This will permanently delete your account (feature coming soon).")

    # --- Optional Gaming Flair ---
    st.markdown("""
        <div style="text-align:center; margin-top:20px;">
            <h3>🌌 Aura Energy Status</h3>
            <progress value="75" max="100" style="width:60%; height:20px;"></progress>
            <p>Energy: 75% — Keep interacting to recharge!</p>
        </div>
    """, unsafe_allow_html=True)



def settings_page():
    import streamlit as st # type: ignore

    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&display=swap');

        body {
            font-family: 'Orbitron', sans-serif;
            background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #1a2a6c);
            background-size: 400% 400%;
            animation: gradientBG 12s ease infinite;
            color: #e0f7fa;
        }

        @keyframes gradientBG {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }

        .setting-card {
            padding: 20px;
            border-radius: 15px;
            background: linear-gradient(135deg, #1e3c72, #2a5298, #00eaff);
            box-shadow: 0 0 25px #00eaff;
            margin-bottom: 20px;
        }

        .setting-card:hover {
            transform: scale(1.02);
            box-shadow: 0 0 40px #00eaff;
            transition: all 0.3s ease-in-out;
        }

        .stButton>button {
            background: linear-gradient(90deg, #00c3ff, #0072ff);
            color: white;
            font-size: 18px;
            font-weight: bold;
            border-radius: 12px;
            box-shadow: 0 0 20px #00c3ff;
            transition: all 0.3s ease-in-out;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #0072ff, #00c3ff);
            transform: scale(1.08);
            box-shadow: 0 0 35px #00eaff;
        }
        </style>
    """, unsafe_allow_html=True)

    st.header("⚙️ Settings — Sonic Ultra Legend Mode")

    st.markdown(
        """
        Welcome to the **settings hub** ⚡  
        Personalize your Aura AI experience and manage system preferences.
        """
    )

    # --- Appearance & Theme ---
    with st.expander("🎨 Appearance & Theme", expanded=True):
        st.info("💡 Use the Streamlit menu (top-right) to toggle Light/Dark mode.")
        theme_choice = st.radio("Select theme accent color:", ["🌌 Cosmic Purple", "🔥 Inferno Red", "🌊 Aqua Blue"])
        st.markdown(f"<div class='setting-card'>Theme set to: <b>{theme_choice}</b></div>", unsafe_allow_html=True)

    # --- Account Settings ---
    with st.expander("👤 Account Settings"):
        if "user" in st.session_state:
            st.write(f"Logged in as: **{st.session_state['user']}**")
        if st.button("🔄 Reset Password"):
            st.warning("Password reset feature coming soon...")

    # --- Data & History ---
    with st.expander("💾 Data & History"):
        st.write("Manage your chat logs and app data.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📤 Export Chat Logs"):
                st.success("✅ Chat logs exported successfully (JSON/TXT).")
        with col2:
            if st.button("🗑️ Clear History"):
                st.error("⚠️ All history cleared!")

    # --- System Info ---
    with st.expander("🖥️ System Info"):
        st.markdown(
            """
            <div class='setting-card'>
                App Version: <b>1.0.0 Ultra</b><br>
                Database: <b>aura_users.db</b><br>
                Mode: <b>Demo / OpenAI Auto-Detect</b>
            </div>
            """, unsafe_allow_html=True
        )

    # --- Aura Energy Bar ---
    st.markdown("""
        <div style="text-align:center; margin-top:20px;">
            <h3>🌌 Aura Energy Status</h3>
            <progress value="65" max="100" style="width:60%; height:20px;"></progress>
            <p>Energy: 65% — Keep exploring to recharge!</p>
        </div>
    """, unsafe_allow_html=True)

# ========== MAIN APP ==========
def main():
    import streamlit as st # type: ignore
    import random

    # --- Neon Chat Bubble CSS ---
    st.markdown("""
    <style>
    /* Chat bubble styles */
    div.stChatMessage { 
        border-radius: 20px; 
        padding: 10px; 
        margin: 5px; 
        box-shadow: 0 0 12px #00eaff; 
        background: linear-gradient(90deg, #00c3ff, #0072ff);
        color: white; 
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize DB once at app start
    init_db()

    st.sidebar.markdown("## 🌌 Aura — Sonic Ultra Legend AI Dashboard")

    if "user" not in st.session_state:
        # --- Sidebar Authentication Menu (Gaming Pro Mode) ---
        st.sidebar.markdown("""
            <style>
            section[data-testid="stSidebar"] {
                background: linear-gradient(180deg, #0f2027, #203a43, #2c5364);
                border-right: 3px solid #00eaff;
                box-shadow: 0 0 25px #00eaff;
            }

            .sidebar-title {
                font-family: 'Cinzel Decorative', cursive;
                font-size: 26px;
                text-align: center;
                color: #00eaff;
                text-shadow: 0 0 12px #00eaff;
                margin-bottom: 20px;
            }

            div[role="radiogroup"] > label {
                background: rgba(0,0,0,0.6);
                border: 2px solid #00eaff;
                border-radius: 14px;
                color: #00eaff !important;
                padding: 12px;
                font-size: 18px;
                font-weight: bold;
                margin: 8px 0;
                cursor: pointer;
                transition: all 0.3s ease-in-out;
                box-shadow: 0 0 12px #00eaff;
            }
            div[role="radiogroup"] > label:hover {
                background: linear-gradient(90deg, #0072ff, #00c3ff);
                color: white !important;
                transform: scale(1.05);
                box-shadow: 0 0 25px #00eaff;
            }
            div[role="radiogroup"] > label[data-checked="true"] {
                background: linear-gradient(90deg, #00c3ff, #0072ff);
                color: white !important;
                box-shadow: 0 0 25px #00eaff, inset 0 0 12px #00eaff;
            }
            </style>
        """, unsafe_allow_html=True)

        st.sidebar.markdown("<div class='sidebar-title'>⚡ Authentication Gate</div>", unsafe_allow_html=True)

        # Sidebar radio (Gaming Pro Mode)
        choice = st.sidebar.radio("Choose your entry:", ["🔐 Login", "📝 Sign Up"], label_visibility="collapsed")

        # Call respective page
        if choice == "🔐 Login":
            login_page()
        else:
            signup_page()

    else:
        # --- Logged-in navigation ---
        st.sidebar.markdown(f"👤 **Welcome, {st.session_state['user']}!**")
        choice = st.sidebar.radio(
            "📂 Navigation",
            ["🏠 Home", "💬 Chat", "📜 History", "⚙️ Profile", "🎨 Settings", "🚪 Logout"]
        )

        if choice == "🏠 Home":
            # --- Quote of the Day ---
            quotes = [
                "Your mind is the battlefield.",
                "AI is your ally.",
                "Dream. Code. Conquer.",
                "Power lies in persistence.",
                "Think fast, act faster."
            ]
            st.markdown(f"💡 **Quote of the Day:** {random.choice(quotes)}")
            home_page()

        elif choice == "💬 Chat":
            chat_page()  # We'll enhance chat_page() to include animation & XP

        elif choice == "📜 History":
            # --- Search Chat History ---
            search = st.text_input("🔍 Search chat history")
            if search:
                filtered = [
                    f"{msg['user']}: {msg['text']}" 
                    for msg in st.session_state.get('chat_log', []) 
                    if search.lower() in msg['text'].lower()
                ]
                st.write(filtered)
            history_page()

        elif choice == "⚙️ Profile":
            profile_page()
        elif choice == "🎨 Settings":
            settings_page()
        elif choice == "🚪 Logout":
            st.session_state.clear()
            st.success("✅ You have been logged out! 🌌")
            st.rerun()


# ----------------- Chat Enhancements -----------------
def animated_ai_response(ai_name, user_input):
    import time
    import streamlit as st # type: ignore

    response_text = f"{ai_name}: Processing your request..."
    message_placeholder = st.empty()
    display_text = ""
    for char in response_text:
        display_text += char
        message_placeholder.markdown(f"```\n{display_text}\n```")
        time.sleep(0.03)
    return f"{ai_name}: Here's your response to '{user_input}'!"


def chat_page():
    import streamlit as st  # type: ignore

    # 🎭 Persona Selection
    persona = st.selectbox("🎭 Choose your AI Persona", [
        "Sonic Ultra", "Mythic Oracle", "Code Whisperer", "Legendary Mentor"
    ])

    # 💬 User Input
    user_input = st.text_input("You:", placeholder="Ask Aura anything...")

    if user_input:
        # 🧙 Animated Typing
        response = animated_ai_response("Aura", user_input)

        # 🏆 XP Boost
        st.session_state['xp'] = st.session_state.get('xp', 0) + 10
        st.write(f"🏆 XP: {st.session_state['xp']}")

        # 🤖 Gemini Response
        final_response = ai_response(user_input, persona)
        st.write(final_response)

        # 📜 Log Chat
        log_chat(st.session_state["user"], persona, user_input, final_response)

# ----------------- Run App -----------------
if __name__ == "__main__":
    main()


# conda activate aura
#streamlit run main.py
# username = akash_sky1904
# password = akash1234