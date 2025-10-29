# 🚀 AURA AI Dashboard

AURA is a full-stack, persona-driven generative AI dashboard built using Streamlit, Gemini API, and SQLite. It transforms a Transformer-based LLM into a secure, personalized, and highly engaging user product.

---

## 🧠 Deep Learning Core: The Transformer Engine

| DL Concept              | AURA Implementation                                      | Key Takeaway                                                                 |
|------------------------|----------------------------------------------------------|------------------------------------------------------------------------------|
| Transformer Architecture | Gemini 2.5 Pro API with Self-Attention                 | Captures long-range context for coherent, intelligent responses             |
| Sequence-to-Sequence    | Chat functionality                                       | Converts user input into generated text using Transformer decoder           |
| Language Modeling       | Text generation                                          | Predicts next word for fluent, human-like output                            |
| Prompt Engineering      | `prompt = f"You are {persona}. Respond to the user: {message}"` | Dynamically tunes model behavior using system prompts                       |

---

## 🧱 Full-Stack Architecture

### 🎨 Frontend (Streamlit)
- Custom "Sonic Ultra Legend" theme via HTML/CSS
- Persona selector (Sage, Analyst, Muse)
- Neon chat bubbles and gamified UI

### 🛡️ Backend & Security
- Python orchestrator handles input, persona logic, Gemini API calls
- Bcrypt password hashing (via passlib) for secure authentication
- SQLite for persistent user data and chat logs

### 🔄 System Flow
1. User enters message and selects persona
2. Backend constructs prompt and calls Gemini API
3. Response is saved to SQLite and rendered on frontend

---

## 📦 Feature Showcase

| Module        | Functionality                     | Technical Mechanism                          |
|---------------|-----------------------------------|----------------------------------------------|
| Home Page     | Gamification layer                | Simulated XP/Level from chat history queries |
| Chat Page     | Persona-based interaction         | `ai_response()` + `log_chat()`               |
| History Page  | Data reporting & export           | `load_chats()` + export to JSON/TXT/PDF      |
| Profile Page  | Secure profile management         | `update_password()` + `update_profile()`     |

---

## 🛠️ Technologies Used

- Python, Streamlit, SQLite
- Google Generative AI (Gemini 2.5 Pro)
- Passlib (bcrypt), dotenv
- Custom CSS, HTML

---

## 📦 Setup Instructions

```bash
conda activate aura
streamlit run main.py
