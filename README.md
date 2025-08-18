# 🌱 Agri-Bot

Agri-Bot is an AI-powered assistant built with **LangChain**, **Streamlit**, and **Google LLM** to help with agricultural insights and information retrieval.  

---

## ⚙️ Setup Instructions

### ✅ Requirements
- **Python**: `3.10.9`
- **Virtual Environment** (recommended)

---

### 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/9085prayas/agri-bot.git
   cd agri-bot
2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   # Windows (PowerShell)
   .venv\Scripts\activate
   # Linux / Mac
   source .venv/bin/activate
3. **Set up environment variables**
   create a .env file in the repo and add your api keys
   ```bash
   GOOGLE_API_KEY=your_google_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
5. **Run the Streamlit app**
   ```bash
   streamlit run app.py


