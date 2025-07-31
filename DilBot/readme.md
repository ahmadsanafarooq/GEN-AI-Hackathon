
# 🤖 DilBot: AI Emotional Support Companion

**Built with ❤️ at CS GIRLIES HACKATHON**

**DilBot** is an AI-powered emotional support companion designed to lend a listening ear and a comforting voice through intelligent understanding and empathetic responses. Whether you're speaking or typing, DilBot listens to your feelings, detects your emotional state, and engages in meaningful, supportive dialogue — all while helping you reflect and grow emotionally.

---

## 🧠 Overview

Mental health matters. **DilBot** is more than just a chatbot — it's a voice-enabled, emotionally intelligent support system. Designed to foster emotional well-being, it offers empathy-driven conversations, motivational quotes, journaling, and even mood tracking, all powered by cutting-edge AI technologies.

---

## 🚀 Key Features

- ✅ **Voice & Text Input**  
  Talk or type — DilBot is here to listen.

- ✅ **Emotion Detection (7 Emotions)**  
  Understands emotions like happy, sad, angry, fearful, disgusted, surprised, and neutral.

- ✅ **Empathetic Responses (LLMs)**  
  Generates emotionally aware replies using **Groq API** + **LLaMA 3 (70B)**.

- ✅ **Text-to-Speech (TTS) Output**  
  Speaks its responses out loud using `gTTS` for a more natural experience.

- ✅ **Emotion-Aligned Quote Generator**  
  Provides motivational or comforting quotes tailored to your mood using `SentenceTransformer` + FAISS.

- ✅ **Mood Tracker Dashboard**  
  Tracks your emotional trends visually with **Altair** charts.

- ✅ **Journaling Mode**  
  Logs your conversations and emotions into a daily journal (`JSON`-based).

- ✅ **Crisis Phrase Detection**  
  Identifies critical emotional states and prompts helpful responses.

- ✅ **Recent Conversation Memory**  
  Keeps track of your last interactions to make conversations more coherent.

---

## 🧰 Tech Stack

| Category              | Tools / Libraries                                                                 |
|-----------------------|------------------------------------------------------------------------------------|
| Frontend              | `Streamlit`                                                                       |
| Language Models       | `Groq API`, `LLaMA 3 (70B)`                                                        |
| Prompt Chaining       | `LangChain`                                                                       |
| Emotion Analysis      | `transformers`, `SentenceTransformer`, `FAISS`                                    |
| Speech                | `SpeechRecognition`, `gTTS`                                                        |
| Visualization         | `Altair`                                                                          |
| Local Storage         | `os`, `json`, `datetime`, `pathlib`, `dotenv`                                     |

---

## 🏗️ System Architecture

```
[User Input (Voice/Text)]
        ↓
[Emotion Detection] ←→ [Quote Generator (FAISS)]
        ↓
[Empathetic Response Generator (LLM)]
        ↓
[Text-to-Speech Output]
        ↓
[Response Display + Mood Tracker + Journal Logging]
```

---

## 👩‍💻 Team: CS GIRLIES HACKATHON

- 🔹 **Habiba Javed** — Documentation + Presentation  
- 🔹 **Zunaira Hawwar** — Backend Logic  
- ✨ **Ahmad Sana Farooq** — Core Backend Terminologies  
- 🔹 **Awais Sajjad** — System Structure & Diagrams  
- 🔹 **Syed Zain Ali Zaidi** — Frontend Development  

---

## 🌐 Live Demo

Try it out on [Hugging Face Spaces](https://huggingface.co/spaces/ahmadsanafarooq/DilBot)

---

## 📁 How to Run Locally

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/DilBot.git
   cd DilBot
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**

   Create a `.env` file and add your keys (Groq API key, etc.)

   ```bash
   GROQ_API_KEY=your_key_here
   ```

4. **Run the app**

   ```bash
   streamlit run app.py
   ```

---


## 📃 License

This project is licensed under the MIT License.

---

> “Sometimes, all we need is someone to listen — even if that someone is an AI.” – *Team DilBot*
