# 📞 Customer Support Call Analysis System

A powerful **Streamlit-based application** to analyze customer support calls and extract key insights using AI/LLMs.

---

## 🚀 Features

* 🎙️ **Speaker-separated transcripts** (Agent vs Customer)
* 🧠 **Sentiment, intent, emotion, urgency** detection
* 📝 **Call summaries** with key points and action items
* 🔍 **RAG-based recommendations** using your company documents
* 📁 Upload and analyze `.wav` or `.mp3` audio recordings
* 📄 Upload internal documents (PDF, TXT) for RAG

---

## 🗂️ Project Structure

```
.
├── app.py                          # Streamlit app entry point
├── config.yaml                    # Custom configuration
├── requirements.txt               # Python dependencies
├── /src
│   ├── stt_module.py              # Speech-to-text with diarization
│   ├── insight_extractor.py       # Sentiment, intent, emotion extraction
│   ├── summarizer.py              # Summary generator
│   ├── rag_module.py              # RAG logic for recommendations
│   ├── config.py                  # Configuration manager
│   └── utils.py                   # Common helper functions
├── /data
│   ├── audio_samples/             # Sample call recordings
│   └── documents/                 # Internal knowledge base
```

---

## ⚙️ Setup Instructions

### 1. 🔽 Clone the Repo

```bash
git clone https://github.com/yourusername/customer-support-analysis.git
cd customer-support-analysis
```

### 2. 🧪 Set Up Environment

Install dependencies:

```bash
pip install -r requirements.txt
```

> ⚠️ If you're using a virtual environment, make sure to activate it first (`venv`, `conda`, etc.).

---

### 3. 🔐 Add API Key via Streamlit Secrets

In **Streamlit Cloud** or locally, store your [OpenRouter API key](https://openrouter.ai):

**Streamlit Cloud Secrets Manager** (recommended for deployment):

```
OPENROUTER_API_KEY = "your_openrouter_key"
```

**Locally (if needed):**

Create a `.streamlit/secrets.toml` file:

```toml
[general]
OPENROUTER_API_KEY = "your_openrouter_key"
```

---

### 4. ▶️ Run the App Locally

```bash
streamlit run app.py
```

Visit: [http://localhost:8501](http://localhost:8501)

---

## 🌐 Live App

You can try the deployed version here:
**🔗 [Live Streamlit App]([https://yourapp-link.streamlit.app](https://customer-support-call-analysis-project-cnxdjg5hpatycgef2zqkae.streamlit.app/))**

---

## 📌 Notes

* Tested with Whisper base and large models. On Streamlit Cloud, `base.en` is recommended to avoid memory issues.
* Audio preprocessing supports `.mp3` and `.wav`. Avoid silent or corrupted files.
* Document upload should be text-based (PDF, TXT) for best RAG results.
