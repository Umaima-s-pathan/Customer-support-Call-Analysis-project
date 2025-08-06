# 📞 Customer Support Call Analysis System

A powerful Streamlit-based application for analyzing customer support calls and extracting insights like:

- 🎙️ Speaker-separated transcripts  
- 🧠 Sentiment, intent & emotion analysis  
- 📝 Summaries of call content  
- 🔍 RAG-based recommendations from internal documents

## 🚀 Features

- Upload audio (WAV, MP3)
- Speaker diarization (Agent vs Customer)
- Sentiment, intent, emotion, urgency detection
- Summary generation via OpenRouter API
- Key points, action items
- RAG system powered by uploaded company docs

## 🗂️ Project Structure

```

.
├── app.py
├── config.yaml
├── requirements.txt
├── /src
│   ├── stt\_module.py
│   ├── insight\_extractor.py
│   ├── summarizer.py
│   ├── rag\_module.py
│   ├── config.py
│   └── utils.py
└── /data
├── audio\_samples/
├── documents/


````

## ⚙️ Setup Instructions

```bash
git clone https://github.com/yourusername/customer-support-analysis.git
cd customer-support-analysis
pip install -r requirements.txt
````

Make sure to securely store your OpenRouter key via Streamlit Secrets:

```toml
# in Streamlit Cloud's secret manager (no need to create .streamlit folder)
OPENROUTER_API_KEY = "your_key_here"
```

Then run locally:

```bash
streamlit run app.py
```

## 🌐 Live App

You can try the app live:
**[Streamlit App Link](https://your-deployed-app-url.streamlit.app)**

---

