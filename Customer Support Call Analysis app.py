import streamlit as st
import traceback

st.set_page_config(page_title="Debug App")

try:
    st.title("🔍 App Startup Debugging")

    # Try all imports inside the try block
    st.write("📦 Importing modules...")
    import os
    import json
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    from src.stt_module import SpeechToTextProcessor
    from src.insight_extractor import InsightExtractor
    from src.summarizer import SummaryGenerator
    from src.rag_module import RAGSystem
    from src.utils import setup_logging, validate_audio_file

    st.success("✅ All modules imported successfully.")

    st.write("⚙️ Setting up components...")
    setup_logging()

    st.write("🎤 Loading STT...")
    stt = SpeechToTextProcessor()

    st.write("🧠 Loading InsightExtractor...")
    insight = InsightExtractor(use_openai=True)

    st.write("📝 Loading SummaryGenerator...")
    summary = SummaryGenerator(use_openai=True)

    st.write("🔎 Loading RAGSystem...")
    rag = RAGSystem(knowledge_base_path="data/documents")

    st.success("✅ All components initialized.")

except Exception as e:
    st.error("❌ App crashed during startup.")
    st.code(str(e))
    st.code(traceback.format_exc())
