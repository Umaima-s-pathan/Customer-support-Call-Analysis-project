import streamlit as st
import os
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.stt_module import SpeechToTextProcessor
from src.insight_extractor import InsightExtractor
from src.summarizer import SummaryGenerator
from src.rag_module import RAGSystem
from src.utils import setup_logging, validate_audio_file

# Configure page
st.set_page_config(
    page_title="Customer Support Analyzer",
    page_icon="📞",
    layout="wide"
)

# Initialize components
@st.cache_resource
def load_components():
    setup_logging()
    return {
        "stt": SpeechToTextProcessor(),
        "insight": InsightExtractor(use_openai=False),
        "summary": SummaryGenerator(use_openai=False),
        "rag": RAGSystem(knowledge_base_path="data/documents")
    }

components = load_components()

# Sidebar for file upload and settings
with st.sidebar:
    st.title("Settings")
    audio_file = st.file_uploader("Upload Call Recording", type=["wav", "mp3"])
    use_rag = st.checkbox("Enable RAG Recommendations", value=True)
    call_id = st.text_input("Call ID (optional)", value="call_001")
    
    st.markdown("---")
    st.markdown("### Sample Data")
    if st.button("Load Sample Data"):
        audio_file = "data/audio_samples/sample_call.wav"  # Should exist
        call_id = "sample_call_001"

# Main app
st.title("📞 Customer Support Call Analysis")
st.markdown("Analyze customer support calls for sentiment, intent, and generate actionable insights.")

if not audio_file:
    st.info("Please upload an audio file to begin analysis")
    st.stop()

# Processing pipeline
with st.status("Analyzing call...", expanded=True) as status:
    # 1. Transcribe audio
    st.write("🔊 Transcribing audio...")
    try:
        transcription = components["stt"].transcribe_audio(audio_file)
        st.session_state.transcription = transcription
    except Exception as e:
        st.error(f"Transcription failed: {str(e)}")
        st.stop()
    
    # 2. Extract insights
    st.write("🧠 Extracting insights...")
    insights = components["insight"].extract_insights(transcription, call_id)
    st.session_state.insights = insights
    
    # 3. Generate summary
    st.write("📝 Generating summary...")
    summary = components["summary"].generate_summary(transcription, insights, call_id)
    st.session_state.summary = summary
    
    # 4. RAG recommendations
    if use_rag:
        st.write("🔍 Generating recommendations...")
        recommendations = components["rag"].generate_recommendations(insights, summary)
        st.session_state.recommendations = recommendations
    
    status.update(label="Analysis complete!", state="complete")

# Display results
tab1, tab2, tab3, tab4 = st.tabs(["Transcript", "Insights", "Summary", "Recommendations"])

with tab1:
    st.subheader("Call Transcript")
    if hasattr(st.session_state, "transcription"):
        segments = st.session_state.transcription.segments
        for seg in segments:
            with st.chat_message(name=seg.speaker.lower()):
                st.write(f"[{seg.start_time:.1f}s] {seg.text}")

with tab2:
    st.subheader("Call Insights")
    if hasattr(st.session_state, "insights"):
        insights = st.session_state.insights
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Overall Sentiment", insights.overall_sentiment.value.title())
            st.metric("Customer Tonality", insights.customer_tonality.value.title())
        
        with col2:
            st.metric("Primary Intent", insights.primary_intent.value.title())
            st.metric("Urgency Level", insights.urgency_level.title())
        
        # Emotion analysis chart
        st.subheader("Emotion Analysis")
        fig, ax = plt.subplots()
        emotions = list(insights.emotion_analysis.keys())
        scores = list(insights.emotion_analysis.values())
        sns.barplot(x=emotions, y=scores, ax=ax)
        ax.set_ylabel("Score")
        st.pyplot(fig)

with tab3:
    st.subheader("Call Summary")
    if hasattr(st.session_state, "summary"):
        summary = st.session_state.summary
        st.write(summary.summary)
        
        if summary.key_points:
            st.subheader("Key Points")
            for point in summary.key_points:
                st.markdown(f"- {point}")
        
        if summary.action_items:
            st.subheader("Action Items")
            for item in summary.action_items:
                st.markdown(f"- {item}")

with tab4:
    if not use_rag:
        st.info("Enable RAG recommendations in the sidebar")
    elif hasattr(st.session_state, "recommendations"):
        rec = st.session_state.recommendations
        st.subheader("Recommended Actions")
        
        for i, action in enumerate(rec.recommendations, 1):
            st.markdown(f"{i}. {action}")
        
        if rec.suggested_actions:
            st.subheader("Suggested Steps")
            for action in rec.suggested_actions:
                st.markdown(f"- {action}")
        
        if rec.relevant_documents:
            with st.expander("Reference Documents"):
                for doc in rec.relevant_documents:
                    st.markdown(f"**{doc['metadata']['title']}** (Score: {doc['similarity_score']:.2f})")
                    st.caption(doc['content'][:200] + "...")

# Download results
if st.button("Download Full Analysis"):
    results = {
        "transcription": [seg.__dict__ for seg in st.session_state.transcription.segments],
        "insights": st.session_state.insights.__dict__,
        "summary": st.session_state.summary.__dict__,
    }
    
    if use_rag:
        results["recommendations"] = st.session_state.recommendations.__dict__
    
    json_str = json.dumps(results, indent=2)
    st.download_button(
        label="Download JSON",
        data=json_str,
        file_name=f"{call_id}_analysis.json",
        mime="application/json"
    )

st.markdown("---")
st.caption("Customer Support Analysis System v1.0")