"""Summary generation module for customer support calls."""

import streamlit as st
import openai
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

# Try to import transformers for local models
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from .config import config_manager
from .utils import setup_logging
from .stt_module import TranscriptionResult
from .insight_extractor import CallInsights


@dataclass
class CallSummary:
    """Generated summary for a customer support call."""
    call_id: str
    summary: str
    key_points: List[str]
    action_items: List[str]
    follow_up_required: bool
    summary_length: int
    confidence_score: float
    generation_method: str
    metadata: Dict[str, Any]


class SummaryGenerator:
    """Advanced summary generation system for customer support calls."""
    
    def __init__(self, use_openai: bool = True):
        """Initialize the summary generator.
        
        Args:
            use_openai: Whether to use OpenAI API for summary generation
        """
        self.logger = setup_logging()
        self.config = config_manager
        self.use_openai = use_openai
        
        # Initialize OpenAI if available
        if self.use_openai:
            self._setup_openai()
            
        # Initialize local models
        self.local_model = None
        if TRANSFORMERS_AVAILABLE:
            self._setup_local_model()
            
    def _setup_openai(self) -> None:
    """Setup OpenRouter API client via Streamlit secrets."""
    try:
        api_key = st.secrets["OPENROUTER_API_KEY"]
        openai.api_key = api_key
        openai.api_base = "https://openrouter.ai/api/v1"

        # Get model from YAML and remap to OpenRouter version
        config_model = self.config.get_openai_config().model
        model_mapping = {
            "gpt-4": "openrouter/gpt-4",
            "gpt-3.5-turbo": "openrouter/gpt-3.5-turbo",
            "gpt-4o": "openrouter/gpt-4o"
        }
        self.openai_model = model_mapping.get(config_model, "openrouter/gpt-3.5-turbo")

        self.logger.info(f"OpenRouter API configured. Using model: {self.openai_model}")
    except Exception as e:
        self.logger.error(f"Failed to configure OpenRouter API: {e}")
        self.use_openai = False
            
    def _setup_local_model(self) -> None:
        """Setup local summarization model."""
        try:
            # Use BART for summarization
            self.local_model = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=-1  # CPU
            )
            self.logger.info("Local summarization model loaded successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to load local summarization model: {e}")
            
    def generate_summary(self, transcription: TranscriptionResult, 
                        insights: Optional[CallInsights] = None,
                        call_id: str = "unknown") -> CallSummary:
        """Generate comprehensive summary for a customer support call.
        
        Args:
            transcription: Transcription result from STT module
            insights: Optional insights from insight extraction
            call_id: Unique identifier for the call
            
        Returns:
            CallSummary object with generated summary and metadata
        """
        self.logger.info(f"Generating summary for call: {call_id}")
        
        try:
            # Generate main summary
            if self.use_openai:
                summary, confidence = self._generate_openai_summary(transcription, insights)
                method = "openai"
            elif self.local_model:
                summary, confidence = self._generate_local_summary(transcription)
                method = "local_model"
            else:
                summary, confidence = self._generate_extractive_summary(transcription)
                method = "extractive"
                
            # Extract key points
            key_points = self._extract_key_points(transcription, insights)
            
            # Identify action items
            action_items = self._identify_action_items(transcription, insights)
            
            # Determine follow-up requirement
            follow_up_required = self._assess_follow_up_need(insights, transcription)
            
            # Create summary object
            call_summary = CallSummary(
                call_id=call_id,
                summary=summary,
                key_points=key_points,
                action_items=action_items,
                follow_up_required=follow_up_required,
                summary_length=len(summary.split()),
                confidence_score=confidence,
                generation_method=method,
                metadata={
                    "original_length": len(transcription.full_text.split()),
                    "compression_ratio": len(summary.split()) / len(transcription.full_text.split()) if transcription.full_text else 0,
                    "call_duration": transcription.duration,
                    "language": transcription.language
                }
            )
            
            self.logger.info(f"Summary generated successfully for call: {call_id}")
            return call_summary
            
        except Exception as e:
            self.logger.error(f"Failed to generate summary: {e}")
            raise
            
    def _generate_openai_summary(self, transcription: TranscriptionResult, 
                                insights: Optional[CallInsights] = None) -> tuple[str, float]:
        """Generate summary using OpenAI API.
        
        Args:
            transcription: Transcription result
            insights: Optional insights data
            
        Returns:
            Tuple of (summary_text, confidence_score)
        """
        try:
            # Prepare context
            context = self._prepare_context_for_summary(transcription, insights)
            
            prompt = f"""
            You are an expert at summarizing customer support calls. Create a concise, professional summary of the following customer support conversation.

            The summary should:
            1. Be 3-5 sentences long
            2. Capture the main issue or request
            3. Mention the resolution or current status
            4. Use professional, clear language
            5. Focus on key facts and outcomes

            Context:
            {context}

            Conversation:
            {transcription.full_text[:2000]}  # Limit text length

            Summary:"""
            
            response = openai.ChatCompletion.create(
                model=self.openai_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3
            )
            
            summary = response.choices[0].message.content.strip()
            return summary, 0.9
            
        except Exception as e:
            self.logger.warning(f"OpenAI summary generation failed: {e}")
            return self._generate_extractive_summary(transcription)
            
    def _generate_local_summary(self, transcription: TranscriptionResult) -> tuple[str, float]:
        """Generate summary using local model.
        
        Args:
            transcription: Transcription result
            
        Returns:
            Tuple of (summary_text, confidence_score)
        """
        try:
            # Prepare text for summarization
            text = transcription.full_text
            
            # BART has input length limits
            max_length = 1024
            if len(text.split()) > max_length:
                # Truncate text while preserving important parts
                words = text.split()
                text = " ".join(words[:max_length])
                
            # Generate summary
            result = self.local_model(
                text,
                max_length=150,
                min_length=50,
                do_sample=False
            )
            
            summary = result[0]['summary_text']
            return summary, 0.8
            
        except Exception as e:
            self.logger.warning(f"Local summary generation failed: {e}")
            return self._generate_extractive_summary(transcription)
            
    def _generate_extractive_summary(self, transcription: TranscriptionResult) -> tuple[str, float]:
        """Generate extractive summary as fallback.
        
        Args:
            transcription: Transcription result
            
        Returns:
            Tuple of (summary_text, confidence_score)
        """
        # Simple extractive summarization
        sentences = transcription.full_text.split('.')
        
        # Score sentences based on important keywords
        important_keywords = [
            'issue', 'problem', 'help', 'resolved', 'fixed', 'account', 
            'billing', 'technical', 'support', 'error', 'working'
        ]
        
        sentence_scores = []
        for sentence in sentences:
            if len(sentence.strip()) < 10:
                continue
                
            score = 0
            sentence_lower = sentence.lower()
            
            # Score based on keyword presence
            for keyword in important_keywords:
                if keyword in sentence_lower:
                    score += 1
                    
            # Bonus for customer/agent indicators
            if 'customer' in sentence_lower or 'agent' in sentence_lower:
                score += 0.5
                
            sentence_scores.append((sentence.strip() + '.', score))
            
        # Sort by score and take top sentences
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [sent[0] for sent in sentence_scores[:3]]
        
        summary = " ".join(top_sentences)
        
        if not summary.strip():
            # Fallback to first few sentences
            summary = ". ".join(sentences[:2]) + "."
            
        return summary, 0.6
        
    def _prepare_context_for_summary(self, transcription: TranscriptionResult, 
                                   insights: Optional[CallInsights] = None) -> str:
        """Prepare context information for summary generation.
        
        Args:
            transcription: Transcription result
            insights: Optional insights data
            
        Returns:
            Context string for summary generation
        """
        context_parts = []
        
        # Add basic call info
        context_parts.append(f"Call Duration: {transcription.duration:.1f} seconds")
        context_parts.append(f"Language: {transcription.language}")
        
        # Add insights if available
        if insights:
            context_parts.append(f"Customer Sentiment: {insights.customer_sentiment.value}")
            context_parts.append(f"Primary Intent: {insights.primary_intent.value}")
            context_parts.append(f"Urgency Level: {insights.urgency_level}")
            context_parts.append(f"Resolution Status: {insights.resolution_status}")
            
        return " | ".join(context_parts)
        
    def _extract_key_points(self, transcription: TranscriptionResult, 
                           insights: Optional[CallInsights] = None) -> List[str]:
        """Extract key points from the conversation.
        
        Args:
            transcription: Transcription result
            insights: Optional insights data
            
        Returns:
            List of key points
        """
        key_points = []
        text = transcription.full_text.lower()
        
        # Extract based on common customer service patterns
        patterns = {
            "Issue Reported": ["issue", "problem", "error", "not working", "broken"],
            "Account Related": ["account", "login", "password", "access"],
            "Billing Inquiry": ["bill", "charge", "payment", "refund", "cost"],
            "Technical Support": ["technical", "software", "app", "website", "connection"],
            "Resolution Provided": ["resolved", "fixed", "working", "solution", "sorted"]
        }
        
        for category, keywords in patterns.items():
            if any(keyword in text for keyword in keywords):
                key_points.append(category)
                
        # Add insights-based key points
        if insights:
            if insights.urgency_level == "high":
                key_points.append("High Priority Issue")
            if insights.customer_sentiment.value == "negative":
                key_points.append("Customer Dissatisfaction")
            if insights.resolution_status == "escalated":
                key_points.append("Issue Escalated")
                
        return key_points[:5]  # Limit to 5 key points
        
    def _identify_action_items(self, transcription: TranscriptionResult, 
                              insights: Optional[CallInsights] = None) -> List[str]:
        """Identify action items from the conversation.
        
        Args:
            transcription: Transcription result
            insights: Optional insights data
            
        Returns:
            List of action items
        """
        action_items = []
        text = transcription.full_text.lower()
        
        # Common action item patterns
        action_patterns = {
            "Follow up within 24 hours": ["follow up", "call back", "check back"],
            "Update account information": ["update", "change", "modify", "account"],
            "Process refund": ["refund", "money back", "reimburse"],
            "Technical investigation required": ["investigate", "look into", "check", "technical"],
            "Escalate to supervisor": ["escalate", "supervisor", "manager", "senior"],
            "Send documentation": ["send", "email", "documentation", "instructions"]
        }
        
        for action, keywords in action_patterns.items():
            if any(keyword in text for keyword in keywords):
                action_items.append(action)
                
        # Add insights-based action items
        if insights:
            if insights.resolution_status == "unresolved":
                action_items.append("Issue requires further attention")
            if insights.urgency_level == "high":
                action_items.append("Priority handling required")
            if insights.customer_sentiment.value == "negative":
                action_items.append("Customer satisfaction follow-up needed")
                
        return action_items[:5]  # Limit to 5 action items
        
    def _assess_follow_up_need(self, insights: Optional[CallInsights], 
                              transcription: TranscriptionResult) -> bool:
        """Assess whether follow-up is required.
        
        Args:
            insights: Optional insights data
            transcription: Transcription result
            
        Returns:
            Boolean indicating if follow-up is needed
        """
        if insights:
            # Follow-up needed if unresolved or escalated
            if insights.resolution_status in ["unresolved", "escalated"]:
                return True
                
            # Follow-up needed for high urgency issues
            if insights.urgency_level == "high":
                return True
                
            # Follow-up needed for very negative sentiment
            if insights.customer_sentiment.value == "negative":
                return True
                
        # Check text for follow-up indicators
        text = transcription.full_text.lower()
        follow_up_indicators = ["follow up", "call back", "check", "monitor", "review"]
        
        if any(indicator in text for indicator in follow_up_indicators):
            return True
            
        return False
        
    def generate_batch_summaries(self, transcriptions: Dict[str, TranscriptionResult],
                                insights_map: Optional[Dict[str, CallInsights]] = None) -> Dict[str, CallSummary]:
        """Generate summaries for multiple calls.
        
        Args:
            transcriptions: Dictionary mapping call IDs to transcription results
            insights_map: Optional dictionary mapping call IDs to insights
            
        Returns:
            Dictionary mapping call IDs to call summaries
        """
        summaries = {}
        
        self.logger.info(f"Generating summaries for {len(transcriptions)} calls")
        
        for call_id, transcription in transcriptions.items():
            try:
                insights = insights_map.get(call_id) if insights_map else None
                summary = self.generate_summary(transcription, insights, call_id)
                summaries[call_id] = summary
                
            except Exception as e:
                self.logger.error(f"Failed to generate summary for {call_id}: {e}")
                continue
                
        self.logger.info(f"Generated {len(summaries)}/{len(transcriptions)} summaries successfully")
        return summaries
        
    def format_summary(self, summary: CallSummary, format_type: str = "text") -> str:
        """Format summary for display or export.
        
        Args:
            summary: CallSummary object
            format_type: Output format (text, json, markdown)
            
        Returns:
            Formatted summary string
        """
        if format_type == "json":
            import json
            return json.dumps(summary.__dict__, indent=2, ensure_ascii=False, default=str)
            
        elif format_type == "markdown":
            md = []
            md.append(f"# Call Summary - {summary.call_id}")
            md.append(f"\n**Summary:** {summary.summary}")
            md.append(f"\n**Key Points:**")
            for point in summary.key_points:
                md.append(f"- {point}")
            md.append(f"\n**Action Items:**")
            for item in summary.action_items:
                md.append(f"- {item}")
            md.append(f"\n**Follow-up Required:** {'Yes' if summary.follow_up_required else 'No'}")
            md.append(f"\n**Confidence Score:** {summary.confidence_score:.2f}")
            return "\n".join(md)
            
        else:  # text format
            text = []
            text.append(f"Call ID: {summary.call_id}")
            text.append(f"Summary: {summary.summary}")
            text.append(f"Key Points: {', '.join(summary.key_points)}")
            text.append(f"Action Items: {', '.join(summary.action_items)}")
            text.append(f"Follow-up Required: {'Yes' if summary.follow_up_required else 'No'}")
            text.append(f"Confidence: {summary.confidence_score:.2f}")
            return "\n".join(text)


def main():
    """CLI interface for testing summary generation."""
    import argparse
    from .stt_module import SpeechToTextProcessor
    from .insight_extractor import InsightExtractor
    
    parser = argparse.ArgumentParser(description="Summary Generation")
    parser.add_argument("--audio", required=True, help="Audio file path")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--format", default="text", choices=["text", "json", "markdown"])
    parser.add_argument("--use-openai", action="store_true", help="Use OpenAI API")
    parser.add_argument("--with-insights", action="store_true", help="Include insights analysis")
    
    args = parser.parse_args()
    
    # Initialize processors
    stt_processor = SpeechToTextProcessor()
    summary_generator = SummaryGenerator(use_openai=args.use_openai)
    
    insights = None
    if args.with_insights:
        insight_extractor = InsightExtractor(use_openai=args.use_openai)
    
    # Process audio
    transcription = stt_processor.transcribe_audio(args.audio)
    
    # Extract insights if requested
    if args.with_insights:
        insights = insight_extractor.extract_insights(transcription, "test_call")
    
    # Generate summary
    summary = summary_generator.generate_summary(transcription, insights, "test_call")
    
    # Output results
    formatted_summary = summary_generator.format_summary(summary, args.format)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(formatted_summary)
        print(f"Summary saved to: {args.output}")
    else:
        print(formatted_summary)


if __name__ == "__main__":
    main()
