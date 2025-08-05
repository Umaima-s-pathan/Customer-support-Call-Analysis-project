"""Insight extraction module for analyzing customer support calls."""

import re
import json
import openai
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# Try to import transformers for local models
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from .config import config_manager
from .utils import setup_logging
from .stt_module import TranscriptionResult


class Sentiment(Enum):
    """Sentiment categories."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class Tonality(Enum):
    """Tonality categories."""
    CALM = "calm"
    ANGRY = "angry"
    POLITE = "polite"
    FRUSTRATED = "frustrated"
    ANXIOUS = "anxious"
    SATISFIED = "satisfied"
    CONFUSED = "confused"


class Intent(Enum):
    """Intent categories."""
    COMPLAINT = "complaint"
    QUERY = "query"
    FEEDBACK = "feedback"
    REQUEST = "request"
    CANCELLATION = "cancellation"
    TECHNICAL_SUPPORT = "technical_support"
    BILLING = "billing"
    GENERAL_INFO = "general_info"


@dataclass
class CallInsights:
    """Structured insights from a customer call."""
    call_id: str
    overall_sentiment: Sentiment
    customer_sentiment: Sentiment
    agent_sentiment: Sentiment
    customer_tonality: Tonality
    agent_tonality: Tonality
    primary_intent: Intent
    secondary_intents: List[Intent]
    confidence_scores: Dict[str, float]
    key_phrases: List[str]
    emotion_analysis: Dict[str, float]
    urgency_level: str  # low, medium, high
    resolution_status: str  # resolved, unresolved, escalated
    metadata: Dict[str, Any]


class InsightExtractor:
    """Advanced insight extraction system for customer support calls."""
    
    def __init__(self, use_openai: bool = True):
        """Initialize the insight extractor.
        
        Args:
            use_openai: Whether to use OpenAI API for analysis
        """
        self.logger = setup_logging()
        self.config = config_manager
        self.use_openai = use_openai
        
        # Initialize OpenAI if available
        if self.use_openai:
            self._setup_openai()
            
        # Initialize local models
        self.local_models = {}
        if TRANSFORMERS_AVAILABLE:
            self._setup_local_models()
            
    def _setup_openai(self) -> None:
        """Setup OpenAI API client."""
        try:
            openai_config = self.config.get_openai_config()
            if openai_config.api_key:
                openai.api_key = openai_config.api_key
                self.openai_model = openai_config.model
                self.logger.info("OpenAI API configured successfully")
            else:
                self.logger.warning("OpenAI API key not found")
                self.use_openai = False
        except Exception as e:
            self.logger.error(f"Failed to setup OpenAI: {e}")
            self.use_openai = False
            
    def _setup_local_models(self) -> None:
        """Setup local transformer models."""
        try:
            # Sentiment analysis model
            self.local_models['sentiment'] = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=-1  # CPU
            )
            
            # Emotion analysis model
            self.local_models['emotion'] = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                device=-1
            )
            
            self.logger.info("Local models loaded successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to load local models: {e}")
            
    def extract_insights(self, transcription: TranscriptionResult, 
                        call_id: str = "unknown") -> CallInsights:
        """Extract comprehensive insights from transcription.
        
        Args:
            transcription: Transcription result from STT module
            call_id: Unique identifier for the call
            
        Returns:
            CallInsights object with extracted insights
        """
        self.logger.info(f"Extracting insights for call: {call_id}")
        
        try:
            # Separate customer and agent text
            customer_text, agent_text = self._separate_speakers(transcription)
            
            # Extract sentiment
            overall_sentiment, customer_sentiment, agent_sentiment, sentiment_scores = \
                self._analyze_sentiment(transcription.full_text, customer_text, agent_text)
            
            # Extract tonality
            customer_tonality, agent_tonality, tonality_scores = \
                self._analyze_tonality(customer_text, agent_text)
            
            # Extract intent
            primary_intent, secondary_intents, intent_scores = \
                self._analyze_intent(transcription.full_text, customer_text)
            
            # Extract key phrases
            key_phrases = self._extract_key_phrases(transcription.full_text)
            
            # Emotion analysis
            emotion_analysis = self._analyze_emotions(customer_text)
            
            # Urgency and resolution analysis
            urgency_level = self._assess_urgency(customer_text, customer_tonality)
            resolution_status = self._assess_resolution(transcription.full_text, agent_text)
            
            # Combine confidence scores
            confidence_scores = {
                **sentiment_scores,
                **tonality_scores,
                **intent_scores
            }
            
            # Create insights object
            insights = CallInsights(
                call_id=call_id,
                overall_sentiment=overall_sentiment,
                customer_sentiment=customer_sentiment,
                agent_sentiment=agent_sentiment,
                customer_tonality=customer_tonality,
                agent_tonality=agent_tonality,
                primary_intent=primary_intent,
                secondary_intents=secondary_intents,
                confidence_scores=confidence_scores,
                key_phrases=key_phrases,
                emotion_analysis=emotion_analysis,
                urgency_level=urgency_level,
                resolution_status=resolution_status,
                metadata={
                    "total_segments": len(transcription.segments),
                    "call_duration": transcription.duration,
                    "language": transcription.language,
                    "processing_method": "openai" if self.use_openai else "local"
                }
            )
            
            self.logger.info(f"Insights extracted successfully for call: {call_id}")
            return insights
            
        except Exception as e:
            self.logger.error(f"Failed to extract insights: {e}")
            raise
            
    def _separate_speakers(self, transcription: TranscriptionResult) -> Tuple[str, str]:
        """Separate customer and agent text from transcription.
        
        Args:
            transcription: Transcription result
            
        Returns:
            Tuple of (customer_text, agent_text)
        """
        customer_segments = []
        agent_segments = []
        
        for segment in transcription.segments:
            if segment.speaker.lower() == "customer":
                customer_segments.append(segment.text)
            else:
                agent_segments.append(segment.text)
                
        customer_text = " ".join(customer_segments)
        agent_text = " ".join(agent_segments)
        
        return customer_text, agent_text
        
    def _analyze_sentiment(self, full_text: str, customer_text: str, 
                          agent_text: str) -> Tuple[Sentiment, Sentiment, Sentiment, Dict[str, float]]:
        """Analyze sentiment for overall call, customer, and agent.
        
        Args:
            full_text: Complete conversation text
            customer_text: Customer's text only
            agent_text: Agent's text only
            
        Returns:
            Tuple of (overall_sentiment, customer_sentiment, agent_sentiment, confidence_scores)
        """
        confidence_scores = {}
        
        if self.use_openai:
            # Use OpenAI for sentiment analysis
            overall_sentiment, overall_score = self._openai_sentiment_analysis(full_text)
            customer_sentiment, customer_score = self._openai_sentiment_analysis(customer_text)
            agent_sentiment, agent_score = self._openai_sentiment_analysis(agent_text)
            
            confidence_scores.update({
                "overall_sentiment": overall_score,
                "customer_sentiment": customer_score,
                "agent_sentiment": agent_score
            })
            
        elif 'sentiment' in self.local_models:
            # Use local model for sentiment analysis
            overall_sentiment, overall_score = self._local_sentiment_analysis(full_text)
            customer_sentiment, customer_score = self._local_sentiment_analysis(customer_text)
            agent_sentiment, agent_score = self._local_sentiment_analysis(agent_text)
            
            confidence_scores.update({
                "overall_sentiment": overall_score,
                "customer_sentiment": customer_score,
                "agent_sentiment": agent_score
            })
            
        else:
            # Fallback to rule-based sentiment analysis
            overall_sentiment = self._rule_based_sentiment(full_text)
            customer_sentiment = self._rule_based_sentiment(customer_text)
            agent_sentiment = self._rule_based_sentiment(agent_text)
            
            confidence_scores.update({
                "overall_sentiment": 0.7,
                "customer_sentiment": 0.7,
                "agent_sentiment": 0.7
            })
            
        return overall_sentiment, customer_sentiment, agent_sentiment, confidence_scores
        
    def _openai_sentiment_analysis(self, text: str) -> Tuple[Sentiment, float]:
        """Analyze sentiment using OpenAI API.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (sentiment, confidence_score)
        """
        if not text.strip():
            return Sentiment.NEUTRAL, 0.0
            
        try:
            prompt = f"""
            Analyze the sentiment of the following customer support conversation text.
            Return only one word: "positive", "negative", or "neutral".
            
            Text: {text[:1000]}  # Limit text length
            
            Sentiment:"""
            
            response = openai.ChatCompletion.create(
                model=self.openai_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.1
            )
            
            result = response.choices[0].message.content.strip().lower()
            
            if "positive" in result:
                return Sentiment.POSITIVE, 0.9
            elif "negative" in result:
                return Sentiment.NEGATIVE, 0.9
            else:
                return Sentiment.NEUTRAL, 0.8
                
        except Exception as e:
            self.logger.warning(f"OpenAI sentiment analysis failed: {e}")
            return self._rule_based_sentiment(text), 0.5
            
    def _local_sentiment_analysis(self, text: str) -> Tuple[Sentiment, float]:
        """Analyze sentiment using local model.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (sentiment, confidence_score)
        """
        if not text.strip():
            return Sentiment.NEUTRAL, 0.0
            
        try:
            result = self.local_models['sentiment'](text[:512])  # Limit text length
            
            label = result[0]['label'].lower()
            score = result[0]['score']
            
            if 'positive' in label:
                return Sentiment.POSITIVE, score
            elif 'negative' in label:
                return Sentiment.NEGATIVE, score
            else:
                return Sentiment.NEUTRAL, score
                
        except Exception as e:
            self.logger.warning(f"Local sentiment analysis failed: {e}")
            return self._rule_based_sentiment(text), 0.5
            
    def _rule_based_sentiment(self, text: str) -> Sentiment:
        """Rule-based sentiment analysis as fallback.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment classification
        """
        if not text.strip():
            return Sentiment.NEUTRAL
            
        text_lower = text.lower()
        
        # Positive indicators
        positive_words = ['thank', 'great', 'excellent', 'good', 'satisfied', 'happy', 'resolved']
        positive_count = sum(1 for word in positive_words if word in text_lower)
        
        # Negative indicators
        negative_words = ['problem', 'issue', 'wrong', 'error', 'bad', 'terrible', 'frustrated', 'angry']
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return Sentiment.POSITIVE
        elif negative_count > positive_count:
            return Sentiment.NEGATIVE
        else:
            return Sentiment.NEUTRAL
            
    def _analyze_tonality(self, customer_text: str, 
                         agent_text: str) -> Tuple[Tonality, Tonality, Dict[str, float]]:
        """Analyze tonality for customer and agent.
        
        Args:
            customer_text: Customer's text
            agent_text: Agent's text
            
        Returns:
            Tuple of (customer_tonality, agent_tonality, confidence_scores)
        """
        customer_tonality = self._detect_tonality(customer_text, is_customer=True)
        agent_tonality = self._detect_tonality(agent_text, is_customer=False)
        
        confidence_scores = {
            "customer_tonality": 0.8,
            "agent_tonality": 0.8
        }
        
        return customer_tonality, agent_tonality, confidence_scores
        
    def _detect_tonality(self, text: str, is_customer: bool = True) -> Tonality:
        """Detect tonality using rule-based approach.
        
        Args:
            text: Text to analyze
            is_customer: Whether the text is from customer or agent
            
        Returns:
            Tonality classification
        """
        if not text.strip():
            return Tonality.CALM
            
        text_lower = text.lower()
        
        # Angry indicators
        angry_patterns = ['angry', 'furious', 'mad', 'outraged', 'disgusted']
        if any(pattern in text_lower for pattern in angry_patterns):
            return Tonality.ANGRY
            
        # Frustrated indicators
        frustrated_patterns = ['frustrated', 'annoyed', 'irritated', 'sick of', 'fed up']
        if any(pattern in text_lower for pattern in frustrated_patterns):
            return Tonality.FRUSTRATED
            
        # Anxious indicators
        anxious_patterns = ['worried', 'concerned', 'anxious', 'nervous', 'urgent']
        if any(pattern in text_lower for pattern in anxious_patterns):
            return Tonality.ANXIOUS
            
        # Confused indicators
        confused_patterns = ['confused', 'don\'t understand', 'not sure', 'unclear']
        if any(pattern in text_lower for pattern in confused_patterns):
            return Tonality.CONFUSED
            
        # Satisfied indicators
        satisfied_patterns = ['satisfied', 'happy', 'pleased', 'glad', 'excellent']
        if any(pattern in text_lower for pattern in satisfied_patterns):
            return Tonality.SATISFIED
            
        # Polite indicators
        polite_patterns = ['please', 'thank you', 'sorry', 'excuse me', 'appreciate']
        if any(pattern in text_lower for pattern in polite_patterns):
            return Tonality.POLITE
            
        return Tonality.CALM
        
    def _analyze_intent(self, full_text: str, 
                       customer_text: str) -> Tuple[Intent, List[Intent], Dict[str, float]]:
        """Analyze customer intent from the conversation.
        
        Args:
            full_text: Complete conversation text
            customer_text: Customer's text only
            
        Returns:
            Tuple of (primary_intent, secondary_intents, confidence_scores)
        """
        # Intent detection patterns
        intent_patterns = {
            Intent.COMPLAINT: ['complaint', 'complain', 'issue', 'problem', 'wrong', 'error', 'bad'],
            Intent.QUERY: ['question', 'ask', 'how', 'what', 'when', 'where', 'why', 'help'],
            Intent.FEEDBACK: ['feedback', 'suggest', 'recommend', 'improve', 'opinion'],
            Intent.REQUEST: ['request', 'need', 'want', 'require', 'would like'],
            Intent.CANCELLATION: ['cancel', 'close', 'terminate', 'stop', 'end'],
            Intent.TECHNICAL_SUPPORT: ['not working', 'broken', 'technical', 'bug', 'fix'],
            Intent.BILLING: ['bill', 'charge', 'payment', 'refund', 'money', 'cost'],
            Intent.GENERAL_INFO: ['information', 'details', 'explain', 'tell me']
        }
        
        text_lower = customer_text.lower()
        intent_scores = {}
        
        for intent, patterns in intent_patterns.items():
            score = sum(1 for pattern in patterns if pattern in text_lower)
            intent_scores[intent.value] = score / len(patterns)
            
        # Find primary intent (highest score)
        primary_intent = max(intent_scores.keys(), key=lambda k: intent_scores[k])
        primary_intent = Intent(primary_intent)
        
        # Find secondary intents (score > 0.3)
        secondary_intents = [
            Intent(intent) for intent, score in intent_scores.items() 
            if score > 0.3 and Intent(intent) != primary_intent
        ]
        
        confidence_scores = {f"intent_{intent}": score for intent, score in intent_scores.items()}
        
        return primary_intent, secondary_intents, confidence_scores
        
    def _extract_key_phrases(self, text: str, max_phrases: int = 10) -> List[str]:
        """Extract key phrases from the conversation.
        
        Args:
            text: Text to analyze
            max_phrases: Maximum number of phrases to extract
            
        Returns:
            List of key phrases
        """
        # Simple keyword extraction based on frequency and importance
        important_patterns = [
            r'\b(?:account|billing|payment|refund|technical|support|help|issue|problem)\b',
            r'\b(?:login|password|access|connection|error|bug|fix)\b',
            r'\b(?:order|delivery|shipping|return|exchange)\b',
            r'\b(?:subscription|service|plan|upgrade|downgrade)\b'
        ]
        
        key_phrases = []
        text_lower = text.lower()
        
        for pattern in important_patterns:
            matches = re.findall(pattern, text_lower)
            key_phrases.extend(matches)
            
        # Remove duplicates and limit count
        key_phrases = list(set(key_phrases))[:max_phrases]
        
        return key_phrases
        
    def _analyze_emotions(self, text: str) -> Dict[str, float]:
        """Analyze emotions in customer text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of emotion scores
        """
        if 'emotion' in self.local_models and text.strip():
            try:
                results = self.local_models['emotion'](text[:512])
                emotions = {result['label'].lower(): result['score'] for result in results}
                return emotions
            except Exception as e:
                self.logger.warning(f"Emotion analysis failed: {e}")
                
        # Fallback emotion analysis
        return {
            "joy": 0.1,
            "anger": 0.3,
            "fear": 0.2,
            "sadness": 0.2,
            "surprise": 0.1,
            "disgust": 0.1
        }
        
    def _assess_urgency(self, customer_text: str, tonality: Tonality) -> str:
        """Assess urgency level of the customer's issue.
        
        Args:
            customer_text: Customer's text
            tonality: Customer's tonality
            
        Returns:
            Urgency level (low, medium, high)
        """
        text_lower = customer_text.lower()
        
        # High urgency indicators
        high_urgency = ['urgent', 'emergency', 'asap', 'immediately', 'critical', 'can\'t access']
        if any(indicator in text_lower for indicator in high_urgency):
            return "high"
            
        # Medium urgency based on tonality
        if tonality in [Tonality.ANGRY, Tonality.FRUSTRATED]:
            return "medium"
            
        # Medium urgency indicators
        medium_urgency = ['soon', 'quickly', 'important', 'need help']
        if any(indicator in text_lower for indicator in medium_urgency):
            return "medium"
            
        return "low"
        
    def _assess_resolution(self, full_text: str, agent_text: str) -> str:
        """Assess whether the issue was resolved.
        
        Args:
            full_text: Complete conversation text
            agent_text: Agent's text
            
        Returns:
            Resolution status (resolved, unresolved, escalated)
        """
        text_lower = full_text.lower()
        agent_lower = agent_text.lower()
        
        # Resolved indicators
        resolved_patterns = ['resolved', 'fixed', 'solved', 'working now', 'thank you', 'satisfied']
        if any(pattern in text_lower for pattern in resolved_patterns):
            return "resolved"
            
        # Escalation indicators
        escalation_patterns = ['escalate', 'supervisor', 'manager', 'transfer', 'follow up']
        if any(pattern in agent_lower for pattern in escalation_patterns):
            return "escalated"
            
        return "unresolved"
        
    def format_insights(self, insights: CallInsights, format_type: str = "json") -> str:
        """Format insights for output.
        
        Args:
            insights: CallInsights object
            format_type: Output format (json, text)
            
        Returns:
            Formatted insights string
        """
        if format_type == "json":
            # Convert enums to strings for JSON serialization
            data = asdict(insights)
            for key, value in data.items():
                if isinstance(value, Enum):
                    data[key] = value.value
                elif isinstance(value, list) and value and isinstance(value[0], Enum):
                    data[key] = [item.value for item in value]
                    
            return json.dumps(data, indent=2, ensure_ascii=False)
            
        else:  # text format
            text = []
            text.append(f"Call ID: {insights.call_id}")
            text.append(f"Overall Sentiment: {insights.overall_sentiment.value}")
            text.append(f"Customer Sentiment: {insights.customer_sentiment.value}")
            text.append(f"Agent Sentiment: {insights.agent_sentiment.value}")
            text.append(f"Customer Tonality: {insights.customer_tonality.value}")
            text.append(f"Agent Tonality: {insights.agent_tonality.value}")
            text.append(f"Primary Intent: {insights.primary_intent.value}")
            text.append(f"Secondary Intents: {[intent.value for intent in insights.secondary_intents]}")
            text.append(f"Urgency Level: {insights.urgency_level}")
            text.append(f"Resolution Status: {insights.resolution_status}")
            text.append(f"Key Phrases: {insights.key_phrases}")
            
            return "\n".join(text)


def main():
    """CLI interface for testing insight extraction."""
    import argparse
    from .stt_module import SpeechToTextProcessor
    
    parser = argparse.ArgumentParser(description="Insight Extraction")
    parser.add_argument("--audio", required=True, help="Audio file path")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--format", default="json", choices=["json", "text"])
    parser.add_argument("--use-openai", action="store_true", help="Use OpenAI API")
    
    args = parser.parse_args()
    
    # Initialize processors
    stt_processor = SpeechToTextProcessor()
    insight_extractor = InsightExtractor(use_openai=args.use_openai)
    
    # Process audio and extract insights
    transcription = stt_processor.transcribe_audio(args.audio)
    insights = insight_extractor.extract_insights(transcription, call_id="test_call")
    
    # Output results
    formatted_insights = insight_extractor.format_insights(insights, args.format)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(formatted_insights)
        print(f"Insights saved to: {args.output}")
    else:
        print(formatted_insights)


if __name__ == "__main__":
    main()