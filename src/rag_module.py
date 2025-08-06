"""RAG (Retrieval-Augmented Generation) module for customer support recommendations."""

import os
import json
import openai
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

# Try to import required RAG libraries
try:
    from sentence_transformers import SentenceTransformer
    import chromadb
    from chromadb.config import Settings
    RAG_LIBRARIES_AVAILABLE = True
except ImportError:
    RAG_LIBRARIES_AVAILABLE = False

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from .config import config_manager
from .utils import setup_logging, chunk_text, save_json, load_json
from .insight_extractor import CallInsights, Intent
from .summarizer import CallSummary


@dataclass
class RAGRecommendation:
    """RAG-generated recommendation for customer support follow-up."""
    call_id: str
    recommendations: List[str]
    relevant_documents: List[Dict[str, Any]]
    confidence_score: float
    reasoning: str
    suggested_actions: List[str]
    escalation_needed: bool
    follow_up_timeline: str
    metadata: Dict[str, Any]


@dataclass
class KnowledgeDocument:
    """Knowledge base document."""
    id: str
    title: str
    content: str
    category: str
    tags: List[str]
    metadata: Dict[str, Any]


class RAGSystem:
    """Advanced RAG system for customer support recommendations."""
    
    def __init__(self, knowledge_base_path: str = "data/documents"):
        """Initialize the RAG system.
        
        Args:
            knowledge_base_path: Path to knowledge base documents
        """
        self.logger = setup_logging()
        self.config = config_manager
        self.knowledge_base_path = Path(knowledge_base_path)
        
        # Initialize components
        self.embedding_model = None
        self.vector_store = None
        self.text_splitter = None
        
        # Setup RAG components
        self._setup_embedding_model()
        self._setup_vector_store()
        self._setup_text_splitter()
        
        # Load knowledge base
        self.documents = []
        self._load_knowledge_base()
        
        # OpenAI setup
        self._setup_openai()
        
    def _setup_embedding_model(self) -> None:
        """Setup sentence transformer for embeddings."""
        if not RAG_LIBRARIES_AVAILABLE:
            self.logger.warning("RAG libraries not available. Limited functionality.")
            return
            
        try:
            rag_config = self.config.get_rag_config()
            self.embedding_model = SentenceTransformer(rag_config.embedding_model)
            self.logger.info(f"Embedding model loaded: {rag_config.embedding_model}")
            
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            
    def _setup_vector_store(self) -> None:
        """Setup ChromaDB vector store."""
        if not RAG_LIBRARIES_AVAILABLE:
            return
            
        try:
            # Create persistent ChromaDB client
            db_path = "data/vector_store"
            os.makedirs(db_path, exist_ok=True)
            
            self.chroma_client = chromadb.PersistentClient(path=db_path)
            
            # Get or create collection
            collection_name = "customer_support_kb"
            try:
                self.vector_store = self.chroma_client.get_collection(collection_name)
                self.logger.info(f"Loaded existing vector store: {collection_name}")
            except:
                self.vector_store = self.chroma_client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                self.logger.info(f"Created new vector store: {collection_name}")
                
        except Exception as e:
            self.logger.error(f"Failed to setup vector store: {e}")
            
    def _setup_text_splitter(self) -> None:
        """Setup text splitter for document chunking."""
        if LANGCHAIN_AVAILABLE:
            rag_config = self.config.get_rag_config()
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=rag_config.chunk_size,
                chunk_overlap=rag_config.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            self.logger.info("LangChain text splitter initialized")
        else:
            self.logger.info("Using simple text chunking")
            
    def _setup_openai(self) -> None:
        """Setup OpenRouter API for generation."""
        try:
            openai_config = self.config.get_openai_config()
            if openai_config.api_key:
                openai.api_key = openai_config.api_key
                openai.api_base = "https://openrouter.ai/api/v1"  # ✅ Set base for OpenRouter
                self.openai_model = openai_config.model
                self.logger.info("OpenRouter API configured for RAG generation")
            else:
                self.logger.warning("OpenRouter API key not found")
        except Exception as e:
            self.logger.error(f"Failed to setup OpenRouter: {e}")

            
    def _load_knowledge_base(self) -> None:
        """Load knowledge base documents from files."""
        if not self.knowledge_base_path.exists():
            self.logger.warning(f"Knowledge base path not found: {self.knowledge_base_path}")
            self._create_sample_knowledge_base()
            return
            
        # Load documents from JSON files
        for json_file in self.knowledge_base_path.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    doc_data = json.load(f)
                    
                document = KnowledgeDocument(
                    id=doc_data.get("id", json_file.stem),
                    title=doc_data.get("title", ""),
                    content=doc_data.get("content", ""),
                    category=doc_data.get("category", "general"),
                    tags=doc_data.get("tags", []),
                    metadata=doc_data.get("metadata", {})
                )
                
                self.documents.append(document)
                
            except Exception as e:
                self.logger.error(f"Failed to load document {json_file}: {e}")
                
        # Load documents from text files
        for txt_file in self.knowledge_base_path.glob("*.txt"):
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                document = KnowledgeDocument(
                    id=txt_file.stem,
                    title=txt_file.stem.replace("_", " ").title(),
                    content=content,
                    category="general",
                    tags=[],
                    metadata={"source": str(txt_file)}
                )
                
                self.documents.append(document)
                
            except Exception as e:
                self.logger.error(f"Failed to load document {txt_file}: {e}")
                
        self.logger.info(f"Loaded {len(self.documents)} knowledge base documents")
        
        # Index documents in vector store
        if self.documents:
            self._index_documents()
            
    def _create_sample_knowledge_base(self) -> None:
        """Create sample knowledge base documents."""
        os.makedirs(self.knowledge_base_path, exist_ok=True)
        
        sample_docs = [
            {
                "id": "login_issues",
                "title": "Login and Authentication Issues",
                "content": """
                Common login issues and solutions:
                
                1. Forgot Password:
                - Use 'Forgot Password' link on login page
                - Check email for reset instructions
                - Reset password using security questions
                
                2. Account Locked:
                - Wait 30 minutes for automatic unlock
                - Contact support for immediate unlock
                - Verify account information
                
                3. Two-Factor Authentication Issues:
                - Check authenticator app time sync
                - Use backup codes if available
                - Contact support to reset 2FA
                
                4. Browser Issues:
                - Clear browser cache and cookies
                - Try incognito/private mode
                - Update browser to latest version
                """,
                "category": "technical_support",
                "tags": ["login", "password", "authentication", "2fa", "browser"],
                "metadata": {"priority": "high"}
            },
            {
                "id": "billing_faq",
                "title": "Billing and Payment FAQ",
                "content": """
                Billing and payment information:
                
                1. Payment Methods:
                - Credit/debit cards accepted
                - PayPal payments supported
                - Bank transfer available
                - No cash payments accepted
                
                2. Billing Cycles:
                - Monthly billing on signup date
                - Annual billing offers 20% discount
                - Pro-rated charges for upgrades
                
                3. Refund Policy:
                - 30-day money-back guarantee
                - Refunds processed within 5-7 business days
                - Partial refunds for downgrades
                
                4. Common Billing Issues:
                - Failed payments: Update payment method
                - Duplicate charges: Contact support immediately
                - Invoice questions: Check billing history
                """,
                "category": "billing",
                "tags": ["billing", "payment", "refund", "invoice", "subscription"],
                "metadata": {"priority": "medium"}
            },
            {
                "id": "service_policies",
                "title": "Service Policies and Procedures",
                "content": """
                Customer service policies:
                
                1. Response Times:
                - Email support: 24 hours
                - Live chat: Immediate
                - Phone support: Business hours only
                
                2. Escalation Process:
                - Level 1: General support agents
                - Level 2: Technical specialists
                - Level 3: Senior management
                
                3. Service Level Agreements:
                - 99.9% uptime guarantee
                - Priority support for premium customers
                - Emergency support available 24/7
                
                4. Data Privacy:
                - GDPR compliant
                - Customer data encrypted
                - No data sharing with third parties
                """,
                "category": "policies",
                "tags": ["policy", "sla", "privacy", "escalation", "response_time"],
                "metadata": {"priority": "low"}
            }
        ]
        
        for doc in sample_docs:
            file_path = self.knowledge_base_path / f"{doc['id']}.json"
            save_json(doc, str(file_path))
            
            # Create document object
            document = KnowledgeDocument(
                id=doc["id"],
                title=doc["title"],
                content=doc["content"],
                category=doc["category"],
                tags=doc["tags"],
                metadata=doc["metadata"]
            )
            
            self.documents.append(document)
            
        self.logger.info("Created sample knowledge base")
        
    def _index_documents(self) -> None:
        """Index documents in vector store for similarity search."""
        if not self.embedding_model or not self.vector_store:
            self.logger.warning("Cannot index documents - missing components")
            return
            
        try:
            # Check if documents are already indexed
            existing_count = self.vector_store.count()
            if existing_count > 0:
                self.logger.info(f"Vector store already contains {existing_count} documents")
                return
                
            # Prepare documents for indexing
            texts = []
            metadatas = []
            ids = []
            
            for doc in self.documents:
                # Split document into chunks
                if self.text_splitter:
                    chunks = self.text_splitter.split_text(doc.content)
                else:
                    chunks = chunk_text(doc.content)
                    
                for i, chunk in enumerate(chunks):
                    if len(chunk.strip()) < 50:  # Skip very short chunks
                        continue
                        
                    texts.append(chunk)
                    ids.append(f"{doc.id}_chunk_{i}")
                    metadatas.append({
                        "document_id": doc.id,
                        "title": doc.title,
                        "category": doc.category,
                        "tags": ",".join(doc.tags),
                        "chunk_index": i
                    })
                    
            if not texts:
                self.logger.warning("No text chunks to index")
                return
                
            # Generate embeddings
            self.logger.info(f"Generating embeddings for {len(texts)} text chunks")
            embeddings = self.embedding_model.encode(texts)
            
            # Add to vector store
            self.vector_store.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            self.logger.info(f"Indexed {len(texts)} document chunks in vector store")
            
        except Exception as e:
            self.logger.error(f"Failed to index documents: {e}")
            
    def retrieve_relevant_documents(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            
        Returns:
            List of relevant documents with scores
        """
        if not self.embedding_model or not self.vector_store:
            self.logger.warning("Cannot retrieve documents - missing components")
            return []
            
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            
            # Search vector store
            results = self.vector_store.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k
            )
            
            # Format results
            relevant_docs = []
            for i in range(len(results['ids'][0])):
                doc = {
                    "id": results['ids'][0][i],
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "similarity_score": 1 - results['distances'][0][i] if 'distances' in results else 1.0
                }
                relevant_docs.append(doc)
                
            self.logger.info(f"Retrieved {len(relevant_docs)} relevant documents")
            return relevant_docs
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve documents: {e}")
            return []
            
    def generate_recommendations(self, insights: CallInsights, 
                               summary: Optional[CallSummary] = None) -> RAGRecommendation:
        """Generate RAG-powered recommendations for customer support follow-up.
        
        Args:
            insights: Call insights from analysis
            summary: Optional call summary
            
        Returns:
            RAGRecommendation with generated recommendations
        """
        self.logger.info(f"Generating RAG recommendations for call: {insights.call_id}")
        
        try:
            # Build query from insights
            query = self._build_query_from_insights(insights)
            
            # Retrieve relevant documents
            relevant_docs = self.retrieve_relevant_documents(query, top_k=3)
            
            # Generate recommendations using OpenAI
            if openai.api_key:
                recommendations, reasoning, confidence = self._generate_openai_recommendations(
                    insights, relevant_docs, summary
                )
            else:
                recommendations, reasoning, confidence = self._generate_rule_based_recommendations(
                    insights, relevant_docs
                )
                
            # Determine escalation and timeline
            escalation_needed = self._assess_escalation_need(insights)
            follow_up_timeline = self._determine_follow_up_timeline(insights)
            
            # Generate suggested actions
            suggested_actions = self._generate_suggested_actions(insights, relevant_docs)
            
            # Create recommendation object
            recommendation = RAGRecommendation(
                call_id=insights.call_id,
                recommendations=recommendations,
                relevant_documents=relevant_docs,
                confidence_score=confidence,
                reasoning=reasoning,
                suggested_actions=suggested_actions,
                escalation_needed=escalation_needed,
                follow_up_timeline=follow_up_timeline,
                metadata={
                    "query": query,
                    "num_relevant_docs": len(relevant_docs),
                    "generation_method": "openai" if openai.api_key else "rule_based",
                    "customer_sentiment": insights.customer_sentiment.value,
                    "primary_intent": insights.primary_intent.value
                }
            )
            
            self.logger.info(f"Generated recommendations for call: {insights.call_id}")
            return recommendation
            
        except Exception as e:
            self.logger.error(f"Failed to generate recommendations: {e}")
            raise
            
    def _build_query_from_insights(self, insights: CallInsights) -> str:
        """Build search query from call insights.
        
        Args:
            insights: Call insights
            
        Returns:
            Search query string
        """
        query_parts = []
        
        # Add primary intent
        query_parts.append(insights.primary_intent.value)
        
        # Add key phrases
        query_parts.extend(insights.key_phrases[:3])  # Top 3 key phrases
        
        # Add context based on sentiment and tonality
        if insights.customer_sentiment.value == "negative":
            query_parts.append("complaint resolution")
            
        if insights.customer_tonality.value in ["angry", "frustrated"]:
            query_parts.append("customer satisfaction")
            
        if insights.urgency_level == "high":
            query_parts.append("urgent support")
            
        return " ".join(query_parts)
        
    def _generate_openai_recommendations(self, insights: CallInsights, 
                                       relevant_docs: List[Dict[str, Any]],
                                       summary: Optional[CallSummary] = None) -> Tuple[List[str], str, float]:
        """Generate recommendations using OpenAI API.
        
        Args:
            insights: Call insights
            relevant_docs: Retrieved relevant documents
            summary: Optional call summary
            
        Returns:
            Tuple of (recommendations, reasoning, confidence_score)
        """
        try:
            # Prepare context
            context = self._prepare_context_for_generation(insights, relevant_docs, summary)
            
            prompt = f"""
            You are an expert customer support manager. Based on the call analysis and relevant company knowledge, provide specific recommendations for follow-up actions.

            Call Analysis:
            - Customer Sentiment: {insights.customer_sentiment.value}
            - Customer Tonality: {insights.customer_tonality.value}  
            - Primary Intent: {insights.primary_intent.value}
            - Urgency Level: {insights.urgency_level}
            - Resolution Status: {insights.resolution_status}

            {context}

            Provide 3-5 specific, actionable recommendations for follow-up. Each recommendation should be practical and based on the available knowledge base information.

            Format your response as:
            RECOMMENDATIONS:
            1. [First recommendation]
            2. [Second recommendation]
            3. [Third recommendation]
            ...

            REASONING:
            [Brief explanation of why these recommendations are appropriate]
            """
            
            response = openai.ChatCompletion.create(
                model=self.openai_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3
            )
            
            result = response.choices[0].message.content.strip()
            
            # Parse response
            recommendations, reasoning = self._parse_openai_response(result)
            
            return recommendations, reasoning, 0.9
            
        except Exception as e:
            self.logger.warning(f"OpenAI recommendation generation failed: {e}")
            return self._generate_rule_based_recommendations(insights, relevant_docs)
            
    def _generate_rule_based_recommendations(self, insights: CallInsights,
                                           relevant_docs: List[Dict[str, Any]]) -> Tuple[List[str], str, float]:
        """Generate recommendations using rule-based approach.
        
        Args:
            insights: Call insights
            relevant_docs: Retrieved relevant documents
            
        Returns:
            Tuple of (recommendations, reasoning, confidence_score)
        """
        recommendations = []
        
        # Intent-based recommendations
        if insights.primary_intent == Intent.COMPLAINT:
            recommendations.append("Follow up within 24 hours to ensure customer satisfaction")
            recommendations.append("Review complaint details and identify improvement areas")
            
        elif insights.primary_intent == Intent.TECHNICAL_SUPPORT:
            recommendations.append("Verify technical issue resolution with customer")
            recommendations.append("Provide additional troubleshooting documentation")
            
        elif insights.primary_intent == Intent.BILLING:
            recommendations.append("Send billing clarification email to customer")
            recommendations.append("Review billing processes to prevent similar issues")
            
        # Sentiment-based recommendations
        if insights.customer_sentiment.value == "negative":
            recommendations.append("Proactive outreach to improve customer satisfaction")
            
        # Urgency-based recommendations
        if insights.urgency_level == "high":
            recommendations.append("Priority handling and immediate supervisor notification")
            
        # Resolution status-based recommendations
        if insights.resolution_status == "unresolved":
            recommendations.append("Schedule follow-up call to complete resolution")
            
        # Add document-based recommendations
        for doc in relevant_docs[:2]:  # Use top 2 relevant documents
            if "follow" in doc["content"].lower():
                recommendations.append("Implement documented follow-up procedures")
                
        reasoning = f"Recommendations based on {insights.primary_intent.value} intent, {insights.customer_sentiment.value} sentiment, and {insights.urgency_level} urgency level."
        
        return recommendations[:5], reasoning, 0.7
        
    def _parse_openai_response(self, response: str) -> Tuple[List[str], str]:
        """Parse OpenAI response into recommendations and reasoning.
        
        Args:
            response: Raw OpenAI response
            
        Returns:
            Tuple of (recommendations_list, reasoning_text)
        """
        recommendations = []
        reasoning = ""
        
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if line.upper().startswith('RECOMMENDATIONS:'):
                current_section = "recommendations"
                continue
            elif line.upper().startswith('REASONING:'):
                current_section = "reasoning"
                continue
                
            if current_section == "recommendations" and line:
                # Extract numbered or bulleted recommendations
                if line.startswith(('1.', '2.', '3.', '4.', '5.', '-', '•')):
                    rec = line.split('.', 1)[-1].strip() if '.' in line else line[1:].strip()
                    recommendations.append(rec)
                    
            elif current_section == "reasoning" and line:
                reasoning += line + " "
                
        return recommendations, reasoning.strip()
        
    def _prepare_context_for_generation(self, insights: CallInsights,
                                      relevant_docs: List[Dict[str, Any]],
                                      summary: Optional[CallSummary] = None) -> str:
        """Prepare context for recommendation generation.
        
        Args:
            insights: Call insights
            relevant_docs: Retrieved documents
            summary: Optional call summary
            
        Returns:
            Context string for generation
        """
        context_parts = []
        
        # Add call summary if available
        if summary:
            context_parts.append(f"Call Summary: {summary.summary}")
            
        # Add relevant knowledge base content
        if relevant_docs:
            context_parts.append("Relevant Knowledge Base Information:")
            for doc in relevant_docs[:2]:  # Limit to prevent context overflow
                context_parts.append(f"- {doc['content'][:200]}...")
                
        return "\n".join(context_parts)
        
    def _assess_escalation_need(self, insights: CallInsights) -> bool:
        """Assess if escalation is needed based on insights.
        
        Args:
            insights: Call insights
            
        Returns:
            Boolean indicating if escalation is needed
        """
        # Escalation criteria
        if insights.urgency_level == "high":
            return True
            
        if insights.customer_sentiment.value == "negative" and insights.customer_tonality.value == "angry":
            return True
            
        if insights.resolution_status == "escalated":
            return True
            
        return False
        
    def _determine_follow_up_timeline(self, insights: CallInsights) -> str:
        """Determine appropriate follow-up timeline.
        
        Args:
            insights: Call insights
            
        Returns:
            Follow-up timeline string
        """
        if insights.urgency_level == "high":
            return "within 4 hours"
        elif insights.customer_sentiment.value == "negative":
            return "within 24 hours"
        elif insights.resolution_status == "unresolved":
            return "within 48 hours"
        else:
            return "within 1 week"
            
    def _generate_suggested_actions(self, insights: CallInsights,
                                  relevant_docs: List[Dict[str, Any]]) -> List[str]:
        """Generate specific suggested actions.
        
        Args:
            insights: Call insights
            relevant_docs: Retrieved documents
            
        Returns:
            List of suggested actions
        """
        actions = []
        
        # Based on intent
        if insights.primary_intent == Intent.TECHNICAL_SUPPORT:
            actions.append("Send technical documentation")
            actions.append("Schedule technical follow-up call")
            
        elif insights.primary_intent == Intent.BILLING:
            actions.append("Send detailed billing explanation")
            actions.append("Offer billing consultation")
            
        # Based on resolution status
        if insights.resolution_status == "unresolved":
            actions.append("Assign to specialist team")
            actions.append("Create follow-up ticket")
            
        # Based on sentiment
        if insights.customer_sentiment.value == "negative":
            actions.append("Customer satisfaction survey")
            actions.append("Manager review required")
            
        return actions[:4]  # Limit to 4 actions
        
    def format_recommendation(self, recommendation: RAGRecommendation, 
                            format_type: str = "text") -> str:
        """Format recommendation for display or export.
        
        Args:
            recommendation: RAGRecommendation object
            format_type: Output format (text, json, markdown)
            
        Returns:
            Formatted recommendation string
        """
        if format_type == "json":
            import json
            return json.dumps(recommendation.__dict__, indent=2, ensure_ascii=False, default=str)
            
        elif format_type == "markdown":
            md = []
            md.append(f"# RAG Recommendations - {recommendation.call_id}")
            md.append(f"\n**Confidence Score:** {recommendation.confidence_score:.2f}")
            md.append(f"\n**Reasoning:** {recommendation.reasoning}")
            md.append(f"\n## Recommendations:")
            for i, rec in enumerate(recommendation.recommendations, 1):
                md.append(f"{i}. {rec}")
            md.append(f"\n## Suggested Actions:")
            for action in recommendation.suggested_actions:
                md.append(f"- {action}")
            md.append(f"\n**Escalation Needed:** {'Yes' if recommendation.escalation_needed else 'No'}")
            md.append(f"**Follow-up Timeline:** {recommendation.follow_up_timeline}")
            return "\n".join(md)
            
        else:  # text format
            text = []
            text.append(f"Call ID: {recommendation.call_id}")
            text.append(f"Confidence: {recommendation.confidence_score:.2f}")
            text.append(f"Reasoning: {recommendation.reasoning}")
            text.append("Recommendations:")
            for i, rec in enumerate(recommendation.recommendations, 1):
                text.append(f"  {i}. {rec}")
            text.append("Suggested Actions:")
            for action in recommendation.suggested_actions:
                text.append(f"  - {action}")
            text.append(f"Escalation Needed: {'Yes' if recommendation.escalation_needed else 'No'}")
            text.append(f"Follow-up Timeline: {recommendation.follow_up_timeline}")
            return "\n".join(text)


def main():
    """CLI interface for testing RAG system."""
    import argparse
    from .stt_module import SpeechToTextProcessor
    from .insight_extractor import InsightExtractor
    from .summarizer import SummaryGenerator
    
    parser = argparse.ArgumentParser(description="RAG Recommendations")
    parser.add_argument("--audio", required=True, help="Audio file path")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--format", default="text", choices=["text", "json", "markdown"])
    parser.add_argument("--kb-path", default="data/documents", help="Knowledge base path")
    
    args = parser.parse_args()
    
    # Initialize all processors
    stt_processor = SpeechToTextProcessor()
    insight_extractor = InsightExtractor()
    summary_generator = SummaryGenerator()
    rag_system = RAGSystem(args.kb_path)
    
    # Process audio through complete pipeline
    print("Processing audio...")
    transcription = stt_processor.transcribe_audio(args.audio)
    
    print("Extracting insights...")
    insights = insight_extractor.extract_insights(transcription, "test_call")
    
    print("Generating summary...")
    summary = summary_generator.generate_summary(transcription, insights, "test_call")
    
    print("Generating RAG recommendations...")
    recommendation = rag_system.generate_recommendations(insights, summary)
    
    # Format and output results
    formatted_recommendation = rag_system.format_recommendation(recommendation, args.format)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(formatted_recommendation)
        print(f"Recommendations saved to: {args.output}")
    else:
        print("\n" + "="*50)
        print("RAG RECOMMENDATIONS")
        print("="*50)
        print(formatted_recommendation)


if __name__ == "__main__":
    main()
