"""Evaluation framework for the customer support analysis system."""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
import numpy as np

# Try to import evaluation libraries
try:
    from rouge_score import rouge_scorer
    from evaluate import load
    EVALUATION_LIBRARIES_AVAILABLE = True
except ImportError:
    EVALUATION_LIBRARIES_AVAILABLE = False

try:
    import jiwer
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False

from .config import config_manager
from .utils import setup_logging, calculate_word_error_rate, save_json, load_json
from .stt_module import SpeechToTextProcessor, TranscriptionResult
from .insight_extractor import InsightExtractor, CallInsights
from .summarizer import SummaryGenerator, CallSummary
from .rag_module import RAGSystem, RAGRecommendation


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    stt_metrics: Dict[str, float]
    insight_metrics: Dict[str, float]
    summary_metrics: Dict[str, float]
    rag_metrics: Dict[str, float]
    overall_metrics: Dict[str, float]
    processing_times: Dict[str, float]
    metadata: Dict[str, Any]


@dataclass
class GroundTruthData:
    """Ground truth data for evaluation."""
    call_id: str
    reference_transcription: str
    reference_sentiment: str
    reference_intent: str
    reference_summary: str
    reference_recommendations: List[str]
    metadata: Dict[str, Any]


class SystemEvaluator:
    """Comprehensive evaluation system for customer support analysis."""
    
    def __init__(self, ground_truth_path: str = "data/ground_truth"):
        """Initialize the evaluation system.
        
        Args:
            ground_truth_path: Path to ground truth data
        """
        self.logger = setup_logging()
        self.config = config_manager
        self.ground_truth_path = Path(ground_truth_path)
        
        # Initialize evaluation metrics
        self.rouge_scorer = None
        self.bleu_metric = None
        self.meteor_metric = None
        
        self._setup_evaluation_metrics()
        
        # Load ground truth data
        self.ground_truth_data = []
        self._load_ground_truth_data()
        
    def _setup_evaluation_metrics(self) -> None:
        """Setup evaluation metrics."""
        if EVALUATION_LIBRARIES_AVAILABLE:
            try:
                # ROUGE scorer for summary evaluation
                self.rouge_scorer = rouge_scorer.RougeScorer(
                    ['rouge1', 'rouge2', 'rougeL'], 
                    use_stemmer=True
                )
                
                # BLEU metric
                self.bleu_metric = load("bleu")
                
                # METEOR metric (if available)
                try:
                    self.meteor_metric = load("meteor")
                except:
                    self.logger.warning("METEOR metric not available")
                    
                self.logger.info("Evaluation metrics initialized successfully")
                
            except Exception as e:
                self.logger.warning(f"Failed to initialize some evaluation metrics: {e}")
        else:
            self.logger.warning("Evaluation libraries not available - using simple metrics")
            
    def _load_ground_truth_data(self) -> None:
        """Load ground truth data from files."""
        if not self.ground_truth_path.exists():
            self.logger.warning(f"Ground truth path not found: {self.ground_truth_path}")
            self._create_sample_ground_truth()
            return
            
        # Load ground truth from JSON files
        for json_file in self.ground_truth_path.glob("*.json"):
            try:
                data = load_json(str(json_file))
                
                ground_truth = GroundTruthData(
                    call_id=data.get("call_id", json_file.stem),
                    reference_transcription=data.get("reference_transcription", ""),
                    reference_sentiment=data.get("reference_sentiment", "neutral"),
                    reference_intent=data.get("reference_intent", "query"),
                    reference_summary=data.get("reference_summary", ""),
                    reference_recommendations=data.get("reference_recommendations", []),
                    metadata=data.get("metadata", {})
                )
                
                self.ground_truth_data.append(ground_truth)
                
            except Exception as e:
                self.logger.error(f"Failed to load ground truth {json_file}: {e}")
                
        self.logger.info(f"Loaded {len(self.ground_truth_data)} ground truth records")
        
    def _create_sample_ground_truth(self) -> None:
        """Create sample ground truth data for demonstration."""
        os.makedirs(self.ground_truth_path, exist_ok=True)
        
        sample_ground_truth = [
            {
                "call_id": "sample_call_001",
                "reference_transcription": "Customer: Hi, I'm having trouble logging into my account. Agent: I can help you with that. Let me check your account status. Customer: Thank you. Agent: I've reset your password. Please check your email for the new password. Customer: Got it, thank you so much!",
                "reference_sentiment": "neutral",
                "reference_intent": "technical_support",
                "reference_summary": "Customer experienced login issues. Agent successfully reset the password and provided email instructions. Issue resolved.",
                "reference_recommendations": [
                    "Follow up within 24 hours to ensure login success",
                    "Send password security best practices guide",
                    "Monitor account for any additional issues"
                ],
                "metadata": {
                    "duration": 120.5,
                    "resolution_status": "resolved",
                    "urgency": "medium"
                }
            },
            {
                "call_id": "sample_call_002", 
                "reference_transcription": "Customer: I'm really frustrated with the billing charges on my account. This is the third time this has happened. Agent: I sincerely apologize for the inconvenience. Let me review your billing history immediately. Customer: This needs to be fixed right now. Agent: I understand your frustration. I'm escalating this to our billing specialist team.",
                "reference_sentiment": "negative",
                "reference_intent": "complaint",
                "reference_summary": "Customer complained about recurring billing issues. Agent apologized and escalated to billing specialists. Issue requires immediate attention.",
                "reference_recommendations": [
                    "Immediate escalation to billing manager",
                    "Comprehensive billing review required",
                    "Customer satisfaction follow-up call",
                    "Process improvement analysis"
                ],
                "metadata": {
                    "duration": 180.2,
                    "resolution_status": "escalated",
                    "urgency": "high"
                }
            }
        ]
        
        for i, gt_data in enumerate(sample_ground_truth):
            file_path = self.ground_truth_path / f"ground_truth_{i+1:03d}.json"
            save_json(gt_data, str(file_path))
            
            # Create ground truth object
            ground_truth = GroundTruthData(
                call_id=gt_data["call_id"],
                reference_transcription=gt_data["reference_transcription"],
                reference_sentiment=gt_data["reference_sentiment"],
                reference_intent=gt_data["reference_intent"],
                reference_summary=gt_data["reference_summary"],
                reference_recommendations=gt_data["reference_recommendations"],
                metadata=gt_data["metadata"]
            )
            
            self.ground_truth_data.append(ground_truth)
            
        self.logger.info("Created sample ground truth data")
        
    def evaluate_stt_performance(self, transcription: TranscriptionResult, 
                                ground_truth: GroundTruthData) -> Dict[str, float]:
        """Evaluate Speech-to-Text performance.
        
        Args:
            transcription: STT output
            ground_truth: Reference transcription
            
        Returns:
            Dictionary of STT evaluation metrics
        """
        metrics = {}
        
        predicted_text = transcription.full_text.lower().strip()
        reference_text = ground_truth.reference_transcription.lower().strip()
        
        # Word Error Rate (WER)
        if JIWER_AVAILABLE:
            try:
                wer = jiwer.wer(reference_text, predicted_text)
                metrics['wer'] = wer
            except:
                metrics['wer'] = calculate_word_error_rate(reference_text, predicted_text)
        else:
            metrics['wer'] = calculate_word_error_rate(reference_text, predicted_text)
            
        # Character Error Rate (CER)
        if JIWER_AVAILABLE:
            try:
                cer = jiwer.cer(reference_text, predicted_text)
                metrics['cer'] = cer
            except:
                metrics['cer'] = self._calculate_character_error_rate(reference_text, predicted_text)
        else:
            metrics['cer'] = self._calculate_character_error_rate(reference_text, predicted_text)
            
        # BLEU Score
        if self.bleu_metric:
            try:
                bleu_score = self.bleu_metric.compute(
                    predictions=[predicted_text],
                    references=[[reference_text]]
                )
                metrics['bleu'] = bleu_score['bleu']
            except:
                metrics['bleu'] = 0.0
        else:
            metrics['bleu'] = self._simple_bleu_score(reference_text, predicted_text)
            
        # Additional metrics
        metrics['length_ratio'] = len(predicted_text) / len(reference_text) if reference_text else 0
        metrics['exact_match'] = 1.0 if predicted_text == reference_text else 0.0
        
        return metrics
        
    def evaluate_insight_performance(self, insights: CallInsights,
                                   ground_truth: GroundTruthData) -> Dict[str, float]:
        """Evaluate insight extraction performance.
        
        Args:
            insights: Extracted insights
            ground_truth: Reference insights
            
        Returns:
            Dictionary of insight evaluation metrics
        """
        metrics = {}
        
        # Sentiment accuracy
        predicted_sentiment = insights.customer_sentiment.value
        reference_sentiment = ground_truth.reference_sentiment
        metrics['sentiment_accuracy'] = 1.0 if predicted_sentiment == reference_sentiment else 0.0
        
        # Intent accuracy
        predicted_intent = insights.primary_intent.value
        reference_intent = ground_truth.reference_intent
        metrics['intent_accuracy'] = 1.0 if predicted_intent == reference_intent else 0.0
        
        # Overall classification accuracy
        correct_predictions = 0
        total_predictions = 2  # sentiment + intent
        
        if predicted_sentiment == reference_sentiment:
            correct_predictions += 1
        if predicted_intent == reference_intent:
            correct_predictions += 1
            
        metrics['classification_accuracy'] = correct_predictions / total_predictions
        
        # Confidence scores
        metrics['average_confidence'] = np.mean(list(insights.confidence_scores.values()))
        
        return metrics
        
    def evaluate_summary_performance(self, summary: CallSummary,
                                   ground_truth: GroundTruthData) -> Dict[str, float]:
        """Evaluate summary generation performance.
        
        Args:
            summary: Generated summary
            ground_truth: Reference summary
            
        Returns:
            Dictionary of summary evaluation metrics
        """
        metrics = {}
        
        predicted_summary = summary.summary.lower().strip()
        reference_summary = ground_truth.reference_summary.lower().strip()
        
        # ROUGE scores
        if self.rouge_scorer:
            try:
                rouge_scores = self.rouge_scorer.score(reference_summary, predicted_summary)
                metrics['rouge1_f1'] = rouge_scores['rouge1'].fmeasure
                metrics['rouge2_f1'] = rouge_scores['rouge2'].fmeasure
                metrics['rougeL_f1'] = rouge_scores['rougeL'].fmeasure
            except:
                metrics.update({
                    'rouge1_f1': 0.0,
                    'rouge2_f1': 0.0, 
                    'rougeL_f1': 0.0
                })
        else:
            # Simple word overlap metric
            pred_words = set(predicted_summary.split())
            ref_words = set(reference_summary.split())
            overlap = len(pred_words.intersection(ref_words))
            union = len(pred_words.union(ref_words))
            metrics['word_overlap'] = overlap / union if union > 0 else 0.0
            
        # BLEU score
        if self.bleu_metric:
            try:
                bleu_score = self.bleu_metric.compute(
                    predictions=[predicted_summary],
                    references=[[reference_summary]]
                )
                metrics['bleu'] = bleu_score['bleu']
            except:
                metrics['bleu'] = 0.0
        else:
            metrics['bleu'] = self._simple_bleu_score(reference_summary, predicted_summary)
            
        # METEOR score (if available)
        if self.meteor_metric:
            try:
                meteor_score = self.meteor_metric.compute(
                    predictions=[predicted_summary],
                    references=[[reference_summary]]
                )
                metrics['meteor'] = meteor_score['meteor']
            except:
                metrics['meteor'] = 0.0
                
        # Length metrics
        metrics['length_ratio'] = len(predicted_summary.split()) / len(reference_summary.split()) if reference_summary else 0
        metrics['compression_ratio'] = summary.metadata.get('compression_ratio', 0.0)
        
        return metrics
        
    def evaluate_rag_performance(self, recommendation: RAGRecommendation,
                               ground_truth: GroundTruthData) -> Dict[str, float]:
        """Evaluate RAG recommendation performance.
        
        Args:
            recommendation: Generated recommendations
            ground_truth: Reference recommendations
            
        Returns:
            Dictionary of RAG evaluation metrics
        """
        metrics = {}
        
        predicted_recs = [rec.lower().strip() for rec in recommendation.recommendations]
        reference_recs = [rec.lower().strip() for rec in ground_truth.reference_recommendations]
        
        # Recommendation coverage
        covered_recommendations = 0
        for ref_rec in reference_recs:
            for pred_rec in predicted_recs:
                # Simple similarity check (word overlap)
                ref_words = set(ref_rec.split())
                pred_words = set(pred_rec.split())
                overlap = len(ref_words.intersection(pred_words))
                if overlap >= len(ref_words) * 0.5:  # 50% word overlap threshold
                    covered_recommendations += 1
                    break
                    
        metrics['recommendation_coverage'] = covered_recommendations / len(reference_recs) if reference_recs else 0.0
        
        # Number of recommendations
        metrics['num_recommendations'] = len(predicted_recs)
        metrics['recommendation_length_avg'] = np.mean([len(rec.split()) for rec in predicted_recs]) if predicted_recs else 0
        
        # Confidence score
        metrics['rag_confidence'] = recommendation.confidence_score
        
        # Relevance metrics (simplified)
        metrics['num_relevant_docs'] = len(recommendation.relevant_documents)
        
        return metrics
        
    def evaluate_system_performance(self, audio_files: List[str],
                                  output_path: Optional[str] = None) -> EvaluationMetrics:
        """Evaluate complete system performance on multiple audio files.
        
        Args:
            audio_files: List of audio file paths
            output_path: Optional path to save evaluation results
            
        Returns:
            EvaluationMetrics with comprehensive evaluation results
        """
        self.logger.info(f"Starting system evaluation on {len(audio_files)} files")
        
        # Initialize processors
        stt_processor = SpeechToTextProcessor()
        insight_extractor = InsightExtractor()
        summary_generator = SummaryGenerator()
        rag_system = RAGSystem()
        
        # Initialize metric containers
        all_stt_metrics = []
        all_insight_metrics = []
        all_summary_metrics = []
        all_rag_metrics = []
        processing_times = {}
        
        # Process each audio file
        for audio_file in audio_files:
            try:
                call_id = Path(audio_file).stem
                self.logger.info(f"Evaluating: {call_id}")
                
                # Find corresponding ground truth
                ground_truth = None
                for gt in self.ground_truth_data:
                    if gt.call_id == call_id or call_id in gt.call_id:
                        ground_truth = gt
                        break
                        
                if not ground_truth:
                    self.logger.warning(f"No ground truth found for {call_id}")
                    continue
                    
                # Process through pipeline with timing
                start_time = time.time()
                
                # STT
                stt_start = time.time()
                transcription = stt_processor.transcribe_audio(audio_file)
                stt_time = time.time() - stt_start
                
                # Insights
                insight_start = time.time()
                insights = insight_extractor.extract_insights(transcription, call_id)
                insight_time = time.time() - insight_start
                
                # Summary
                summary_start = time.time()
                summary = summary_generator.generate_summary(transcription, insights, call_id)
                summary_time = time.time() - summary_start
                
                # RAG
                rag_start = time.time()
                recommendation = rag_system.generate_recommendations(insights, summary)
                rag_time = time.time() - rag_start
                
                total_time = time.time() - start_time
                
                # Record processing times
                processing_times[call_id] = {
                    'stt_time': stt_time,
                    'insight_time': insight_time,
                    'summary_time': summary_time,
                    'rag_time': rag_time,
                    'total_time': total_time
                }
                
                # Evaluate each component
                stt_metrics = self.evaluate_stt_performance(transcription, ground_truth)
                insight_metrics = self.evaluate_insight_performance(insights, ground_truth)
                summary_metrics = self.evaluate_summary_performance(summary, ground_truth)
                rag_metrics = self.evaluate_rag_performance(recommendation, ground_truth)
                
                all_stt_metrics.append(stt_metrics)
                all_insight_metrics.append(insight_metrics)
                all_summary_metrics.append(summary_metrics)
                all_rag_metrics.append(rag_metrics)
                
            except Exception as e:
                self.logger.error(f"Evaluation failed for {audio_file}: {e}")
                continue
                
        # Aggregate metrics
        aggregated_stt = self._aggregate_metrics(all_stt_metrics)
        aggregated_insights = self._aggregate_metrics(all_insight_metrics)
        aggregated_summary = self._aggregate_metrics(all_summary_metrics)
        aggregated_rag = self._aggregate_metrics(all_rag_metrics)
        
        # Calculate overall metrics
        overall_metrics = {
            'total_files_processed': len(all_stt_metrics),
            'average_processing_time': np.mean([times['total_time'] for times in processing_times.values()]),
            'average_accuracy': np.mean([
                aggregated_insights.get('classification_accuracy', 0),
                aggregated_stt.get('bleu', 0),
                aggregated_summary.get('rouge1_f1', 0)
            ])
        }
        
        # Average processing times
        avg_processing_times = {}
        if processing_times:
            for key in ['stt_time', 'insight_time', 'summary_time', 'rag_time', 'total_time']:
                avg_processing_times[key] = np.mean([times[key] for times in processing_times.values()])
        
        # Create evaluation results
        evaluation_metrics = EvaluationMetrics(
            stt_metrics=aggregated_stt,
            insight_metrics=aggregated_insights,
            summary_metrics=aggregated_summary,
            rag_metrics=aggregated_rag,
            overall_metrics=overall_metrics,
            processing_times=avg_processing_times,
            metadata={
                'evaluation_timestamp': time.time(),
                'num_files_evaluated': len(all_stt_metrics),
                'ground_truth_records': len(self.ground_truth_data)
            }
        )
        
        # Save results if output path provided
        if output_path:
            self._save_evaluation_results(evaluation_metrics, output_path)
            
        self.logger.info("System evaluation completed successfully")
        return evaluation_metrics
        
    def _aggregate_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics across multiple evaluations.
        
        Args:
            metrics_list: List of metric dictionaries
            
        Returns:
            Aggregated metrics dictionary
        """
        if not metrics_list:
            return {}
            
        aggregated = {}
        
        # Get all metric keys
        all_keys = set()
        for metrics in metrics_list:
            all_keys.update(metrics.keys())
            
        # Calculate mean for each metric
        for key in all_keys:
            values = [metrics.get(key, 0) for metrics in metrics_list]
            aggregated[f"{key}_mean"] = np.mean(values)
            aggregated[f"{key}_std"] = np.std(values)
            aggregated[f"{key}_min"] = np.min(values)
            aggregated[f"{key}_max"] = np.max(values)
            
        return aggregated
        
    def _calculate_character_error_rate(self, reference: str, hypothesis: str) -> float:
        """Calculate Character Error Rate (CER)."""
        if len(reference) == 0:
            return 1.0 if len(hypothesis) > 0 else 0.0
            
        # Simple CER calculation using edit distance
        d = np.zeros((len(reference) + 1, len(hypothesis) + 1))
        
        for i in range(len(reference) + 1):
            d[i][0] = i
        for j in range(len(hypothesis) + 1):
            d[0][j] = j
            
        for i in range(1, len(reference) + 1):
            for j in range(1, len(hypothesis) + 1):
                if reference[i-1] == hypothesis[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = min(d[i-1][j] + 1,      # deletion
                                 d[i][j-1] + 1,        # insertion
                                 d[i-1][j-1] + 1)      # substitution
                                 
        return d[len(reference)][len(hypothesis)] / len(reference)
        
    def _simple_bleu_score(self, reference: str, hypothesis: str) -> float:
        """Simple BLEU score calculation."""
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        
        if not hyp_words:
            return 0.0
            
        # 1-gram precision
        ref_1grams = set(ref_words)
        hyp_1grams = set(hyp_words)
        precision_1 = len(ref_1grams.intersection(hyp_1grams)) / len(hyp_1grams)
        
        # Brevity penalty
        bp = 1.0 if len(hyp_words) > len(ref_words) else np.exp(1 - len(ref_words) / len(hyp_words))
        
        return bp * precision_1
        
    def _save_evaluation_results(self, metrics: EvaluationMetrics, output_path: str) -> None:
        """Save evaluation results to file.
        
        Args:
            metrics: Evaluation metrics
            output_path: Output file path
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert to dictionary for JSON serialization
        results = asdict(metrics)
        
        save_json(results, output_path)
        self.logger.info(f"Evaluation results saved to: {output_path}")
        
    def generate_evaluation_report(self, metrics: EvaluationMetrics) -> str:
        """Generate human-readable evaluation report.
        
        Args:
            metrics: Evaluation metrics
            
        Returns:
            Formatted evaluation report
        """
        report = []
        report.append("=" * 60)
        report.append("CUSTOMER SUPPORT ANALYSIS SYSTEM - EVALUATION REPORT")
        report.append("=" * 60)
        
        # Overall metrics
        report.append("\n📊 OVERALL PERFORMANCE")
        report.append("-" * 30)
        report.append(f"Files Processed: {metrics.overall_metrics.get('total_files_processed', 0)}")
        report.append(f"Average Processing Time: {metrics.overall_metrics.get('average_processing_time', 0):.2f}s")
        report.append(f"Average Accuracy: {metrics.overall_metrics.get('average_accuracy', 0):.2f}")
        
        # STT Performance
        report.append("\n🎤 SPEECH-TO-TEXT PERFORMANCE")
        report.append("-" * 30)
        report.append(f"Word Error Rate: {metrics.stt_metrics.get('wer_mean', 0):.3f} ± {metrics.stt_metrics.get('wer_std', 0):.3f}")
        report.append(f"Character Error Rate: {metrics.stt_metrics.get('cer_mean', 0):.3f} ± {metrics.stt_metrics.get('cer_std', 0):.3f}")
        report.append(f"BLEU Score: {metrics.stt_metrics.get('bleu_mean', 0):.3f} ± {metrics.stt_metrics.get('bleu_std', 0):.3f}")
        
        # Insight Performance
        report.append("\n🧠 INSIGHT EXTRACTION PERFORMANCE")
        report.append("-" * 30)
        report.append(f"Sentiment Accuracy: {metrics.insight_metrics.get('sentiment_accuracy_mean', 0):.3f}")
        report.append(f"Intent Accuracy: {metrics.insight_metrics.get('intent_accuracy_mean', 0):.3f}")
        report.append(f"Overall Classification Accuracy: {metrics.insight_metrics.get('classification_accuracy_mean', 0):.3f}")
        
        # Summary Performance
        report.append("\n📝 SUMMARY GENERATION PERFORMANCE")
        report.append("-" * 30)
        report.append(f"ROUGE-1 F1: {metrics.summary_metrics.get('rouge1_f1_mean', 0):.3f} ± {metrics.summary_metrics.get('rouge1_f1_std', 0):.3f}")
        report.append(f"ROUGE-2 F1: {metrics.summary_metrics.get('rouge2_f1_mean', 0):.3f} ± {metrics.summary_metrics.get('rouge2_f1_std', 0):.3f}")
        report.append(f"ROUGE-L F1: {metrics.summary_metrics.get('rougeL_f1_mean', 0):.3f} ± {metrics.summary_metrics.get('rougeL_f1_std', 0):.3f}")
        
        # RAG Performance
        report.append("\n🔍 RAG RECOMMENDATION PERFORMANCE")
        report.append("-" * 30)
        report.append(f"Recommendation Coverage: {metrics.rag_metrics.get('recommendation_coverage_mean', 0):.3f}")
        report.append(f"Average Recommendations: {metrics.rag_metrics.get('num_recommendations_mean', 0):.1f}")
        report.append(f"RAG Confidence: {metrics.rag_metrics.get('rag_confidence_mean', 0):.3f}")
        
        # Processing Times
        report.append("\n⏱️ PROCESSING TIMES")
        report.append("-" * 30)
        report.append(f"STT Processing: {metrics.processing_times.get('stt_time', 0):.2f}s")
        report.append(f"Insight Extraction: {metrics.processing_times.get('insight_time', 0):.2f}s")
        report.append(f"Summary Generation: {metrics.processing_times.get('summary_time', 0):.2f}s")
        report.append(f"RAG Recommendations: {metrics.processing_times.get('rag_time', 0):.2f}s")
        report.append(f"Total Processing: {metrics.processing_times.get('total_time', 0):.2f}s")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


def main():
    """CLI interface for system evaluation."""
    import argparse
    import glob
    
    parser = argparse.ArgumentParser(description="System Evaluation")
    parser.add_argument("--audio-dir", required=True, help="Directory containing audio files")
    parser.add_argument("--output", help="Output file path for results")
    parser.add_argument("--ground-truth", default="data/ground_truth", help="Ground truth data path")
    parser.add_argument("--format", default="text", choices=["text", "json"], help="Output format")
    
    args = parser.parse_args()
    
    # Find audio files
    audio_files = []
    for ext in ['*.wav', '*.mp3', '*.m4a']:
        audio_files.extend(glob.glob(os.path.join(args.audio_dir, ext)))
        
    if not audio_files:
        print(f"No audio files found in {args.audio_dir}")
        return
        
    print(f"Found {len(audio_files)} audio files for evaluation")
    
    # Initialize evaluator
    evaluator = SystemEvaluator(args.ground_truth)
    
    # Run evaluation
    print("Starting system evaluation...")
    metrics = evaluator.evaluate_system_performance(audio_files)
    
    # Generate report
    if args.format == "json":
        import json
        report = json.dumps(asdict(metrics), indent=2, default=str)
    else:
        report = evaluator.generate_evaluation_report(metrics)
    
    # Output results
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Evaluation report saved to: {args.output}")
    else:
        print("\n" + report)


if __name__ == "__main__":
    main()