"""Speech-to-Text processing module with speaker diarization."""

import os
import torch
import whisper
import librosa
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import tempfile
from pydub import AudioSegment
import logging
from .config import config_manager
from .utils import setup_logging, validate_audio_file, get_audio_duration


@dataclass
class TranscriptionSegment:
    """Represents a transcription segment with speaker information."""
    start_time: float
    end_time: float
    text: str
    speaker: str
    confidence: float = 0.0


@dataclass
class TranscriptionResult:
    """Complete transcription result with metadata."""
    segments: List[TranscriptionSegment]
    full_text: str
    language: str
    duration: float
    processing_time: float
    metadata: Dict[str, Any]


class SpeechToTextProcessor:
    """Advanced Speech-to-Text processor with speaker diarization."""
    
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """Initialize the STT processor.
        
        Args:
            model_name: Whisper model name (tiny, base, small, medium, large, large-v2, large-v3)
            device: Device to use (auto, cpu, cuda)
        """
        self.logger = setup_logging()
        self.config = config_manager.get_stt_config()
        
        # Model configuration
        self.model_name = model_name or self.config.model.split('/')[-1]
        self.device = device or self._get_optimal_device()
        
        # Initialize Whisper model
        self.model = None
        self._load_model()
        
        # Speaker diarization setup
        self.use_diarization = True
        self._setup_diarization()
        
    def _get_optimal_device(self) -> str:
        """Determine optimal device for processing."""
        if self.config.device != "auto":
            return self.config.device
            
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
            
    def _load_model(self) -> None:
        """Load Whisper model."""
        try:
            self.logger.info(f"Loading Whisper model: {self.model_name} on {self.device}")
            self.model = whisper.load_model(self.model_name, device=self.device)
            self.logger.info("Whisper model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load Whisper model: {e}")
            # Fallback to smaller model
            try:
                self.model_name = "base"
                self.model = whisper.load_model("base", device="cpu")
                self.logger.warning("Loaded fallback model: base on CPU")
            except Exception as fallback_error:
                raise RuntimeError(f"Failed to load any Whisper model: {fallback_error}")
                
    def _setup_diarization(self) -> None:
        """Setup speaker diarization (simplified version)."""
        try:
            # In a production system, you would use pyannote.audio or similar
            # For this prototype, we'll use a simplified approach based on audio features
            self.diarization_available = True
            self.logger.info("Speaker diarization setup completed")
            
        except Exception as e:
            self.logger.warning(f"Speaker diarization not available: {e}")
            self.diarization_available = False
            
    def preprocess_audio(self, audio_path: str) -> str:
        """Preprocess audio file for optimal transcription.
        
        Args:
            audio_path: Path to input audio file
            
        Returns:
            Path to preprocessed audio file
        """
        try:
            # Load audio using pydub for format compatibility
            audio = AudioSegment.from_file(audio_path)
            
            # Convert to mono
            if audio.channels > 1:
                audio = audio.set_channels(1)
                self.logger.debug("Converted audio to mono")
                
            # Normalize audio level
            audio = audio.normalize()
            
            # Set sample rate to 16kHz for Whisper
            audio = audio.set_frame_rate(16000)
            
            # Export to temporary WAV file
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            audio.export(temp_file.name, format='wav')
            
            self.logger.debug(f"Preprocessed audio saved to: {temp_file.name}")
            return temp_file.name
            
        except Exception as e:
            self.logger.error(f"Audio preprocessing failed: {e}")
            return audio_path  # Return original if preprocessing fails
            
    def _simple_speaker_diarization(self, audio_data: np.ndarray, 
                                  segments: List[Dict]) -> List[TranscriptionSegment]:
        """Simplified speaker diarization based on audio features.
        
        Args:
            audio_data: Audio waveform data
            segments: Whisper transcription segments
            
        Returns:
            List of transcription segments with speaker labels
        """
        result_segments = []
        
        try:
            # Simple approach: alternate speakers based on silence gaps
            # In production, use proper diarization models
            
            current_speaker = "Customer"
            speaker_change_threshold = 2.0  # seconds of silence to change speaker
            
            for i, segment in enumerate(segments):
                start_time = segment['start']
                end_time = segment['end']
                text = segment['text'].strip()
                
                # Determine speaker based on patterns and timing
                if i > 0:
                    prev_end = segments[i-1]['end']
                    silence_duration = start_time - prev_end
                    
                    # Change speaker if long silence or specific patterns
                    if (silence_duration > speaker_change_threshold or 
                        self._detect_speaker_change_patterns(text)):
                        current_speaker = "Agent" if current_speaker == "Customer" else "Customer"
                
                # Create transcription segment
                result_segments.append(TranscriptionSegment(
                    start_time=start_time,
                    end_time=end_time,
                    text=text,
                    speaker=current_speaker,
                    confidence=segment.get('avg_logprob', 0.0)
                ))
                
        except Exception as e:
            self.logger.warning(f"Speaker diarization failed: {e}")
            # Fallback: alternate speakers
            for i, segment in enumerate(segments):
                speaker = "Customer" if i % 2 == 0 else "Agent"
                result_segments.append(TranscriptionSegment(
                    start_time=segment['start'],
                    end_time=segment['end'],
                    text=segment['text'].strip(),
                    speaker=speaker,
                    confidence=segment.get('avg_logprob', 0.0)
                ))
                
        return result_segments
        
    def _detect_speaker_change_patterns(self, text: str) -> bool:
        """Detect speaker change based on text patterns.
        
        Args:
            text: Transcribed text
            
        Returns:
            True if speaker change detected
        """
        # Common agent phrases
        agent_patterns = [
            "thank you for calling",
            "how can I help",
            "let me check",
            "I understand",
            "I can help you with",
            "is there anything else"
        ]
        
        # Common customer phrases  
        customer_patterns = [
            "I have a problem",
            "I need help",
            "my account",
            "I can't",
            "it's not working"
        ]
        
        text_lower = text.lower()
        
        # Check for agent patterns
        if any(pattern in text_lower for pattern in agent_patterns):
            return True
            
        return False
        
    def transcribe_audio(self, audio_path: str, language: Optional[str] = None) -> TranscriptionResult:
        """Transcribe audio file with speaker diarization.
        
        Args:
            audio_path: Path to audio file
            language: Language code (auto-detect if None)
            
        Returns:
            TranscriptionResult with segments and metadata
        """
        if not validate_audio_file(audio_path):
            raise ValueError(f"Unsupported audio format: {audio_path}")
            
        start_time = torch.cuda.Event(enable_timing=True) if self.device == "cuda" else None
        end_time = torch.cuda.Event(enable_timing=True) if self.device == "cuda" else None
        
        try:
            self.logger.info(f"Starting transcription of: {Path(audio_path).name}")
            
            if start_time:
                start_time.record()
            import time
            process_start = time.time()
            
            # Preprocess audio
            processed_audio_path = self.preprocess_audio(audio_path)
            
            # Get audio duration
            duration = get_audio_duration(audio_path)
            
            # Transcribe with Whisper
            language_code = language or self.config.language
            if language_code == "auto":
                language_code = None
                
            result = self.model.transcribe(
                processed_audio_path,
                language=language_code,
                word_timestamps=True,
                verbose=False
            )
            
            # Clean up temporary file
            if processed_audio_path != audio_path:
                try:
                    os.unlink(processed_audio_path)
                except:
                    pass
                    
            # Load audio data for diarization
            audio_data, sr = librosa.load(audio_path, sr=16000)
            
            # Apply speaker diarization
            if self.diarization_available and result['segments']:
                segments = self._simple_speaker_diarization(audio_data, result['segments'])
            else:
                # No diarization - alternate speakers
                segments = []
                for i, segment in enumerate(result['segments']):
                    speaker = "Customer" if i % 2 == 0 else "Agent"
                    segments.append(TranscriptionSegment(
                        start_time=segment['start'],
                        end_time=segment['end'],
                        text=segment['text'].strip(),
                        speaker=speaker,
                        confidence=segment.get('avg_logprob', 0.0)
                    ))
            
            # Calculate processing time
            if end_time:
                end_time.record()
                torch.cuda.synchronize()
                processing_time = start_time.elapsed_time(end_time) / 1000.0
            else:
                processing_time = time.time() - process_start
            
            # Create full text
            full_text = " ".join([seg.text for seg in segments])
            
            # Detected language
            detected_language = result.get('language', 'unknown')
            
            # Create result
            transcription = TranscriptionResult(
                segments=segments,
                full_text=full_text,
                language=detected_language,
                duration=duration,
                processing_time=processing_time,
                metadata={
                    "model": self.model_name,
                    "device": self.device,
                    "file_path": audio_path,
                    "num_segments": len(segments),
                    "avg_confidence": np.mean([seg.confidence for seg in segments]) if segments else 0.0
                }
            )
            
            self.logger.info(f"Transcription completed in {processing_time:.2f}s")
            self.logger.info(f"Detected language: {detected_language}")
            self.logger.info(f"Generated {len(segments)} segments")
            
            return transcription
            
        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            raise
            
    def transcribe_batch(self, audio_files: List[str], 
                        language: Optional[str] = None) -> Dict[str, TranscriptionResult]:
        """Transcribe multiple audio files.
        
        Args:
            audio_files: List of audio file paths
            language: Language code (auto-detect if None)
            
        Returns:
            Dictionary mapping file paths to transcription results
        """
        results = {}
        
        self.logger.info(f"Starting batch transcription of {len(audio_files)} files")
        
        for i, audio_file in enumerate(audio_files):
            try:
                self.logger.info(f"Processing file {i+1}/{len(audio_files)}: {Path(audio_file).name}")
                result = self.transcribe_audio(audio_file, language)
                results[audio_file] = result
                
            except Exception as e:
                self.logger.error(f"Failed to transcribe {audio_file}: {e}")
                continue
                
        self.logger.info(f"Batch transcription completed. {len(results)}/{len(audio_files)} files processed successfully")
        return results
        
    def format_transcription(self, result: TranscriptionResult, 
                           format_type: str = "detailed") -> str:
        """Format transcription result for display.
        
        Args:
            result: Transcription result
            format_type: Format type (detailed, simple, srt)
            
        Returns:
            Formatted transcription string
        """
        if format_type == "simple":
            return result.full_text
            
        elif format_type == "srt":
            # SRT subtitle format
            srt_content = []
            for i, segment in enumerate(result.segments, 1):
                start_time = self._seconds_to_srt_time(segment.start_time)
                end_time = self._seconds_to_srt_time(segment.end_time)
                srt_content.append(f"{i}\n{start_time} --> {end_time}\n[{segment.speaker}] {segment.text}\n")
            return "\n".join(srt_content)
            
        else:  # detailed
            formatted = []
            formatted.append(f"Language: {result.language}")
            formatted.append(f"Duration: {result.duration:.2f}s")
            formatted.append(f"Processing Time: {result.processing_time:.2f}s")
            formatted.append(f"Segments: {len(result.segments)}")
            formatted.append("-" * 50)
            
            for segment in result.segments:
                timestamp = f"[{segment.start_time:.1f}-{segment.end_time:.1f}]"
                formatted.append(f"{timestamp} {segment.speaker}: {segment.text}")
                
            return "\n".join(formatted)
            
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')
        
    def save_transcription(self, result: TranscriptionResult, output_path: str, 
                          format_type: str = "json") -> None:
        """Save transcription result to file.
        
        Args:
            result: Transcription result
            output_path: Output file path
            format_type: Output format (json, txt, srt)
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if format_type == "json":
            import json
            data = {
                "full_text": result.full_text,
                "language": result.language,
                "duration": result.duration,
                "processing_time": result.processing_time,
                "metadata": result.metadata,
                "segments": [
                    {
                        "start_time": seg.start_time,
                        "end_time": seg.end_time,
                        "text": seg.text,
                        "speaker": seg.speaker,
                        "confidence": seg.confidence
                    }
                    for seg in result.segments
                ]
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        else:
            formatted_text = self.format_transcription(result, format_type)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(formatted_text)
                
        self.logger.info(f"Transcription saved to: {output_path}")


def main():
    """CLI interface for testing STT module."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Speech-to-Text Processing")
    parser.add_argument("--audio", required=True, help="Audio file path")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--language", help="Language code (auto, en, hi)")
    parser.add_argument("--model", default="base", help="Whisper model size")
    parser.add_argument("--format", default="detailed", choices=["detailed", "simple", "srt", "json"])
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = SpeechToTextProcessor(model_name=args.model)
    
    # Process audio
    result = processor.transcribe_audio(args.audio, args.language)
    
    # Save or display result
    if args.output:
        format_type = "json" if args.output.endswith('.json') else args.format
        processor.save_transcription(result, args.output, format_type)
    else:
        print(processor.format_transcription(result, args.format))


if __name__ == "__main__":
    main()
