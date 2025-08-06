"""Utility functions for the customer support analysis system."""

import os
import json
import logging
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import wave
import numpy as np
import librosa
from pydub import AudioSegment


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)
    """Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger("customer_support_analysis")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger


def generate_call_id(audio_path: str, timestamp: Optional[datetime] = None) -> str:
    """Generate unique call ID based on file path and timestamp.
    
    Args:
        audio_path: Path to audio file
        timestamp: Optional timestamp, uses current time if None
        
    Returns:
        Unique call ID
    """
    if timestamp is None:
        timestamp = datetime.now()
        
    # Create hash from file path and timestamp
    hash_input = f"{audio_path}_{timestamp.isoformat()}"
    hash_obj = hashlib.md5(hash_input.encode())
    return f"call_{hash_obj.hexdigest()[:8]}"


def save_json(data: Dict[str, Any], file_path: str) -> None:
    """Save data to JSON file.
    
    Args:
        data: Data to save
        file_path: Output file path
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def load_json(file_path: str) -> Dict[str, Any]:
    """Load data from JSON file.
    
    Args:
        file_path: Input file path
        
    Returns:
        Loaded data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def validate_audio_file(file_path: str) -> bool:
    valid_extensions = ['.wav', '.mp3', '.m4a', '.flac']
    return any(file_path.endswith(ext) for ext in valid_extensions)
    """Validate if file is a supported audio format.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        True if valid audio file, False otherwise
    """
    supported_formats = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
    file_ext = Path(file_path).suffix.lower()
    return file_ext in supported_formats


def convert_audio_format(input_path: str, output_path: str, 
                        target_format: str = "wav", sample_rate: int = 16000) -> str:
    """Convert audio file to target format.
    
    Args:
        input_path: Input audio file path
        output_path: Output audio file path
        target_format: Target audio format (wav, mp3, etc.)
        sample_rate: Target sample rate
        
    Returns:
        Path to converted audio file
    """
    # Load audio file
    audio = AudioSegment.from_file(input_path)
    
    # Convert to target sample rate
    audio = audio.set_frame_rate(sample_rate)
    
    # Convert to mono if stereo
    if audio.channels > 1:
        audio = audio.set_channels(1)
        
    # Export in target format
    audio.export(output_path, format=target_format)
    
    return output_path


def get_audio_duration(file_path: str) -> float:
    audio, sr = librosa.load(file_path, sr=None)
    return librosa.get_duration(y=audio, sr=sr)
    """Get audio file duration in seconds.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Duration in seconds
    """
    audio = AudioSegment.from_file(file_path)
    return len(audio) / 1000.0  # Convert milliseconds to seconds


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks.
    
    Args:
        text: Input text
        chunk_size: Maximum chunk size in characters
        overlap: Overlap between chunks in characters
        
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
        
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence endings
            for i in range(end, max(start + chunk_size // 2, start), -1):
                if text[i] in '.!?':
                    end = i + 1
                    break
                    
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
            
        start = end - overlap
        
    return chunks


def clean_transcription(text: str) -> str:
    """Clean and normalize transcription text.
    
    Args:
        text: Raw transcription text
        
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove common transcription artifacts
    artifacts = ['[MUSIC]', '[NOISE]', '[INAUDIBLE]', '[SILENCE]']
    for artifact in artifacts:
        text = text.replace(artifact, '')
        
    # Normalize punctuation
    text = text.replace('..', '.').replace('??', '?').replace('!!', '!')
    
    return text.strip()


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string (e.g., "2m 30s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def create_directory_structure(base_path: str) -> None:
    """Create the required directory structure for the project.
    
    Args:
        base_path: Base project directory path
    """
    directories = [
        "data/audio_samples",
        "data/transcriptions", 
        "data/documents",
        "data/ground_truth",
        "data/outputs",
        "logs",
        "results"
    ]
    
    for directory in directories:
        full_path = os.path.join(base_path, directory)
        os.makedirs(full_path, exist_ok=True)


def calculate_word_error_rate(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate (WER) between reference and hypothesis.
    
    Args:
        reference: Reference (ground truth) text
        hypothesis: Hypothesis (predicted) text
        
    Returns:
        WER as a float between 0 and 1
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    # Simple WER calculation using edit distance
    # In production, use jiwer library for more accurate calculation
    
    if len(ref_words) == 0:
        return 1.0 if len(hyp_words) > 0 else 0.0
        
    # Levenshtein distance calculation
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
    
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
        
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(d[i-1][j] + 1,      # deletion
                             d[i][j-1] + 1,        # insertion
                             d[i-1][j-1] + 1)      # substitution
                             
    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


class ProgressTracker:
    """Simple progress tracking utility."""
    
    def __init__(self, total: int, description: str = "Processing"):
        """Initialize progress tracker.
        
        Args:
            total: Total number of items to process
            description: Description of the process
        """
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = datetime.now()
        
    def update(self, increment: int = 1) -> None:
        """Update progress.
        
        Args:
            increment: Number of items processed
        """
        self.current += increment
        percentage = (self.current / self.total) * 100
        
        elapsed = datetime.now() - self.start_time
        if self.current > 0:
            eta = elapsed * (self.total - self.current) / self.current
            eta_str = f"ETA: {eta.total_seconds():.0f}s"
        else:
            eta_str = "ETA: unknown"
            
        print(f"\r{self.description}: {self.current}/{self.total} "
              f"({percentage:.1f}%) {eta_str}", end="", flush=True)
              
        if self.current >= self.total:
            print()  # New line when complete
