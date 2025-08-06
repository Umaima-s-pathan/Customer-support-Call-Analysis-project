"""Configuration management for the customer support analysis system."""

import os
import yaml
import streamlit as st
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class OpenAIConfig:
    """OpenAI API configuration."""
    api_key: str
    model: str = "gpt-4"
    temperature: float = 0.3
    max_tokens: int = 1000


@dataclass
class STTConfig:
    """Speech-to-Text configuration."""
    model: str = "openai/whisper-large-v3"
    language: str = "auto"
    device: str = "auto"
    compute_type: str = "float16"


@dataclass
class RAGConfig:
    """RAG system configuration."""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 3
    similarity_threshold: float = 0.7
    vector_store: str = "chromadb"


class ConfigManager:
    """Manages configuration loading and access."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
        
    def _get_default_config_path(self) -> str:
        """Get default configuration file path."""
        return str(Path(__file__).parent.parent / "config.yaml")
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                
            # Replace environment variables
            config = self._replace_env_vars(config)
            return config
            
        except FileNotFoundError:
            print(f"Configuration file not found: {self.config_path}")
            return self._get_default_config()
        except yaml.YAMLError as e:
            print(f"Error parsing configuration file: {e}")
            return self._get_default_config()
            
    def _replace_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Replace environment variable placeholders in config with Streamlit secrets."""
    
        def replace_recursive(obj):
            if isinstance(obj, dict):
                return {k: replace_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_recursive(item) for item in obj]
            elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
                secret_key = obj[2:-1]
                return st.secrets.get(secret_key, obj)  # fallback to original if not found
            return obj
                
        return replace_recursive(config)
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file is not found."""
        return {
            "openai": {"api_key": "", "model": "gpt-4"},
            "stt": {"model": "openai/whisper-large-v3", "language": "auto"},
            "rag": {"embedding_model": "sentence-transformers/all-MiniLM-L6-v2"},
            "processing": {"batch_size": 4, "max_workers": 2}
        }
        
    def get_openai_config(self) -> OpenAIConfig:
        """Get OpenAI configuration."""
        openai_config = self.config.get("openai", {})
        return OpenAIConfig(
            api_key=openai_config.get("api_key", ""),
            model=openai_config.get("model", "gpt-4"),
            temperature=openai_config.get("temperature", 0.3),
            max_tokens=openai_config.get("max_tokens", 1000)
        )
        
    def get_stt_config(self) -> STTConfig:
        """Get Speech-to-Text configuration."""
        stt_config = self.config.get("stt", {})
        return STTConfig(
            model=stt_config.get("model", "openai/whisper-large-v3"),
            language=stt_config.get("language", "auto"),
            device=stt_config.get("device", "auto"),
            compute_type=stt_config.get("compute_type", "float16")
        )
        
    def get_rag_config(self) -> RAGConfig:
        """Get RAG configuration."""
        rag_config = self.config.get("rag", {})
        return RAGConfig(
            embedding_model=rag_config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
            chunk_size=rag_config.get("chunk_size", 500),
            chunk_overlap=rag_config.get("chunk_overlap", 50),
            top_k=rag_config.get("top_k", 3),
            similarity_threshold=rag_config.get("similarity_threshold", 0.7),
            vector_store=rag_config.get("vector_store", "chromadb")
        )
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation, e.g., 'openai.model')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value


# Global configuration instance
config_manager = ConfigManager()
