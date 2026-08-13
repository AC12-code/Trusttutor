"""Central configuration, all overridable via environment variables.

Design choice: everything has a sensible default so the repo runs with zero
setup except an LLM key. Retrieval defaults to TF-IDF (no model download) so a
live demo never breaks; flip RETRIEVER=faiss to show the embeddings path.
"""
import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # dotenv is optional
    pass


@dataclass
class Settings:
    # LLM
    llm_provider: str = os.getenv("LLM_PROVIDER", "anthropic")  # anthropic | openai | google | mock
    model: str = os.getenv("MODEL", "claude-3-5-sonnet-20241022")  # override to your model
    temperature: float = float(os.getenv("TEMPERATURE", "0"))

    # Retrieval
    retriever: str = os.getenv("RETRIEVER", "tfidf")  # tfidf | faiss
    top_k: int = int(os.getenv("TOP_K", "4"))
    chunk_words: int = int(os.getenv("CHUNK_WORDS", "80"))

    # Grounding gate: if the best retrieved chunk scores below this, we refuse.
    grounding_threshold: float = float(os.getenv("GROUNDING_THRESHOLD", "0.08"))

    # Optional second gate: ask the LLM to confirm the context is sufficient.
    use_llm_grader: bool = os.getenv("USE_LLM_GRADER", "false").lower() == "true"


settings = Settings()
