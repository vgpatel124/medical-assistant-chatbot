"""LLM initialization and management."""

from enum import Enum
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from core.config import Config


class ModelType(Enum):
    """Available model types."""
    GROQ = "groq"
    LOCAL_BASE = "local_base"      # Llama without DPO training
    LOCAL_DPO = "local_dpo"        # Llama with DPO training


class LLMManager:
    """Manages LLM and embedding models."""

    _llm = None
    _local_llm = None
    _embeddings = None
    _model_type = ModelType.GROQ

    @classmethod
    def set_model_type(cls, model_type: ModelType):
        """Set which model to use."""
        cls._model_type = model_type
        cls._llm = None
        cls._local_llm = None

    @classmethod
    def get_llm(cls, model_type: ModelType = None, temperature: float = None):
        """Get or create the LLM instance.

        Pass temperature to get a fresh instance with a custom value
        (bypasses the singleton — used for A/B variant generation).
        """
        if model_type is not None:
            cls._model_type = model_type

        if cls._model_type == ModelType.GROQ:
            if temperature is not None:
                Config.validate(require_llm=True)
                return ChatGroq(
                    model=Config.LLM_MODEL,
                    temperature=temperature,
                    api_key=Config.GROQ_API_KEY,
                    streaming=True,
                )
            return cls._get_groq_llm()
        elif cls._model_type == ModelType.LOCAL_BASE:
            return cls._get_local_llm(use_adapter=False)
        elif cls._model_type == ModelType.LOCAL_DPO:
            return cls._get_local_llm(use_adapter=True)

    @classmethod
    def _get_groq_llm(cls) -> ChatGroq:
        """Get cached Groq LLM (default temperature)."""
        if cls._llm is None:
            Config.validate(require_llm=True)
            cls._llm = ChatGroq(
                model=Config.LLM_MODEL,
                temperature=Config.TEMPERATURE,
                api_key=Config.GROQ_API_KEY,
                streaming=True,
            )
        return cls._llm

    @classmethod
    def _get_local_llm(cls, use_adapter: bool = False):
        """Get local LLM (base or DPO-trained)."""
        if cls._local_llm is None:
            from core.llm_local import LocalLLM, LocalLLMWrapper

            adapter_path = Config.LOCAL_ADAPTER_PATH if use_adapter else None
            model_desc = "DPO-trained" if use_adapter else "base"
            print(f"Loading local {model_desc} model: {Config.LOCAL_BASE_MODEL}")

            local = LocalLLM(
                base_model=Config.LOCAL_BASE_MODEL,
                adapter_path=adapter_path,
            )
            cls._local_llm = LocalLLMWrapper(local)
        return cls._local_llm

    @classmethod
    def get_embeddings(cls) -> FastEmbedEmbeddings:
        """Get or create the embeddings instance."""
        if cls._embeddings is None:
            cls._embeddings = FastEmbedEmbeddings(model_name=Config.EMBEDDING_MODEL)
        return cls._embeddings
