# rag.py
"""
RAG utility for formula retrieval using HuggingFace embeddings.

Dependencies
------------
pip install langchain langchain-huggingface faiss-cpu sentence-transformers
# or   pip install faiss-gpu   if you have FAISS with CUDA
"""

from __future__ import annotations

import os
from typing import List, Tuple

from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


class RAG:
    """A minimal RAG class that retrieves the *k* most similar formula blocks."""

    def __init__(
        self,
        doc_path: str = "data/web_formula.txt",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        embeddings_dir: str = "data/formula_embeddings_hf",
        normalize_embeddings: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        doc_path : str
            Path to the plain-text knowledge base. Each document is the text
            between <<FORMULA START>> and <<FORMULA END>> markers.
        embedding_model : str
            HuggingFace model name for embeddings.
            Options: 'sentence-transformers/all-MiniLM-L6-v2' (fast, 384 dim)
                     'sentence-transformers/all-mpnet-base-v2' (better quality, 768 dim)
        embeddings_dir : str
            Directory to cache the FAISS index.
        normalize_embeddings : bool
            Whether to L2-normalize vectors before similarity search (recommended).
        """
        # Initialize HuggingFace embeddings (runs locally, no API key needed)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},  # Use 'cuda' if GPU available
            encode_kwargs={'normalize_embeddings': normalize_embeddings}
        )

        # Try loading a precomputed FAISS index
        if os.path.isdir(embeddings_dir):
            self.vectorstore = FAISS.load_local(
                embeddings_dir,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            return

        # Otherwise, load & split the source file
        if not os.path.exists(doc_path):
            raise FileNotFoundError(f"Knowledge base not found: {doc_path}")

        # 1. Load the entire file and split by special markers
        with open(doc_path, "r", encoding="utf-8") as fh:
            text = fh.read()

        self.documents: List[Document] = []
        start_marker = "<<FORMULA START>>"
        end_marker = "<<FORMULA END>>"
        pos = 0
        block_idx = 0

        while True:
            start = text.find(start_marker, pos)
            if start == -1:
                break
            start += len(start_marker)
            end = text.find(end_marker, start)
            if end == -1:
                break

            content = text[start:end].strip()
            if content:
                self.documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "block_index": block_idx,
                            "source_file": doc_path,
                        },
                    )
                )
                block_idx += 1

            pos = end + len(end_marker)

        if not self.documents:
            raise ValueError(
                "No documents found between <<FORMULA START>> and <<FORMULA END>> markers."
            )

        # 2. Create embeddings & vector store
        self.vectorstore = FAISS.from_documents(
            self.documents,
            self.embeddings,
        )

        os.makedirs(embeddings_dir, exist_ok=True)
        self.vectorstore.save_local(embeddings_dir)

    def retrieve(self, query: str, k: int = 1) -> List[Tuple[str, float]]:
        """
        Return up to k most similar formula blocks for the given query.

        Parameters
        ----------
        query : str
            Arbitrary text used as retrieval key.
        k : int, default 1
            Number of top results wanted.

        Returns
        -------
        List[Tuple[str, float]]
            Each tuple is (block_text, similarity_score). Higher score == closer.
        """
        if k < 1:
            raise ValueError("k must be >= 1")
        query = self.trim_before_phrase(query)
        hits = self.vectorstore.similarity_search_with_relevance_scores(query, k=k)
        return [(doc.page_content, float(score)) for doc, score in hits]

    def __len__(self) -> int:
        """Return the number of indexed formula blocks."""
        return int(self.vectorstore.index.ntotal)

    @staticmethod
    def trim_before_phrase(text: str) -> str:
        """
        If the phrase "You should use the patient's medical values" appears in the text,
        return only the portion before that phrase. Otherwise, return the original text.
        """
        phrase = "You should use the patient's medical values"
        idx = text.find(phrase)
        if idx != -1:
            return text[:idx]
        return text
