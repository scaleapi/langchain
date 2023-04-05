import os
import uuid
from typing import Any, Iterable, List, Optional

import requests

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.base import VectorStore

try:
    import spellbook as sb
except ImportError:
    raise ValueError(
        'Could not import redis python package. '
        'Please install it with `pip install spellbook`.'
    )


class SpellbookVectorStore(VectorStore):
    """Wrapper around Spellbook vector store."""

    def __init__(
        self,
        name: str,
        embedding_function: Optional[Embeddings] = None,
    ):
        """Initialize with Spellbook client."""
        self._vector_store = sb.VectorStore(name=name)
        self._embedding_function = embedding_function or OpenAIEmbeddings()

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        query_embedding = self._embedding_function.embed_query(query)
        res = self._vector_store.similarity_search(
            query_embedding=query_embedding, k=k
        )
        return [
            Document(
                page_content=item['text'], metadata=item.get('metadata', {})
            )
            for item in res['data']['items']
        ]

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> List[str]:
        doc_ids = []
        texts_batch, metadatas_batch = [], []
        for i, text in enumerate(texts):
            if metadatas:
                metadata = metadatas[i]
            else:
                metadata = {}
            texts_batch.append(text)
            metadatas_batch.append(metadata)

            if i % batch_size == 32:
                doc_ids.extend(
                    self._add_texts_batch(
                        self._vector_store,
                        texts_batch,
                        metadatas_batch,
                        self._embedding_function,
                    )
                )
                texts_batch, metadatas_batch = [], []

        doc_ids.extend(
            self._add_texts_batch(
                self._vector_store,
                texts_batch,
                metadatas_batch,
                self._embedding_function,
            )
        )
        return doc_ids

    @staticmethod
    def _add_texts_batch(
        vector_store,
        texts: List[str],
        metadatas: List[dict],
        embedding: Embeddings,
    ) -> List[str]:
        embeddings = embedding.embed_documents(texts)

        res = vector_store.upload_documents_with_embeddings(
            items=[
                sb.DocumentWithEmbedding(text=t, embedding=e, metadata=m)
                for t, e, m in zip(texts, embeddings, metadatas)
            ]
        )
        return [doc['id'] for doc in res['data']['items']]

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Optional[Embeddings] = None,
        metadatas: Optional[List[dict]] = None,
        batch_size: int = 32,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> VectorStore:
        name = name or str(uuid.uuid4())
        # Create vector store, error if name exists
        vector_store = sb.VectorStore.create(name=name)

        # Embed and upload documents in batches
        embedding = embedding or OpenAIEmbeddings()
        for l in range(0, len(texts), batch_size):
            r = min(l + batch_size, len(texts))
            texts_batch = texts[l:r]
            if metadatas:
                metadatas_batch = metadatas[l:r]
            else:
                metadatas_batch = [{} for _ in range(l, r)]
            cls._add_texts_batch(
                vector_store,
                texts_batch,
                metadatas_batch,
                embedding,
            )
        return cls(name=name)

    def delete_vector_store(self):
        self._vector_store.delete()

    def delete_documents(
        self,
        ids: List[str],
    ):
        res = self._vector_store.delete_documents(ids)
        return [doc['id'] for doc in res['data']['items']]
