import os
import uuid
from typing import Any, Iterable, List, Optional

import requests

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.base import VectorStore

SPELLBOOK_ENDPOINT = 'https://canary.dashboard.scale.com/spellbook/api/v1'


class Spellbook(VectorStore):
    """Wrapper around Spellbook vector store."""

    def __init__(
        self,
        name: Optional[str] = None,
        embedding_function: Optional[Embeddings] = None,
    ):
        """Initialize with Spellbook client."""
        spellbook_api_key = os.getenv('SPELLBOOK_API_KEY')
        if not spellbook_api_key:
            raise ValueError(
                'Could not retrieve Spellbook API key. '
                'Ensure your SPELLBOOK_API_KEY env variable is properly set.'
            )
        self._api_key = spellbook_api_key
        self._vector_store_name = name or str(uuid.uuid4())
        self._embedding_function = embedding_function or OpenAIEmbeddings()

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        query_embedding = self._embedding_function.embed_query(query)
        return requests.post(
            url=f'{SPELLBOOK_ENDPOINT}/similaritySearch',
            headers={'authorization': f'Basic {self._api_key}'},
            json={
                'vectorStoreName': self._vector_store_name,
                'queryEmbedding': query_embedding,
                'k': k,
            },
        ).json()

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        for i, text in enumerate(texts):
            embedding = self._embedding_function.embed_documents([text])[0]
            if metadatas:
                metadata = metadatas[i]
            else:
                metadata = {}
            requests.post(
                url=f'{SPELLBOOK_ENDPOINT}/upload',
                headers={'authorization': f'Basic {self._api_key}'},
                json={
                    'vectorStoreName': self._vector_store_name,
                    'items': [
                        {
                            'text': text,
                            'embedding': embedding,
                            'metadata': metadata,
                        }
                    ],
                },
            )
        return []

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
        spellbook_api_key = os.getenv('SPELLBOOK_API_KEY')
        if not spellbook_api_key:
            raise ValueError(
                'Could not retrieve Spellbook API key. '
                'Ensure your SPELLBOOK_API_KEY env variable is properly set.'
            )

        # Create vector store, error if name exists
        requests.post(
            url=f'{SPELLBOOK_ENDPOINT}/createVectorStore',
            headers={'authorization': f'Basic {spellbook_api_key}'},
            json={'name': name},
        )

        # Embed and upload documents in batches
        ret = cls(name, embedding)
        embedding = embedding or OpenAIEmbeddings()
        for i in range(0, len(texts), batch_size):
            i_end = min(i + batch_size, len(texts))
            texts_batch = texts[i:i_end]
            embeddings_batch = embedding.embed_documents(texts_batch)
            if metadatas:
                metadatas_batch = metadatas[i:i_end]
            else:
                metadatas_batch = [{} for _ in range(i, i_end)]
            requests.post(
                url=f'{SPELLBOOK_ENDPOINT}/upload',
                headers={'authorization': f'Basic {spellbook_api_key}'},
                json={
                    'vectorStoreName': name,
                    'items': [
                        {
                            'text': text,
                            'embedding': embedding,
                            'metadata': metadata,
                        }
                        for text, embedding, metadata in zip(
                            texts_batch, embeddings_batch, metadatas_batch
                        )
                    ],
                },
            )
        return ret
