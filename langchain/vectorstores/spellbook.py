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
        api_key: str,
        name: str,
        embedding_function: Optional[Embeddings] = None,
    ):
        """Initialize with Spellbook client."""
        self._api_key = api_key
        self._vector_store_name = name
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
        api_key: str,
        name: str,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        embedding_function: Optional[Embeddings] = None,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> VectorStore:
        # Create vector store, error if name exists
        requests.post(
            url=f'{SPELLBOOK_ENDPOINT}/createVectorStore',
            headers={'authorization': f'Basic {api_key}'},
            json={'name': name},
        )

        # Embed and upload documents in batches
        ret = cls(api_key, name, embedding_function)
        embedding_function = embedding_function or OpenAIEmbeddings()
        for i in range(0, len(texts), batch_size):
            i_end = min(i + batch_size, len(texts))
            texts_batch = texts[i:i_end]
            embeddings_batch = embedding_function.embed_documents(texts_batch)
            if metadatas:
                metadatas_batch = metadatas[i:i_end]
            else:
                metadatas_batch = [{} for _ in range(i, i_end)]
            requests.post(
                url=f'{SPELLBOOK_ENDPOINT}/upload',
                headers={'authorization': f'Basic {api_key}'},
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
