import os
import uuid
from typing import Any, Iterable, List, Optional

import requests

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.base import VectorStore

SPELLBOOK_ENDPOINT = 'https://canary.dashboard.scale.com/spellbook/api/v1'


class SpellbookVectorStore(VectorStore):
    """Wrapper around Spellbook vector store."""

    def __init__(
        self,
        name: Optional[str] = None,
        embedding_function: Optional[Embeddings] = None,
    ):
        """Initialize with Spellbook client."""
        self._vector_store_name = name or str(uuid.uuid4())
        self._embedding_function = embedding_function or OpenAIEmbeddings()

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        query_embedding = self._embedding_function.embed_query(query)
        res = self._post_request(
            route='similarity_search',
            json={
                'vectorStoreName': self._vector_store_name,
                'queryEmbedding': query_embedding,
                'k': k,
            },
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
                        texts_batch,
                        metadatas_batch,
                        self._embedding_function,
                        self._vector_store_name,
                    )
                )
                texts_batch, metadatas_batch = [], []

        doc_ids.extend(
            self._add_texts_batch(
                texts_batch,
                metadatas_batch,
                self._embedding_function,
                self._vector_store_name,
            )
        )
        return doc_ids

    @staticmethod
    def _add_texts_batch(
        texts: List[str],
        metadatas: List[dict],
        embedding: Embeddings,
        name: str,
    ) -> List[str]:
        embeddings = embedding.embed_documents(texts)
        res = SpellbookVectorStore._post_request(
            route='upload_documents',
            json={
                'vectorStoreName': name,
                'items': [
                    {
                        'text': text,
                        'embedding': embedding,
                        'metadata': metadata,
                    }
                    for text, embedding, metadata in zip(
                        texts, embeddings, metadatas
                    )
                ],
            },
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
        cls._post_request(
            route='create_vector_store',
            json={'name': name},
        )

        # Embed and upload documents in batches
        ret = cls(name, embedding)
        embedding = embedding or OpenAIEmbeddings()
        for l in range(0, len(texts), batch_size):
            r = min(l + batch_size, len(texts))
            texts_batch = texts[l:r]
            embeddings_batch = embedding.embed_documents(texts_batch)
            if metadatas:
                metadatas_batch = metadatas[l:r]
            else:
                metadatas_batch = [{} for _ in range(l, r)]
            cls._post_request(
                route='upload_documents',
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

    def delete_vector_store(self):
        self._post_request(
            route='delete_vector_store',
            json={'vectorStoreName': self._vector_store_name},
        )

    def delete_documents(
        self,
        ids: List[str],
    ):
        res = self._post_request(
            route='delete_documents',
            json={
                'vectorStoreName': self._vector_store_name,
                'ids': ids,
            },
        )
        return [doc['id'] for doc in res['data']['items']]

    @staticmethod
    def _post_request(
        route: str, json: dict, spellbook_api_key: Optional[str] = None
    ):
        if not spellbook_api_key:
            spellbook_api_key = os.getenv('SPELLBOOK_API_KEY')
            if not spellbook_api_key:
                raise ValueError(
                    'Could not retrieve Spellbook API key. '
                    'Ensure your SPELLBOOK_API_KEY env variable is properly set.'
                )
        res = requests.post(
            url=f'{SPELLBOOK_ENDPOINT}/{route}',
            headers={'authorization': f'Basic {spellbook_api_key}'},
            json=json,
        ).json()
        if not res['ok']:
            # TODO: parse and format specifically for each route
            raise ValueError(res['error'])
        return res
