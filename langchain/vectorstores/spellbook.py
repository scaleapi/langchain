import json
import uuid
from typing import Any, Callable, Iterable, List, Mapping, Optional

import numpy as np

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.utils import get_from_dict_or_env
from langchain.vectorstores.base import VectorStore


class Spellbook(VectorStore):
    def __init__(self, api_key: str, connector_name: str):
        pass

    def similarity_search(
            self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        pass

    # add_texts and from_texts can be only used for "files" connectors

    def add_texts(
            self,
            texts: Iterable[str],
            metadatas: Optional[List[dict]] = None,
            **kwargs: Any,
    ) -> List[str]:
        pass

    def from_texts(
            cls,
            texts: List[str],
            embedding: Embeddings,
            metadatas: Optional[List[dict]] = None,
            **kwargs: Any,
    ) -> VectorStore:
        pass
