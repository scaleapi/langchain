"""Loader that loads HN."""
from typing import Any, List

from langchain.docstore.document import Document
from langchain.document_loaders import BaseLoader


class SpellbookLoader(BaseLoader):
    def __init__(self, api_key: str, connector_name: str) -> None:
        ...

    def load(self) -> List[Document]:
        ...

