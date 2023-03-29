from langchain.vectorstores import Spellbook, VectorStore
from langchain.embeddings.openai import OpenAIEmbeddings
import requests

# create a vector store via REST api
res = requests.post('https://api.spellbook.scale.com/create_vector_store', json={
    "name": "cat facts",
}, headers={
    "authorization": "...",
})
assert res.status_code == 200

# add documents to vectorstores
spellbook: VectorStore = Spellbook(api_key="...", name="cat facts")
embeddings = OpenAIEmbeddings()
spellbook.add_texts(texts=[
    "cats are fun",
], embeddings=embeddings)

# query
query = "what's up?"
query_embedding = embeddings.embed_query(query)

docs = spellbook.similarity_search_by_vector(query_embedding)
