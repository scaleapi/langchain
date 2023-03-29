from langchain.document_loaders import HNLoader
from langchain.vectorstores.spellbook import Spellbook


def main():
    loader = HNLoader('https://news.ycombinator.com/')
    docs = loader.load()

    # create a new vector store from docs
    # sb = Spellbook.from_texts(texts=[doc.page_content for doc in docs])

    # or fetch and upload to an existing vector store
    name = 'my existing vector store'
    sb = Spellbook(name=name)  # requires OPENAI_API_KEY env var to be set
    sb.add_texts(texts=[doc.page_content for doc in docs])

    # query vector store
    query = 'The coolest hack'
    print('Query:', query, '\n\nDocuments:\n', sb.similarity_search(query))


if __name__ == '__main__':
    main()
