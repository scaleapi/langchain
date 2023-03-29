from langchain.document_loaders import HNLoader
from langchain.vectorstores.spellbook import Spellbook


def main():
    loader = HNLoader('https://news.ycombinator.com/')
    docs = loader.load()

    name = 'test collection'
    # requires OPENAI_API_KEY env var to be set
    sb = Spellbook(
        api_key=input('Enter your Spellbook deployment key:\n'), name=name
    )
    sb.add_texts(texts=[doc.page_content for doc in docs])

    query = 'The coolest hack'
    print('Query:', query, '\n\nDocuments:\n', sb.similarity_search(query))


if __name__ == '__main__':
    main()
