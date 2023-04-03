from langchain.document_loaders import HNLoader
from langchain.vectorstores.spellbook import SpellbookVectorStore

if __name__ == '__main__':
    loader = HNLoader('https://news.ycombinator.com/')
    docs = loader.load()

    name = 'my vector store'

    sb = SpellbookVectorStore.from_texts(
        texts=[doc.page_content for doc in docs[:1]],
        name=name,
    )

    ids = sb.add_documents(docs[1:])
    print('added', ids)

    sources = sb.similarity_search('large language models')
    for i, source in enumerate(sources, 1):
        print(f'{i}.\t{source.page_content}')

    ids = sb.delete_documents(ids)
    print('deleted', ids)

    SpellbookVectorStore(name).delete_vector_store()
