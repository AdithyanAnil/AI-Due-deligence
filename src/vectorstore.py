import os

from langchain_chroma import Chroma


def create_vectorstore(
    texts,
    embeddings,
    session_dir
):

    chroma_dir = os.path.join(
        session_dir,
        "chroma_db"
    )

    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=chroma_dir
    )

    return vectorstore


def create_retriever(vectorstore):

    return vectorstore.as_retriever(
        search_kwargs={
            "k": 3
        }
    )