from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=['\n\n','\n']
    )
    texts=text_splitter.split_documents(documents)

    return texts