import os
from langchain_community.document_loaders import PyPDFLoader


def load_uploaded_pdf(uploaded_file, session_dir):

    upload_dir = os.path.join(
        session_dir,
        "uploads"
    )

    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(
        upload_dir,
        uploaded_file.name
    )

    file_bytes = uploaded_file.getvalue()

    if not file_bytes:
        raise ValueError("Uploaded PDF is empty.")

    with open(file_path, "wb") as file:
        file.write(file_bytes)

    if os.path.getsize(file_path) == 0:
        raise ValueError("PDF was saved as an empty file.")

    print("PDF loading...")

    loader = PyPDFLoader(file_path)

    documents = loader.load()

    return documents