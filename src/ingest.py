import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from config import (
    DOCS_DIR, CHROMA_DIR, COLLECTION_NAME,
    EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP,
)


def load_documents():
    docs = []
    for path in DOCS_DIR.iterdir():
        if path.suffix == ".txt":
            text = path.read_text(encoding="utf-8")
        elif path.suffix == ".pdf":
            reader = PdfReader(str(path))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            continue
        docs.append({"name": path.name, "text": text})
    return docs


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + size].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap
    return chunks


def main():
    model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    ids, texts, metas = [], [], []
    for doc in load_documents():
        for i, chunk in enumerate(chunk_text(doc["text"])):
            ids.append(f'{doc["name"]}-{i}')
            texts.append(chunk)
            metas.append({"source": doc["name"], "chunk": i})

    embeddings = model.encode(
        [f"passage: {t}" for t in texts], normalize_embeddings=True
    ).tolist()

    collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metas)
    print(f"{len(texts)} تکه از اسناد ایندکس شد.")


if __name__ == "__main__":
    main()