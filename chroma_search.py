import sys  #-- for command line arguments
import chromadb 
from loader import load_docs
from chunker import chunk_docs
from embedder import embed

#-- Create a Chroma client and collection first

client = chromadb.PersistentClient(path="chroma_db")  #-- PersistentClient will store the database on disk at the specified path which in this case is "chroma_db" a folder.
COLLECTION_NAME = "my_docs"


def get_collection():
    #-- Get the collection if it exists, otherwise create it
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  #-- Use cosine distance
    )

def build_index():
    """Indexing phase: load -> chunk -> embed -> store in Chroma."""
    #-- Start fresh so old chunks from earlier settings dont linger

    try:
        client.delete_collection(COLLECTION_NAME)  #-- Delete the collection if it exists to start fresh
    except Exception:
        pass  #-- If it doesn't exist, ignore the error

    docs = load_documents()
    chunks = chunk_documents(docs, max_words=150, overlap_words=30)
    texts = [c["text"] for c in chunks]


    #-- Create a new collection and upsert the chunks with their embeddings
    collection = get_collection()
    collection.upsert(
        ids=[c["chunk_id"] for c in chunks],
        documents=texts,
        embeddings=embed(texts).tolist(),
        metadatas=[{"source": c["source"]} for c in chunks],
    )
    print(f"Indexed {collection.count()} chunks.")



def search(question, k=3, source=None):
    """Retrieval phase: embed the question, ask Chroma for nearest chunks."""
    collection = get_collection()
    result = collection.query(
        query_embeddings=embed([question]).tolist(),
        n_results=k,
        where={"source": source} if source else None,   # optional filter
    )
    # Chroma returns lists-of-lists (one inner list per query); we sent one query
    hits = []
    for i in range(len(result["ids"][0])):
        hits.append({
            "id": result["ids"][0][i],
            "text": result["documents"][0][i],
            "source": result["metadatas"][0][i]["source"],
            "distance": result["distances"][0][i],
        })
    return hits


if __name__ == "__main__":
    if "--rebuild" in sys.argv or get_collection().count() == 0:
        build_index()

    while True:
        question = input("\nAsk (or 'exit'): ")
        if question == "exit":
            break
        for h in search(question):
            print(f"\n[distance {h['distance']:.3f}] {h['source']} ({h['id']})")
            print(h["text"][:300], "...")