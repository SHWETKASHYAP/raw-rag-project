from loader import load_docs
from chunker import chunk_docs
from embedder import embed

#--- 1.Load docs -> 2. Chunk docs -> 3. Embed chunks 
docs = load_docs()
chunks = chunk_docs(docs)
chunk_vectors = embed([c["text"] for c in chunks])   #--([]) , [] Since embed expects a list of texts, we use a list comprehension to extract the "text" field from each chunk in chunks. This creates a new list containing only the text of each chunk, which is then passed to the embed function to generate their corresponding vector embeddings.
print(f"Embedded {len(chunks)} chunks, shape {chunk_vectors.shape}")

#Retrieval: embed the question, compare to every chunk
def search(question, k=3):
    q_vec = embed([question])[0]   #-- embed returns a list of vectors, we want the first (and only) one for the question
    scores = chunk_vectors @ q_vec   #-- dot product of all chunk vectors with the question vector, gives a score for each chunk
                                    #-- where @ -> is the matrix multiplication in python and since vectors are normalized, this gives dot product
    top_ids = scores.argsort()[::-1][:k]  #-- argsort returns the indices that would sort the array, [::-1] reverses it to get descending order, [:k] takes the top k
    return [(chunks[i], float(scores[i])) for i in top_ids]  #-- return the top k chunks and their scores as a list of tuples


while True:
    question = input("\nAsk (or type 'exit' to quit): ")
    if question == "exit":
        break
    for chunk, score in search(question):
        print(f"\n[{score:.3f}] {chunk['source']} ({chunk['chunk_id']})")
        print(chunk["text"][:300], "...")