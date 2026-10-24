from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed(texts):
    """Turn a list of texts into a list of vectors using the sentence-transformers model"""
    return model.encode(texts, normalize_embeddings=True)  #-- normalize to unit length for cosine similarity

if __name__ == "__main__":
    sentences = [
        "How do I get my money back",
        "What is the refund policy?",
        "The cat sat on the sofa.",
    ]
    vectors = embed(sentences)

    print("Shape: ", vectors.shape) #-- print the shape of the resulting array i.e.dimensions which in this case is (3, 384) for 3 sentences and 384-dimensional embeddings
    print("First 5 numbers of vector 0 : ", vectors[0][:5])

    #-- since the vectors are normalized, the dot product is equivalent to cosine similarity
    print("\nSimiliarity matrix: ")
    print(vectors @ vectors.T)  #-- print the similarity matrix by taking the dot product of the vectors with their transpose