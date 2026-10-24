
def chunk_text(text, max_words=100, overlap_words=20):
    """Split text into chunks of roughly max_words, with overlap"""
    
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    current_chunk = []   #-- holds words in the chunk being built

    for para in paragraphs:
        words = para.split()

        #if adding these para would overflow , close the current chunk
        if current_chunk and len(current_chunk) + len(words) > max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap_words:] #-- keep the last overlap_words for the next chunk using slicing string method

        current_chunk.extend(words) #-- add the words of the current paragraph to the current chunk

        #-- if the current chunk is too big, split it into smaller chunks
        while len(current_chunk) > max_words:
            chunks.append(" ".join(current_chunk[:max_words]))
            current_chunk = current_chunk[max_words - overlap_words:]


    #-- add the last chunk if it is not empty
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def chunk_docs(docs):
    """Turn a list of documents into a list of chunks, keeping metadata"""
    all_chunks = []
    for doc in docs:
        pieces = chunk_text(doc["text"])
        for i, piece in enumerate(pieces):
            all_chunks.append({
                "text": piece,
                "source": doc["source"],
                "chunk_id": f"{doc['source']}-{i}",   #-- unique ID per chunk, combining source and index
            })

    return all_chunks


if __name__ == "__main__":
    from loader import load_docs
    docs = load_docs()
    chunks = chunk_docs(docs)
    print(f"{len(docs)} documents -> {len(chunks)} chunks\n")
    for c in chunks[:3]:  #-- print first 3 chunks for inspection
        print(c["chunk_id"],"|", len(c["text"].split()), "words")
        print(c["text"][:200], "...\n") #-- print first 200 characters of the chunk