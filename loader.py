from pathlib import Path

def load_docs(folder = "data"):
    """ Read every text file in the folder and return a list of docs"""
    documents = []

    for path in Path(folder).glob("*.txt"):
        text = path.read_text(encoding="utf-8")
        documents.append({
            "text": text,
            "source": path.name   #-- metadata
        })
    return documents


#--This allows the file to act both as an executable script and as an importable module
#--when loder.py is run directly , python sets "__name__" to "__main__", so the code inside the if block will be executed.
#--When loader.py is imported as a module, "__name__" is set to "loader", and the code inside the if block will not be executed.
if __name__ == "__main__":
    docs = load_docs()
    if not docs:
       print("No documents found in the folder.")
    for d in docs:
        print(d["source"], len(d["text"]), "characters")