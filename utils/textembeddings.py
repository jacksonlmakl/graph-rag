import os
import faiss
import numpy as np
import chromadb
import sentence_transformers
from sentence_transformers import SentenceTransformer
from itertools import combinations

def create_embeddings():
    # Load BERT model (lightweight and efficient)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Directory containing .txt files
    TEXT_DIR = "files/"
    
    def read_txt_files(directory):
        """Reads all .txt files in the specified directory and returns a list of (filename, content)."""
        documents = []
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                file_path = os.path.join(directory, filename)
                with open(file_path, "r", encoding="utf-8") as file:
                    documents.append((filename, file.read()))
        return documents
    
    # Step 1: Read all text files
    docs = read_txt_files(TEXT_DIR)
    doc_names = [doc[0] for doc in docs]
    doc_texts = [doc[1] for doc in docs]
    print(doc_names)
    
    # Step 2: Generate BERT embeddings
    embeddings = model.encode(doc_texts, convert_to_numpy=True)
    
    # Step 3: Store embeddings in FAISS (Vector Database)
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity search
    faiss_index.add(embeddings)  # Add vectors to FAISS index
    
    # Save FAISS index to disk
    faiss.write_index(faiss_index, "vector_index.faiss")
    np.save("doc_names.npy", np.array(doc_names))  # Save filenames for reference
    
    print("✅ Documents embedded and stored in FAISS successfully!")
    
    # Alternative: Store in ChromaDB
    chroma_client = chromadb.PersistentClient(path="./chromadb_store")
    collection = chroma_client.get_or_create_collection("documents")
    
    # Add embeddings to ChromaDB
    for i, (name, embedding) in enumerate(zip(doc_names, embeddings)):
        collection.add(
            ids=[str(i)],
            embeddings=[embedding.tolist()],
            metadatas=[{"filename": name}]
        )
    
    print("✅ Documents stored in ChromaDB successfully!")


def search(query,window):
    # Load FAISS index and document names
    faiss_index = faiss.read_index("vector_index.faiss")
    if not window:
        doc_names = np.load("doc_names.npy", allow_pickle=True)
    else:
        doc_names=np.array(window)
    
    # Load BERT model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Encode search query
    
    query_embedding = model.encode([query], convert_to_numpy=True)
    
    # Perform similarity search
    k = 10  # Get more results initially
    max_distance = 1.0  # Define your similarity threshold (lower is better)
    
    distances, indices = faiss_index.search(query_embedding, k)
    
    # Filter results based on threshold
    filtered_results = [
        (doc_names[idx], distances[0][i])  # (document name, distance score)
        for i, idx in enumerate(indices[0]) if distances[0][i] <= max_distance
    ]
    
    # Print filtered results
    if filtered_results:
        print("\nRelevant Results (Below Threshold):")
        for doc, dist in filtered_results:
            print(f"Document: {doc} | Distance: {dist:.4f}")
    else:
        print("\nNo results found within the similarity threshold.")
    return filtered_results

def generate_pairs(strings):
    """
    Generates all possible 2-pair combinations from a list of strings.
    
    Args:
        strings (list): List of strings.
    
    Returns:
        list: List of tuple pairs.
    """
    return list(combinations(strings, 2))


def compare_files(file1, file2):
    """
    Compares the embeddings of two .txt files stored in the FAISS vector database
    and returns the distance between them.
    """
    # Load FAISS index and document names
    faiss_index = faiss.read_index("vector_index.faiss")
    doc_names = np.load("doc_names.npy", allow_pickle=True)

    # Ensure files exist in the vector database
    if file1 not in doc_names or file2 not in doc_names:
        print(f"❌ One or both files not found in the vector database: {file1}, {file2}")
        return None

    # Get index positions of the files
    index1 = int(np.where(doc_names == file1)[0][0])  # Convert to int
    index2 = int(np.where(doc_names == file2)[0][0])  # Convert to int

    # Extract embeddings for the two files
    embedding1 = np.zeros((faiss_index.d,), dtype=np.float32)
    embedding2 = np.zeros((faiss_index.d,), dtype=np.float32)

    faiss_index.reconstruct(index1, embedding1)  # Correctly retrieve vector
    faiss_index.reconstruct(index2, embedding2)

    # Compute L2 (Euclidean) distance
    distance = np.linalg.norm(embedding1 - embedding2)

    print(f"✅ Distance between '{file1}' and '{file2}': {distance:.4f}")
    return distance
