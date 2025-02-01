import os
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

def extract_topics_bertopic(file1, file2):
    """
    Uses BERTopic (Transformer-based topic modeling) to extract and compare topics from two files.
    """
    # Load a transformer-based embedding model (fast & accurate)
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Read text files
    with open(file1, "r", encoding="utf-8") as f:
        text1 = f.read()
    with open(file2, "r", encoding="utf-8") as f:
        text2 = f.read()

    # Combine both texts into a list for topic modeling
    documents = [text1, text2]

    # Create BERTopic model
    topic_model = BERTopic(embedding_model=embedding_model)

    # Extract topics
    topics, probs = topic_model.fit_transform(documents)

    # Get topic representations
    topic_info = topic_model.get_topic_info()
    
    # Compute topic similarity
    topic_vectors = topic_model.topic_embeddings
    similarity_score = topic_model.approximate_distribution_similarity(topic_vectors[0], topic_vectors[1])

    print(f"\n✅ BERTopic Similarity between '{file1}' and '{file2}': {similarity_score:.4f}\n")

    # Display topics
    print(f"🔹 Topics in '{file1}':", topic_model.get_topic(0))
    print(f"\n🔹 Topics in '{file2}':", topic_model.get_topic(1))

    return similarity_score

# Example Usage
file1 = "files/file1.txt"
file2 = "files/file2.txt"
extract_topics_bertopic(file1, file2)
