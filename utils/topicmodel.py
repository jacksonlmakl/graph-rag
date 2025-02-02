import os
import umap
import textwrap
import hdbscan
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load embedding model globally
if "embedding_model" not in globals():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def split_text(text, chunk_size=50):
    """Splits text into smaller chunks to prevent memory overload."""
    return textwrap.wrap(text, width=chunk_size)

def preprocess_topic_words(topic_words):
    """Removes stop words and short words from extracted topic words."""
    if not topic_words or topic_words is False:  # Check if topics exist
        return []
    return [word for word in topic_words if word.lower() not in ENGLISH_STOP_WORDS and len(word) > 2]

def compare_topics(file1, file2, db):
    """
    Uses BERTopic to extract and compare topics from two files,
    ensuring that no topics are discarded due to clustering issues.
    """
    file1 = "files/" + file1
    file2 = "files/" + file2

    # Read text files
    with open(file1, "r", encoding="utf-8") as f:
        text1 = f.read()
    with open(file2, "r", encoding="utf-8") as f:
        text2 = f.read()

    # Split text into smaller chunks
    documents = split_text(text1) + split_text(text2)

    # Configure UMAP (low-dimensional space for clustering)
    umap_model = umap.UMAP(n_neighbors=10, n_components=5, min_dist=0.1, random_state=42)

    # Configure HDBSCAN (ensures every document gets a topic)
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, prediction_data=True)

    # Create BERTopic model with adjusted settings
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=True  # Ensures probability-based topic assignment
    )

    # Extract topics
    topics, _ = topic_model.fit_transform(documents)

    # Extract topic words for both files
    topic_words_file1 = preprocess_topic_words([word for word, _ in topic_model.get_topic(0) or []])
    topic_words_file2 = preprocess_topic_words([word for word, _ in topic_model.get_topic(1) or []])

    # Handle cases where no topics were found
    if not topic_words_file1 or not topic_words_file2:
        print(f"\n❌ No meaningful topics found even after forcing topic extraction: '{file1}', '{file2}'")
        return []

    # Find common topics
    common_topics = list(set(topic_words_file1).intersection(set(topic_words_file2)))

    # Print and return the common topics
    if common_topics:
        print(f"\n✅ Common Topics between '{file1}' and '{file2}': {common_topics}")
        for i in common_topics:
            db.create_node(i,"Topic")
            db.create_relationship(file2, i, "CONTAINS")
            db.create_relationship(file1, i, "CONTAINS")
    else:
        print(f"\n❌ No common topics found between '{file1}' and '{file2}', but topics were extracted.")

    return common_topics
