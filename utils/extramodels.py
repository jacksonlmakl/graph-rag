import os
import nltk
import string
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter


# Ensure stopwords are available
nltk.download("stopwords")
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))

# Custom list of very common words that TF-IDF might still pick up
SUPER_COMMON_WORDS = {
    "said", "one", "also", "like", "many", "much", "new", "old", "time", "year",
    "people", "first", "make", "way", "even", "two", "day", "still", "know"
}

# Load Hugging Face NER model (DistilBERT)
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")

def read_text(file1, file2):
    """Reads and returns the content of two text files."""
    file1_path = os.path.join("files", file1)
    file2_path = os.path.join("files", file2)

    with open(file1_path, "r", encoding="utf-8") as f1, open(file2_path, "r", encoding="utf-8") as f2:
        text1, text2 = f1.read(), f2.read()
    
    return text1, text2

### 🔹 1️⃣ Function: SHARES_NAMED_ENTITIES ###
def shares_named_entities(file1, file2, db):
    """Finds shared named entities (People, Locations, Organizations) between two text files using Hugging Face Transformers."""
    text1, text2 = read_text(file1, file2)

    # Extract named entities
    entities1 = {ent["word"] for ent in ner_pipeline(text1) if ent["entity_group"] in {"PER", "ORG", "LOC"}}

    entities2 = {ent["word"] for ent in ner_pipeline(text2) if ent["entity_group"] in {"PER", "ORG", "LOC"}}

        
    # Find common entities
    common_entities = entities1.intersection(entities2)
    common_entities=[i for i in common_entities if len(i)>4]
    for i in common_entities:
        db.create_node(i,"Entity")
        db.create_relationship(file2, i, "CONTAINS")
        db.create_relationship(file1, i, "CONTAINS")
    print(f"✅ SHARED NAMED ENTITIES between '{file1}' and '{file2}': {common_entities}")
    return list(common_entities)


### 🔹 2️⃣ Function: CONTAINS_COMMON_KEYWORDS (Now Filters Out Common Words) ###
def contains_common_keywords(file1, file2, db,num_keywords=10):
    """Extracts and compares top keywords between two text files using TF-IDF, filtering stopwords and super common words."""
    text1, text2 = read_text(file1, file2)

    # Preprocess text: Remove punctuation and stopwords
    def preprocess(text):
        text = text.lower().translate(str.maketrans("", "", string.punctuation))
        return " ".join([word for word in text.split() if word not in STOPWORDS and word not in SUPER_COMMON_WORDS])

    processed_texts = [preprocess(text1), preprocess(text2)]

    # Compute TF-IDF vectors
    vectorizer = TfidfVectorizer(max_features=100)
    tfidf_matrix = vectorizer.fit_transform(processed_texts)

    # Extract top keywords for each document
    feature_names = vectorizer.get_feature_names_out()
    top_keywords1 = set(feature_names[tfidf_matrix[0].toarray().argsort()[0][-num_keywords:]])
    top_keywords2 = set(feature_names[tfidf_matrix[1].toarray().argsort()[0][-num_keywords:]])

    # Find common keywords
    common_keywords = top_keywords1.intersection(top_keywords2)
    common_entities=[i for i in common_entities if len(i)>=3]
    print(f"✅ COMMON KEYWORDS between '{file1}' and '{file2}': {common_keywords}")
    for i in common_keywords:
        db.create_node(i,"Keyword")
        db.create_relationship(file2, i, "CONTAINS")
        db.create_relationship(file1, i, "CONTAINS")
    return list(common_keywords)



# Main function for execution
if __name__ == "__main__":
    file1 = "file1.txt"
    file2 = "file2.txt"

    print("\n🔹 Named Entities:")
    shares_named_entities(file1, file2)

    print("\n🔹 Sentiment Analysis:")
    shares_sentiment(file1, file2)

    print("\n🔹 Common Keywords:")
    contains_common_keywords(file1, file2)
