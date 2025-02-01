from utils.graphdb import GraphDB
from utils.textembeddings import create_embeddings, search, compare_files,generate_pairs
from utils.topicmodel import extract_topics_bertopic
import os

GRAPHDB_URI = "bolt://localhost:7687"
GRAPHDB_AUTH = ("neo4j", "jackson123")



db = GraphDB(GRAPHDB_URI, GRAPHDB_AUTH)

file_names=[i for i in os.listdir('files') if i.endswith(".txt")]

#Generate all possible pairs of files
file_pairs=generate_pairs(file_names)

#Create nodes for each file in graph db
for name in file_names:
    db.create_node(name)

#Create & Save embeddings in vector db
create_embeddings()

# Create Relationships
for file1,file2 in file_pairs:
    distance=compare_files(file1,file2)
    threshold=1
    if distance<threshold:
        db.create_relationship(file1, file2, "SIMILAR_TO")
        print(f"\t*RELATIONSHIP CREATED {file1} & {file2} \nDistance: {distance}")
        
db.close()
extract_topics_bertopic(file1,file2)