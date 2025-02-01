from utils.graphdb import GraphDB
from utils.textembeddings import create_embeddings, search, compare_files, generate_pairs
from utils.topicmodel import compare_topics
import os
from utils.extramodels import shares_named_entities, contains_common_keywords


GRAPHDB_URI = "bolt://localhost:7687"
GRAPHDB_AUTH = ("neo4j", "jackson123")


def process():
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
    
        common_topics=compare_topics(file1, file2)
        if len(common_topics)>0:
            db.create_relationship(file1, file2, "TOPICS_IN_COMMON_WITH")
            print(f"\t*RELATIONSHIP CREATED {file1} & {file2} \n Topics In Common")
        
        print("\n🔹 Named Entities:")
        has_common_name_entities= True if len([i for i in shares_named_entities(file1, file2, db=db) if len(str(i))>3]) > 3 else False
        print("\n🔹 Common Keywords:")
        has_common_key_words= True if len(contains_common_keywords(file1, file2, db=db)) > 3 else False
    
        if has_common_name_entities:
            db.create_relationship(file1, file2, "ENTITIES_IN_COMMON_WITH")
            print(f"\t*RELATIONSHIP CREATED {file1} & {file2} \n Entities In Common")
        if has_common_key_words:
            db.create_relationship(file1, file2, "KEYWORDS_IN_COMMON_WITH")
            print(f"\t*RELATIONSHIP CREATED {file1} & {file2} \n Key Words In Common")
                
    db.close()