from neo4j import GraphDatabase

GRAPHDB_URI = "bolt://localhost:7687"
GRAPHDB_AUTH = ("neo4j", "jackson123")

class GraphDB:
    def __init__(self, uri, auth):
        self.driver = GraphDatabase.driver(uri, auth=auth)

    def close(self):
        self.driver.close()

    # Create a generic node
    def create_node(self, name, label="Node"):
        with self.driver.session() as session:
            session.run(f"MERGE (n:{label} {{name: $name}})", name=name)

    # Create a tag
    def create_tag(self, name):
        self.create_node(name, label="Tag")

    # Create a relationship between two nodes
    def create_relationship(self, node1, node2, rel_type):
        with self.driver.session() as session:
            query = f"""
            MATCH (a {{name: $node1}}), (b {{name: $node2}})
            MERGE (a)-[:{rel_type}]->(b)
            """
            session.run(query, node1=node1, node2=node2)

    # Create a relationship between a node and a tag
    def tag_node(self, node_name, tag_name):
        self.create_relationship(node_name, tag_name, "TAGGED_WITH")

    # Get all nodes of a given label
    def get_nodes(self, label="Node"):
        with self.driver.session() as session:
            result = session.run(f"MATCH (n:{label}) RETURN n.name AS name")
            return [record["name"] for record in result]

    # Get all relationships of a given type
    def get_relationships(self, node_name):
        with self.driver.session() as session:
            query = """
            MATCH (a {name: $node_name})-[r]->(b)
            RETURN type(r) AS relationship, b.name AS connected_node
            """
            result = session.run(query, node_name=node_name)
            return [{"relationship": record["relationship"], "node": record["connected_node"]} for record in result]

if __name__ == "main":
    # Initialize Database
    db = GraphDB(GRAPHDB_URI, GRAPHDB_AUTH)
    
    # Create Nodes and Tags
    db.create_node("node1")
    db.create_node("node2")
    db.create_tag("tag1")
    db.create_tag("tag2")
    
    # Create Relationships
    db.create_relationship("node1", "node2", "SIMILAR")
    
    
    db.tag_node("node1", "tag1")
    db.tag_node("node2", "tag2")
    
    # Fetch Nodes and Relationships
    print("Nodes:", db.get_nodes())  # Output: ['Jackson', 'Alex']
    print("Tags:", db.get_nodes(label="Tag"))  # Output: ['Engineer', 'Data Scientist']
    print("Node 1's Relationships:", db.get_relationships("node1"))
    
    # Close Connection
    db.close()
