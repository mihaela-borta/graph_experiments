import networkx as nx
from graspologic.partition import hierarchical_leiden
from collections import defaultdict

from llama_index.core.llms import ChatMessage
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.llms.openai import OpenAI
import re
import logging
from llama_index.core import Document
from llama_index.core.schema import TextNode, NodeRelationship
from typing import List

# Set up logging
logger = logging.getLogger(__name__)

class GraphRAGStore(Neo4jPropertyGraphStore):
    community_summary = {}
    entity_info = None
    max_cluster_size = 5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info("GraphRAGStore initialized")

    def generate_community_summary(self, text):
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "You are provided with a set of relationships from a knowledge graph, each represented as "
                    "entity1->entity2->relation->relationship_description. Your task is to create a summary of these "
                    "relationships. The summary should include the names of the entities involved and a concise synthesis "
                    "of the relationship descriptions. The goal is to capture the most critical and relevant details that "
                    "highlight the nature and significance of each relationship. Ensure that the summary is coherent and "
                    "integrates the information in a way that emphasizes the key aspects of the relationships."
                ),
            ),
            ChatMessage(role="user", content=text),
        ]
        response = OpenAI().chat(messages)
        clean_response = re.sub(r"^assistant:\s*", "", str(response)).strip()
        logger.info(f"Generated community summary: {clean_response[:100]}...")  # Log first 100 chars
        return clean_response

    def build_communities(self):
        logger.info("Starting community building process")
        nx_graph = self._create_nx_graph()
        if nx_graph.number_of_edges() == 0:
            logger.warning("No relationships found in the graph. Skipping community building.")
            return
        community_hierarchical_clusters = hierarchical_leiden(
            nx_graph, max_cluster_size=self.max_cluster_size
        )
        self.entity_info, community_info = self._collect_community_info(
            nx_graph, community_hierarchical_clusters
        )
        self._summarize_communities(community_info)
        logger.info(f"Community building completed. {len(community_info)} communities found.")

    def _create_nx_graph(self):
        logger.info("Creating NetworkX graph from triplets")
        nx_graph = nx.Graph()
        triplets = self.get_triplets()
        for entity1, relation, entity2 in triplets:
            nx_graph.add_node(entity1.name)
            nx_graph.add_node(entity2.name)
            nx_graph.add_edge(
                relation.source_id,
                relation.target_id,
                relationship=relation.label,
                description=relation.properties["relationship_description"],
            )
        logger.info(f"NetworkX graph created with {nx_graph.number_of_nodes()} nodes and {nx_graph.number_of_edges()} edges")
        return nx_graph

    def _collect_community_info(self, nx_graph, clusters):
        logger.info("Collecting community information")
        entity_info = defaultdict(set)
        community_info = defaultdict(list)

        for item in clusters:
            node = item.node
            cluster_id = item.cluster

            entity_info[node].add(cluster_id)

            for neighbor in nx_graph.neighbors(node):
                edge_data = nx_graph.get_edge_data(node, neighbor)
                if edge_data:
                    detail = f"{node} -> {neighbor} -> {edge_data['relationship']} -> {edge_data['description']}"
                    community_info[cluster_id].append(detail)

        entity_info = {k: list(v) for k, v in entity_info.items()}
        logger.info(f"Collected information for {len(community_info)} communities")
        return dict(entity_info), dict(community_info)

    def _summarize_communities(self, community_info):
        logger.info("Summarizing communities")
        for community_id, details in community_info.items():
            details_text = "\n".join(details) + "."
            self.community_summary[community_id] = self.generate_community_summary(details_text)
        logger.info(f"Summarized {len(self.community_summary)} communities")

    def get_community_summaries(self):
        if not self.community_summary:
            logger.info("Community summaries not found. Building communities.")
            self.build_communities()
        return self.community_summary

    def get_node_count(self):
        with self._driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) as count")
            count = result.single()["count"]
            logger.info(f"Node count: {count}")
            return count

    def get_rel_count(self):
        with self._driver.session() as session:
            result = session.run("MATCH ()-->() RETURN count(*) as count")
            count = result.single()["count"]
            logger.info(f"Relationship count: {count}")
            return count

    def get_all_nodes(self) -> List[Document]:
        logger.info("Fetching all nodes from the database")
        query = """
            MATCH (n:__Node__)
            RETURN n.id AS id, n.text AS text, n.metadata AS metadata, 
                   labels(n) AS labels, n.name AS name
        """
        documents = []
        with self._driver.session() as session:
            result = session.run(query)
            for record in result:
                node_id = record["id"]
                text = record["text"]
                metadata = record["metadata"] or {}
                labels = record["labels"]
                name = record["name"]

                if "Chunk" in labels and text:
                    # This is a document chunk
                    doc = Document(text=text, id_=node_id, metadata=metadata)
                    documents.append(doc)
                elif "__Entity__" in labels and name:
                    # This is an entity node
                    node = TextNode(text=name, id_=node_id, metadata=metadata)
                    documents.append(node)
                elif "__Relation__" in labels:
                    # This is a relationship node
                    # You might want to handle this differently depending on your needs
                    pass
                else:
                    logger.warning(f"Skipping node with unexpected structure: id={node_id}, labels={labels}")

        logger.info(f"Fetched {len(documents)} valid nodes from the database")
        return documents

    def add_node(self, node):
        if node.id_ and node.text and node.text.strip():
            super().add_node(node)
            logger.info(f"Node added to graph store: {node.id_}")
        else:
            logger.warning(f"Skipped adding invalid node: id={node.id_}, text_length={len(node.text) if node.text else 0}")

   