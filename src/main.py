import os
import warnings
import pandas as pd
from dotenv import load_dotenv
from typing import List, Dict, Any, Union
from pathlib import Path
import sys
import argparse

from llama_index.core import Document, PropertyGraphIndex
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SentenceSplitter

from graphrag_extractor import GraphRAGExtractor, parse_fn
from graphrag_store import GraphRAGStore
from graphrag_query_engine import GraphRAGQueryEngine

from tenacity import retry, wait_exponential, stop_after_attempt, wait_fixed, wait_chain, before_sleep_log
from ratelimit import limits
import logging
import asyncio
from asyncio import Semaphore
from tenacity import RetryCallState
import concurrent.futures

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")


# Constants
KG_TRIPLET_EXTRACT_TMPL = """
-Goal-
Given a text document, identify all entities and their entity types from the text and all relationships among the identified entities.
Given the text, extract up to {max_knowledge_triplets} entity-relation triplets.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: Type of the entity
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"$$$$<entity_name>$$$$<entity_type>$$$$<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relation: relationship between source_entity and target_entity
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other

Format each relationship as ("relationship"$$$$<source_entity>$$$$<target_entity>$$$$<relation>$$$$<relationship_description>)

3. When finished, output.

-Real Data-
######################
text: {text}
######################
output:"""

def get_env_vars(env_file: Union[str, Path]='.env.dev') -> Dict:
    """Loads environment variables from file and returns them as a dict"""
    initial_env_vars = dict(os.environ)
    load_dotenv(dotenv_path=env_file)

    current_env_vars = dict(os.environ)
    loaded_vars = {key: current_env_vars[key] for key in current_env_vars if key not in initial_env_vars or current_env_vars[key] != initial_env_vars[key]}

    return loaded_vars


def read_data(num_samples: int = 5) -> List[Document]:
    logger.info(f"Will serve: {num_samples} documents")
    news = pd.read_csv(
        "https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/news_articles.csv"
    )[:num_samples]
    
    documents = [
        Document(text=f"{row['title']}: {row['text']}")
        for _, row in news.iterrows()
    ]
    return documents

def setup_llm():
    return OpenAI(model="gpt-4o")

def create_nodes(documents: List[Document]) -> List[Any]:
    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
    return splitter.get_nodes_from_documents(documents)

def print_data_info(documents: List[Document], nodes: List[Any]):
    total_chars = sum(len(doc.text) for doc in documents)
    total_size_kb = sys.getsizeof(documents) / 1024
    
    print(f"Debug: Total number of documents: {len(documents)}")
    print(f"Debug: Total characters in documents: {total_chars}")
    print(f"Debug: Approximate size of documents in memory: {total_size_kb:.2f} KB")
    print(f"Debug: Total number of nodes after splitting: {len(nodes)}")

    # token_counter = TokenCounter()
    # total_tokens = sum(token_counter.get_num_tokens(node.text) for node in nodes)
    # print(f"Debug: Estimated total tokens in nodes: {total_tokens}")

def setup_graph_store() -> GraphRAGStore:
    return GraphRAGStore(
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        url=os.getenv("NEO4J_URI", "bolt://neo4j:7687"),  # Use the service name as hostname
    )


def wait_and_log(retry_state: RetryCallState) -> None:
    logger.info(f"Rate limit reached. Waiting before retrying... (Attempt {retry_state.attempt_number})")
    return 60  # Return the number of seconds to wait

async def process_node(node, kg_extractor, semaphore):
    async with semaphore:
        return await kg_extractor._aextract(node)

@retry(
    wait=wait_chain(
        wait_fixed(60),
        wait_exponential(multiplier=1, min=4, max=300)  # Increased max wait time
    ),
    stop=stop_after_attempt(5),
    before_sleep=wait_and_log
)
@limits(calls=5, period=60)
async def build_index_with_rate_limit(nodes: List[Any], graph_store: GraphRAGStore, llm: Any, use_local_responses: bool) -> PropertyGraphIndex:
    kg_extractor = GraphRAGExtractor(
        llm=llm,
        extract_prompt=KG_TRIPLET_EXTRACT_TMPL,
        max_paths_per_chunk=2,
        parse_fn=parse_fn,
        use_local_responses=use_local_responses,
        raw_responses_file="/app/data/raw_llm_responses.json"
    )
    
    semaphore = Semaphore(2)  # Reduce to 2 concurrent API calls
    
    processed_nodes = []
    for i in range(0, len(nodes), 2):  # Process 2 nodes at a time
        batch = nodes[i:i+2]
        logger.info(f"Processing batch {i//2 + 1} with {len(batch)} nodes")
        batch_processed = await asyncio.gather(*[process_node(node, kg_extractor, semaphore) for node in batch])
        processed_nodes.extend(batch_processed)
        logger.info(f"Batch {i//2 + 1} processed. Total processed nodes: {len(processed_nodes)}")
        await asyncio.sleep(1)  # Add a small delay between batches
    
    # Create the index in a separate thread to avoid blocking the event loop
    with concurrent.futures.ThreadPoolExecutor() as executor:
        index = await asyncio.get_event_loop().run_in_executor(
            executor,
            lambda: PropertyGraphIndex(
                nodes=processed_nodes,
                kg_extractors=[kg_extractor],
                property_graph_store=graph_store,
                show_progress=True,
            )
        )
    
    logger.info(f"PropertyGraphIndex created with {len(processed_nodes)} nodes")
    return index

def create_query_engine(index: PropertyGraphIndex, llm: Any) -> GraphRAGQueryEngine:
    return GraphRAGQueryEngine(
        graph_store=index.property_graph_store,
        llm=llm,
        index=index,
        similarity_top_k=10,
    )


async def main(use_local_responses: bool):
    # Load environment variables
    ENV_FILE = '/app/config/.env.dev'
    _ = get_env_vars(ENV_FILE)
    logger.info("Environment variables loaded successfully.")
    
    # Read data (now using only 5 documents)
    documents = read_data(50)
    logger.info(f"Data read successfully. Number of documents: {len(documents)}")
    
    # Setup LLM
    llm = setup_llm()
    logger.info("LLM setup completed.")
    
    # Create nodes
    nodes = create_nodes(documents)
    logger.info(f"Nodes created successfully. Number of nodes: {len(nodes)}")
    
    # Print data info
    print_data_info(documents, nodes)
    
    # Setup graph store
    graph_store = setup_graph_store()
    logger.info("Graph store setup completed.")
    
    # Check if graph already exists
    node_count = graph_store.get_node_count()
    if node_count > 0:
        logger.info(f"Graph already exists with {node_count} nodes.")

        index = PropertyGraphIndex.from_existing(
            property_graph_store=graph_store,
            embed_kg_nodes=True,
        )
    else:
        # Build index with rate limiting
        index = await build_index_with_rate_limit(nodes, graph_store, llm, use_local_responses)
        logger.info(f"Index built successfully with {len(nodes)} nodes.")

    """  
    # Build communities
    if graph_store.get_rel_count() > 0:
        await build_communities_async(index.property_graph_store)
        logger.info("Communities built successfully.")
    else:
        logger.warning("No relationships found in the graph. Skipping community building.")
    """
    # Verify if the graph was built
    node_count = graph_store.get_node_count()
    relation_count = graph_store.get_rel_count()
    logger.info(f"Graph verification: {node_count} nodes and {relation_count} relations found.")
    
    if node_count == 0 or relation_count == 0:
        logger.warning("Graph may not have been built successfully. Please check the process.")
    else:
        logger.info("Graph appears to have been built successfully.")
    
    logger.info(f"""DEBUG: {index.property_graph_store.get_triplets()[10]}""")


    # Create query engine
    query_engine = create_query_engine(index, llm)
    logger.info("Query engine created successfully.")
    
    # Example queries
    queries = [
        "What are the main news discussed in the document?",
        "What are the main news in energy sector?",
    ]
    
    for i, query in enumerate(queries, 1):
        if i==1:
            logger.info(f"\nProcessing query {i}: '{query}'")
            response = await query_engine.custom_query(query)  # Use aquery instead of query
            logger.info(f"Query {i}: {query}")
            logger.info(f"Response {i}: {response}\n")
        
    logger.info("All queries processed successfully.")

async def build_communities_async(property_graph_store):
    await asyncio.get_event_loop().run_in_executor(
        None,
        property_graph_store.build_communities
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the GraphRAG system")
    parser.add_argument("--use-local-responses", action="store_true", help="Use and store local LLM responses")
    args = parser.parse_args()

    logger.info(f"use_local_responses: {args.use_local_responses}")

    asyncio.run(main(use_local_responses=args.use_local_responses))

