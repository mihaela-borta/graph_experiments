import asyncio
import re
import json
import os
from typing import Any, List, Callable, Optional, Union
from tenacity import retry, wait_exponential, stop_after_attempt, wait_fixed, wait_chain, before_sleep_log
from ratelimit import limits, sleep_and_retry, RateLimitException
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from llama_index.core.async_utils import run_jobs
from llama_index.core.indices.property_graph.utils import default_parse_triplets_fn
from llama_index.core.graph_stores.types import EntityNode, KG_NODES_KEY, KG_RELATIONS_KEY, Relation
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.prompts.default_prompts import DEFAULT_KG_TRIPLET_EXTRACT_PROMPT
from llama_index.core.schema import TransformComponent, BaseNode
from llama_index.core import Settings

def wait_and_log(retry_state):
    logger.info(f"Rate limit reached in extractor. Waiting before retrying... (Attempt {retry_state.attempt_number})")
    return 60  # Return the number of seconds to wait

class GraphRAGExtractor(TransformComponent):
    llm: LLM
    extract_prompt: PromptTemplate
    parse_fn: Callable
    num_workers: int
    max_paths_per_chunk: int
    use_local_responses: bool
    raw_responses_file: str

    def __init__(
        self,
        llm: Optional[LLM] = None,
        extract_prompt: Optional[Union[str, PromptTemplate]] = None,
        parse_fn: Callable = default_parse_triplets_fn,
        max_paths_per_chunk: int = 10,
        num_workers: int = 4,
        use_local_responses: bool = False,
        raw_responses_file: str = "/app/data/raw_llm_responses.json",
    ) -> None:
        if isinstance(extract_prompt, str):
            extract_prompt = PromptTemplate(extract_prompt)

        super().__init__(
            llm=llm or Settings.llm,
            extract_prompt=extract_prompt or DEFAULT_KG_TRIPLET_EXTRACT_PROMPT,
            parse_fn=parse_fn,
            num_workers=num_workers,
            max_paths_per_chunk=max_paths_per_chunk,
            use_local_responses=use_local_responses,  # Add this line
            raw_responses_file=raw_responses_file,    # Add this line
        )

        self.use_local_responses = use_local_responses
        self.raw_responses_file = raw_responses_file

    @classmethod
    def class_name(cls) -> str:
        return "GraphExtractor"

    def __call__(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        return asyncio.run(
            self.acall(nodes, show_progress=show_progress, **kwargs)
        )

    @retry(
        wait=wait_chain(
            wait_fixed(60),  # Always wait 60 seconds first
            wait_exponential(multiplier=1, min=4, max=60)  # Then use exponential backoff
        ),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.INFO)
    )
    @limits(calls=5, period=60)  # Adjust these values based on your rate limits
    async def _aextract(self, node: BaseNode) -> BaseNode:
        assert hasattr(node, "text")

        text = node.get_content(metadata_mode="llm")
        try:
            start_time = time.time()
            
            if self.use_local_responses:
                llm_response = self._load_raw_response(text)
                if llm_response:
                    logger.info(f"Using local response for text: {text[:50]}...")
                    prediction_source = "local file"
                else:
                    logger.info(f"No local response found. Querying OpenAI for text: {text[:50]}...")
                    prediction_source = "LLM API call"
                    llm_response = await self.llm.apredict(
                        self.extract_prompt,
                        text=text,
                        max_knowledge_triplets=self.max_paths_per_chunk,
                    )
                    self._store_raw_response(text, llm_response)
            else:
                logger.info(f"Querying OpenAI for text: {text[:50]}...")
                prediction_source = "LLM API call"
                llm_response = await self.llm.apredict(
                    self.extract_prompt,
                    text=text,
                    max_knowledge_triplets=self.max_paths_per_chunk,
                )

            end_time = time.time()

            logger.info(f"Getting LLM prediction from {prediction_source} took {end_time - start_time:.2f} seconds")
            #logger.info(f"Full LLM response: {llm_response}")

            entities, entities_relationship = self.parse_fn(llm_response)
            logger.info(f"Extracted {len(entities)} entities and {len(entities_relationship)} relationships")
        except RateLimitException:
            logger.warning("Rate limit exceeded. Retrying...")
            raise
        except Exception as e:
            logger.error(f"Failed to extract entities and relationships: {str(e)}")
            entities = []
            entities_relationship = []

        existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
        existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])
        entity_metadata = node.metadata.copy()
        for entity, entity_type, description in entities:
            entity_metadata["entity_description"] = description
            entity_node = EntityNode(
                name=entity, label=entity_type, properties=entity_metadata
            )
            existing_nodes.append(entity_node)

        relation_metadata = node.metadata.copy()
        for triple in entities_relationship:
            subj, obj, rel, description = triple
            relation_metadata["relationship_description"] = description
            rel_node = Relation(
                label=rel,
                source_id=subj,
                target_id=obj,
                properties=relation_metadata,
            )

            existing_relations.append(rel_node)

        node.metadata[KG_NODES_KEY] = existing_nodes
        node.metadata[KG_RELATIONS_KEY] = existing_relations
        return node

    async def acall(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        logger.info(f"Starting extraction for {len(nodes)} nodes")
        jobs = []
        for node in nodes:
            jobs.append(self._aextract(node))

        return await run_jobs(
            jobs,
            workers=self.num_workers,
            show_progress=show_progress,
            desc="Extracting paths from text",
        )

    def _store_raw_response(self, text: str, response: str) -> None:
        try:
            if os.path.exists(self.raw_responses_file):
                with open(self.raw_responses_file, 'r') as f:
                    data = json.load(f)
            else:
                data = {}
            
            if text not in data:  # Only add if not already present
                data[text] = response
            
                with open(self.raw_responses_file, 'w') as f:
                    json.dump(data, f, indent=2)
            else:
                logger.info(f"Response for text already exists: {text[:50]}...")
        except Exception as e:
            logger.error(f"Failed to store raw LLM response: {str(e)}")

    def _load_raw_response(self, text: str) -> str:
        try:
            with open(self.raw_responses_file, 'r') as f:
                data = json.load(f)
            response = data.get(text, "")
            if not response:
                logger.info(f"No local response found for text: {text[:50]}...")
            return response
        except Exception as e:
            logger.error(f"Failed to load raw LLM response: {str(e)}")
            return ""

def parse_fn(response_str: str) -> Any:
    logger.info(f"Raw LLM response: {response_str}")
    entity_pattern = r'\("entity"\$\$\$\$"?(.+?)"?\$\$\$\$"?(.+?)"?\$\$\$\$"?(.+?)"?\)'
    relationship_pattern = r'\("relationship"\$\$\$\$"?(.+?)"?\$\$\$\$"?(.+?)"?\$\$\$\$"?(.+?)"?\$\$\$\$"?(.+?)"?\)'
    entities = re.findall(entity_pattern, response_str)
    relationships = re.findall(relationship_pattern, response_str)
    logger.info(f"Parsed entities: {entities}")
    logger.info(f"Parsed relationships: {relationships}")
    return entities, relationships