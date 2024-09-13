from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.llms import LLM, ChatMessage
from llama_index.core import PropertyGraphIndex
from graphrag_store import GraphRAGStore  # Add this import
import re
import asyncio
from typing import List

class GraphRAGQueryEngine(CustomQueryEngine):
    graph_store: GraphRAGStore
    index: PropertyGraphIndex
    llm: LLM
    similarity_top_k: int = 20

    def custom_query(self, query_str: str) -> str:
        # This method is required by CustomQueryEngine
        return asyncio.run(self.aquery(query_str))

    async def aquery(self, query_str: str) -> str:
        entities = await self.get_entities(query_str, self.similarity_top_k)
        community_ids = self.retrieve_entity_communities(self.graph_store.entity_info, entities)
        community_summaries = self.graph_store.get_community_summaries()
        
        community_answers = await asyncio.gather(*[
            self.generate_answer_from_summary(community_summary, query_str)
            for id, community_summary in community_summaries.items()
            if id in community_ids
        ])

        final_answer = await self.aggregate_answers(community_answers)
        return final_answer

    async def get_entities(self, query_str: str, similarity_top_k: int) -> List[str]:
        nodes_retrieved = await self.index.as_retriever(
            similarity_top_k=similarity_top_k
        ).aretrieve(query_str)

        entities = set()
        pattern = r"(\w+(?:\s+\w+)*)\s*\({[^}]*}\)\s*->\s*([^(]+?)\s*\({[^}]*}\)\s*->\s*(\w+(?:\s+\w+)*)"

        for node in nodes_retrieved:
            matches = re.findall(pattern, node.text, re.DOTALL)
            for match in matches:
                subject, obj = match[0], match[2]
                entities.add(subject)
                entities.add(obj)

        return list(entities)

    def retrieve_entity_communities(self, entity_info, entities):
        community_ids = set()
        for entity in entities:
            if entity in entity_info:
                community_ids.update(entity_info[entity])
        return list(community_ids)

    async def generate_answer_from_summary(self, community_summary: str, query: str) -> str:
        prompt = (
            f"Given the community summary: {community_summary}, "
            f"how would you answer the following query? Query: {query}"
        )
        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(role="user", content="I need an answer based on the above information."),
        ]
        response = await self.llm.achat(messages)
        cleaned_response = re.sub(r"^assistant:\s*", "", str(response)).strip()
        return cleaned_response

    async def aggregate_answers(self, community_answers: List[str]) -> str:
        prompt = "Combine the following intermediate answers into a final, concise response."
        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(role="user", content=f"Intermediate answers: {community_answers}"),
        ]
        final_response = await self.llm.achat(messages)
        cleaned_final_response = re.sub(r"^assistant:\s*", "", str(final_response)).strip()
        return cleaned_final_response