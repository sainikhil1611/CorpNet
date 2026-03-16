"""Prompts for community summarisation."""

COMMUNITY_SUMMARY_SYSTEM = (
    "You are a summarisation expert. "
    "Given a set of entities and their relationships that form a community, "
    "produce a concise thematic summary."
)

COMMUNITY_SUMMARY_PROMPT = """\
The following entities and relationships form a community in a knowledge graph.
Summarise the main theme(s) and key information of this community in 2-4 sentences.

Entities:
{entities_text}

Relationships:
{relations_text}

Summary:
"""
