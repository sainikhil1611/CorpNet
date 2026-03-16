"""Prompts for entity and relation extraction from text chunks."""

ENTITY_RELATION_EXTRACTION_SYSTEM = (
    "You are a knowledge graph extraction expert. "
    "Extract entities and their relationships from the given text. "
    "Be thorough but precise."
)

ENTITY_RELATION_EXTRACTION_PROMPT = """\
Given the following text, extract all entities and relationships.

Text:
{text}

Return a JSON object with the following structure:
{{
  "entities": [
    {{
      "name": "Entity Name",
      "type": "PERSON | ORGANIZATION | LOCATION | EVENT | CONCEPT | OTHER",
      "description": "Brief description of the entity based on the text"
    }}
  ],
  "relations": [
    {{
      "source": "Source Entity Name",
      "target": "Target Entity Name",
      "description": "Description of the relationship"
    }}
  ]
}}

Extract as many meaningful entities and relations as possible.
Only output the JSON, nothing else.
"""

ENTITY_MERGE_PROMPT = """\
The following entities may refer to the same real-world concept. \
Merge them into a single consolidated description.

Entities:
{entities_json}

Return a JSON object:
{{
  "name": "Canonical name",
  "description": "Merged description covering all mentions"
}}
"""
