GROUNDING_SYSTEM_PROMPT = """You answer questions using only the supplied retrieved context.
Do not invent facts, sources, or citations that are not supported by the context.
If the context is insufficient, say that clearly and explain what is uncertain.
Distinguish supported facts from uncertainty.
Cite the supplied chunk IDs that support your answer by returning them in cited_chunk_ids.
Return only JSON matching the requested response schema.
"""


def build_grounding_prompt(query: str, context: str) -> str:
    return (
        f"{GROUNDING_SYSTEM_PROMPT}\n\n"
        "Retrieved context:\n"
        f"{context}\n\n"
        f"User query: {query}"
    )
