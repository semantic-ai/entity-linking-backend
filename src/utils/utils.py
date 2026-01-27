from qdrant_client.models import ScoredPoint

def format_docs(docs: list[ScoredPoint]) -> str:
    """Format a list of documents."""
    return "\n".join(_format_doc(doc) for doc in docs)


def _format_doc(doc: ScoredPoint) -> str:
    """Format a single document, with special formatting based on doc type (sparql, schema)."""
    if not doc.payload:
        return ""
    doc_meta: dict[str, str] = doc.payload.get("metadata", {})
    if doc_meta.get("answer"):
        doc_lang = ""
        doc_type = str(doc_meta.get("doc_type", "")).lower()
        if "query" in doc_type:
            doc_lang = f"sparql\n#+ endpoint: {doc_meta.get('endpoint_url', 'undefined')}"
        elif "schema" in doc_type:
            doc_lang = "shex"
        return f"{doc.payload['page_content']}:\n\n```{doc_lang}\n{doc_meta.get('answer')}\n```"
    # Generic formatting:
    meta = "".join(f" {k}={v!r}" for k, v in doc_meta.items())
    if meta:
        meta = f" {meta}"
    return f"{meta}\n{doc.payload['page_content']}\n"
