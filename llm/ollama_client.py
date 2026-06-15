import requests
import json


def check_ollama_available() -> bool:
    try:
        r = requests.get('http://localhost:11434/api/tags', timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def list_available_models() -> list:
    try:
        r = requests.get('http://localhost:11434/api/tags', timeout=3)
        if r.status_code == 200:
            return [m['name'] for m in r.json().get('models', [])]
        return []
    except Exception:
        return []


def _build_context_block(keyword_hits: list, max_hits: int = 15) -> str:
    lines = []
    top = sorted(keyword_hits, key=lambda x: x['count'], reverse=True)[:max_hits]
    for h in top:
        sample = h['contexts'][0] if h['contexts'] else '(no context)'
        lines.append(f"  - \"{h['keyword']}\" ({h['count']}x): ...{sample}...")
    return '\n'.join(lines) if lines else 'None'


def analyze_document(
    text_excerpt: str,
    keyword_hits: list,
    tfidf_terms: list,
    model: str = 'llama3.2'
) -> dict:

    context_block = _build_context_block(keyword_hits)

    tfidf_summary = ', '.join([f"{t['term']} ({t['score']:.2f})" for t in tfidf_terms[:6]]) or 'None'

    prompt = f"""You are a strict document security analyst. Your job is to assess documents before they are shared externally.

Below are keyword matches found in the document. Use these specific excerpts to make your assessment — do NOT read or summarize the full document text blindly. Focus on what the keyword contexts reveal.

Keyword matches found in document:
{context_block}

Top topic signals from the document (TF-IDF):
{tfidf_summary}

First 600 characters of the document for context:
{text_excerpt[:600]}

Based ONLY on the above, respond with this exact JSON structure:
{{
  "summary": "2-3 sentences describing what this document is about",
  "document_type": "one of: KYC Form, Financial Record, Credential Document, Medical Record, Legal Contract, Policy Document, Offer Letter, Academic Report, General Document",
  "primary_risk": "1-2 sentences on the specific security risk of sharing this externally",
  "risk_level": "LOW or MEDIUM or HIGH",
  "sensitive_data_types": ["list", "of", "specific", "data", "types", "found"],
  "recommendation": "one specific action to take before sharing"
}}"""

    fallback = {
        "summary": "Ollama analysis failed or timed out.",
        "document_type": "Unknown",
        "primary_risk": "Could not assess. Check that Ollama is running and the model is pulled.",
        "risk_level": "UNKNOWN",
        "sensitive_data_types": [],
        "recommendation": "Run 'ollama serve' then 'ollama pull llama3.2' and retry."
    }

    try:
        r = requests.post(
            'http://localhost:11434/api/generate',
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.1, "num_predict": 500}
            },
            timeout=120
        )
        if r.status_code == 200:
            raw = r.json().get('response', '').strip()
            raw = raw.removeprefix('```json').removeprefix('```').removesuffix('```').strip()
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return fallback
        return fallback
    except Exception:
        return fallback
