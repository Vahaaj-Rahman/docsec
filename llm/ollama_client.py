import requests
import json

def check_ollama_available() -> bool:
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=3)
        return response.status_code == 200
    except Exception:
        return False

def list_available_models() -> list:
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=3)
        if response.status_code == 200:
            data = response.json()
            return [model['name'] for model in data.get('models', [])]
        return []
    except Exception:
        return []

def analyze_document(
    text_excerpt: str,
    keyword_hits: list,
    tfidf_terms: list,
    model: str = 'llama3.2:3b'
) -> dict:
    
    # Format keyword summary
    top_10_hits = sorted(keyword_hits, key=lambda x: x['count'], reverse=True)[:10]
    keyword_summary = ", ".join([f"{k['keyword']} ({k['count']} times)" for k in top_10_hits])
    if not keyword_summary:
        keyword_summary = "None found"
        
    # Format tfidf summary
    top_5_tfidf = tfidf_terms[:5]
    tfidf_summary = ", ".join([f"{t['term']} ({t['score']:.3f})" for t in top_5_tfidf])
    if not tfidf_summary:
        tfidf_summary = "None found"

    prompt = f"""You are a document security analyst. Analyze this document and respond ONLY with a valid JSON object. No explanation, no markdown, no preamble.

Document excerpt:
{text_excerpt}

Keywords found: {keyword_summary}
Document topic signals (TF-IDF): {tfidf_summary}

Respond with exactly this JSON structure:
{{
  "summary": "2-3 sentence description of what this document is about",
  "document_type": "one of: KYC Form, Financial Record, Credential Document, Medical Record, Legal Contract, Policy Document, Offer Letter, General Document",
  "primary_risk": "1-2 sentences on the main security risk of sharing this externally",
  "risk_level": "LOW or MEDIUM or HIGH",
  "sensitive_data_types": ["list", "of", "data", "types", "found"],
  "recommendation": "one specific action to take before sharing this document"
}}
"""

    fallback = {
        "summary": "Ollama not available or analysis failed.",
        "document_type": "Unknown",
        "primary_risk": "Unable to analyze — run 'ollama serve' and ensure model is pulled.",
        "risk_level": "UNKNOWN",
        "sensitive_data_types": [],
        "recommendation": "Start Ollama: run 'ollama serve' then 'ollama pull llama3.2:3b'"
    }

    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 400}
            },
            timeout=60
        )
        if response.status_code == 200:
            result_text = response.json().get('response', '')
            try:
                # Basic cleanup in case model returned markdown fences
                cleaned_text = result_text.strip()
                if cleaned_text.startswith('```json'):
                    cleaned_text = cleaned_text[7:]
                elif cleaned_text.startswith('```'):
                    cleaned_text = cleaned_text[3:]
                if cleaned_text.endswith('```'):
                    cleaned_text = cleaned_text[:-3]
                
                return json.loads(cleaned_text.strip())
            except json.JSONDecodeError:
                return fallback
        return fallback
    except Exception:
        return fallback
