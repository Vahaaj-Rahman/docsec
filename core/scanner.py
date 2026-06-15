import regex as re
from collections import defaultdict


def build_fuzzy_pattern(keyword: str) -> re.Pattern:
    """
    Build a regex pattern that:
    - Is case-insensitive
    - Allows flexible whitespace between words (handles extraction artifacts)
    - Allows up to 1 fuzzy error (substitution/insertion/deletion) for typo tolerance
    - Handles optional plural 's' suffix
    """
    stripped = keyword.strip()
    words = stripped.split()
    # Escape each word individually, then join with flexible whitespace
    escaped_words = [re.escape(w) for w in words]
    flexible = r'\s+'.join(escaped_words)
    # Fuzzy: up to 1 error, allow optional trailing s for plurals
    pattern_str = r'(?i)\b(?:' + flexible + r's?){e<=1}\b'
    return re.compile(pattern_str)


def build_exact_pattern(keyword: str) -> re.Pattern:
    """
    Build a regex pattern that:
    - Is case-insensitive
    - Allows flexible whitespace between words
    - Handles optional plural 's' suffix
    - Does NOT allow fuzzy errors
    """
    stripped = keyword.strip()
    words = stripped.split()
    escaped_words = [re.escape(w) for w in words]
    flexible = r'\s+'.join(escaped_words)
    pattern_str = r'(?i)\b(?:' + flexible + r's?)\b'
    return re.compile(pattern_str)


# ---------------------------------------------------------------------------
# PII Auto-Detection
# ---------------------------------------------------------------------------

PII_PATTERNS = {
    'Email Address': (r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,7}\b', 'personal', 4.0),
    'Phone Number':  (r'\b(?:\+?[\d]{1,3}[\s\-.])?(?:\(?\d{3}\)?[\s\-.])\d{3}[\s\-\.]\d{4}\b', 'personal', 3.5),
    'Indian Aadhaar': (r'\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b', 'identity', 5.0),
    'Indian PAN':    (r'\b[A-Z]{5}[0-9]{4}[A-Z]\b', 'identity', 5.0),
    'Credit / Debit Card': (r'\b(?:\d[ \-]?){13,16}\b', 'financial', 4.5),
    'SSN / National ID': (r'\b\d{3}-\d{2}-\d{4}\b', 'identity', 4.5),
    'Date of Birth': (r'\b(?:\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})\b', 'personal', 3.0),
    'IP Address':    (r'\b(?:\d{1,3}\.){3}\d{1,3}\b', 'technical', 2.5),
    'URL / Web Link': (r'https?://[^\s"\'<>]+', 'technical', 1.5),
}

def scan_pii(text: str) -> list:
    results = []
    if not text:
        return results
    text_len = len(text)
    
    for label, (pat, cat, weight) in PII_PATTERNS.items():
        try:
            matches = list(re.finditer(pat, text))
        except Exception:
            continue
        if not matches:
            continue
        
        contexts = []
        for i, match in enumerate(matches):
            if i >= 50:
                break
            start = max(0, match.start() - 60)
            end   = min(text_len, match.end() + 60)
            snippet = text[start:end]
            original = match.group(0)
            snippet = snippet.replace(original, f">>>{original}<<<")
            snippet = re.sub(r'\s+', ' ', snippet.strip())
            contexts.append(snippet)
        
        results.append({
            'keyword': f'[Auto PII] {label}',
            'category': cat,
            'weight': weight,
            'count': len(matches),
            'contexts': contexts,
            'is_pii': True
        })
    
    return results


NER_PATTERNS = {
    'Organisation Name': (
        r'\b(?:Ltd|Limited|Pvt|Inc|Corp|LLC|LLP|Technologies|Solutions|Services|Systems|Enterprises|Consulting|Industries)\b',
        'general', 1.0
    ),
    'Designation / Role': (
        r'\b(?:CEO|CTO|CFO|Director|Manager|Engineer|Analyst|Consultant|Developer|Intern|President|VP|Head of)\b',
        'general', 1.0
    ),
}

def scan_ner(text: str) -> list:
    results = []
    if not text:
        return results
    text_len = len(text)
    
    for label, (pat, cat, weight) in NER_PATTERNS.items():
        try:
            matches = list(re.finditer(pat, text, re.IGNORECASE))
        except Exception:
            continue
        if not matches:
            continue
        
        contexts = []
        for i, match in enumerate(matches):
            if i >= 30:
                break
            start = max(0, match.start() - 60)
            end   = min(text_len, match.end() + 60)
            snippet = text[start:end]
            original = match.group(0)
            snippet = snippet.replace(original, f">>>{original}<<<")
            snippet = re.sub(r'\s+', ' ', snippet.strip())
            contexts.append(snippet)
        
        results.append({
            'keyword': f'[Auto NER] {label}',
            'category': cat,
            'weight': weight,
            'count': len(matches),
            'contexts': contexts,
            'is_ner': True
        })
    
    return results



def scan_document(text: str, keywords_list: list, include_pii: bool = True, include_ner: bool = True, fuzzy_mode: bool = False) -> list:
    results = []
    
    # Auto-detect PII and NER first
    if include_pii:
        results.extend(scan_pii(text))
    if include_ner:
        results.extend(scan_ner(text))
    
    if not text:
        return results

    text_length = len(text)
    
    # User-defined keyword scanning
    for kw_dict in keywords_list:
        keyword  = kw_dict['keyword']
        category = kw_dict['category']
        weight   = kw_dict['weight']
        
        try:
            if fuzzy_mode:
                pattern = build_fuzzy_pattern(keyword)
            else:
                pattern = build_exact_pattern(keyword)
            matches = list(re.finditer(pattern, text))
        except Exception:
            continue
        
        # Deduplicate by start position (fuzzy can overlap)
        unique_matches = {}
        for m in matches:
            if m.start() not in unique_matches:
                unique_matches[m.start()] = m
        matches = sorted(unique_matches.values(), key=lambda m: m.start())
        count = len(matches)
        
        if count > 0:
            contexts = []
            for i, match in enumerate(matches):
                if i >= 50:
                    break
                start_idx = max(0, match.start() - 60)
                end_idx   = min(text_length, match.end() + 60)
                snippet   = text[start_idx:end_idx]
                original  = match.group(0)
                snippet   = snippet.replace(original, f">>>{original}<<<")
                snippet   = re.sub(r'\s+', ' ', snippet.replace('\n', ' ').strip())
                contexts.append(snippet)
            
            results.append({
                'keyword': keyword,
                'category': category,
                'weight': weight,
                'count': count,
                'contexts': contexts
            })
    
    results.sort(key=lambda x: x['count'], reverse=True)
    return results


def scan_multiple(docs: dict, keywords_list: list) -> dict:
    return {filename: scan_document(text, keywords_list) for filename, text in docs.items()}


def get_keyword_frequency_table(scan_results: dict) -> dict:
    pdfs = list(scan_results.keys())
    
    keyword_totals = defaultdict(int)
    for results_list in scan_results.values():
        for res in results_list:
            keyword_totals[res['keyword']] += res['count']
    
    if not keyword_totals:
        return {'pdfs': [], 'keywords': [], 'matrix': []}
    
    top_keywords = [k for k, _ in sorted(keyword_totals.items(), key=lambda x: x[1], reverse=True)[:20]]
    
    matrix = []
    for pdf in pdfs:
        doc_results = {r['keyword']: r['count'] for r in scan_results[pdf]}
        matrix.append([doc_results.get(kw, 0) for kw in top_keywords])
    
    return {'pdfs': pdfs, 'keywords': top_keywords, 'matrix': matrix}
