import regex as re
from collections import defaultdict


def build_exact_pattern(keyword: str) -> re.Pattern:
    stripped = keyword.strip()
    words = stripped.split()
    escaped = [re.escape(w) for w in words]
    flexible = r'\s+'.join(escaped)
    return re.compile(r'(?i)\b' + flexible + r's?\b')


def build_fuzzy_pattern(keyword: str) -> re.Pattern:
    stripped = keyword.strip()
    words = stripped.split()
    escaped = [re.escape(w) for w in words]
    flexible = r'\s+'.join(escaped)
    return re.compile(r'(?i)\b(?:' + flexible + r's?){e<=1}\b')


PII_PATTERNS = {
    'Email Address':       (r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,7}\b',                'personal', 4.0),
    'Phone Number':        (r'\b(?:\+?[\d]{1,3}[\s\-.])?(?:\(?\d{3}\)?[\s\-.])\d{3}[\s\-\.]\d{4}\b', 'personal', 3.5),
    'Indian Aadhaar':      (r'\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b',                                        'identity', 5.0),
    'Indian PAN':          (r'\b[A-Z]{5}[0-9]{4}[A-Z]\b',                                             'identity', 5.0),
    'Credit/Debit Card':   (r'\b(?:4\d{12}(?:\d{3})?|5[1-5]\d{14}|3[47]\d{13})\b',                  'financial', 4.5),
    'SSN / National ID':   (r'\b\d{3}-\d{2}-\d{4}\b',                                                 'identity', 4.5),
    'IP Address':          (r'\b(?:\d{1,3}\.){3}\d{1,3}\b',                                           'technical', 2.5),
    'URL / Web Link':      (r'https?://[^\s"\'<>]+',                                                   'technical', 1.5),
}

NER_PATTERNS = {
    'Organisation Name': (
        r'\b(?:Ltd|Limited|Pvt|Inc|Corp|LLC|LLP|Technologies|Solutions|Services|Systems|Enterprises|Consulting|Industries)\b',
        'general', 1.0
    ),
    'Designation / Role': (
        r'\b(?:CEO|CTO|CFO|Director|Manager|Engineer|Analyst|Consultant|Developer|Intern|President|VP)\b',
        'general', 1.0
    ),
}


def _extract_contexts(matches, text, limit=50):
    text_len = len(text)
    contexts = []
    for i, match in enumerate(matches):
        if i >= limit:
            break
        start = max(0, match.start() - 60)
        end   = min(text_len, match.end() + 60)
        snippet = text[start:end]
        snippet = snippet.replace(match.group(0), f">>>{match.group(0)}<<<")
        snippet = re.sub(r'\s+', ' ', snippet.strip())
        contexts.append(snippet)
    return contexts


def scan_pii(text: str) -> list:
    results = []
    if not text:
        return results
    for label, (pat, cat, weight) in PII_PATTERNS.items():
        try:
            matches = list(re.finditer(pat, text))
        except Exception:
            continue
        if not matches:
            continue
        results.append({
            'keyword': f'[Auto PII] {label}',
            'category': cat,
            'weight': weight,
            'count': len(matches),
            'contexts': _extract_contexts(matches, text),
            'is_pii': True
        })
    return results


def scan_ner(text: str) -> list:
    results = []
    if not text:
        return results
    for label, (pat, cat, weight) in NER_PATTERNS.items():
        try:
            matches = list(re.finditer(pat, text, re.IGNORECASE))
        except Exception:
            continue
        if not matches:
            continue
        results.append({
            'keyword': f'[Auto NER] {label}',
            'category': cat,
            'weight': weight,
            'count': len(matches),
            'contexts': _extract_contexts(matches, text, limit=30),
            'is_ner': True
        })
    return results


def scan_document(text: str, keywords_list: list,
                  include_pii: bool = True,
                  include_ner: bool = True,
                  fuzzy_mode: bool = False) -> list:
    results = []

    if include_pii:
        results.extend(scan_pii(text))
    if include_ner:
        results.extend(scan_ner(text))

    if not text:
        return results

    text_length = len(text)

    for kw_dict in keywords_list:
        keyword  = kw_dict['keyword']
        category = kw_dict['category']
        weight   = kw_dict['weight']

        try:
            pattern = build_fuzzy_pattern(keyword) if fuzzy_mode else build_exact_pattern(keyword)
            matches = list(re.finditer(pattern, text))
        except Exception:
            continue

        unique_matches = {}
        for m in matches:
            if m.start() not in unique_matches:
                unique_matches[m.start()] = m
        matches = sorted(unique_matches.values(), key=lambda m: m.start())
        count = len(matches)

        if count > 0:
            results.append({
                'keyword':  keyword,
                'category': category,
                'weight':   weight,
                'count':    count,
                'contexts': _extract_contexts(matches, text),
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
