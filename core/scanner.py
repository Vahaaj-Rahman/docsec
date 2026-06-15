import re
from collections import defaultdict

def build_pattern(keyword: str) -> re.Pattern:
    stripped = keyword.strip()
    escaped = re.escape(stripped)
    flexible = r'\s+'.join(escaped.split(r'\ '))
    return re.compile(r'\b' + flexible + r'\b', re.IGNORECASE)

def scan_document(text: str, keywords_list: list) -> list:
    results = []
    if not text:
        return results

    text_length = len(text)
    
    for kw_dict in keywords_list:
        keyword = kw_dict['keyword']
        category = kw_dict['category']
        weight = kw_dict['weight']
        
        pattern = build_pattern(keyword)
        matches = list(re.finditer(pattern, text))
        count = len(matches)
        
        if count > 0:
            contexts = []
            for i, match in enumerate(matches):
                if i >= 5:
                    break
                
                start_idx = max(0, match.start() - 60)
                end_idx = min(text_length, match.end() + 60)
                
                snippet = text[start_idx:end_idx]
                
                # Highlight the matched text
                original_match = match.group(0)
                snippet = snippet.replace(original_match, f">>>{original_match}<<<")
                
                # Clean snippet
                snippet = snippet.strip()
                snippet = snippet.replace('\n', ' ')
                snippet = re.sub(r'\s+', ' ', snippet)
                
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
    all_results = {}
    for filename, text in docs.items():
        all_results[filename] = scan_document(text, keywords_list)
    return all_results

def get_keyword_frequency_table(scan_results: dict) -> dict:
    pdfs = list(scan_results.keys())
    
    # Collect all unique keywords that appear at least once
    keyword_totals = defaultdict(int)
    for filename, results_list in scan_results.items():
        for res in results_list:
            keyword_totals[res['keyword']] += res['count']
            
    if not keyword_totals:
        return {'pdfs': [], 'keywords': [], 'matrix': []}
        
    sorted_keywords = sorted(keyword_totals.items(), key=lambda x: x[1], reverse=True)[:20]
    top_keywords = [k[0] for k in sorted_keywords]
    
    matrix = []
    for pdf in pdfs:
        row = []
        doc_results = {r['keyword']: r['count'] for r in scan_results[pdf]}
        for kw in top_keywords:
            row.append(doc_results.get(kw, 0))
        matrix.append(row)
        
    return {
        'pdfs': pdfs,
        'keywords': top_keywords,
        'matrix': matrix
    }
