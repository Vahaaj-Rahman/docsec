from sklearn.feature_extraction.text import TfidfVectorizer

def run_tfidf(texts_dict: dict) -> dict:
    if len(texts_dict) < 1:
        return {}
        
    filenames = list(texts_dict.keys())
    corpus = list(texts_dict.values())
    
    # Handle case where all documents are empty
    if all(not text.strip() for text in corpus):
        return {fn: [] for fn in filenames}
        
    vectorizer = TfidfVectorizer(
        max_features=100,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z0-9\s]{1,}\b'
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(corpus)
        feature_names = vectorizer.get_feature_names_out()
    except ValueError:
        # Happens if corpus only contains stop words or is empty
        return {fn: [] for fn in filenames}
        
    results = {}
    for i, filename in enumerate(filenames):
        doc_vector = tfidf_matrix[i].tocoo()
        tuples = zip(doc_vector.col, doc_vector.data)
        sorted_items = sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)[:15]
        
        doc_terms = []
        for feature_index, score in sorted_items:
            doc_terms.append({
                'term': feature_names[feature_index],
                'score': float(score)
            })
            
        results[filename] = doc_terms
        
    return results

def get_document_topic(tfidf_results: dict, filename: str) -> str:
    top_terms = tfidf_results.get(filename, [])[:5]
    if not top_terms:
        return 'General Document'
        
    top_words = [t['term'].lower() for t in top_terms]
    
    identity_signals = {'aadhaar', 'pan', 'passport', 'dob', 'date of birth', 'kyc'}
    financial_signals = {'salary', 'ctc', 'bank', 'ifsc', 'invoice', 'payment', 'gst'}
    credential_signals = {'password', 'key', 'token', 'secret', 'credential', 'auth'}
    medical_signals = {'diagnosis', 'medical', 'patient', 'prescription', 'clinical'}
    legal_signals = {'agreement', 'contract', 'clause', 'terms', 'liability', 'parties'}
    
    scores = {
        'Identity / KYC Document': sum(1 for w in top_words if any(s in w for s in identity_signals)),
        'Financial Document': sum(1 for w in top_words if any(s in w for s in financial_signals)),
        'Credential / Security Document': sum(1 for w in top_words if any(s in w for s in credential_signals)),
        'Medical Record': sum(1 for w in top_words if any(s in w for s in medical_signals)),
        'Legal / Contract Document': sum(1 for w in top_words if any(s in w for s in legal_signals))
    }
    
    max_score = max(scores.values())
    if max_score == 0:
        return 'General Document'
        
    # Get the key with the max score
    best_match = max(scores, key=scores.get)
    return best_match
