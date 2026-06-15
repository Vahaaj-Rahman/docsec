import re
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------------------------------------------------------
# Heading / Title patterns for direct document type detection
# ---------------------------------------------------------------------------

HEADING_RULES = [
    # Pattern                                             Label
    (r'\binternship\s+report\b',                         'Internship Report'),
    (r'\bproject\s+report\b',                            'Project Report'),
    (r'\btraining\s+report\b',                           'Training Report'),
    (r'\bcurriculum\s+vitae\b|\bresume\b',               'Resume / CV'),
    (r'\boffer\s+letter\b',                              'Offer Letter'),
    (r'\bappointment\s+letter\b',                        'Appointment Letter'),
    (r'\binvoice\b|\btax\s+invoice\b',                   'Invoice / Receipt'),
    (r'\breceipt\b',                                     'Receipt'),
    (r'\bpurchase\s+order\b',                            'Purchase Order'),
    (r'\bnon[\s\-]*disclosure\s+agreement\b|\bnda\b',    'NDA / Legal Contract'),
    (r'\bservice\s+agreement\b|\bcontract\b',            'Legal Contract'),
    (r'\bmedical\s+report\b|\bpatient\s+record\b',       'Medical Record'),
    (r'\bprescription\b',                                'Medical Prescription'),
    (r'\bpolicy\b',                                      'Policy Document'),
    (r'\bbalance\s+sheet\b|\bincome\s+statement\b',      'Financial Statement'),
    (r'\bpay\s+slip\b|\bsalary\s+slip\b',               'Payslip'),
    (r'\bkyc\b|\bknow\s+your\s+customer\b',             'KYC Document'),
    (r'\bpassport\b',                                    'Passport / ID Document'),
    (r'\baadhar\b|\baadhaar\b',                         'Aadhaar Card'),
    (r'\bread\s*me\b',                                   'Technical README'),
    (r'\bapi\s+documentation\b|\bapi\s+reference\b',    'API Documentation'),
    (r'\bthesis\b|\bdissertation\b',                     'Academic Thesis'),
    (r'\bresearch\s+paper\b|\bjournal\s+article\b',      'Research Paper'),
    (r'\bmeeting\s+minutes\b|\bminutes\s+of\s+meeting\b','Meeting Minutes'),
    (r'\bcertificate\b',                                 'Certificate'),
]

# TF-IDF based signal dictionaries for fallback classification
CATEGORY_SIGNALS = {
    'Identity / KYC Document':          {'aadhaar', 'pan', 'passport', 'dob', 'date birth', 'kyc', 'national id', 'voter'},
    'Financial Document':               {'salary', 'ctc', 'bank', 'ifsc', 'invoice', 'payment', 'gst', 'account', 'debit', 'credit', 'tax', 'finance', 'revenue', 'balance'},
    'Credential / Security Document':   {'password', 'key', 'token', 'secret', 'credential', 'auth', 'api', 'certificate', 'ssl', 'ssh', 'access'},
    'Medical Record':                   {'diagnosis', 'medical', 'patient', 'prescription', 'clinical', 'health', 'doctor', 'hospital', 'medicine'},
    'Legal / Contract Document':        {'agreement', 'contract', 'clause', 'terms', 'liability', 'parties', 'hereinafter', 'jurisdiction', 'arbitration'},
    'Academic / Report Document':       {'report', 'project', 'internship', 'thesis', 'abstract', 'conclusion', 'chapter', 'student', 'university', 'college', 'research'},
    'HR / Employment Document':         {'salary', 'designation', 'joining', 'offer', 'employee', 'employer', 'ctc', 'appraisal', 'probation', 'resignation'},
    'Technical Document':               {'api', 'endpoint', 'server', 'database', 'deployment', 'repository', 'function', 'algorithm', 'software', 'system'},
}


def run_tfidf(texts_dict: dict) -> dict:
    if not texts_dict:
        return {}
    
    filenames = list(texts_dict.keys())
    corpus    = list(texts_dict.values())
    
    if all(not text.strip() for text in corpus):
        return {fn: [] for fn in filenames}
    
    vectorizer = TfidfVectorizer(
        max_features=150,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z]{1,}\b'   # standard, clean token pattern
    )
    
    try:
        tfidf_matrix  = vectorizer.fit_transform(corpus)
        feature_names = vectorizer.get_feature_names_out()
    except ValueError:
        return {fn: [] for fn in filenames}
    
    results = {}
    for i, filename in enumerate(filenames):
        doc_vec = tfidf_matrix[i].tocoo()
        sorted_items = sorted(zip(doc_vec.col, doc_vec.data), key=lambda x: x[1], reverse=True)[:15]
        
        results[filename] = [
            {'term': str(feature_names[col]), 'score': float(score)}
            for col, score in sorted_items
        ]
    
    return results


def get_document_topic(tfidf_results: dict, filename: str, full_text: str = "") -> str:
    """
    Two-stage classification:
    Stage 1 — Scan first 1200 chars of document for explicit heading keywords (most reliable)
    Stage 2 — Fall back to TF-IDF term scoring across expanded signal sets
    """
    # Stage 1: Header-based detection (first ~1200 characters)
    header_text = full_text[:1200].lower() if full_text else ""
    
    for pattern, label in HEADING_RULES:
        if re.search(pattern, header_text, re.IGNORECASE):
            return label
    
    # Stage 2: TF-IDF term scoring
    top_terms = tfidf_results.get(filename, [])[:12]
    if not top_terms:
        return 'General Document'
    
    top_words = [t['term'].lower() for t in top_terms]
    
    scores = {}
    for category, signals in CATEGORY_SIGNALS.items():
        score = sum(1 for w in top_words if any(s in w for s in signals))
        scores[category] = score
    
    max_score = max(scores.values())
    if max_score == 0:
        return 'General Document'
    
    return max(scores, key=scores.get)


def get_risk_score(scan_results: list) -> tuple[float, str]:
    """Calculate an overall weighted risk score for a document."""
    if not scan_results:
        return 0.0, 'LOW'
    
    total_weighted = sum(r['weight'] * r['count'] for r in scan_results)
    
    if total_weighted >= 30:
        return total_weighted, 'HIGH'
    elif total_weighted >= 10:
        return total_weighted, 'MEDIUM'
    else:
        return total_weighted, 'LOW'
