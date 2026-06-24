import sqlite3
import os
import json

DB_PATH = os.path.join(os.path.dirname(__file__), 'keywords.db')

def get_connection():
    return sqlite3.connect(DB_PATH)

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS keywords (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        keyword TEXT UNIQUE NOT NULL COLLATE NOCASE,
        category TEXT DEFAULT 'general',
        weight REAL DEFAULT 1.0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Heading rules table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS heading_rules (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pattern TEXT UNIQUE NOT NULL,
        label TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Category signals table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS category_signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        category TEXT UNIQUE NOT NULL,
        words TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Config table to prevent re-seeding if user deletes all items
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS app_config (
        key TEXT PRIMARY KEY,
        value TEXT
    )
    ''')
    
    cursor.execute("SELECT value FROM app_config WHERE key = 'is_seeded'")
    is_seeded = cursor.fetchone()
    
    if not is_seeded:
        # Check if we are migrating an existing DB that already has data
        cursor.execute("SELECT COUNT(*) FROM keywords")
        has_data = cursor.fetchone()[0] > 0
        if has_data:
            cursor.execute("INSERT INTO app_config (key, value) VALUES ('is_seeded', '1')")
        else:
            # Seed keywords
            seed_data = [
                ('aadhaar', 'identity', 3.5),
                ('pan number', 'identity', 3.5),
                ('passport', 'identity', 3.0),
                ('date of birth', 'personal', 2.5),
                ('mobile number', 'personal', 2.0),
                ('bank account', 'financial', 3.0),
                ('ifsc', 'financial', 2.5),
                ('salary', 'financial', 2.5),
                ('ctc', 'financial', 2.5),
                ('gst number', 'financial', 2.0),
                ('password', 'credential', 3.5),
                ('api key', 'credential', 3.5),
                ('secret key', 'credential', 3.5),
                ('access token', 'credential', 3.0),
                ('confidential', 'classification', 2.0),
                ('internal only', 'classification', 2.0),
                ('not for distribution', 'classification', 2.5),
                ('restricted', 'classification', 2.0),
                ('medical', 'health', 2.0),
                ('diagnosis', 'health', 2.5),
                ('foundation model', 'technical', 1.5),
                ('large language model', 'technical', 1.5),
                ('training data', 'technical', 1.5)
            ]
            cursor.executemany('''
                INSERT INTO keywords (keyword, category, weight)
                VALUES (?, ?, ?)
            ''', seed_data)
        
            # Seed heading rules
            seed_rules = [
                (r'\binternship\s+report\b', 'Internship Report'),
                (r'\bproject\s+report\b', 'Project Report'),
                (r'\btraining\s+report\b', 'Training Report'),
                (r'\bcurriculum\s+vitae\b|\bresume\b', 'Resume / CV'),
                (r'\boffer\s+letter\b', 'Offer Letter'),
                (r'\bappointment\s+letter\b', 'Appointment Letter'),
                (r'\binvoice\b|\btax\s+invoice\b', 'Invoice / Receipt'),
                (r'\breceipt\b', 'Receipt'),
                (r'\bpurchase\s+order\b', 'Purchase Order'),
                (r'\bnon[\s\-]*disclosure\s+agreement\b|\bnda\b', 'NDA / Legal Contract'),
                (r'\bservice\s+agreement\b|\bcontract\b', 'Legal Contract'),
                (r'\bmedical\s+report\b|\bpatient\s+record\b', 'Medical Record'),
                (r'\bprescription\b', 'Medical Prescription'),
                (r'\bpolicy\b', 'Policy Document'),
                (r'\bbalance\s+sheet\b|\bincome\s+statement\b', 'Financial Statement'),
                (r'\bpay\s+slip\b|\bsalary\s+slip\b', 'Payslip'),
                (r'\bkyc\b|\bknow\s+your\s+customer\b', 'KYC Document'),
                (r'\bpassport\b', 'Passport / ID Document'),
                (r'\baadhar\b|\baadhaar\b', 'Aadhaar Card'),
                (r'\bread\s*me\b', 'Technical README'),
                (r'\bapi\s+documentation\b|\bapi\s+reference\b', 'API Documentation'),
                (r'\bthesis\b|\bdissertation\b', 'Academic Thesis'),
                (r'\bresearch\s+paper\b|\bjournal\s+article\b', 'Research Paper'),
                (r'\bmeeting\s+minutes\b|\bminutes\s+of\s+meeting\b', 'Meeting Minutes'),
                (r'\bcertificate\b', 'Certificate'),
            ]
            cursor.executemany('''
                INSERT INTO heading_rules (pattern, label)
                VALUES (?, ?)
            ''', seed_rules)
        
            # Seed category signals
            seed_signals = [
                ('Identity / KYC Document', json.dumps(['aadhaar', 'pan', 'passport', 'dob', 'date birth', 'kyc', 'national id', 'voter'])),
                ('Financial Document', json.dumps(['salary', 'ctc', 'bank', 'ifsc', 'invoice', 'payment', 'gst', 'account', 'debit', 'credit', 'tax', 'finance', 'revenue', 'balance'])),
                ('Credential / Security Document', json.dumps(['password', 'key', 'token', 'secret', 'credential', 'auth', 'api', 'certificate', 'ssl', 'ssh', 'access'])),
                ('Medical Record', json.dumps(['diagnosis', 'medical', 'patient', 'prescription', 'clinical', 'health', 'doctor', 'hospital', 'medicine'])),
                ('Legal / Contract Document', json.dumps(['agreement', 'contract', 'clause', 'terms', 'liability', 'parties', 'hereinafter', 'jurisdiction', 'arbitration'])),
                ('Academic / Report Document', json.dumps(['report', 'project', 'internship', 'thesis', 'abstract', 'conclusion', 'chapter', 'student', 'university', 'college', 'research'])),
                ('HR / Employment Document', json.dumps(['salary', 'designation', 'joining', 'offer', 'employee', 'employer', 'ctc', 'appraisal', 'probation', 'resignation'])),
                ('Technical Document', json.dumps(['api', 'endpoint', 'server', 'database', 'deployment', 'repository', 'function', 'algorithm', 'software', 'system'])),
            ]
            cursor.executemany('''
                INSERT INTO category_signals (category, words)
                VALUES (?, ?)
            ''', seed_signals)
            
            cursor.execute("INSERT INTO app_config (key, value) VALUES ('is_seeded', '1')")
    conn.commit()
    conn.close()

# ---------------------------------------------------------------------------
# Keywords CRUD
# ---------------------------------------------------------------------------

def get_all_keywords():
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT id, keyword, category, weight, created_at FROM keywords ORDER BY category, keyword ASC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_keywords_by_category():
    """Return keywords grouped by category."""
    all_kws = get_all_keywords()
    grouped = {}
    for kw in all_kws:
        cat = kw['category']
        if cat not in grouped:
            grouped[cat] = []
        grouped[cat].append(kw)
    return grouped

def add_keyword(keyword, category, weight):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO keywords (keyword, category, weight)
            VALUES (?, ?, ?)
        ''', (keyword, category, weight))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False
    except Exception:
        return False

def update_keyword(keyword_id, new_keyword, new_category, new_weight):
    """Update keyword text, category, and weight."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE keywords SET keyword = ?, category = ?, weight = ?
            WHERE id = ?
        ''', (new_keyword, new_category, new_weight, keyword_id))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False
    except Exception:
        return False

def delete_keyword(keyword_id):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM keywords WHERE id = ?", (keyword_id,))
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False

def delete_keywords_bulk(keyword_ids):
    """Delete multiple keywords at once."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        placeholders = ','.join('?' * len(keyword_ids))
        cursor.execute(f"DELETE FROM keywords WHERE id IN ({placeholders})", keyword_ids)
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False

def search_keywords(query):
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    search_pattern = f"%{query}%"
    cursor.execute('''
        SELECT id, keyword, category, weight, created_at 
        FROM keywords 
        WHERE keyword LIKE ? 
        ORDER BY category, keyword ASC
    ''', (search_pattern,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_keyword_by_id(keyword_id):
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT id, keyword, category, weight, created_at FROM keywords WHERE id = ?", (keyword_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def update_keyword_weight(keyword_id, new_weight):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE keywords SET weight = ? WHERE id = ?", (new_weight, keyword_id))
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False

# ---------------------------------------------------------------------------
# Heading Rules CRUD
# ---------------------------------------------------------------------------

def get_all_heading_rules():
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT id, pattern, label, created_at FROM heading_rules ORDER BY label ASC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def add_heading_rule(pattern, label):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO heading_rules (pattern, label) VALUES (?, ?)", (pattern, label))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False
    except Exception:
        return False

def update_heading_rule(rule_id, pattern, label):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE heading_rules SET pattern = ?, label = ? WHERE id = ?", (pattern, label, rule_id))
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False

def delete_heading_rule(rule_id):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM heading_rules WHERE id = ?", (rule_id,))
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False

# ---------------------------------------------------------------------------
# Category Signals CRUD
# ---------------------------------------------------------------------------

def get_all_category_signals():
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT id, category, words, created_at FROM category_signals ORDER BY category ASC")
    rows = cursor.fetchall()
    conn.close()
    result = []
    for row in rows:
        d = dict(row)
        d['words_list'] = json.loads(d['words'])
        result.append(d)
    return result

def add_category_signal(category, words_list):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO category_signals (category, words) VALUES (?, ?)", 
                       (category, json.dumps(words_list)))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False
    except Exception:
        return False

def update_category_signal(signal_id, category, words_list):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE category_signals SET category = ?, words = ? WHERE id = ?", 
                       (category, json.dumps(words_list), signal_id))
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False

def delete_category_signal(signal_id):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM category_signals WHERE id = ?", (signal_id,))
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False

def get_category_signals_dict():
    """Return signals as dict for TF-IDF engine use: {category: set_of_words}"""
    signals = get_all_category_signals()
    return {s['category']: set(s['words_list']) for s in signals}
