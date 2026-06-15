import sqlite3
import os

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
    
    cursor.execute("SELECT COUNT(*) FROM keywords")
    if cursor.fetchone()[0] == 0:
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
        
    conn.commit()
    conn.close()

def get_all_keywords():
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT id, keyword, category, weight, created_at FROM keywords ORDER BY keyword ASC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

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

def search_keywords(query):
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    search_pattern = f"%{query}%"
    cursor.execute('''
        SELECT id, keyword, category, weight, created_at 
        FROM keywords 
        WHERE keyword LIKE ? 
        ORDER BY keyword ASC
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
