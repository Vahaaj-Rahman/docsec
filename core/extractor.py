import fitz
import pdfplumber
import io
import re

def extract_text(pdf_bytes: bytes, filename: str) -> dict:
    result = {
        'filename': filename,
        'text': '',
        'page_count': 0,
        'char_count': 0,
        'word_count': 0,
        'method': '',
        'error': None
    }
    
    try:
        doc = fitz.open(stream=pdf_bytes, filetype='pdf')
        result['page_count'] = len(doc)
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        
        full_text = '\n\n'.join(text_parts)
        
        if len(full_text.strip()) > 100:
            result['method'] = 'pymupdf'
            cleaned_text = _clean_text(full_text)
            result['text'] = cleaned_text
            result['char_count'] = len(cleaned_text)
            result['word_count'] = len(cleaned_text.split())
            return result
        else:
            raise ValueError("PyMuPDF extracted too few characters.")
            
    except Exception as e_fitz:
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                result['page_count'] = len(pdf.pages)
                text_parts = []
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text_parts.append(extracted)
                
                full_text = '\n\n'.join(text_parts)
                result['method'] = 'pdfplumber'
                cleaned_text = _clean_text(full_text)
                result['text'] = cleaned_text
                result['char_count'] = len(cleaned_text)
                result['word_count'] = len(cleaned_text.split())
                return result
        except Exception as e_plumber:
            result['error'] = f"fitz error: {str(e_fitz)} | pdfplumber error: {str(e_plumber)}"
            return result

def extract_metadata(pdf_bytes: bytes) -> dict:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype='pdf')
        meta = doc.metadata
        return {
            'author': meta.get('author', ''),
            'title': meta.get('title', ''),
            'subject': meta.get('subject', ''),
            'creator': meta.get('creator', ''),
            'creationDate': meta.get('creationDate', ''),
            'modDate': meta.get('modDate', ''),
            'keywords': meta.get('keywords', '')
        }
    except Exception:
        return {
            'author': '',
            'title': '',
            'subject': '',
            'creator': '',
            'creationDate': '',
            'modDate': '',
            'keywords': ''
        }

def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()
