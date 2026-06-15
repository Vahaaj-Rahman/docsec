import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import io

from core.extractor import extract_text, extract_metadata
from core.scanner import scan_document, scan_multiple, get_keyword_frequency_table
from core.tfidf_engine import run_tfidf, get_document_topic
from db.db_manager import init_db, get_all_keywords, add_keyword, delete_keyword, search_keywords
from llm.ollama_client import check_ollama_available, list_available_models, analyze_document

# PAGE CONFIG
st.set_page_config(
    page_title="DocScan — PDF Security Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS INJECTION
st.markdown("""
<style>
/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stTextInput label { color: #8b949e !important; }

/* Buttons */
.stButton > button {
    background: #1f6feb; color: #ffffff;
    border: none; border-radius: 8px;
    padding: 0.45rem 1.4rem; font-weight: 500;
    font-size: 0.875rem; transition: background 0.2s;
    width: 100%;
}
.stButton > button:hover { background: #388bfd; border: none; }

/* Metric cards */
.metric-card {
    background: #161b22; border: 1px solid #21262d;
    border-radius: 12px; padding: 1rem 1.25rem;
    text-align: center;
}
.metric-card .value {
    font-size: 2rem; font-weight: 600; color: #58a6ff; 
    line-height: 1.2;
}
.metric-card .label {
    font-size: 0.75rem; color: #8b949e; 
    text-transform: uppercase; letter-spacing: 0.05em;
    margin-top: 0.25rem;
}

/* Context snippet styling */
.context-snippet {
    background: #161b22; border-left: 3px solid #1f6feb;
    border-radius: 0 6px 6px 0; padding: 0.5rem 0.75rem;
    margin: 0.35rem 0; font-size: 0.82rem;
    font-family: 'Courier New', monospace;
    color: #c9d1d9; line-height: 1.5;
    word-break: break-word;
}
.context-snippet .hit { 
    color: #ffa657; font-weight: 600;
    background: rgba(255,166,87,0.15);
    padding: 1px 3px; border-radius: 3px;
}

/* Category badges */
.badge {
    display: inline-block; padding: 2px 10px;
    border-radius: 20px; font-size: 0.72rem;
    font-weight: 500; letter-spacing: 0.02em;
}
.badge-identity    { background: #3d1515; color: #f97171; border: 1px solid #6b2020; }
.badge-financial   { background: #152415; color: #5acd5a; border: 1px solid #1f4a1f; }
.badge-credential  { background: #261535; color: #c084fc; border: 1px solid #4a2070; }
.badge-classification { background: #0d1f3c; color: #60a5fa; border: 1px solid #1a3a6b; }
.badge-personal    { background: #2e2010; color: #fbbf24; border: 1px solid #5a3d10; }
.badge-health      { background: #0d2e2e; color: #2dd4bf; border: 1px solid #0f5050; }
.badge-technical   { background: #1a1a2e; color: #818cf8; border: 1px solid #2d2d5a; }
.badge-general     { background: #1c1c1c; color: #9ca3af; border: 1px solid #333; }

/* Risk level indicators */
.risk-high   { color: #f97171; font-weight: 600; }
.risk-medium { color: #fbbf24; font-weight: 600; }
.risk-low    { color: #5acd5a; font-weight: 600; }
.risk-unknown { color: #8b949e; font-weight: 600; }

/* PDF result cards */
.pdf-header {
    background: #161b22; border: 1px solid #21262d;
    border-radius: 10px; padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
}

/* TF-IDF term pills */
.tfidf-pill {
    display: inline-block; background: #0d2030;
    border: 1px solid #1f4060; color: #58a6ff;
    border-radius: 20px; padding: 3px 12px;
    font-size: 0.78rem; margin: 3px;
}

/* Ollama status indicator */
.ollama-online  { color: #5acd5a; font-size: 0.8rem; }
.ollama-offline { color: #f97171; font-size: 0.8rem; }

/* Hide default Streamlit footer and menu */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# SESSION STATE INIT
init_db()

if 'scan_results' not in st.session_state:
    st.session_state.scan_results = {}
if 'extracted_texts' not in st.session_state:
    st.session_state.extracted_texts = {}
if 'tfidf_results' not in st.session_state:
    st.session_state.tfidf_results = {}
if 'llm_analyses' not in st.session_state:
    st.session_state.llm_analyses = {}
if 'uploaded_file_names' not in st.session_state:
    st.session_state.uploaded_file_names = []

# SIDEBAR LAYOUT
with st.sidebar:
    st.markdown("<h1>🔍 DocScan</h1>", unsafe_allow_html=True)
    st.markdown("### PDF Security Analyzer")
    st.markdown("---")

    st.markdown("### AI Backend")
    ollama_ok = check_ollama_available()
    if ollama_ok:
        st.markdown("<div class='ollama-online'>● Ollama online</div>", unsafe_allow_html=True)
        models = list_available_models()
        if not models:
            st.markdown("<small style='color:#f97171'>No models found. Run `ollama pull llama3.2:3b`</small>", unsafe_allow_html=True)
        else:
            default_index = models.index('llama3.2:3b') if 'llama3.2:3b' in models else 0
            st.selectbox("Model", models, index=default_index, key="ollama_model")
    else:
        st.markdown("<div class='ollama-offline'>● Ollama offline</div>", unsafe_allow_html=True)
        st.info("Start with: ollama serve")
        st.code("ollama pull llama3.2:3b")
        
    st.markdown("---")
    
    st.markdown("### Keyword Database")
    all_kws = get_all_keywords()
    st.markdown(f'<div class="metric-card"><div class="value">{len(all_kws)}</div><div class="label">Total Keywords</div></div>', unsafe_allow_html=True)
    
    search_query = st.text_input("🔍 Search keywords", key="kw_search")
    if search_query:
        display_keywords = search_keywords(search_query)
    else:
        display_keywords = all_kws
        
    for kw in display_keywords:
        col1, col2 = st.columns([0.85, 0.15])
        with col1:
            st.markdown(
                f"<span>{kw['keyword']}</span> "
                f"<span class='badge badge-{kw['category'].lower()}'>{kw['category']}</span> "
                f"<span style='color:#8b949e;font-size:0.75rem'>({kw['weight']})</span>",
                unsafe_allow_html=True
            )
        with col2:
            if st.button("×", key=f"del_{kw['id']}"):
                delete_keyword(kw['id'])
                st.rerun()
                
    if not display_keywords and search_query:
        st.write("No keywords match your search")
        
    with st.expander("+ Add New Keyword"):
        new_kw = st.text_input("Keyword or phrase", help="Multi-word phrases work: 'foundation model', 'bank account'")
        new_cat = st.selectbox("Category", ['general','identity','financial','credential','classification','personal','health','technical'])
        new_weight = st.slider("Risk weight", 0.5, 5.0, 1.0, 0.5, help="Higher weight = contributes more to risk score")
        if st.button("Add Keyword"):
            if new_kw.strip():
                success = add_keyword(new_kw.strip().lower(), new_cat, new_weight)
                if success:
                    st.success(f"Added: '{new_kw.strip()}'")
                    st.rerun()
                else:
                    st.error("Keyword already exists or error occurred")
            else:
                st.warning("Please enter a keyword")

# MAIN AREA TABS
tab1, tab2, tab3 = st.tabs(["📄 Scan Documents", "📊 Analysis", "🤖 AI Insights"])

# TAB 1: SCAN DOCUMENTS
with tab1:
    st.header("Upload & Scan PDFs")
    
    uploaded_files = st.file_uploader(
        "Drop PDFs here or click to browse",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more PDF files for keyword analysis"
    )
    
    if uploaded_files:
        current_filenames = [f.name for f in uploaded_files]
        if current_filenames != st.session_state.uploaded_file_names:
            st.session_state.scan_results = {}
            st.session_state.extracted_texts = {}
            st.session_state.tfidf_results = {}
            st.session_state.llm_analyses = {}
            st.session_state.uploaded_file_names = current_filenames
            
        st.markdown(f"**{len(uploaded_files)} file(s) ready to scan**")
        for f in uploaded_files:
            st.write(f"- {f.name} ({f.size/1024:.1f} KB)")
            
        col1, col2 = st.columns([2, 1])
        with col1:
            scan_btn = st.button("Scan All Documents")
        with col2:
            run_ai_analysis = False
            if ollama_ok:
                run_ai_analysis = st.checkbox("Run AI Analysis (requires Ollama)", value=False)
                
        if scan_btn:
            keywords = get_all_keywords()
            if not keywords:
                st.warning("No keywords found in database.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"Extracting: {file.name}...")
                    pdf_bytes = file.read()
                    extraction = extract_text(pdf_bytes, file.name)
                    
                    if extraction['error']:
                        st.error(f"Failed to extract {file.name}: {extraction['error']}")
                        continue
                        
                    status_text.text(f"Scanning keywords: {file.name}...")
                    keyword_results = scan_document(extraction['text'], keywords)
                    
                    st.session_state.extracted_texts[file.name] = extraction['text']
                    st.session_state.scan_results[file.name] = {
                        'keywords': keyword_results,
                        'page_count': extraction['page_count'],
                        'char_count': extraction['char_count'],
                        'word_count': extraction['word_count'],
                        'method': extraction['method'],
                        'error': extraction['error']
                    }
                    progress_bar.progress((i + 1) / len(uploaded_files))
                    
                status_text.text("Running TF-IDF analysis across corpus...")
                st.session_state.tfidf_results = run_tfidf(st.session_state.extracted_texts)
                
                if run_ai_analysis and check_ollama_available():
                    for filename in st.session_state.extracted_texts:
                        status_text.text(f"AI analyzing: {filename}...")
                        text_excerpt = st.session_state.extracted_texts[filename][:2000]
                        kw_hits = st.session_state.scan_results[filename]['keywords']
                        tfidf_terms = st.session_state.tfidf_results.get(filename, [])
                        model = st.session_state.get('ollama_model', 'llama3.2:3b')
                        
                        analysis = analyze_document(text_excerpt, kw_hits, tfidf_terms, model)
                        st.session_state.llm_analyses[filename] = analysis
                        
                progress_bar.progress(1.0)
                status_text.text("✓ Scan complete!")
                st.rerun()

    if st.session_state.scan_results:
        st.subheader(f"Results — {len(st.session_state.scan_results)} document(s) scanned")
        
        for filename, result in st.session_state.scan_results.items():
            with st.expander(f"📄 {filename}", expanded=True):
                cols = st.columns(4)
                kw_hits = result['keywords']
                total_hits = sum(k['count'] for k in kw_hits)
                unique_kw = len(kw_hits)
                
                cols[0].markdown(f'<div class="metric-card"><div class="value">{total_hits}</div><div class="label">Total Hits</div></div>', unsafe_allow_html=True)
                cols[1].markdown(f'<div class="metric-card"><div class="value">{unique_kw}</div><div class="label">Unique Keywords</div></div>', unsafe_allow_html=True)
                cols[2].markdown(f'<div class="metric-card"><div class="value">{result["page_count"]}</div><div class="label">Pages</div></div>', unsafe_allow_html=True)
                cols[3].markdown(f'<div class="metric-card"><div class="value">{result["word_count"]}</div><div class="label">Words</div></div>', unsafe_allow_html=True)
                
                if filename in st.session_state.tfidf_results:
                    st.markdown("**Document topic signals (TF-IDF):**")
                    tfidf_html = ' '.join([
                        f'<span class="tfidf-pill">{t["term"]} <small>({t["score"]:.2f})</small></span>'
                        for t in st.session_state.tfidf_results[filename][:8]
                    ])
                    st.markdown(tfidf_html, unsafe_allow_html=True)
                    
                    topic_label = get_document_topic(st.session_state.tfidf_results, filename)
                    st.markdown(f"**Detected document type:** {topic_label}")
                    
                if filename in st.session_state.llm_analyses:
                    analysis = st.session_state.llm_analyses[filename]
                    risk = analysis.get('risk_level', 'UNKNOWN')
                    risk_class = f"risk-{risk.lower()}"
                    
                    st.markdown("---")
                    st.markdown("**🤖 AI Security Analysis**")
                    
                    ai_col1, ai_col2 = st.columns([3, 1])
                    ai_col1.markdown(f"*{analysis.get('summary', '')}*")
                    ai_col2.markdown(
                      f'<div style="text-align:center"><div class="metric-card">'
                      f'<div class="{risk_class}" style="font-size:1.3rem">{risk}</div>'
                      f'<div class="label">Risk Level</div></div></div>',
                      unsafe_allow_html=True
                    )
                    
                    st.markdown(f"**Primary risk:** {analysis.get('primary_risk', '')}")
                    st.markdown(f"**Recommendation:** {analysis.get('recommendation', '')}")
                    
                    data_types = analysis.get('sensitive_data_types', [])
                    if data_types:
                        types_html = ' '.join([f'<span class="badge badge-general">{t}</span>' for t in data_types])
                        st.markdown("**Sensitive data types found:** " + types_html, unsafe_allow_html=True)
                        
                if kw_hits:
                    top_kw = sorted(kw_hits, key=lambda x: x['count'], reverse=True)[:12]
                    cat_colors = {
                        'identity': '#f97171', 'financial': '#5acd5a',
                        'credential': '#c084fc', 'classification': '#60a5fa',
                        'personal': '#fbbf24', 'health': '#2dd4bf',
                        'technical': '#818cf8', 'general': '#9ca3af'
                    }
                    bar_colors = [cat_colors.get(k['category'], '#9ca3af') for k in top_kw]
                    
                    fig = go.Figure(go.Bar(
                        x=[k['count'] for k in top_kw],
                        y=[k['keyword'] for k in top_kw],
                        orientation='h',
                        marker_color=bar_colors,
                        text=[str(k['count']) for k in top_kw],
                        textposition='outside',
                        hovertemplate='<b>%{y}</b><br>Count: %{x}<extra></extra>'
                    ))
                    fig.update_layout(
                        paper_bgcolor='#161b22', plot_bgcolor='#161b22',
                        font_color='#c9d1d9', height=max(250, len(top_kw) * 32),
                        margin=dict(l=20, r=60, t=20, b=20),
                        xaxis=dict(showgrid=False, color='#8b949e'),
                        yaxis=dict(showgrid=False, color='#c9d1d9', autorange='reversed'),
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                st.markdown("**Keyword matches with context:**")
                for kw_result in sorted(kw_hits, key=lambda x: x['count'], reverse=True):
                    cat = kw_result['category']
                    badge_html = f'<span class="badge badge-{cat}">{cat}</span>'
                    
                    st.markdown(
                      f"**{kw_result['keyword']}** {badge_html} — "
                      f"**{kw_result['count']}** occurrence(s) | weight: {kw_result['weight']}",
                      unsafe_allow_html=True
                    )
                    
                    for context in kw_result['contexts'][:3]:
                        display = context.replace('>>>', '<span class="hit">').replace('<<<', '</span>')
                        st.markdown(
                          f'<div class="context-snippet">{display}</div>',
                          unsafe_allow_html=True
                        )
                        
                    if len(kw_result['contexts']) > 3:
                        with st.expander("Select to view all occurrences"):
                            for context in kw_result['contexts'][3:]:
                                display = context.replace('>>>', '<span class="hit">').replace('<<<', '</span>')
                                st.markdown(
                                  f'<div class="context-snippet">{display}</div>',
                                  unsafe_allow_html=True
                                )
                    st.markdown("---")

# TAB 2: ANALYSIS
with tab2:
    if not st.session_state.scan_results:
        st.info("Upload and scan PDFs first")
    else:
        st.header("Cross-Document Analysis")
        
        st.subheader("Document Summary")
        summary_data = []
        for filename, result in st.session_state.scan_results.items():
            kw_hits = result['keywords']
            top_kw = sorted(kw_hits, key=lambda x: x['count'], reverse=True)[0]['keyword'] if kw_hits else 'None'
            
            summary_data.append({
                'Document': filename,
                'Pages': result['page_count'],
                'Unique Keywords': len(kw_hits),
                'Total Hits': sum(k['count'] for k in kw_hits),
                'Top Keyword': top_kw,
                'Extraction Method': result['method']
            })
            
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
        
        if len(st.session_state.scan_results) > 1:
            st.subheader("Keyword Frequency Heatmap")
            st.caption("Rows = documents, Columns = keywords. Color intensity = occurrence count.")
            
            formatted_results = {fn: res['keywords'] for fn, res in st.session_state.scan_results.items()}
            freq_table = get_keyword_frequency_table(formatted_results)
            
            if freq_table['keywords']:
                fig_heat = go.Figure(go.Heatmap(
                    z=freq_table['matrix'],
                    x=freq_table['keywords'],
                    y=freq_table['pdfs'],
                    colorscale=[[0, '#161b22'], [0.3, '#0d2030'], [0.6, '#1f6feb'], [1.0, '#58a6ff']],
                    showscale=True,
                    hovertemplate='Doc: %{y}<br>Keyword: %{x}<br>Count: %{z}<extra></extra>'
                ))
                fig_heat.update_layout(
                    paper_bgcolor='#0d1117', plot_bgcolor='#0d1117',
                    font_color='#c9d1d9',
                    height=max(300, len(freq_table['pdfs']) * 50 + 100),
                    margin=dict(l=20, r=20, t=20, b=120),
                    xaxis=dict(tickangle=-45, color='#8b949e', tickfont_size=11),
                    yaxis=dict(color='#c9d1d9')
                )
                st.plotly_chart(fig_heat, use_container_width=True)
                
        st.subheader("Keyword Hits by Category (all documents)")
        cat_totals = {}
        for res in st.session_state.scan_results.values():
            for kw in res['keywords']:
                cat_totals[kw['category']] = cat_totals.get(kw['category'], 0) + kw['count']
                
        if cat_totals:
            cat_colors = {
                'identity': '#f97171', 'financial': '#5acd5a',
                'credential': '#c084fc', 'classification': '#60a5fa',
                'personal': '#fbbf24', 'health': '#2dd4bf',
                'technical': '#818cf8', 'general': '#9ca3af'
            }
            cat_color_list = [cat_colors.get(c, '#9ca3af') for c in cat_totals.keys()]
            
            fig_donut = go.Figure(go.Pie(
                labels=list(cat_totals.keys()),
                values=list(cat_totals.values()),
                hole=0.55,
                marker_colors=cat_color_list,
                textinfo='label+percent',
                hovertemplate='%{label}: %{value} hits<extra></extra>'
            ))
            fig_donut.update_layout(
                paper_bgcolor='#0d1117', font_color='#c9d1d9',
                height=380, showlegend=True,
                legend=dict(bgcolor='#161b22', bordercolor='#21262d')
            )
            st.plotly_chart(fig_donut, use_container_width=True)

# TAB 3: AI INSIGHTS
with tab3:
    st.header("🤖 AI Security Insights")
    
    if not check_ollama_available():
        st.info("Ollama is not running. AI insights require Ollama installed locally.\n\nInstall: `curl -fsSL https://ollama.ai/install.sh | sh`\nThen run: `ollama serve`\nThen pull model: `ollama pull llama3.2:3b`\n\nAll analysis runs 100% on your machine — no data leaves your system.")
    elif not st.session_state.scan_results:
        st.write("Scan documents first, then run AI analysis from the Scan tab.")
    elif not st.session_state.llm_analyses:
        st.write("Re-scan with 'Run AI Analysis' checkbox enabled to see AI insights.")
    else:
        for filename, analysis in st.session_state.llm_analyses.items():
            with st.expander(f"🤖 {filename}", expanded=True):
                risk = analysis.get('risk_level', 'UNKNOWN')
                
                ai_c1, ai_c2 = st.columns([4, 1])
                ai_c1.subheader(f"📄 {filename}")
                ai_c2.markdown(
                    f'<div class="metric-card"><div class="risk-{risk.lower()}" '
                    f'style="font-size:1.5rem;font-weight:700">{risk}</div>'
                    f'<div class="label">Risk Level</div></div>',
                    unsafe_allow_html=True
                )
                
                st.markdown(f"**Document type:** {analysis.get('document_type', 'Unknown')}")
                st.markdown(f"**Summary:** {analysis.get('summary', '')}")
                st.info(f"⚠️ **Primary risk:** {analysis.get('primary_risk', '')}")
                st.success(f"✅ **Recommendation:** {analysis.get('recommendation', '')}")
                
                data_types = analysis.get('sensitive_data_types', [])
                if data_types:
                    types_html = ' '.join([f'<span class="badge badge-general">{t}</span>' for t in data_types])
                    st.markdown("**Sensitive data types identified:**")
                    st.markdown(types_html, unsafe_allow_html=True)
