import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from core.extractor import extract_text
from core.scanner import scan_document, get_keyword_frequency_table
from core.tfidf_engine import run_tfidf, get_document_topic, get_risk_score
from db.db_manager import init_db, get_all_keywords, add_keyword, delete_keyword, search_keywords
from llm.ollama_client import check_ollama_available, list_available_models, analyze_document

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocScan — Secure Document Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #21262d; }
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label { color: #8b949e !important; }

.stButton > button {
    background: #1f6feb; color: #fff; border: none;
    border-radius: 8px; padding: 0.45rem 1.4rem;
    font-weight: 500; font-size: 0.875rem;
    transition: background 0.2s; width: 100%;
}
.stButton > button:hover { background: #388bfd; }

.metric-card {
    background: #161b22; border: 1px solid #21262d;
    border-radius: 12px; padding: 1rem 1.25rem; text-align: center;
}
.metric-card .value { font-size: 2rem; font-weight: 600; color: #58a6ff; line-height: 1.2; }
.metric-card .label { font-size: 0.75rem; color: #8b949e; text-transform: uppercase;
    letter-spacing: 0.05em; margin-top: 0.25rem; }

.context-snippet {
    background: #161b22; border-left: 3px solid #1f6feb;
    border-radius: 0 6px 6px 0; padding: 0.5rem 0.75rem; margin: 0.35rem 0;
    font-size: 0.82rem; font-family: 'Courier New', monospace;
    color: #c9d1d9; line-height: 1.5; word-break: break-word;
}
.context-snippet .hit {
    color: #ffa657; font-weight: 600;
    background: rgba(255,166,87,0.15); padding: 1px 3px; border-radius: 3px;
}

.badge { display: inline-block; padding: 2px 10px; border-radius: 20px;
    font-size: 0.72rem; font-weight: 500; letter-spacing: 0.02em; }
.badge-identity    { background:#3d1515; color:#f97171; border:1px solid #6b2020; }
.badge-financial   { background:#152415; color:#5acd5a; border:1px solid #1f4a1f; }
.badge-credential  { background:#261535; color:#c084fc; border:1px solid #4a2070; }
.badge-classification { background:#0d1f3c; color:#60a5fa; border:1px solid #1a3a6b; }
.badge-personal    { background:#2e2010; color:#fbbf24; border:1px solid #5a3d10; }
.badge-health      { background:#0d2e2e; color:#2dd4bf; border:1px solid #0f5050; }
.badge-technical   { background:#1a1a2e; color:#818cf8; border:1px solid #2d2d5a; }
.badge-general     { background:#1c1c1c; color:#9ca3af; border:1px solid #333; }

.risk-HIGH    { color:#f97171; font-weight:700; }
.risk-MEDIUM  { color:#fbbf24; font-weight:700; }
.risk-LOW     { color:#5acd5a; font-weight:700; }
.risk-UNKNOWN { color:#8b949e; font-weight:700; }

.tfidf-pill {
    display:inline-block; background:#0d2030; border:1px solid #1f4060;
    color:#58a6ff; border-radius:20px; padding:3px 12px; font-size:0.78rem; margin:3px;
}
.ollama-online  { color:#5acd5a; font-size:0.82rem; font-weight:600; }
.ollama-offline { color:#f97171; font-size:0.82rem; font-weight:600; }
.meta-row { display:flex; gap:0.5rem; margin:0.2rem 0; font-size:0.82rem; }
.meta-key { color:#8b949e; min-width:130px; font-weight:500; }
.meta-val { color:#c9d1d9; }
.section-divider { border:none; border-top:1px solid #21262d; margin:1rem 0; }
#MainMenu {visibility:hidden;} footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ── Init ───────────────────────────────────────────────────────────────────────
init_db()

DEFAULTS = ['scan_results','extracted_texts','tfidf_results','llm_analyses',
            'uploaded_file_names','doc_metadata']
for key in DEFAULTS:
    if key not in st.session_state:
        st.session_state[key] = {} if key != 'uploaded_file_names' else []

SUPPORTED_TYPES = ['pdf','docx','xlsx','pptx']

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h1 style='margin-bottom:0'>🔍 DocScan</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8b949e;margin-top:0'>Secure Document Analyzer</p>", unsafe_allow_html=True)
    st.markdown("---")

    # AI backend status
    st.markdown("### 🤖 AI Backend")
    ollama_ok = check_ollama_available()
    if ollama_ok:
        st.markdown("<div class='ollama-online'>● Ollama online</div>", unsafe_allow_html=True)
        models = list_available_models()
        if models:
            default_idx = models.index('llama3.2') if 'llama3.2' in models else 0
            st.selectbox("Model", models, index=default_idx, key="ollama_model")
        else:
            st.warning("No models found. Run: `ollama pull llama3.2`")
    else:
        st.markdown("<div class='ollama-offline'>● Ollama offline</div>", unsafe_allow_html=True)
        st.info("AI Insights tab disabled.\nTo enable: install Ollama and run `ollama serve`")

    st.markdown("---")

    # Keyword database
    st.markdown("### 🗂 Keyword Database")
    all_kws = get_all_keywords()
    st.markdown(
        f'<div class="metric-card"><div class="value">{len(all_kws)}</div>'
        f'<div class="label">Total Keywords</div></div>', unsafe_allow_html=True)
    st.markdown("")

    search_q = st.text_input("🔍 Search keywords", key="kw_search")
    display_kws = search_keywords(search_q) if search_q else all_kws

    for kw in display_kws:
        c1, c2 = st.columns([0.85, 0.15])
        with c1:
            st.markdown(
                f"<span style='font-size:0.85rem'>{kw['keyword']}</span> "
                f"<span class='badge badge-{kw['category']}'>{kw['category']}</span> "
                f"<span style='color:#8b949e;font-size:0.72rem'>w:{kw['weight']}</span>",
                unsafe_allow_html=True)
        with c2:
            if st.button("×", key=f"del_{kw['id']}"):
                delete_keyword(kw['id'])
                st.rerun()

    if not display_kws and search_q:
        st.caption("No keywords match.")

    with st.expander("＋ Add New Keyword"):
        new_kw  = st.text_input("Keyword / phrase", placeholder="e.g. bank account")
        new_cat = st.selectbox("Category",
            ['general','identity','financial','credential','classification','personal','health','technical'])
        new_wt  = st.slider("Risk weight", 0.5, 5.0, 1.5, 0.5,
            help="Higher = more sensitive. Used in risk score calculation.")
        if st.button("Add Keyword", key="add_kw_btn"):
            if new_kw.strip():
                ok = add_keyword(new_kw.strip().lower(), new_cat, new_wt)
                st.success(f"Added '{new_kw.strip()}'") if ok else st.error("Keyword already exists.")
                st.rerun()
            else:
                st.warning("Please enter a keyword.")

# ── Tabs ────────────────────────────────────────────────────────────────────────
tab_scan, tab_analysis, tab_ai = st.tabs(["📄 Scan Documents", "📊 Analysis", "🤖 AI Insights"])

# =============================================================================
# TAB 1 — SCAN
# =============================================================================
with tab_scan:
    st.header("Upload & Scan Documents")
    st.caption("Supports: PDF · DOCX · XLSX · PPTX")

    uploaded_files = st.file_uploader(
        "Drop files here or click to browse",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True)

    if uploaded_files:
        cur_names = [f.name for f in uploaded_files]
        if cur_names != st.session_state.uploaded_file_names:
            for key in ['scan_results','extracted_texts','tfidf_results','llm_analyses','doc_metadata']:
                st.session_state[key] = {}
            st.session_state.uploaded_file_names = cur_names

        st.markdown(f"**{len(uploaded_files)} file(s) ready**")
        for f in uploaded_files:
            st.write(f"- {f.name} ({f.size/1024:.1f} KB)")

        # Controls row
        col_scan, col_pii, col_ner, col_ai_chk = st.columns([2, 1, 1, 1.5])
        with col_scan:
            scan_btn = st.button("🔍 Scan All Documents", use_container_width=True)
        with col_pii:
            do_pii = st.checkbox("Auto PII", value=True, help="Detect emails, phone, Aadhaar, PAN, etc.")
        with col_ner:
            do_ner = st.checkbox("Auto NER", value=True, help="Detect roles, org names, etc.")
        with col_ai_chk:
            run_ai = False
            if ollama_ok:
                run_ai = st.checkbox("Run AI Analysis", value=False,
                    help="Requires Ollama running locally")

        if scan_btn:
            keywords = get_all_keywords()
            prog = st.progress(0)
            status = st.empty()
            n = len(uploaded_files)

            for i, file in enumerate(uploaded_files):
                status.text(f"Extracting: {file.name}…")
                file_bytes = file.read()
                extraction = extract_text(file_bytes, file.name)

                if extraction['error']:
                    st.error(f"❌ {file.name}: {extraction['error']}")
                    prog.progress((i + 1) / n)
                    continue

                status.text(f"Scanning: {file.name}…")
                kw_results = scan_document(
                    extraction['text'], keywords,
                    include_pii=do_pii, include_ner=do_ner)

                st.session_state.extracted_texts[file.name] = extraction['text']
                st.session_state.doc_metadata[file.name]    = extraction.get('metadata', {})
                st.session_state.scan_results[file.name] = {
                    'keywords':   kw_results,
                    'page_count': extraction['page_count'],
                    'char_count': extraction['char_count'],
                    'word_count': extraction['word_count'],
                    'method':     extraction['method'],
                }
                prog.progress((i + 1) / n)

            status.text("Running TF-IDF…")
            st.session_state.tfidf_results = run_tfidf(st.session_state.extracted_texts)

            if run_ai and ollama_ok:
                for fname in st.session_state.extracted_texts:
                    status.text(f"AI: {fname}…")
                    analysis = analyze_document(
                        text_excerpt=st.session_state.extracted_texts[fname][:2000],
                        keyword_hits=st.session_state.scan_results[fname]['keywords'],
                        tfidf_terms=st.session_state.tfidf_results.get(fname, []),
                        model=st.session_state.get('ollama_model', 'llama3.2')
                    )
                    st.session_state.llm_analyses[fname] = analysis

            status.text("✅ Scan complete!")
            prog.progress(1.0)
            st.rerun()

    # ── Results ──────────────────────────────────────────────────────────────
    if st.session_state.scan_results:
        st.subheader(f"Results — {len(st.session_state.scan_results)} document(s)")

        CAT_COLORS = {
            'identity':'#f97171','financial':'#5acd5a','credential':'#c084fc',
            'classification':'#60a5fa','personal':'#fbbf24','health':'#2dd4bf',
            'technical':'#818cf8','general':'#9ca3af'
        }

        for fname, result in st.session_state.scan_results.items():
            with st.expander(f"📄 {fname}", expanded=True):
                kw_hits   = result['keywords']
                total_hits = sum(k['count'] for k in kw_hits)
                unique_kw  = len(kw_hits)
                risk_score, risk_label = get_risk_score(kw_hits)

                # ── Metrics row
                mc = st.columns(5)
                mc[0].markdown(f'<div class="metric-card"><div class="value">{total_hits}</div><div class="label">Total Hits</div></div>', unsafe_allow_html=True)
                mc[1].markdown(f'<div class="metric-card"><div class="value">{unique_kw}</div><div class="label">Signals</div></div>', unsafe_allow_html=True)
                mc[2].markdown(f'<div class="metric-card"><div class="value">{result["page_count"]}</div><div class="label">Pages</div></div>', unsafe_allow_html=True)
                mc[3].markdown(f'<div class="metric-card"><div class="value">{result["word_count"]}</div><div class="label">Words</div></div>', unsafe_allow_html=True)
                mc[4].markdown(
                    f'<div class="metric-card"><div class="value risk-{risk_label}">'
                    f'{risk_label}</div><div class="label">Risk Level</div></div>',
                    unsafe_allow_html=True)

                st.markdown("")

                # ── Document Metadata
                meta = st.session_state.doc_metadata.get(fname, {})
                if meta and any(v for v in meta.values()):
                    with st.expander("📋 Document Metadata"):
                        for k, v in meta.items():
                            if v:
                                st.markdown(
                                    f'<div class="meta-row">'
                                    f'<span class="meta-key">{k.capitalize()}</span>'
                                    f'<span class="meta-val">{v}</span></div>',
                                    unsafe_allow_html=True)

                # ── TF-IDF topic signals
                tfidf_data = st.session_state.tfidf_results.get(fname, [])
                if tfidf_data:
                    topic = get_document_topic(
                        st.session_state.tfidf_results, fname,
                        full_text=st.session_state.extracted_texts.get(fname, ""))
                    st.markdown(f"**📂 Detected document type:** `{topic}`")
                    pills = ' '.join([
                        f'<span class="tfidf-pill">{t["term"]} <small>({t["score"]:.2f})</small></span>'
                        for t in tfidf_data[:10]])
                    st.markdown("**TF-IDF topic signals:**")
                    st.markdown(pills, unsafe_allow_html=True)

                # ── AI inline summary (if run)
                if fname in st.session_state.llm_analyses:
                    analysis = st.session_state.llm_analyses[fname]
                    ai_risk  = analysis.get('risk_level', 'UNKNOWN')
                    st.markdown("---")
                    st.markdown("**🤖 AI Security Summary**")
                    a1, a2 = st.columns([4, 1])
                    a1.markdown(f"*{analysis.get('summary', '')}*")
                    a2.markdown(
                        f'<div class="metric-card"><div class="risk-{ai_risk}" style="font-size:1.2rem">'
                        f'{ai_risk}</div><div class="label">AI Risk</div></div>',
                        unsafe_allow_html=True)
                    st.markdown(f"**Primary risk:** {analysis.get('primary_risk','')}")
                    st.markdown(f"**Recommendation:** {analysis.get('recommendation','')}")

                # ── Bar chart
                if kw_hits:
                    top12 = sorted(kw_hits, key=lambda x: x['count'], reverse=True)[:12]
                    fig = go.Figure(go.Bar(
                        x=[k['count'] for k in top12],
                        y=[k['keyword'] for k in top12],
                        orientation='h',
                        marker_color=[CAT_COLORS.get(k['category'], '#9ca3af') for k in top12],
                        text=[str(k['count']) for k in top12],
                        textposition='outside',
                        hovertemplate='<b>%{y}</b><br>Count: %{x}<extra></extra>'
                    ))
                    fig.update_layout(
                        paper_bgcolor='#161b22', plot_bgcolor='#161b22',
                        font_color='#c9d1d9',
                        height=max(250, len(top12) * 34),
                        margin=dict(l=20, r=60, t=10, b=10),
                        xaxis=dict(showgrid=False, color='#8b949e'),
                        yaxis=dict(showgrid=False, color='#c9d1d9', autorange='reversed'),
                        showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

                # ── Keyword detail cards
                st.markdown("**Keyword & Signal Matches:**")
                pii_hits = [k for k in kw_hits if k.get('is_pii')]
                ner_hits = [k for k in kw_hits if k.get('is_ner')]
                kw_user  = [k for k in kw_hits if not k.get('is_pii') and not k.get('is_ner')]

                def render_hits(hits):
                    for kw_r in sorted(hits, key=lambda x: x['count'], reverse=True):
                        cat = kw_r['category']
                        st.markdown(
                            f"**{kw_r['keyword']}** "
                            f"<span class='badge badge-{cat}'>{cat}</span> — "
                            f"**{kw_r['count']}** occurrence(s) | weight: {kw_r['weight']}",
                            unsafe_allow_html=True)
                        for ctx in kw_r['contexts'][:3]:
                            disp = ctx.replace('>>>', '<span class="hit">').replace('<<<', '</span>')
                            st.markdown(f'<div class="context-snippet">{disp}</div>', unsafe_allow_html=True)
                        if len(kw_r['contexts']) > 3:
                            with st.expander(f"▼ View all {kw_r['count']} occurrences"):
                                for ctx in kw_r['contexts'][3:]:
                                    disp = ctx.replace('>>>', '<span class="hit">').replace('<<<', '</span>')
                                    st.markdown(f'<div class="context-snippet">{disp}</div>', unsafe_allow_html=True)
                        st.markdown("")

                if pii_hits:
                    st.markdown("##### 🔴 Auto-Detected PII")
                    render_hits(pii_hits)
                if ner_hits:
                    st.markdown("##### 🟡 Auto-Detected Named Entities")
                    render_hits(ner_hits)
                if kw_user:
                    st.markdown("##### 🔵 User-Defined Keywords")
                    render_hits(kw_user)
                if not kw_hits:
                    st.info("No sensitive keywords detected in this document.")

# =============================================================================
# TAB 2 — ANALYSIS
# =============================================================================
with tab_analysis:
    if not st.session_state.scan_results:
        st.info("Upload and scan documents first.")
    else:
        st.header("Cross-Document Analysis")

        # Summary table
        summary_rows = []
        for fname, res in st.session_state.scan_results.items():
            kw = res['keywords']
            rs, rl = get_risk_score(kw)
            top_kw = sorted(kw, key=lambda x: x['count'], reverse=True)
            summary_rows.append({
                'Document': fname,
                'Pages': res['page_count'],
                'Words': res['word_count'],
                'Signals Found': len(kw),
                'Total Hits': sum(k['count'] for k in kw),
                'Risk Level': rl,
                'Risk Score': round(rs, 1),
                'Top Signal': top_kw[0]['keyword'] if top_kw else '—',
                'Extraction': res['method'],
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        # Heatmap (only if > 1 doc)
        if len(st.session_state.scan_results) > 1:
            st.subheader("Keyword Frequency Heatmap")
            st.caption("Rows = documents · Columns = top keywords · Intensity = count")
            formatted = {fn: res['keywords'] for fn, res in st.session_state.scan_results.items()}
            ft = get_keyword_frequency_table(formatted)
            if ft['keywords']:
                fig_heat = go.Figure(go.Heatmap(
                    z=ft['matrix'], x=ft['keywords'], y=ft['pdfs'],
                    colorscale=[[0,'#161b22'],[0.3,'#0d2030'],[0.6,'#1f6feb'],[1.0,'#58a6ff']],
                    showscale=True,
                    hovertemplate='Doc: %{y}<br>Keyword: %{x}<br>Count: %{z}<extra></extra>'
                ))
                fig_heat.update_layout(
                    paper_bgcolor='#0d1117', plot_bgcolor='#0d1117', font_color='#c9d1d9',
                    height=max(300, len(ft['pdfs']) * 55 + 100),
                    margin=dict(l=20, r=20, t=20, b=140),
                    xaxis=dict(tickangle=-45, tickfont_size=11, color='#8b949e'),
                    yaxis=dict(color='#c9d1d9'))
                st.plotly_chart(fig_heat, use_container_width=True)

        # Category donut
        st.subheader("Hits by Category")
        cat_totals = {}
        for res in st.session_state.scan_results.values():
            for kw in res['keywords']:
                cat_totals[kw['category']] = cat_totals.get(kw['category'], 0) + kw['count']
        if cat_totals:
            CAT_COLORS2 = {
                'identity':'#f97171','financial':'#5acd5a','credential':'#c084fc',
                'classification':'#60a5fa','personal':'#fbbf24','health':'#2dd4bf',
                'technical':'#818cf8','general':'#9ca3af'
            }
            fig_d = go.Figure(go.Pie(
                labels=list(cat_totals.keys()), values=list(cat_totals.values()),
                hole=0.55,
                marker_colors=[CAT_COLORS2.get(c,'#9ca3af') for c in cat_totals.keys()],
                textinfo='label+percent',
                hovertemplate='%{label}: %{value} hits<extra></extra>'
            ))
            fig_d.update_layout(
                paper_bgcolor='#0d1117', font_color='#c9d1d9', height=380,
                legend=dict(bgcolor='#161b22', bordercolor='#21262d'))
            st.plotly_chart(fig_d, use_container_width=True)

# =============================================================================
# TAB 3 — AI INSIGHTS
# =============================================================================
with tab_ai:
    st.header("🤖 AI Security Insights")

    if not ollama_ok:
        st.warning("**Ollama is not running.** AI Insights are disabled.")
        st.markdown("""
**To enable AI Insights:**
1. Download Ollama from [ollama.com](https://ollama.com/download)
2. Install and it will run automatically in your system tray
3. Pull the model: `ollama pull llama3.2`
4. Refresh this page — the sidebar will show **● Ollama online**

> All analysis runs 100% on your local machine. No data is sent to the cloud.
        """)
    elif not st.session_state.scan_results:
        st.info("Scan documents first (Tab 1), then enable 'Run AI Analysis' checkbox before scanning.")
    elif not st.session_state.llm_analyses:
        st.info("No AI analysis found. Re-scan with the **'Run AI Analysis'** checkbox enabled.")
    else:
        for fname, analysis in st.session_state.llm_analyses.items():
            with st.expander(f"🤖 {fname}", expanded=True):
                risk = analysis.get('risk_level', 'UNKNOWN')
                rc1, rc2 = st.columns([4, 1])
                rc1.subheader(fname)
                rc2.markdown(
                    f'<div class="metric-card"><div class="risk-{risk}" style="font-size:1.5rem">'
                    f'{risk}</div><div class="label">Risk Level</div></div>',
                    unsafe_allow_html=True)

                st.markdown(f"**Document type:** {analysis.get('document_type','Unknown')}")
                st.markdown(f"**Summary:** {analysis.get('summary','')}")
                st.info(f"⚠️ **Primary risk:** {analysis.get('primary_risk','')}")
                st.success(f"✅ **Recommendation:** {analysis.get('recommendation','')}")

                dtypes = analysis.get('sensitive_data_types', [])
                if dtypes:
                    st.markdown("**Sensitive data types:**")
                    st.markdown(' '.join(
                        [f'<span class="badge badge-general">{t}</span>' for t in dtypes]),
                        unsafe_allow_html=True)
