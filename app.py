import streamlit as st
from datetime import datetime
from Agent_orchestration import agent

# ────────────────────────────────────────────────────────────────────────────
# Page Config
# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Edynapse — AI Content Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ────────────────────────────────────────────────────────────────────────────
# Global Styles
# ────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --bg:        #0A0C10;
    --surface:   #111318;
    --border:    #1E2230;
    --border-hi: #2E3450;
    --accent:    #00E5C3;
    --accent-dim:#00E5C320;
    --accent2:   #4F6EF7;
    --accent2-dim:#4F6EF715;
    --text:      #E8EAF0;
    --muted:     #6B7280;
    --pass:      #10B981;
    --fail:      #F43F5E;
    --warn:      #F59E0B;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stAppViewContainer"] { padding: 0 !important; }
[data-testid="stHeader"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border-hi); border-radius: 2px; }

/* ── Topbar ── */
.edynapse-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 48px;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
    position: sticky;
    top: 0;
    z-index: 100;
}
.edynapse-logo {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 22px;
    letter-spacing: -0.5px;
    color: var(--text);
}
.edynapse-logo span { color: var(--accent); }
.topbar-badge {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    font-weight: 500;
    color: var(--accent);
    background: var(--accent-dim);
    border: 1px solid var(--accent)40;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* ── Hero ── */
.hero-section {
    padding: 72px 48px 52px;
    border-bottom: 1px solid var(--border);
    position: relative;
    overflow: hidden;
}
.hero-section::before {
    content: '';
    position: absolute;
    top: -60px; left: -60px;
    width: 400px; height: 400px;
    background: radial-gradient(circle, var(--accent)08 0%, transparent 70%);
    pointer-events: none;
}
.hero-section::after {
    content: '';
    position: absolute;
    bottom: -80px; right: 100px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, var(--accent2)0A 0%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: var(--accent);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.hero-eyebrow::before {
    content: '';
    display: inline-block;
    width: 24px; height: 1px;
    background: var(--accent);
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: clamp(32px, 4vw, 52px);
    line-height: 1.08;
    letter-spacing: -1.5px;
    color: var(--text);
    margin-bottom: 16px;
}
.hero-title .hl { color: var(--accent); }
.hero-sub {
    font-size: 15px;
    color: var(--muted);
    font-weight: 300;
    max-width: 520px;
    line-height: 1.7;
}

/* ── Main Layout ── */
.main-grid {
    display: grid;
    grid-template-columns: 340px 1fr;
    min-height: calc(100vh - 160px);
}
.panel-left {
    border-right: 1px solid var(--border);
    padding: 36px 32px;
    background: var(--surface);
}
.panel-right {
    padding: 36px 48px;
    background: var(--bg);
}

/* ── Panel Labels ── */
.panel-label {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.panel-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Form Controls ── */
.field-label {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 8px;
}
div[data-baseweb="select"] > div {
    background: var(--bg) !important;
    border: 1px solid var(--border-hi) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: border-color 0.2s;
}
div[data-baseweb="select"] > div:hover { border-color: var(--accent) !important; }
div[data-baseweb="select"] svg { fill: var(--muted) !important; }

input[data-testid="stTextInput"] {
    background: var(--bg) !important;
    border: 1px solid var(--border-hi) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Streamlit input wrappers */
[data-testid="stTextInput"] > div > div > input {
    background: var(--bg) !important;
    border: 1px solid var(--border-hi) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}

/* ── Generate Button ── */
[data-testid="stFormSubmitButton"] > button {
    width: 100% !important;
    background: var(--accent) !important;
    color: #0A0C10 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 14px 24px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    letter-spacing: 0.5px !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    margin-top: 8px !important;
}
[data-testid="stFormSubmitButton"] > button:hover {
    background: #00FFD9 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px var(--accent)30 !important;
}

/* ── Stat Cards ── */
.stat-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-bottom: 28px;
}
.stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
}
.stat-value {
    font-family: 'Syne', sans-serif;
    font-size: 24px;
    font-weight: 800;
    color: var(--text);
    line-height: 1;
    margin-bottom: 4px;
}
.stat-label {
    font-family: 'DM Mono', monospace;
    font-size: 9px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--muted);
}

/* ── Status Badge ── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 20px;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 28px;
}
.status-pass {
    color: var(--pass);
    background: #10B98115;
    border: 1px solid #10B98130;
}
.status-fail {
    color: var(--fail);
    background: #F43F5E15;
    border: 1px solid #F43F5E30;
}
.status-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
}
.dot-pass { background: var(--pass); }
.dot-fail { background: var(--fail); }

/* ── Content Card ── */
.content-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 28px 32px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
.content-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: var(--accent);
}
.content-card-title {
    font-family: 'Syne', sans-serif;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 16px;
}
.explanation-text {
    font-size: 15px;
    line-height: 1.8;
    color: #C8CAD4;
    font-weight: 300;
}

/* ── MCQ Card ── */
.mcq-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px 28px;
    margin-bottom: 12px;
    transition: border-color 0.2s, transform 0.2s;
    position: relative;
}
.mcq-card:hover {
    border-color: var(--border-hi);
    transform: translateX(3px);
}
.mcq-number {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    color: var(--accent);
    letter-spacing: 1px;
    margin-bottom: 10px;
}
.mcq-question {
    font-family: 'DM Sans', sans-serif;
    font-size: 15px;
    font-weight: 500;
    color: var(--text);
    margin-bottom: 16px;
    line-height: 1.5;
}
.mcq-options {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-bottom: 16px;
}
.mcq-option {
    padding: 9px 14px;
    border-radius: 6px;
    font-size: 13px;
    color: #9CA3AF;
    background: var(--bg);
    border: 1px solid var(--border);
    font-family: 'DM Sans', sans-serif;
    transition: all 0.15s;
}
.mcq-option:hover { border-color: var(--border-hi); color: var(--text); }
.mcq-option.correct {
    color: var(--pass);
    background: #10B98110;
    border-color: #10B98140;
    font-weight: 500;
}
.mcq-answer-label {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    color: var(--pass);
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
    padding: 0 !important;
    margin-bottom: 28px !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    color: var(--muted) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    padding: 12px 20px !important;
    transition: all 0.2s !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}

/* ── Inspector ── */
.inspector-block {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    margin-bottom: 16px;
    overflow: hidden;
}
.inspector-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 20px;
    border-bottom: 1px solid var(--border);
    background: #0D0F14;
}
.inspector-title {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--muted);
}
.inspector-body { padding: 16px 20px; }

/* ── Log Line ── */
.log-line {
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    color: var(--muted);
    padding: 6px 0;
    border-bottom: 1px solid var(--border);
    display: flex;
    gap: 12px;
}
.log-time { color: var(--accent)80; }
.log-msg { color: #9CA3AF; }

/* ── Download Button ── */
[data-testid="stDownloadButton"] > button {
    background: transparent !important;
    border: 1px solid var(--border-hi) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 1px !important;
    padding: 10px 20px !important;
    transition: all 0.2s !important;
}
[data-testid="stDownloadButton"] > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

/* ── Empty State ── */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 420px;
    gap: 16px;
    opacity: 0.5;
}
.empty-icon {
    font-size: 40px;
    filter: grayscale(1);
}
.empty-text {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
}

/* ── JSON Viewer ── */
[data-testid="stJson"] {
    background: var(--bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
}

/* ── Warning/Error ── */
[data-testid="stAlert"] {
    background: var(--surface) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# Topbar
# ────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="edynapse-topbar">
    <div class="edynapse-logo">Edy<span>napse</span></div>
    <div class="topbar-badge">⚡ AI Content Engine</div>
</div>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# Hero
# ────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-section">
    <div class="hero-eyebrow">Content Generation Platform</div>
    <div class="hero-title">Curriculum-Aligned Content,<br><span class="hl">Generated in Seconds.</span></div>
    <div class="hero-sub">Edynapse's agentic pipeline generates, reviews, and refines educational content with zero manual effort — built for EdTech platforms at scale.</div>
</div>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# Main Two-Column Layout
# ────────────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1.1, 2.6], gap="small")

# ═══════════════════════════════════════
# LEFT PANEL — Controls
# ═══════════════════════════════════════
with col_left:
    st.markdown('<div style="padding: 36px 8px 36px 48px;">', unsafe_allow_html=True)
    st.markdown('<div class="panel-label">Configuration</div>', unsafe_allow_html=True)

    with st.form("input_form"):
        st.markdown('<div class="field-label">Grade Level</div>', unsafe_allow_html=True)
        grade = st.selectbox(
            label="Grade Level",
            options=list(range(1, 13)),
            format_func=lambda x: f"Grade {x}",
            label_visibility="collapsed"
        )

        st.markdown('<div class="field-label" style="margin-top:20px;">Topic</div>', unsafe_allow_html=True)
        topic = st.text_input(
            label="Topic",
            placeholder="e.g. The Water Cycle",
            label_visibility="collapsed"
        )

        st.markdown('<div style="margin-top: 28px;"></div>', unsafe_allow_html=True)
        submit = st.form_submit_button("⚡ Generate Content")

    # Pipeline info
    st.markdown("""
    <div style="margin-top: 32px; padding-top: 24px; border-top: 1px solid var(--border);">
        <div class="panel-label">Pipeline</div>
        <div style="display:flex; flex-direction:column; gap:12px; margin-top: 4px;">
            <div style="display:flex; align-items:center; gap:12px;">
                <div style="width:28px; height:28px; border-radius:6px; background:var(--accent-dim); border:1px solid var(--accent)30; display:flex; align-items:center; justify-content:center; font-size:12px;">🧠</div>
                <div>
                    <div style="font-size:12px; color:var(--text); font-weight:500;">Generator</div>
                    <div style="font-size:11px; color:var(--muted);">Gemini 2.5 Flash</div>
                </div>
            </div>
            <div style="width:1px; height:16px; background:var(--border); margin-left:13px;"></div>
            <div style="display:flex; align-items:center; gap:12px;">
                <div style="width:28px; height:28px; border-radius:6px; background:var(--accent2-dim); border:1px solid var(--accent2)30; display:flex; align-items:center; justify-content:center; font-size:12px;">🔍</div>
                <div>
                    <div style="font-size:12px; color:var(--text); font-weight:500;">Reviewer</div>
                    <div style="font-size:11px; color:var(--muted);">Quality Gate</div>
                </div>
            </div>
            <div style="width:1px; height:16px; background:var(--border); margin-left:13px;"></div>
            <div style="display:flex; align-items:center; gap:12px;">
                <div style="width:28px; height:28px; border-radius:6px; background:#F59E0B10; border:1px solid #F59E0B30; display:flex; align-items:center; justify-content:center; font-size:12px;">🔁</div>
                <div>
                    <div style="font-size:12px; color:var(--text); font-weight:500;">Refinement</div>
                    <div style="font-size:11px; color:var(--muted);">Auto-retry on fail</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════
# RIGHT PANEL — Output
# ═══════════════════════════════════════
with col_right:
    st.markdown('<div style="padding: 36px 48px 36px 24px;">', unsafe_allow_html=True)

    if not submit:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">⚡</div>
            <div class="empty-text">Awaiting generation request</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        if not topic.strip():
            st.warning("⚠️  Please enter a topic to continue.")
            st.stop()

        logs = []
        t0 = datetime.now()
        logs.append((t0.strftime('%H:%M:%S'), "Agent invocation started"))

        with st.spinner("Running Edynapse pipeline…"):
            result = agent.invoke({
                "grade": grade,
                "topic": topic.strip(),
                "generator_output": None,
                "reviewer_output": None,
                "retry_count": 0,
            })

        t1 = datetime.now()
        logs.append((t1.strftime('%H:%M:%S'), "Generation complete"))

        generator_output = result.get("generator_output")
        reviewer_output  = result.get("reviewer_output")
        retry_count      = result.get("retry_count", 0)
        elapsed          = round((t1 - t0).total_seconds(), 1)

        if not generator_output:
            st.error("Pipeline returned no content. Please retry.")
            st.stop()

        logs.append((t1.strftime('%H:%M:%S'), f"Review status: {reviewer_output.status if reviewer_output else 'N/A'}"))

        # ── Stat Row ──
        review_status = reviewer_output.status if reviewer_output else "unknown"
        status_color  = "#10B981" if review_status == "pass" else "#F43F5E"

        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-card">
                <div class="stat-value">{elapsed}s</div>
                <div class="stat-label">Generation Time</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{retry_count}</div>
                <div class="stat-label">Retries Used</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">5</div>
                <div class="stat-label">MCQs Generated</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" style="color:{status_color}; font-size:16px; padding-top:4px;">{'APPROVED' if review_status == 'pass' else 'FLAGGED'}</div>
                <div class="stat-label">Review Decision</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Tabs ──
        tab_output, tab_inspector = st.tabs(["OUTPUT", "INSPECTOR"])

        # ════════════════════════
        # TAB 1 — Final Output
        # ════════════════════════
        with tab_output:
            # Status badge
            if review_status == "pass":
                st.markdown("""
                <div class="status-badge status-pass">
                    <div class="status-dot dot-pass"></div>
                    Review Passed — Production Ready
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="status-badge status-fail">
                    <div class="status-dot dot-fail"></div>
                    Review Flagged — Best-Effort Output
                </div>
                """, unsafe_allow_html=True)

            # Explanation
            content = generator_output.model_dump()
            st.markdown(f"""
            <div class="content-card">
                <div class="content-card-title">📘 Explanation — Grade {grade} · {topic}</div>
                <div class="explanation-text">{content['explanation']}</div>
            </div>
            """, unsafe_allow_html=True)

            # MCQs
            st.markdown('<div class="panel-label" style="margin-top:28px;">Multiple Choice Questions</div>', unsafe_allow_html=True)

            for i, mcq in enumerate(content["mcqs"], 1):
                correct_letter = mcq["answer"].strip().upper()

                options_html = ""
                for opt in mcq["options"]:
                    letter = opt[0].upper()
                    is_correct = (letter == correct_letter)
                    cls = "mcq-option correct" if is_correct else "mcq-option"
                    options_html += f'<div class="{cls}">{opt}</div>'

                st.markdown(f"""
                <div class="mcq-card">
                    <div class="mcq-number">Q{i:02d}</div>
                    <div class="mcq-question">{mcq['question']}</div>
                    <div class="mcq-options">{options_html}</div>
                    <div class="mcq-answer-label">✓ Correct Answer: {correct_letter}</div>
                </div>
                """, unsafe_allow_html=True)

            # ── Markdown Export Builder ──
            def build_markdown(content: dict, grade: int, topic: str,
                               review_status: str, elapsed: float,
                               retry_count: int) -> str:
                now = datetime.now().strftime("%B %d, %Y · %H:%M")
                status_icon = "✅ Passed" if review_status == "pass" else "⚠️ Flagged"

                lines = [
                    f"# {topic}",
                    f"> **Grade {grade} Educational Content** — Generated by Edynapse AI",
                    "",
                    "---",
                    "",
                    "## 📋 Generation Metadata",
                    "",
                    f"| Field | Value |",
                    f"|---|---|",
                    f"| Generated On | {now} |",
                    f"| Grade Level | Grade {grade} |",
                    f"| Topic | {topic} |",
                    f"| Review Status | {status_icon} |",
                    f"| Generation Time | {elapsed}s |",
                    f"| Retries Used | {retry_count} |",
                    "",
                    "---",
                    "",
                    "## 📘 Explanation",
                    "",
                    content["explanation"],
                    "",
                    "---",
                    "",
                    "## 📝 Multiple Choice Questions",
                    "",
                ]

                for i, mcq in enumerate(content["mcqs"], 1):
                    correct = mcq["answer"].strip().upper()
                    lines.append(f"### Q{i}. {mcq['question']}")
                    lines.append("")
                    for opt in mcq["options"]:
                        letter = opt[0].upper()
                        marker = "✅" if letter == correct else "○"
                        lines.append(f"{marker} {opt}")
                    lines.append("")
                    lines.append(f"> **Answer:** `{correct}`")
                    lines.append("")

                lines += [
                    "---",
                    "",
                    "*This document was auto-generated by the Edynapse AI Content Engine.*",
                    "*Review all content before distribution to students.*",
                ]

                return "\n".join(lines)

            md_content = build_markdown(
                content, grade, topic, review_status, elapsed, retry_count
            )
            fname = f"edynapse_{topic.lower().replace(' ', '_')}_grade{grade}.md"

            st.markdown('<div style="margin-top: 24px;"></div>', unsafe_allow_html=True)
            st.download_button(
                label="⬇  Export as Markdown",
                data=md_content,
                file_name=fname,
                mime="text/markdown"
            )

        # ════════════════════════
        # TAB 2 — Inspector
        # ════════════════════════
        with tab_inspector:
            # Generator output
            st.markdown("""
            <div class="inspector-block">
                <div class="inspector-header">
                    <div class="inspector-title">🧠 Generator Output</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.json(generator_output.model_dump())

            # Reviewer output
            st.markdown("""
            <div class="inspector-block" style="margin-top:16px;">
                <div class="inspector-header">
                    <div class="inspector-title">🔍 Reviewer Output</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if reviewer_output:
                st.json(reviewer_output.model_dump())
                if reviewer_output.status == "fail" and reviewer_output.feedback:
                    st.markdown('<div class="panel-label" style="margin-top:20px;">Reviewer Feedback</div>', unsafe_allow_html=True)
                    for fb in reviewer_output.feedback:
                        st.markdown(f"""
                        <div style="padding: 10px 16px; background: #F43F5E08; border: 1px solid #F43F5E25;
                                    border-radius: 6px; margin-bottom: 8px; font-size: 13px; color: #F87171;
                                    font-family: 'DM Sans', sans-serif; line-height: 1.5;">
                            · {fb}
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.markdown('<div style="color: var(--muted); font-size:13px;">No reviewer output available.</div>', unsafe_allow_html=True)

            # Execution logs
            st.markdown('<div class="panel-label" style="margin-top:24px;">Execution Log</div>', unsafe_allow_html=True)
            log_html = ""
            for ts, msg in logs:
                log_html += f'<div class="log-line"><span class="log-time">{ts}</span><span class="log-msg">{msg}</span></div>'
            st.markdown(f'<div style="background:var(--surface); border:1px solid var(--border); border-radius:8px; padding: 12px 16px;">{log_html}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
