import streamlit as st
from datetime import datetime

# Import your compiled agent
from Agent_orchestration import agent

st.set_page_config(
    page_title="Educational Content Generator",
    page_icon="📘",
    layout="wide"
)

st.title("📘 Educational Content Generator")
st.caption("LangGraph-based AI agent with review and refinement")

# ------------------- INPUTS -------------------
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        grade = st.selectbox(
            "Select Grade",
            options=[1, 2, 3, 4, 5, 6, 7, 8]
        )

    with col2:
        topic = st.text_input(
            "Enter Topic",
            placeholder="e.g. Types of angles"
        )

    submit = st.form_submit_button("Generate Content")

# ------------------- RUN AGENT -------------------
if submit:
    if not topic.strip():
        st.warning("Please enter a topic.")
        st.stop()

    logs = []
    logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Agent invocation started")

    with st.spinner("Generating, reviewing, and refining content..."):
        result = agent.invoke({
            "grade": grade,
            "topic": topic,
            "generator_output": None,
            "reviewer_output": None,
            "retry_count": 0
        })

    logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Agent invocation completed")

    generator_output = result.get("generator_output")
    reviewer_output = result.get("reviewer_output")

    # ------------------- TABS -------------------
    tab1, tab2 = st.tabs(["📖 Final Output", "🧠 Agent Inspector"])

    # =================== TAB 1 ===================
    with tab1:
        st.subheader("Final Educational Content")

        if not generator_output:
            st.error("No content generated.")
            st.stop()

        # If reviewer failed, warn user
        if reviewer_output and reviewer_output.status == "fail":
            st.warning("Content generated but did not fully pass review.")

        content = generator_output.model_dump()

        st.markdown("### 📘 Explanation")
        st.write(content["explanation"])

        st.markdown("### 📝 MCQs")
        for i, mcq in enumerate(content["mcqs"], start=1):
            with st.expander(f"Question {i}"):
                st.write(mcq["question"])
                for opt in mcq["options"]:
                    st.write(f"- {opt}")
                st.markdown(f"**Correct Answer:** `{mcq['answer']}`")

    # =================== TAB 2 ===================
    with tab2:
        st.subheader("🔍 Structured Node Outputs")

        # -------- Generator Output --------
        st.markdown("### Generator Output")
        st.json(generator_output.model_dump())

        # -------- Reviewer Output --------
        st.markdown("### Reviewer Output")
        if reviewer_output:
            st.json(reviewer_output.model_dump())
        else:
            st.write("No reviewer output.")

        # -------- Retry Count --------
        st.markdown("### Retry Count")
        st.code(result.get("retry_count", 0))

        # -------- Logs --------
        st.markdown("### Execution Logs")
        for log in logs:
            st.write(log)

        # -------- Download --------
        st.download_button(
            "⬇️ Download Content (JSON)",
            data=generator_output.model_dump_json(indent=2),
            file_name="educational_content.json",
            mime="application/json"
        )
