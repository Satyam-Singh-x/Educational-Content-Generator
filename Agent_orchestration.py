"""
Agentic Educational Content Pipeline using LangGraph + Google Gemini.

Flow:
    START → generator → reviewer → (pass → END | fail → generator [once] → END)
"""

# ────────────────────────────────────────────────────────────────────────────
# Imports
# ────────────────────────────────────────────────────────────────────────────
import os
from typing import List, Literal, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# ────────────────────────────────────────────────────────────────────────────
# Environment — works on both Streamlit Cloud (st.secrets) and local (.env)
# ────────────────────────────────────────────────────────────────────────────
def _resolve_google_api_key() -> str:
    """
    Resolution order:
      1. st.secrets["GOOGLE_API_KEY"]  — Streamlit Cloud deployment
      2. os.environ["GOOGLE_API_KEY"]  — local .env via dotenv or shell export
    Raises RuntimeError if neither source provides the key.
    """
    # ── Streamlit Cloud path ──
    try:
        import streamlit as st
        key = st.secrets.get("GOOGLE_API_KEY", "")
        if key:
            return key
    except Exception:
        pass  # streamlit not available or secrets not configured

    # ── Local / CI path ──
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # python-dotenv not installed in cloud env — that is fine

    key = os.getenv("GOOGLE_API_KEY", "")
    if key:
        return key

    raise RuntimeError(
        "GOOGLE_API_KEY not found.\n"
        "  Streamlit Cloud: add it under App Settings -> Secrets as GOOGLE_API_KEY=\"...\"\n"
        "  Local: set it in your .env file or as a shell environment variable."
    )

os.environ["GOOGLE_API_KEY"] = _resolve_google_api_key()

# ────────────────────────────────────────────────────────────────────────────
# LLM
# ────────────────────────────────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

# ────────────────────────────────────────────────────────────────────────────
# Data Schemas
# ────────────────────────────────────────────────────────────────────────────

class MCQ(BaseModel):
    """A single multiple-choice question."""

    question: str = Field(description="The question text, ending with a '?'.")
    options: List[str] = Field(
        description=(
            "Exactly 4 options labelled A, B, C, D. "
            "Example: ['A) Paris', 'B) London', 'C) Berlin', 'D) Rome']"
        ),
        min_length=4,
        max_length=4,
    )
    answer: str = Field(
        description="The correct option label only, e.g. 'A' or 'B'."
    )


class Content(BaseModel):
    """Full educational content package produced by the generator."""

    explanation: str = Field(
        description=(
            "A structured, grade-appropriate explanation of the topic. "
            "Must contain: a brief intro paragraph, 2-4 key concept points, "
            "and a closing summary sentence."
        )
    )
    mcqs: List[MCQ] = Field(
        description="Exactly 5 MCQs derived strictly from the explanation.",
        min_length=5,
        max_length=5,
    )


class Review(BaseModel):
    """Structured review decision produced by the reviewer."""

    status: Literal["pass", "fail"] = Field(
        description="'pass' if content meets all criteria, 'fail' otherwise."
    )
    feedback: List[str] = Field(
        description=(
            "Empty list on 'pass'. On 'fail', a list of concise, "
            "actionable issues the generator must fix."
        ),
        default_factory=list,
    )


class State(TypedDict):
    """Shared pipeline state passed between all nodes."""

    grade: int
    topic: str
    generator_output: Optional[Content]
    reviewer_output: Optional[Review]
    retry_count: int


# ────────────────────────────────────────────────────────────────────────────
# System Prompts
# ────────────────────────────────────────────────────────────────────────────

GENERATOR_SYSTEM_PROMPT = """\
ROLE
────
You are an expert Educational Content Generation Agent specialised in
creating structured, curriculum-aligned learning material for students
from Grade 1 through Grade 12.

OBJECTIVE
─────────
Given a grade level and topic, produce a single `Content` JSON object
that teaches the topic clearly and tests understanding through MCQs.

OUTPUT SCHEMA (strictly enforced)
──────────────────────────────────
{
  "explanation": "<string>",   // structured explanation — see rules below
  "mcqs": [                    // exactly 5 items
    {
      "question": "<string ending with ?>",
      "options":  ["A) ...", "B) ...", "C) ...", "D) ..."],  // exactly 4
      "answer":   "<single capital letter: A | B | C | D>"
    }
  ]
}

EXPLANATION RULES
─────────────────
1. Language & vocabulary must match the grade level:
   - Grade 1-4  → very simple words, short sentences, concrete examples
   - Grade 5-8  → moderate vocabulary, real-world analogies allowed
   - Grade 9-12 → formal academic language, technical terms with brief definitions
2. Structure MUST follow this three-part format:
   a. INTRO  — 1 paragraph introducing what the topic is and why it matters
   b. CONCEPTS — 2 to 4 numbered key points, each 1-3 sentences long
   c. SUMMARY — 1 closing sentence that ties all concepts together
3. Length target: 120-250 words for Grades 1-6; 200-400 words for Grades 7-12.
4. Do NOT use markdown headers inside the explanation string.

MCQ RULES
─────────
1. Generate EXACTLY 5 MCQs — no more, no fewer.
2. Every MCQ must be answerable solely from the explanation above.
   Do NOT include facts not covered in the explanation.
3. Each question must test a DIFFERENT concept from the explanation.
4. Options format: each string must begin with "A) ", "B) ", "C) ", or "D) ".
5. Only ONE option may be correct; the remaining three must be plausible
   distractors (not obviously wrong).
6. The `answer` field contains only the capital letter (A, B, C, or D).
7. Avoid trivial or trick questions.

EXAMPLE OUTPUT (Grade 4 / Topic: Photosynthesis)
─────────────────────────────────────────────────
{
  "explanation": "Plants are living things that make their own food using sunlight. ...[INTRO]... 1. Plants have a green pigment called chlorophyll that captures sunlight. 2. ... [CONCEPTS] ... In short, photosynthesis is nature's way of turning light into life. [SUMMARY]",
  "mcqs": [
    {
      "question": "What does a plant use to capture sunlight?",
      "options": ["A) Roots", "B) Chlorophyll", "C) Water", "D) Soil"],
      "answer": "B"
    },
    ...4 more MCQs...
  ]
}

REVISION INSTRUCTIONS (only when feedback is provided)
───────────────────────────────────────────────────────
- Fix ONLY the issues listed in the feedback block.
- Do NOT change parts of the content that were not flagged.
- Preserve the full three-part structure of the explanation.
- Re-generate only the affected MCQs if MCQ issues were flagged.
"""


REVIEWER_SYSTEM_PROMPT = """\
ROLE
────
You are a Senior Educational Content Reviewer responsible for quality-
assuring learning material before it reaches students.

OBJECTIVE
─────────
Evaluate the provided `Content` JSON object against the rubric below.
Return a `Review` JSON object with your verdict and, if failing, a
precise list of actionable issues for the generator to fix.

OUTPUT SCHEMA (strictly enforced)
──────────────────────────────────
{
  "status":   "pass" | "fail",
  "feedback": ["<issue 1>", "<issue 2>", ...]  // empty array on "pass"
}

EVALUATION RUBRIC
─────────────────
Grade Appropriateness (weight: HIGH)
  ✓ Vocabulary and sentence complexity suit the stated grade level.
  ✓ Concepts are neither too advanced nor too simplistic.
  ✗ Fail if: jargon unexplained for lower grades; or topic oversimplified
    to the point of inaccuracy for higher grades.

Explanation Quality (weight: HIGH)
  ✓ Follows the three-part structure: INTRO → CONCEPTS → SUMMARY.
  ✓ All key concepts of the topic are introduced before any question.
  ✓ Factually accurate.
  ✗ Fail if: missing a structural section; factual errors present;
    explanation is too short (<100 words) or padded without substance.

MCQ Quality (weight: HIGH)
  ✓ Exactly 5 MCQs present.
  ✓ Every question is answerable from the explanation alone.
  ✓ Each question targets a different concept.
  ✓ Options are formatted as "A) ...", "B) ...", "C) ...", "D) ...".
  ✓ `answer` is a single capital letter (A | B | C | D).
  ✓ Distractors are plausible (not obviously wrong).
  ✗ Fail if: fewer/more than 5 MCQs; any question introduces facts not
    in the explanation; duplicate concept coverage; malformed options or
    answer field; all distractors are obviously incorrect.

FEEDBACK FORMAT (on "fail")
────────────────────────────
Each item in `feedback` must:
  - Reference the specific component (e.g., "MCQ 3", "Explanation INTRO").
  - State the exact problem.
  - State the required fix.

Example feedback items:
  - "MCQ 2: question introduces the concept of mitosis which is not covered
     in the explanation. Rewrite to test only concepts present in the text."
  - "Explanation SUMMARY: missing closing summary sentence. Add a single
     sentence that ties all key concepts together."
  - "MCQ 4 options: option C is obviously wrong ('D) The Sun is cold').
     Replace with a plausible distractor."

DECISION RULE
─────────────
- Return "pass" only when ALL rubric criteria are satisfied.
- Return "fail" as soon as ANY criterion fails; list ALL issues found.
- Never return "pass" with non-empty feedback or "fail" with empty feedback.
"""

# ────────────────────────────────────────────────────────────────────────────
# Node Functions
# ────────────────────────────────────────────────────────────────────────────

def generator(state: State) -> dict:
    """Generate (or regenerate) educational content for the given state."""

    gen_llm = llm.with_structured_output(Content)

    # Build optional feedback block for retry runs
    feedback_block = ""
    if state.get("reviewer_output") and state["retry_count"] > 0:
        feedback_items = "\n".join(
            f"  - {fb}" for fb in state["reviewer_output"].feedback
        )
        feedback_block = (
            "\n⚠ REVISION REQUEST\n"
            "The previous content failed the quality review.\n"
            "Fix ONLY the issues listed below — do not change anything else:\n"
            f"{feedback_items}\n"
        )

    messages = [
        SystemMessage(content=GENERATOR_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Grade: {state['grade']}\n"
                f"Topic: {state['topic']}\n"
                f"{feedback_block}"
            )
        ),
    ]

    response = gen_llm.invoke(messages)
    return {"generator_output": response}


def reviewer(state: State) -> dict:
    """Review the generated content and return a pass/fail decision."""

    if state["generator_output"] is None:
        raise ValueError("reviewer node called before generator produced output.")

    review_llm = llm.with_structured_output(Review)

    messages = [
        SystemMessage(content=REVIEWER_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Grade: {state['grade']}\n"
                f"Topic: {state['topic']}\n\n"
                "Generated Content:\n"
                f"{state['generator_output'].model_dump_json(indent=2)}"
            )
        ),
    ]

    response = review_llm.invoke(messages)
    return {"reviewer_output": response}


# ────────────────────────────────────────────────────────────────────────────
# Routing Logic
# ────────────────────────────────────────────────────────────────────────────

MAX_RETRIES = 1  # Generator gets exactly one retry after a failed review


def route_after_review(state: State) -> str:
    """
    Decide whether to retry generation or end the pipeline.

    Returns:
        "retry" → send back to generator (only if retries remain)
        "end"   → terminate pipeline
    """
    review = state["reviewer_output"]

    if review.status == "pass":
        return "end"

    if review.status == "fail" and state["retry_count"] < MAX_RETRIES:
        return "retry"

    # Exhausted retries → surface best-effort output
    return "end"


# ────────────────────────────────────────────────────────────────────────────
# Graph Assembly
# ────────────────────────────────────────────────────────────────────────────

def build_agent() -> StateGraph:
    graph = StateGraph(State)

    graph.add_node("generator", generator)
    graph.add_node("reviewer", reviewer)

    graph.add_edge(START, "generator")
    graph.add_edge("generator", "reviewer")

    graph.add_conditional_edges(
        "reviewer",
        route_after_review,
        {
            "retry": "generator",
            "end": END,
        },
    )

    return graph.compile()


agent = build_agent()

# ────────────────────────────────────────────────────────────────────────────
# Helper — increment retry counter inside the generator (state mutation fix)
# ────────────────────────────────────────────────────────────────────────────
# NOTE: LangGraph state is immutable inside nodes; the retry counter is
# incremented via a dedicated thin wrapper so the router sees the new value.

_original_generator = generator


def generator(state: State) -> dict:  # noqa: F811  (intentional override)
    result = _original_generator(state)
    return {**result, "retry_count": state["retry_count"] + 1}


# Rebuild with the counter-aware generator
def build_agent() -> StateGraph:  # noqa: F811
    graph = StateGraph(State)
    graph.add_node("generator", generator)
    graph.add_node("reviewer", reviewer)
    graph.add_edge(START, "generator")
    graph.add_edge("generator", "reviewer")
    graph.add_conditional_edges(
        "reviewer",
        route_after_review,
        {"retry": "generator", "end": END},
    )
    return graph.compile()


agent = build_agent()

# ────────────────────────────────────────────────────────────────────────────
# Entry Point
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    initial_state: State = {
        "grade": 6,
        "topic": "The Water Cycle",
        "generator_output": None,
        "reviewer_output": None,
        "retry_count": 0,
    }

    final_state = agent.invoke(initial_state)

    content: Content = final_state["generator_output"]
    review: Review   = final_state["reviewer_output"]

    print(f"\n{'═' * 60}")
    print(f"  TOPIC : {initial_state['topic']}  |  GRADE : {initial_state['grade']}")
    print(f"  STATUS: {review.status.upper()}")
    print(f"{'═' * 60}\n")
    print("EXPLANATION\n" + "─" * 40)
    print(content.explanation)
    print("\nMCQs\n" + "─" * 40)
    for i, mcq in enumerate(content.mcqs, 1):
        print(f"\nQ{i}. {mcq.question}")
        for opt in mcq.options:
            print(f"    {opt}")
        print(f"    ✓ Answer: {mcq.answer}")

    if review.status == "fail":
        print("\nREVIEWER FEEDBACK (unresolved)\n" + "─" * 40)
        for fb in review.feedback:
            print(f"  • {fb}")
