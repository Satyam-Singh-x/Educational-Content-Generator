"""
Agentic Educational Content Pipeline using LangGraph + Groq (free tier).

Flow:
    START → generator → reviewer → (pass → END | fail → generator [once] → END)

Key design decisions:
    - Groq does not support .with_structured_output(), so all JSON parsing
      is handled manually via _parse_json() with a strict extraction strategy.
    - Prompts use a JSON-fence contract: the model is told to output ONLY a
      raw JSON block wrapped in ```json ... ``` so it is unambiguous to extract.
    - Pydantic models are used for validation AFTER parsing, not before.
"""

# ────────────────────────────────────────────────────────────────────────────
# Imports
# ────────────────────────────────────────────────────────────────────────────
import json
import os
import re
from typing import List, Literal, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI # Added
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError
from typing_extensions import TypedDict

# ────────────────────────────────────────────────────────────────────────────
# Environment — Streamlit Cloud (st.secrets) → local .env → shell env
# ────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────
# Environment & Keys
# ────────────────────────────────────────────────────────────────────────────

def _get_secret(key_name: str) -> str:
    """Helper to fetch keys from Streamlit secrets or environment."""
    try:
        import streamlit as st
        if key_name in st.secrets:
            return st.secrets[key_name]
    except Exception:
        pass
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
        
    return os.getenv(key_name, "")

GROQ_KEY = _get_secret("GROQ_API_KEY")
GOOGLE_KEY = _get_secret("GOOGLE_API_KEY")

if not GROQ_KEY and not GOOGLE_KEY:
    raise RuntimeError("Neither GROQ_API_KEY nor GOOGLE_API_KEY found.")

# ────────────────────────────────────────────────────────────────────────────
# LLM Initializations
# ────────────────────────────────────────────────────────────────────────────

# Primary Model
groq_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    max_tokens=4096,
    api_key=GROQ_KEY
)

# Fallback Model
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", # Note: Using 2.0 as 2.5 is not yet a standard release string
    temperature=0.3,
    google_api_key=GOOGLE_KEY
)

def smart_invoke(messages):
    """Try Groq first, fallback to Gemini if Groq fails (e.g., restricted account)."""
    try:
        if not GROQ_KEY:
            raise ValueError("Groq Key missing")
        return groq_llm.invoke(messages)
    except Exception as e:
        print(f"--- Groq failed, switching to Gemini fallback. Error: {e} ---")
        return gemini_llm.invoke(messages)
# ────────────────────────────────────────────────────────────────────────────
# Data Schemas  (Pydantic — used for validation after manual JSON parse)
# ────────────────────────────────────────────────────────────────────────────

class MCQ(BaseModel):
    """A single multiple-choice question."""
    question: str = Field(description="Question text ending with '?'")
    options: List[str] = Field(description="Exactly 4 options: ['A) ...', 'B) ...', 'C) ...', 'D) ...']")
    answer: str = Field(description="Single capital letter: A | B | C | D")


class Content(BaseModel):
    """Full educational content package produced by the generator node."""
    explanation: str = Field(description="Structured grade-appropriate explanation (INTRO + CONCEPTS + SUMMARY)")
    mcqs: List[MCQ] = Field(description="Exactly 5 MCQs derived from the explanation")


class Review(BaseModel):
    """Structured review verdict produced by the reviewer node."""
    status: Literal["pass", "fail"] = Field(description="'pass' or 'fail'")
    feedback: List[str] = Field(default_factory=list, description="Empty on pass; actionable issues on fail")


class State(TypedDict):
    """Shared mutable pipeline state passed between all LangGraph nodes."""
    grade: int
    topic: str
    generator_output: Optional[Content]
    reviewer_output: Optional[Review]
    retry_count: int


# ────────────────────────────────────────────────────────────────────────────
# JSON Parser  (replaces with_structured_output for Groq)
# ────────────────────────────────────────────────────────────────────────────

def _parse_json(raw: str, label: str) -> dict:
    """
    Extract and parse a JSON object from a model response.

    Strategy (in order):
      1. Pull content from a ```json ... ``` fence if present.
      2. Find the first balanced { ... } block.
      3. Attempt raw string parse as last resort.

    Raises ValueError with diagnostics on complete failure.
    """
    # Step 1: JSON fence
    fence_match = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Step 2: first balanced { ... } block
    start = raw.find("{")
    if start != -1:
        depth = 0
        for i, ch in enumerate(raw[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = raw[start: i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break

    # Step 3: raw fallback
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass

    raise ValueError(
        f"[{label}] Could not extract valid JSON from model response.\n"
        f"Raw output (first 500 chars):\n{raw[:500]}"
    )


# ────────────────────────────────────────────────────────────────────────────
# System Prompts
# ────────────────────────────────────────────────────────────────────────────

GENERATOR_SYSTEM_PROMPT = """\
ROLE
You are an expert Educational Content Generation Agent specialised in
creating structured, curriculum-aligned learning material for students
from Grade 1 through Grade 12.

OBJECTIVE
Given a grade level and a topic, produce educational content that:
  1. Explains the topic clearly at the correct grade level.
  2. Tests understanding with exactly 5 multiple-choice questions (MCQs).

OUTPUT FORMAT  CRITICAL - READ CAREFULLY
You MUST respond with ONLY a single JSON object wrapped in a json fence.
Do NOT write any text before or after the fence.
Do NOT include explanations, apologies, or commentary outside the fence.
The entire response must look exactly like this:

```json
{
  "explanation": "<structured explanation string>",
  "mcqs": [
    {
      "question": "<question ending with ?>",
      "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
      "answer": "<A or B or C or D>"
    },
    { ... },
    { ... },
    { ... },
    { ... }
  ]
}
```

EXPLANATION RULES
The explanation string must follow this exact three-part structure:

  PART 1 - INTRO (1 paragraph)
    Introduce the topic: what it is and why it matters.

  PART 2 - CONCEPTS (2 to 4 numbered points)
    Each point is 1-3 sentences. Number them: "1. ...", "2. ...", etc.

  PART 3 - SUMMARY (1 sentence)
    A closing sentence that ties all concepts together.

Language calibration by grade:
  Grade 1-4  : Very simple words, short sentences, everyday examples. No jargon.
  Grade 5-8  : Moderate vocabulary. Real-world analogies encouraged.
               Technical terms must be defined in plain language.
  Grade 9-12 : Formal academic language. Technical terms with brief inline
               definitions. Expect prior subject knowledge.

Length targets:
  Grade 1-6  : 120-250 words
  Grade 7-12 : 200-400 words

STRICT PROHIBITIONS for explanation:
  Do NOT use markdown headers (##, ###, **, etc.) inside the string.
  Do NOT include newline escape sequences as literal text.
  Do NOT reference the MCQs inside the explanation.

MCQ RULES
1. Generate EXACTLY 5 MCQ objects - no more, no fewer.
2. Each MCQ must test a DIFFERENT concept from the explanation.
3. Every question must be answerable solely from the explanation.
   Do NOT introduce any fact not present in the explanation text.
4. Options MUST be an array of exactly 4 strings, each starting with
   "A) ", "B) ", "C) ", or "D) " (capital letter, closing parenthesis, space).
5. The "answer" field contains ONLY the single capital letter (A, B, C, or D).
   Do NOT write "A)" or "Option A" - just the bare letter.
6. Exactly ONE option is correct. The other three are plausible distractors
   that are wrong but not obviously absurd.
7. Avoid trivial or trick questions.

COMPLETE EXAMPLE (Grade 4 / Topic: Photosynthesis)

```json
{
  "explanation": "Plants are amazing living things that can make their own food using sunlight. This process is called photosynthesis, and it is very important because it also produces the oxygen we breathe. 1. Plants have a green pigment called chlorophyll found in their leaves. Chlorophyll captures energy from sunlight and uses it to power food-making. 2. During photosynthesis, plants take in carbon dioxide from the air through tiny pores called stomata, and absorb water through their roots. 3. Using the captured sunlight energy, plants convert carbon dioxide and water into glucose (a type of sugar) and release oxygen as a by-product. 4. The glucose made by plants is used as energy for growing, flowering, and producing seeds. In short, photosynthesis is nature's way of turning sunlight, water, and air into food and oxygen that support almost all life on Earth.",
  "mcqs": [
    {
      "question": "What is the name of the green pigment in plants that captures sunlight?",
      "options": ["A) Glucose", "B) Chlorophyll", "C) Stomata", "D) Carbon dioxide"],
      "answer": "B"
    },
    {
      "question": "Through which tiny pores do plants absorb carbon dioxide from the air?",
      "options": ["A) Roots", "B) Chlorophyll", "C) Stomata", "D) Seeds"],
      "answer": "C"
    },
    {
      "question": "What gas do plants release as a by-product of photosynthesis?",
      "options": ["A) Carbon dioxide", "B) Nitrogen", "C) Hydrogen", "D) Oxygen"],
      "answer": "D"
    },
    {
      "question": "What do plants primarily use the glucose produced during photosynthesis for?",
      "options": ["A) Absorbing water", "B) Energy for growth", "C) Capturing sunlight", "D) Releasing carbon dioxide"],
      "answer": "B"
    },
    {
      "question": "Which two raw materials do plants use in photosynthesis?",
      "options": ["A) Oxygen and glucose", "B) Nitrogen and stomata", "C) Carbon dioxide and water", "D) Chlorophyll and seeds"],
      "answer": "C"
    }
  ]
}
```

REVISION MODE (active only when a REVISION REQUEST block is present)
When you receive a REVISION REQUEST:
  - Fix ONLY the specific issues listed. Do not alter anything else.
  - Preserve the three-part explanation structure.
  - Re-generate only the MCQs that were explicitly flagged.
  - Output the full corrected JSON (not just the changed parts).
"""


REVIEWER_SYSTEM_PROMPT = """\
ROLE
You are a Senior Educational Content Reviewer responsible for quality-
assuring AI-generated learning material before it reaches students.

OBJECTIVE
Evaluate the provided Content JSON against the rubric below.
Return a structured verdict as a JSON object.

OUTPUT FORMAT  CRITICAL - READ CAREFULLY
You MUST respond with ONLY a single JSON object wrapped in a json fence.
Do NOT write any text before or after the fence.
Do NOT include explanations, preamble, or commentary outside the fence.

On pass:
```json
{
  "status": "pass",
  "feedback": []
}
```

On fail:
```json
{
  "status": "fail",
  "feedback": [
    "<specific actionable issue 1>",
    "<specific actionable issue 2>"
  ]
}
```

FIELD RULES:
  "status"   : exactly the string "pass" or "fail". No other values allowed.
  "feedback" : MUST be an empty array [] when status is "pass".
               MUST be a non-empty array of strings when status is "fail".
               Never return "pass" with non-empty feedback.
               Never return "fail" with an empty feedback array.

EVALUATION RUBRIC

CRITERION 1 - Grade Appropriateness  [HIGH WEIGHT]
  PASS:
    Vocabulary and sentence complexity match the stated grade level.
    Concepts are neither too advanced nor too simplistic.
  FAIL:
    Unexplained jargon for Grades 1-6.
    Topic oversimplified to the point of factual inaccuracy for Grades 9-12.

CRITERION 2 - Explanation Quality  [HIGH WEIGHT]
  PASS:
    Contains all three parts: INTRO paragraph, numbered CONCEPTS, SUMMARY.
    All concepts tested in MCQs are introduced in the explanation first.
    Content is factually accurate.
    Word count is within grade-appropriate range.
  FAIL:
    Any structural part (INTRO, CONCEPTS, or SUMMARY) is missing.
    Contains factual errors.
    Fewer than 100 words total.
    Contains markdown headers or formatting artifacts.

CRITERION 3 - MCQ Quality  [HIGH WEIGHT]
  PASS:
    Exactly 5 MCQ objects present.
    Every question is fully answerable from the explanation alone.
    Each question tests a different concept.
    All options start with "A) ", "B) ", "C) ", or "D) ".
    The "answer" field is a single capital letter (A, B, C, or D).
    Distractors are plausible (not obviously wrong or absurd).
  FAIL:
    Fewer or more than 5 MCQs.
    Any question introduces a fact not covered in the explanation.
    Two or more questions test the same concept.
    Any option is malformed (missing prefix, wrong format).
    "answer" field is not a bare single capital letter.
    Any distractor is obviously wrong (nonsensical or irrelevant).

FEEDBACK FORMAT (required when status = fail)
Each feedback string must follow this pattern:
  "[Component]: [exact problem]. [required fix]."

Component examples: "MCQ 3", "Explanation INTRO", "MCQ 5 options", "Explanation SUMMARY"

Good feedback examples:
  "MCQ 2: The question references mitosis, which is not covered in the explanation. Rewrite to test a concept present in the text."
  "Explanation SUMMARY: The closing summary sentence is missing. Add one sentence that ties all key concepts together."
  "MCQ 4 options: Option D is obviously wrong. Replace with a plausible distractor."

DECISION RULE
  Return "pass" ONLY when every single criterion above is fully satisfied.
  Return "fail" the moment ANY criterion fails; list ALL issues found.
  Do not be lenient. Partial compliance = fail.
"""


# ────────────────────────────────────────────────────────────────────────────
# Node Functions
# ────────────────────────────────────────────────────────────────────────────

def generator(state: State) -> dict:
    """
    Generate (or regenerate on retry) educational content.
    Calls Groq, extracts JSON manually, validates with Pydantic.
    """
    feedback_block = ""
    if state.get("reviewer_output") and state["retry_count"] > 0:
        feedback_items = "\n".join(
            f"  - {fb}" for fb in state["reviewer_output"].feedback
        )
        feedback_block = (
            "\nREVISION REQUEST\n"
            "The previous output failed review. Fix ONLY the issues below.\n"
            "Do not alter anything that was not flagged.\n"
            f"{feedback_items}\n"
        )

    messages = [
        SystemMessage(content=GENERATOR_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Grade: {state['grade']}\n"
                f"Topic: {state['topic']}\n"
                f"{feedback_block}"
                "\nRemember: respond with ONLY the ```json ... ``` block. Nothing else."
            )
        ),
    ]

    raw = llm.invoke(messages).content
    parsed = _parse_json(raw, label="generator")

    try:
        content = Content(**parsed)
    except ValidationError as e:
        raise ValueError(f"[generator] Pydantic validation failed:\n{e}\nParsed: {parsed}") from e

    return {"generator_output": content, "retry_count": state["retry_count"] + 1}


def reviewer(state: State) -> dict:
    """
    Review the generated content. Calls Groq, extracts JSON manually,
    validates with Pydantic.
    """
    if state["generator_output"] is None:
        raise ValueError("reviewer node called before generator produced output.")

    content_json = state["generator_output"].model_dump_json(indent=2)

    messages = [
        SystemMessage(content=REVIEWER_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Grade: {state['grade']}\n"
                f"Topic: {state['topic']}\n\n"
                "Content to review:\n"
                f"{content_json}\n\n"
                "Remember: respond with ONLY the ```json ... ``` block. Nothing else."
            )
        ),
    ]

    raw = llm.invoke(messages).content
    parsed = _parse_json(raw, label="reviewer")

    try:
        review = Review(**parsed)
    except ValidationError as e:
        raise ValueError(f"[reviewer] Pydantic validation failed:\n{e}\nParsed: {parsed}") from e

    return {"reviewer_output": review}


# ────────────────────────────────────────────────────────────────────────────
# Routing Logic
# ────────────────────────────────────────────────────────────────────────────

MAX_RETRIES = 1


def route_after_review(state: State) -> str:
    """
    Pure routing function - no state mutation.
    retry_count is already incremented inside the generator node.
    """
    review = state["reviewer_output"]

    if review.status == "pass":
        return "end"

    if review.status == "fail" and state["retry_count"] < MAX_RETRIES:
        return "retry"

    return "end"


# ────────────────────────────────────────────────────────────────────────────
# Graph Assembly
# ────────────────────────────────────────────────────────────────────────────

def build_agent():
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
            "end":   END,
        },
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
    review:  Review  = final_state["reviewer_output"]

    print(f"\n{'=' * 60}")
    print(f"  TOPIC  : {initial_state['topic']}  |  GRADE : {initial_state['grade']}")
    print(f"  STATUS : {review.status.upper()}")
    print(f"  RETRIES: {final_state['retry_count']}")
    print(f"{'=' * 60}\n")
    print("EXPLANATION\n" + "-" * 40)
    print(content.explanation)
    print("\nMCQs\n" + "-" * 40)
    for i, mcq in enumerate(content.mcqs, 1):
        print(f"\nQ{i}. {mcq.question}")
        for opt in mcq.options:
            print(f"    {opt}")
        print(f"    Answer: {mcq.answer}")

    if review.status == "fail":
        print("\nREVIEWER FEEDBACK (unresolved)\n" + "-" * 40)
        for fb in review.feedback:
            print(f"  - {fb}")
