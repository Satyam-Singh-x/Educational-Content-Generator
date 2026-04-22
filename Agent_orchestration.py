import json
import os
import re
from typing import List, Literal, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError
from typing_extensions import TypedDict

# ────────────────────────────────────────────────────────────────────────────
# Environment & Keys Resolution
# ────────────────────────────────────────────────────────────────────────────

def _get_secret(key_name: str) -> str:
    """Helper to fetch keys from Streamlit secrets, then environment, then .env."""
    # 1. Try Streamlit Secrets
    try:
        import streamlit as st
        if key_name in st.secrets:
            return st.secrets[key_name]
    except Exception:
        pass
    
    # 2. Try .env file (Local)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
        
    # 3. Try Environment Variables
    return os.getenv(key_name, "")

GROQ_KEY = _get_secret("GROQ_API_KEY")
GOOGLE_KEY = _get_secret("GOOGLE_API_KEY")

if not GROQ_KEY and not GOOGLE_KEY:
    raise RuntimeError(
        "Missing API Keys. Please provide GROQ_API_KEY or GOOGLE_API_KEY "
        "in Streamlit Secrets or a .env file."
    )

# ────────────────────────────────────────────────────────────────────────────
# LLM Initializations
# ────────────────────────────────────────────────────────────────────────────

# Primary: Groq Llama 3.3
groq_llm = None
if GROQ_KEY:
    groq_llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=4096,
        api_key=GROQ_KEY
    )

# Fallback: Gemini 2.0 Flash
gemini_llm = None
if GOOGLE_KEY:
    gemini_llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,
        google_api_key=GOOGLE_KEY
    )

def smart_invoke(messages):
    """
    Tries Groq first. If it fails (Rate limits, Account Restricted, etc.),
    it falls back to Gemini automatically.
    """
    if groq_llm:
        try:
            return groq_llm.invoke(messages)
        except Exception as e:
            print(f"--- Groq invocation failed. Falling back to Gemini. Error: {e} ---")
    
    if gemini_llm:
        return gemini_llm.invoke(messages)
    
    raise RuntimeError("No working LLM available to handle the request.")

# ────────────────────────────────────────────────────────────────────────────
# Data Schemas (Pydantic)
# ────────────────────────────────────────────────────────────────────────────

class MCQ(BaseModel):
    question: str
    options: List[str]
    answer: str

class Content(BaseModel):
    explanation: str
    mcqs: List[MCQ]

class Review(BaseModel):
    status: Literal["pass", "fail"]
    feedback: List[str] = Field(default_factory=list)

class State(TypedDict):
    grade: int
    topic: str
    generator_output: Optional[Content]
    reviewer_output: Optional[Review]
    retry_count: int

# ────────────────────────────────────────────────────────────────────────────
# Utils: JSON Parser
# ────────────────────────────────────────────────────────────────────────────

def _parse_json(raw: str, label: str) -> dict:
    # Try regex for code block
    fence_match = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
    candidate = fence_match.group(1).strip() if fence_match else raw.strip()
    
    # Try to find the first '{' and last '}' if simple parse fails
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(candidate[start:end+1])
            except:
                pass
    
    raise ValueError(f"[{label}] Failed to extract JSON from: {raw[:200]}...")

# ────────────────────────────────────────────────────────────────────────────
# System Prompts (Shortened for space, keep your original full prompts here)
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
    feedback_block = ""
    if state.get("reviewer_output") and state["retry_count"] > 0:
        fb_text = "\n".join(state["reviewer_output"].feedback)
        feedback_block = f"\nREVISION REQUEST: Fix these issues:\n{fb_text}"

    messages = [
        SystemMessage(content=GENERATOR_SYSTEM_PROMPT),
        HumanMessage(content=f"Grade: {state['grade']}\nTopic: {state['topic']}{feedback_block}")
    ]

    # FIX: Use smart_invoke instead of llm.invoke
    response = smart_invoke(messages)
    parsed = _parse_json(response.content, label="generator")
    
    return {
        "generator_output": Content(**parsed), 
        "retry_count": state["retry_count"] + 1
    }

def reviewer(state: State) -> dict:
    content_json = state["generator_output"].model_dump_json()
    messages = [
        SystemMessage(content=REVIEWER_SYSTEM_PROMPT),
        HumanMessage(content=f"Review this content for Grade {state['grade']}:\n{content_json}")
    ]

    # FIX: Use smart_invoke instead of llm.invoke
    response = smart_invoke(messages)
    parsed = _parse_json(response.content, label="reviewer")
    
    return {"reviewer_output": Review(**parsed)}

# ────────────────────────────────────────────────────────────────────────────
# Graph Logic
# ────────────────────────────────────────────────────────────────────────────

def route_after_review(state: State) -> str:
    if state["reviewer_output"].status == "pass":
        return "end"
    return "retry" if state["retry_count"] < 2 else "end" # Allowing 1 retry

def build_agent():
    workflow = StateGraph(State)
    workflow.add_node("generator", generator)
    workflow.add_node("reviewer", reviewer)
    
    workflow.add_edge(START, "generator")
    workflow.add_edge("generator", "reviewer")
    
    workflow.add_conditional_edges(
        "reviewer",
        route_after_review,
        {"retry": "generator", "end": END}
    )
    
    return workflow.compile()

agent = build_agent()

# ────────────────────────────────────────────────────────────────────────────
# Entry Point (Testing)
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_state: State = {
        "grade": 7,
        "topic": "Photosynthesis",
        "generator_output": None,
        "reviewer_output": None,
        "retry_count": 0
    }
    result = agent.invoke(test_state)
    print("Final Status:", result["reviewer_output"].status)
    print("Explanation:", result["generator_output"].explanation[:100], "...")
