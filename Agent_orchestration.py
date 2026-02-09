from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import TypedDict, List, Literal, Optional
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
import os

# -------------------- ENV SETUP --------------------
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# -------------------- LLM --------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)

# -------------------- SCHEMAS ----------------------

class MCQ(BaseModel):
    question: str
    options: List[str]
    answer: str


class Content(BaseModel):
    explanation: str
    mcqs: List[MCQ]


class Review(BaseModel):
    status: Literal["pass", "fail"]
    feedback: List[str]


class State(TypedDict):
    grade: int
    topic: str
    generator_output: Optional[Content]
    reviewer_output: Optional[Review]
    retry_count: int


# -------------------- PROMPTS ----------------------

GENERATOR_SYSTEM_PROMPT = """
You are an educational content generation agent.

Generate structured educational content for the given grade and topic.
Follow the provided output schema strictly.

Rules:
- Use grade-appropriate language
- Introduce concepts before questions
- Keep explanations concise
- MCQs must be based only on the explanation
"""

REVIEWER_SYSTEM_PROMPT = """
You are an educational content reviewer.

Review the generated content for:
- Grade appropriateness
- Concept coverage
- MCQ correctness and clarity

If all criteria are met, return status "pass".
Otherwise, return status "fail" with actionable feedback.
"""


# -------------------- NODES ------------------------

def generator(state: State) -> dict:
    gen_llm = llm.with_structured_output(Content)

    feedback = (
        state["reviewer_output"].feedback
        if state.get("reviewer_output")
        else None
    )

    feedback_block = ""
    if feedback and state["retry_count"] == 0:
        feedback_block = (
            "\nThe previous output failed review.\n"
            "Fix only the issues listed below:\n"
            + "\n".join(f"- {fb}" for fb in feedback)
        )

    messages = [
        SystemMessage(content=GENERATOR_SYSTEM_PROMPT),
        HumanMessage(
            content=f"""
Grade: {state['grade']}
Topic: {state['topic']}
{feedback_block}
"""
        )
    ]

    response = gen_llm.invoke(messages)
    return {"generator_output": response}


def reviewer(state: State) -> dict:
    if state["generator_output"] is None:
        raise ValueError("Reviewer called without generator output")

    review_llm = llm.with_structured_output(Review)

    messages = [
        SystemMessage(content=REVIEWER_SYSTEM_PROMPT),
        HumanMessage(
            content=f"""
Grade: {state['grade']}
Topic: {state['topic']}

Generated Content:
{state['generator_output'].model_dump_json(indent=2)}
"""
        )
    ]

    response = review_llm.invoke(messages)
    return {"reviewer_output": response}


def decide_next_step(state: State) -> str:
    review = state["reviewer_output"]

    if review.status == "pass":
        return "end"

    if review.status == "fail" and state["retry_count"] == 0:
        state["retry_count"] += 1
        return "retry"

    return "end"


# -------------------- GRAPH ------------------------

g = StateGraph(State)

g.add_node("generator", generator)
g.add_node("review", reviewer)

g.add_edge(START, "generator")
g.add_edge("generator", "review")

g.add_conditional_edges(
    "review",
    decide_next_step,
    {
        "retry": "generator",
        "end": END
    }
)

agent = g.compile()
