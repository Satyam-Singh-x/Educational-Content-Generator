📘 Educational Content Generation Agent

LangGraph · Gemini · Streamlit

Live Demo: 

Overview

This project implements an AI-powered educational content generation system designed to create grade-appropriate explanations and MCQs for school students.
The system uses a graph-based agent architecture with automated review and controlled refinement to ensure quality and structure.

The application is built using:

LangGraph for agent orchestration

Gemini (Google Generative AI) for content generation and evaluation

Pydantic for strict schema enforcement

Streamlit for an interactive web interface

Key Features

Structured Content Generation

Generates explanations and multiple-choice questions (MCQs)

Output strictly follows predefined Pydantic schemas

Automated Review Agent

Reviews generated content for:

Grade appropriateness

Concept coverage

MCQ correctness and clarity

Returns a deterministic pass / fail decision with actionable feedback

Controlled Refinement

If content fails review, the system performs one guided retry

Reviewer feedback is injected into regeneration

Prevents infinite loops or uncontrolled retries

Transparent Agent Inspector

View structured outputs from each node

Inspect reviewer feedback and retry count

Execution logs for debugging and evaluation

Deployable Streamlit UI

Clean, user-friendly interface

Ready for deployment on Streamlit Cloud

System Architecture

The agent workflow is implemented using LangGraph:

START

  ↓
  
Generator Node

  ↓
  
Reviewer Node

  ↓
(pass) ─────────▶END
  ↓
  
(fail + retry_count = 0)

  ↓
  
Generator (refined)

  ↓
  
Reviewer

  ↓
  
END



Core Nodes

Generator

Produces structured educational content using Gemini

Optionally refines output based on reviewer feedback

Reviewer

Evaluates generated content against explicit criteria

Outputs structured review results

Decision Logic

Routes execution based on review outcome

Enforces a single retry policy

Tech Stack

Python 3.10+

LangGraph

LangChain

Google Gemini (Flash)

Pydantic

Streamlit

python-dotenv

Project Structure
.
├── app.py                  # Streamlit UI

├── Agent_orchestration.py  # LangGraph agent and nodes

├── requirements.txt

└── README.md

Setup & Installation

1️⃣ Clone the Repository

git clone https://github.com/Satyam-Singh-x/Educational-Content-Generator/
cd Educational-Content-Generator

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Configure API Key

Create a .env file (local development only):

GOOGLE_API_KEY=your_api_key_here


For Streamlit Cloud, add the key under:

App Settings → Secrets

Running the Application

streamlit run app.py


Then open the local URL provided by Streamlit.

Usage

Select a grade level

Enter a topic

Click Generate Content

View:

Final explanation and MCQs

Agent Inspector tab with:

Generator output

Reviewer feedback

Retry count

Execution logs

Download structured JSON output if needed

Design Principles

Single Source of Truth

generator_output always holds the latest structured content

Strict Schema Enforcement

All LLM outputs are validated using Pydantic models

Deterministic Control Flow

No memory agents

No uncontrolled loops

Clear retry limits

Transparency

Reviewer decisions and feedback are exposed


No hidden post-processing

Future Improvements (Optional)

Add grade-specific reviewer strictness

Support additional question types

Add authentication for multi-user usage

Persist logs and outputs for analytics

Author

Satyam
AI / ML Developer
Built as part of an AI Developer Assessment
