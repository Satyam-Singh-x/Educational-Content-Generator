<div align="center">

# ⚡ Edynapse — AI Educational Content Engine






**Curriculum-aligned educational content — generated, reviewed, and refined autonomously.**

Live demo: https://educational-content-generator-bysatyam.streamlit.app/

---

</div>

## 📌 Overview

**Edynapse** is a B2B EdTech platform that uses an **agentic AI pipeline** to generate structured educational content for Grades 1–12. Built on **LangGraph** and powered by **Groq's LLaMA 3.3 70B** (free tier), it autonomously generates explanations and MCQs, reviews them against a quality rubric, and self-corrects on failure — all without human intervention.

> 🆓 This edition runs entirely on **Groq's free API** — no billing required.

---

## 🏗️ Architecture
```
START
  │
  ▼
┌─────────────┐     parsed JSON     ┌──────────────┐
│  Generator  │ ──── Content ──────▶│   Reviewer   │
│   Node      │                     │    Node      │
└─────────────┘                     └──────┬───────┘
       ▲                                   │
       │            fail + retries left    │ pass
       │◀──────────────────────────────────┤
       │                                   │
  (retry once)                             ▼
                                          END
```

| Node | Role | Model |
|---|---|---|
| **Generator** | Produces `explanation` + 5 MCQs via JSON-fence prompting | LLaMA 3.3 70B (Groq) |
| **Reviewer** | Evaluates against a 3-criterion rubric, returns `pass`/`fail` | LLaMA 3.3 70B (Groq) |
| **Router** | Sends back to Generator once on `fail`; ends on `pass` or retry exhaustion | — |

---

## ✨ Features

- 🧠 **Agentic self-correction** — automatically retries once on review failure
- 🔧 **Manual JSON parsing** — robust 3-stage extractor replaces `with_structured_output()` (unsupported on Groq)
- ✅ **Pydantic post-validation** — all model output validated after parsing
- 🎓 **Grade-aware generation** — calibrated language for Grades 1–12
- 📝 **5 MCQs per lesson** — each testing a different concept from the explanation
- 🔍 **Rubric-based review** — grade appropriateness, explanation quality, MCQ quality
- 📥 **Markdown export** — download a beautifully structured `.md` file
- 🖥️ **Professional B2B UI** — dark theme Streamlit dashboard with KPI stats
- 🆓 **100% free to run** — Groq free tier, no credit card needed

---

## 🗂️ Project Structure
```
edynapse-groq/
│
├── app.py                  # Streamlit frontend (B2B dashboard UI)
├── Agent_orchestration.py  # LangGraph pipeline (generator + reviewer + router)
├── requirements.txt        # Python dependencies
├── .env                    # Local secrets (never commit this)
├── .env.example            # Template for environment variables
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- A Groq API key (free) → [Get one here](https://console.groq.com/keys)

### 1. Clone the repository
```bash
git clone https://github.com/Satyam-Singh-x/Educational-Content-Generator.git
cd Educational-Content-Generator
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
```bash
cp .env.example .env
```

Edit `.env`:
```env
GROQ_API_KEY="your-groq-api-key-here"
```

### 4. Run locally
```bash
streamlit run app.py
```

---

## ☁️ Deploying to Streamlit Cloud

1. Push your repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo and set `app.py` as the main file
4. Open **Advanced settings → Secrets** and add:
```toml
GROQ_API_KEY = "your-groq-api-key-here"
```

5. Click **Deploy** ✅

> ⚠️ Never commit your `.env` file or API key to GitHub.

---

## 📦 Dependencies
```
streamlit
langgraph
langchain
langchain-groq
groq
python-dotenv
pydantic
langchain-core
```

---

## 🔑 Environment Variables

| Variable | Description | Required |
|---|---|---|
| `GROQ_API_KEY` | Groq API key for LLaMA 3.3 70B access | ✅ Yes |

---

## 🧠 How the Pipeline Works

### Why Manual JSON Parsing?

Groq does not support LangChain's `with_structured_output()`. Instead, Edynapse uses a **JSON-fence prompting contract** combined with a **3-stage extractor**:
```
Stage 1 → Extract from ```json ... ``` fence  (primary — always enforced in prompt)
Stage 2 → Find first balanced { ... } brace block  (fallback)
Stage 3 → Attempt raw string parse  (last resort)
```

All extracted dicts are then validated through **Pydantic models** before entering pipeline state.

### Generator Node
Receives `grade` and `topic`, sends a JSON-fence prompt to **LLaMA 3.3 70B on Groq**. Returns a validated `Content` object containing:
- A three-part explanation (INTRO → CONCEPTS → SUMMARY)
- Exactly 5 MCQs, each testing a different concept

### Reviewer Node
Receives the `Content` object, evaluates it against a 3-criterion rubric:
1. **Grade Appropriateness** — language and vocabulary match the grade
2. **Explanation Quality** — structure, accuracy, and word count
3. **MCQ Quality** — format, concept coverage, distractor plausibility

Returns a validated `Review` object with `status: "pass" | "fail"` and component-level feedback like:
```
"MCQ 2: References mitosis which is not in the explanation. Rewrite to test a covered concept."
```

### Router
- `pass` → pipeline ends, output surfaced to UI
- `fail` + retries remaining → feedback injected into Generator prompt
- `fail` + retries exhausted → best-effort output surfaced to UI

> The retry counter is incremented inside the **Generator node's return dict**, not the router — respecting LangGraph's immutable state contract.

---

## 📸 UI Overview

| Section | Description |
|---|---|
| **Hero** | Edynapse branding, product tagline |
| **Left Panel** | Grade selector, topic input, pipeline diagram |
| **Stat Row** | Generation time, retries used, MCQ count, review status |
| **Output Tab** | Explanation card + MCQ cards with highlighted correct answers |
| **Inspector Tab** | Raw JSON from generator + reviewer, execution logs |
| **Export** | One-click Markdown download |

---

## ⚖️ Groq vs Gemini Edition

| Feature | Groq Edition | Gemini Edition |
|---|---|---|
| Model | LLaMA 3.3 70B | Gemini 2.5 Flash |
| Structured output | Manual JSON parsing | `with_structured_output()` |
| Cost | **Free** | Pay-per-use |
| Speed | Ultra-fast inference | Fast |
| JSON reliability | Prompt-engineered fence contract | Native schema binding |
| Best for | Prototyping, demos, free deployment | Production, higher reliability |

---

## 📄 License

MIT License © 2024 Edynapse

---

<div align="center">
Built with ⚡ by the Edynapse team
</div>
