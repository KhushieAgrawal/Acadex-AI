# Acadex AI – Agentic Academic Intelligence & Intervention System

## Overview

Acadex AI is an agentic AI-powered academic monitoring and intervention system designed to identify students who may be at academic risk and recommend timely interventions. It uses multiple AI agents to analyze student performance, evaluate risk factors, validate recommendations, and support faculty decision-making.

The project demonstrates how agentic AI can be applied in the education domain to support data-driven academic monitoring using synthetic student data.

## Problem Statement

Educational institutions often face challenges in identifying students who are experiencing academic difficulties at an early stage. Student information, such as attendance, assessment scores, and assignment completion, may require manual analysis, making timely intervention difficult.

Acadex AI addresses this challenge by analyzing academic indicators and generating risk assessments and actionable recommendations for faculty members.

## Objectives

- Monitor student attendance, academic performance, and assignment completion.
- Identify students who may require academic support.
- Use specialized AI agents to analyze data and generate recommendations.
- Validate outputs before presenting them to faculty.
- Provide an interactive dashboard for exploring student data and running the agent workflow.
- Demonstrate responsible human-in-the-loop academic intervention.

## Key Features

- **Student Dashboard:** Displays student statistics and academic indicators.
- **Risk Assessment:** Identifies students with high, moderate, or low academic risk.
- **Multi-Agent Workflow:** Uses specialized agents for planning, research, execution, critique, software-readiness checks, and business automation.
- **Gemini Integration:** Uses Google's Gemini model for supported agent tasks.
- **Synthetic Dataset:** Includes 500 synthetic student records for proof-of-concept testing.
- **Output Validation:** Checks workflow results before presenting recommendations.
- **Human-in-the-Loop:** Keeps intervention messages as drafts requiring faculty review.

## Agent Architecture

The system consists of the following components:

1. **Planner Agent:** Organizes the workflow and identifies the analysis required.
2. **Research Agent:** Examines academic indicators and relevant student information.
3. **Executor Agent:** Applies deterministic Python rules to calculate academic risk and generate structured results.
4. **Critic Agent:** Reviews the generated analysis and recommendations for consistency.
5. **Software Development Agent:** Checks dataset structure, required fields, and record identifiers.
6. **Business Automation Agent:** Prepares actionable intervention suggestions for faculty review.

The agents work together through an orchestrated workflow. Gemini supports language-based agent tasks, while Python performs rule-based calculations. The system also includes fallback behavior for cases where the language model is unavailable.

## Technology Stack

- **Programming Language:** Python
- **Frontend:** Streamlit
- **Backend API:** FastAPI
- **AI Integration:** Google Gemini
- **Data Processing:** Pandas
- **API Testing:** FastAPI Swagger UI
- **Dataset:** Synthetic academic student records

## System Architecture

The workflow follows this sequence:

1. Student records are loaded from a CSV dataset.
2. The Streamlit dashboard allows users to explore student information.
3. The dashboard communicates with the FastAPI backend.
4. The backend orchestrates the specialized agents.
5. Python-based rules and Gemini-supported analysis generate the assessment.
6. Validation checks are performed on the results.
7. The system displays the risk assessment and recommended intervention for faculty review.

Refer to the architecture diagram in the `docs` folder for a visual representation of the workflow.

## Sample Output

For an example student record, the system may identify:

- **Risk Level:** Moderate
- **Attendance:** 69.9%
- **Current Average:** 67.9
- **Previous Score:** 54.7
- **Score Change:** +13.2
- **Assignment Completion:** 7 out of 10
- **Suggested Action:** Faculty check-in during the current week

These values are illustrative results from the synthetic dataset and demonstrate the system's intended workflow. They are not real student records or validated predictions of future academic outcomes.

## Installation and Setup

### Prerequisites

- Python 3.10 or a compatible version
- pip
- A Google Gemini API key for Gemini-powered agent execution

### 1. Clone the Repository

```bash
git clone https://github.com/KhushieAgrawal/Acedex-AI.git
cd Acedex-AI
```

### 2. Install Dependencies

Install the dashboard dependencies:

```bash
pip install -r requirements.txt
```

Install the backend dependencies in a separate terminal:

```bash
pip install -r backend/requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root and configure your Gemini API key and model:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=your_configured_gemini_model
```

Use the model name supported by your configured Gemini API. Never commit your real API key to GitHub.

### 4. Start the Backend

From the project root, run:

```bash
uvicorn backend.main:app --reload
```

The API documentation should be available at:

`http://127.0.0.1:8000/docs`

### 5. Start the Streamlit Dashboard

Open another terminal in the project root and run:

```bash
streamlit run streamlit_app.py
```

Open the local URL shown in the terminal, usually:

`http://localhost:8501`

## Validation and Testing

The proof of concept was tested using synthetic student records. The backend workflow was checked through the FastAPI Swagger interface, including Gemini-supported agent execution and output validation.

The system demonstrates:
- Successful execution of the multi-agent workflow.
- Rule-based academic risk assessment.
- Structured recommendations for faculty review.
- Validation of generated workflow results.

These tests demonstrate proof-of-concept functionality and do not establish real-world predictive accuracy.

## Limitations

- The system uses synthetic data rather than actual institutional student records.
- Risk assessment depends on the indicators and rules implemented in the prototype.
- Recommendations require faculty review and should not be treated as automated academic decisions.
- Language-model outputs may contain errors or inconsistencies.
- The current implementation has not been validated through a real-world institutional deployment.

## Future Scope

- Integration with institutional learning management systems.
- Evaluation using approved, anonymized real-world academic data.
- Improved risk assessment using validated machine learning models.
- Personalized academic support recommendations.
- Faculty feedback and intervention outcome tracking.
- Stronger privacy, access control, monitoring, and audit mechanisms.

## Responsible AI Considerations

Acadex AI is designed as a decision-support tool, not a replacement for faculty judgment. Student data must be handled securely, recommendations should be reviewed by authorized faculty, and academic interventions should consider individual circumstances. Synthetic data is used in this proof of concept to avoid exposing real student information.

## Project Information

**Project:** Acadex AI – Agentic Academic Intelligence & Intervention System  
**Domain:** Education  
**Project Type:** Agentic AI Proof of Concept  
**Frameworks:** Streamlit, FastAPI, Google Gemini  
**Dataset:** 500 synthetic student records

---

*Developed as a CA-3 Agentic AI proof-of-concept project.*
