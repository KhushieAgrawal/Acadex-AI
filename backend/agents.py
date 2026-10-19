from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import os
import time
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from google import genai
except ImportError:
    genai = None

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def llm_generate(prompt: str) -> dict[str, Any]:
    """Call Gemini when configured; otherwise return a clearly labelled fallback."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or genai is None:
        return {"text": "", "mode": "python_fallback", "error": "Gemini is not configured; using deterministic fallback."}
    last_error = "Gemini request failed."
    for attempt in range(2):
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config={"http_options": {"timeout": 20000}}
            )
            text = (getattr(response, "text", None) or "").strip()
            if not text:
                raise ValueError("Gemini returned an empty response")
            return {"text": text, "mode": "gemini", "error": None}
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt == 0:
                time.sleep(1)
    return {"text": "", "mode": "python_fallback", "error": last_error[:300]}


def _json_from_llm(result: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    if result.get("mode") != "gemini":
        return fallback
    raw = result["text"].strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else fallback
    except (ValueError, TypeError):
        return fallback


DATA_PATH = Path(__file__).parent / "data" / "acadex_ai_synthetic_students_500.csv"
REQUIRED_COLUMNS = [
    "student_id", "department", "semester", "previous_semester_score",
    "attendance_pct", "assignments_completed", "assignments_total",
    "mathematics_marks", "programming_marks", "data_structures_marks",
    "communication_marks", "current_average", "performance_change",
    "risk_category", "suggested_intervention"
]

def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Dataset is missing required columns: " + ", ".join(missing))
    return df

def completion_pct(row: pd.Series) -> float | None:
    total = float(row.get("assignments_total", 0) or 0)
    if total <= 0:
        return None
    return round(float(row.get("assignments_completed", 0)) / total * 100, 1)

def calculate_risk(row: pd.Series) -> dict[str, Any]:
    attendance = float(row["attendance_pct"])
    average = float(row["current_average"])
    change = float(row["performance_change"])
    completion = completion_pct(row)
    reasons = []

    if attendance < 65:
        reasons.append(f"Attendance is {attendance:.1f}%, below the 65% high-risk threshold")
    if average < 50:
        reasons.append(f"Current average is {average:.1f}, below 50")
    if change <= -10:
        reasons.append(f"Performance fell by {abs(change):.1f} points")
    if completion is not None and completion < 50:
        reasons.append(f"Assignment completion is {completion:.1f}%")

    if reasons:
        risk = "High"
        intervention = "Faculty check-in within 2 working days; review attendance, missed work, and subject support."
    else:
        moderate_reasons = []
        if attendance < 80:
            moderate_reasons.append(f"Attendance is {attendance:.1f}%, below 80%")
        if average < 65:
            moderate_reasons.append(f"Current average is {average:.1f}, below 65")
        if change < -3:
            moderate_reasons.append(f"Performance is down {abs(change):.1f} points")
        if completion is not None and completion < 80:
            moderate_reasons.append(f"Assignment completion is {completion:.1f}%")
        if moderate_reasons:
            risk = "Moderate"
            reasons = moderate_reasons
            intervention = "Faculty check-in this week; agree on one attendance or assignment improvement goal."
        else:
            risk = "Low"
            reasons = ["No configured high- or moderate-risk thresholds were triggered"]
            intervention = "Continue regular progress monitoring and positive feedback."

    if completion is None:
        reasons.append("Assignment completion could not be calculated because assignments_total is zero")
    return {
        "calculated_risk": risk,
        "completion_pct": completion,
        "risk_reasons": reasons,
        "recommended_intervention": intervention,
    }

# 1. Planner Agent

def planner_agent(request: str, student_id: str) -> dict[str, Any]:
    fallback_tasks = [
        "Retrieve the selected student's synthetic record",
        "Review attendance, assignment completion, marks, and performance trend",
        "Calculate a transparent rule-based risk category",
        "Validate the record and recommendation",
        "Prepare a faculty follow-up task and report summary",
    ]
    llm = llm_generate(
        "You are the Planner Agent in Acadex AI, an academic-support PoC using synthetic data. "
        "Return JSON only with key plan: an array of 4-6 concise ordered tasks. Preserve Python-based risk calculation and critic validation. "
        f"Faculty request: {request!r}. Student identifier: {student_id!r}. Do not infer student facts."
    )
    parsed = _json_from_llm(llm, {"plan": fallback_tasks})
    tasks = parsed.get("plan", fallback_tasks)
    if not isinstance(tasks, list) or not tasks or not all(isinstance(x, str) for x in tasks):
        tasks = fallback_tasks
    return {
        "agent": "Planner Agent", "status": "completed",
        "execution_mode": llm["mode"], "llm_error": llm.get("error"),
        "summary": f"{'Gemini' if llm['mode'] == 'gemini' else 'Python fallback'} produced a plan with {len(tasks)} ordered tasks for {student_id}.",
        "plan": tasks[:8], "request": request,
    }

# 2. Research Agent

def research_agent(row: pd.Series) -> dict[str, Any]:
    completion = completion_pct(row)
    subjects = {
        "Mathematics": float(row["mathematics_marks"]),
        "Programming": float(row["programming_marks"]),
        "Data Structures": float(row["data_structures_marks"]),
        "Communication": float(row["communication_marks"]),
    }
    weakest = min(subjects, key=subjects.get)
    evidence = {
        "attendance_pct": float(row["attendance_pct"]),
        "current_average": float(row["current_average"]),
        "previous_semester_score": float(row["previous_semester_score"]),
        "performance_change": float(row["performance_change"]),
        "assignments_completed": int(row["assignments_completed"]),
        "assignments_total": int(row["assignments_total"]),
        "completion_pct": completion, "subject_marks": subjects, "lowest_subject": weakest,
    }
    llm = llm_generate(
        "You are the Research Agent for an educational support proof of concept. Interpret ONLY the supplied synthetic evidence. "
        "Return JSON only with keys summary (2-3 factual sentences), strengths (array of strings), areas_to_review (array of strings). "
        "Do not diagnose, invent causes, or change numeric values. Treat this as support triage, not a consequential decision. Evidence: "
        + json.dumps(evidence)
    )
    fallback = {
        "summary": "Reviewed the student's academic and engagement indicators using the supplied synthetic record.",
        "strengths": [f"{k} score: {v:.1f}" for k, v in subjects.items() if v >= 70][:3],
        "areas_to_review": [f"{weakest} is the lowest-marked subject", f"Attendance is {evidence['attendance_pct']:.1f}%", f"Assignment completion is {completion}%" if completion is not None else "Assignment completion unavailable"],
    }
    narrative = _json_from_llm(llm, fallback)
    if not isinstance(narrative.get("summary"), str): narrative = fallback
    return {
        "agent": "Research Agent", "status": "completed", "execution_mode": llm["mode"], "llm_error": llm.get("error"),
        "summary": narrative.get("summary", fallback["summary"]), "evidence": evidence,
        "strengths": narrative.get("strengths", fallback["strengths"]),
        "areas_to_review": narrative.get("areas_to_review", fallback["areas_to_review"]),
    }

# 3. Executor Agent
def executor_agent(row: pd.Series) -> dict[str, Any]:
    result = calculate_risk(row)
    return {
        "agent": "Executor Agent",
        "status": "completed",
        "summary": f"Calculated {result['calculated_risk'].lower()} risk from explicit Python rules.",
        **result,
    }

# 4. Critic Agent

def critic_agent(row: pd.Series, research: dict, execution: dict) -> dict[str, Any]:
    issues = []
    for col in ["attendance_pct", "current_average", "previous_semester_score"]:
        val = float(row[col])
        if not 0 <= val <= 100: issues.append(f"{col} is outside the expected 0–100 range")
    if int(row["assignments_completed"]) < 0: issues.append("assignments_completed cannot be negative")
    if int(row["assignments_total"]) < 0: issues.append("assignments_total cannot be negative")
    if int(row["assignments_completed"]) > int(row["assignments_total"]): issues.append("assignments_completed exceeds assignments_total")
    independently_recomputed = calculate_risk(row)["calculated_risk"]
    if independently_recomputed != execution["calculated_risk"]: issues.append("Executor risk does not match independent risk recomputation")
    if not execution.get("recommended_intervention"): issues.append("No intervention recommendation was produced")

    rule_passed = len(issues) == 0
    checks = [
        {"name": "Required numeric ranges", "passed": all(0 <= float(row[c]) <= 100 for c in ["attendance_pct", "current_average", "previous_semester_score"])},
        {"name": "Assignment counts", "passed": int(row["assignments_completed"]) >= 0 and int(row["assignments_total"]) >= 0 and int(row["assignments_completed"]) <= int(row["assignments_total"])},
        {"name": "Independent risk recomputation", "passed": independently_recomputed == execution["calculated_risk"]},
        {"name": "Intervention exists", "passed": bool(execution.get("recommended_intervention"))},
    ]
    llm = llm_generate(
        "You are the Critic Agent. Review the following machine-calculated academic support result for consistency. "
        "Python checks are authoritative; do not change the calculated risk or overrule failed checks. Return JSON only with keys: review_summary (string), concerns (array of strings). "
        "Do not infer hidden facts. Result: " + json.dumps({"research": research, "execution": execution, "python_checks": checks, "python_issues": issues}, default=str)
    )
    fallback = {"review_summary": "LLM review unavailable; deterministic validation remains authoritative." if llm["mode"] != "gemini" else "LLM review completed.", "concerns": []}
    review = _json_from_llm(llm, fallback)
    passed = rule_passed
    return {
        "agent": "Critic Agent", "status": "passed" if passed else "needs_review",
        "execution_mode": llm["mode"], "llm_error": llm.get("error"),
        "summary": ("Python validation passed. " if passed else "Python validation found issues. ") + str(review.get("review_summary", fallback["review_summary"])),
        "checks": checks, "issues": issues, "llm_concerns": review.get("concerns", []),
        "human_review_required": not passed,
    }

# 5. Software Development Agent
def software_development_agent(df: pd.DataFrame | None = None) -> dict[str, Any]:
    try:
        data = df if df is not None else load_data()
        missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
        checks = [
            {"name": "Dataset loaded", "passed": len(data) > 0},
            {"name": "Required schema present", "passed": len(missing) == 0},
            {"name": "Student ID uniqueness", "passed": bool(data["student_id"].is_unique)},
        ]
        passed = all(c["passed"] for c in checks)
        return {
            "agent": "Software Development Agent",
            "status": "passed" if passed else "needs_review",
            "summary": "Application readiness checks passed." if passed else "Application readiness checks found issues.",
            "checks": checks,
            "row_count": int(len(data)),
            "missing_columns": missing,
            "note": "This agent supports development/testing and does not modify source code at runtime.",
        }
    except Exception as exc:
        return {
            "agent": "Software Development Agent",
            "status": "needs_review",
            "summary": "Application readiness check failed.",
            "checks": [{"name": "Dataset/schema readiness", "passed": False}],
            "error": str(exc),
            "note": "This agent supports development/testing and does not modify source code at runtime.",
        }

# 6. Business Automation Agent

def business_automation_agent(row: pd.Series, execution: dict, critic: dict, research: dict | None = None) -> dict[str, Any]:
    student_id = str(row["student_id"])
    risk = execution["calculated_risk"]
    if risk == "High": due, priority = "Within 2 working days", "Urgent"
    elif risk == "Moderate": due, priority = "This week", "Normal"
    else: due, priority = "Next scheduled review", "Routine"
    action = {
        "task_title": f"{priority} faculty follow-up — {student_id}", "priority": priority,
        "suggested_owner": "Academic advisor / faculty mentor", "target_time": due,
        "action": execution["recommended_intervention"],
        "approval": "Requires faculty review before any real-world action",
    }
    facts = {
        "student_id": student_id, "department": str(row["department"]), "semester": int(row["semester"]),
        "risk": risk, "attendance_pct": float(row["attendance_pct"]), "current_average": float(row["current_average"]),
        "performance_change": float(row["performance_change"]), "risk_reasons": execution["risk_reasons"],
        "recommended_intervention": execution["recommended_intervention"], "critic_status": critic["status"],
        "human_review_required": critic["human_review_required"], "due": due, "priority": priority,
    }
    llm = llm_generate(
        "You are Acadex AI's Business Automation Agent. Draft a respectful, non-judgmental faculty follow-up message and concise intervention summary. "
        "Use only supplied facts, do not add unsupported claims, do not send anything, and explicitly keep faculty approval. Return JSON only with keys: report_summary, faculty_message, task_action. Facts: "
        + json.dumps(facts)
    )
    fallback = {
        "report_summary": (f"{student_id} ({row['department']}, semester {row['semester']}) is classified as {risk} risk by the Python rule-based baseline. "
            f"Attendance: {float(row['attendance_pct']):.1f}%; current average: {float(row['current_average']):.1f}; performance change: {float(row['performance_change']):+.1f} points. "
            f"Validation: {'passed' if critic['status'] == 'passed' else 'requires human review'}. Recommended next step: {execution['recommended_intervention']}"),
        "faculty_message": f"Please review {student_id}'s synthetic academic indicators and consider a supportive check-in. {execution['recommended_intervention']} This draft requires faculty review.",
        "task_action": execution["recommended_intervention"],
    }
    generated = _json_from_llm(llm, fallback)
    report = generated.get("report_summary") if isinstance(generated.get("report_summary"), str) else fallback["report_summary"]
    faculty_message = generated.get("faculty_message") if isinstance(generated.get("faculty_message"), str) else fallback["faculty_message"]
    task_action = generated.get("task_action") if isinstance(generated.get("task_action"), str) else fallback["task_action"]
    action["action"] = task_action
    return {
        "agent": "Business Automation Agent", "status": "prepared", "execution_mode": llm["mode"], "llm_error": llm.get("error"),
        "summary": f"{'Gemini' if llm['mode'] == 'gemini' else 'Python fallback'} prepared a faculty follow-up draft; nothing was sent automatically.",
        "follow_up_task": action, "faculty_message": faculty_message, "report_summary": report,
    }

def run_workflow(student_id: str, request: str = "Identify academic risk and recommend an intervention.") -> dict[str, Any]:
    df = load_data()
    matched = df[df["student_id"].astype(str) == str(student_id)]
    if matched.empty:
        raise KeyError(f"Student ID {student_id} was not found")
    row = matched.iloc[0]

    plan = planner_agent(request, student_id)
    research = research_agent(row)
    execution = executor_agent(row)
    critic = critic_agent(row, research, execution)
    dev_check = software_development_agent(df)
    automation = business_automation_agent(row, execution, critic, research)

    llm_agents = [a for a in [plan, research, critic, automation] if a.get("execution_mode") == "gemini"]
    return {
        "llm_provider": "Google Gemini",
        "llm_configured": bool(os.getenv("GEMINI_API_KEY", "").strip()) and genai is not None,
        "llm_agents_used": len(llm_agents),
        "workflow_execution_mode": "gemini_and_python" if llm_agents else "python_fallback",
        "student": serialize_student(row),
        "workflow_status": "completed" if critic["status"] == "passed" and dev_check["status"] == "passed" else "needs_review",
        "agents": [plan, research, execution, critic, dev_check, automation],
        "final_result": {
            "calculated_risk": execution["calculated_risk"],
            "risk_reasons": execution["risk_reasons"],
            "recommended_intervention": execution["recommended_intervention"],
            "reference_risk_category": str(row["risk_category"]),
            "reference_suggested_intervention": str(row["suggested_intervention"]),
            "validation_passed": critic["status"] == "passed",
            "human_review_required": critic["human_review_required"] or dev_check["status"] != "passed",
            "follow_up_task": automation["follow_up_task"],
            "report_summary": automation["report_summary"],
        },
        "disclaimer": "Synthetic-data proof of concept. Rule-based triage is illustrative and must not be used as the sole basis for consequential decisions about real students.",
    }

def serialize_student(row: pd.Series) -> dict[str, Any]:
    out = {}
    for key, value in row.to_dict().items():
        if pd.isna(value):
            out[key] = None
        elif hasattr(value, "item"):
            out[key] = value.item()
        else:
            out[key] = value
    out["completion_pct"] = completion_pct(row)
    computed = calculate_risk(row)
    out["calculated_risk"] = computed["calculated_risk"]
    out["risk_reasons"] = computed["risk_reasons"]
    out["recommended_intervention"] = computed["recommended_intervention"]
    return out

def get_students(department: str | None = None, risk: str | None = None, search: str | None = None) -> list[dict[str, Any]]:
    df = load_data()
    rows = []
    for _, row in df.iterrows():
        student = serialize_student(row)
        if department and department != "All" and student["department"] != department:
            continue
        if risk and risk != "All" and student["calculated_risk"] != risk:
            continue
        if search and search.lower() not in str(student["student_id"]).lower():
            continue
        rows.append(student)
    return rows

def get_summary() -> dict[str, Any]:
    df = load_data()
    students = [serialize_student(row) for _, row in df.iterrows()]
    counts = {risk: sum(s["calculated_risk"] == risk for s in students) for risk in ["High", "Moderate", "Low"]}
    departments = sorted(df["department"].dropna().astype(str).unique().tolist())
    return {
        "total_students": len(students),
        "risk_counts": counts,
        "average_attendance": round(float(df["attendance_pct"].mean()), 1),
        "average_score": round(float(df["current_average"].mean()), 1),
        "departments": departments,
        "reference_label_match_pct": round(100 * sum(s["calculated_risk"].lower() == str(s["risk_category"]).lower() for s in students) / max(len(students), 1), 1),
    }
