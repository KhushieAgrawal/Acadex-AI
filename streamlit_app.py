from pathlib import Path
import sys
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.agents import load_data, serialize_student, run_workflow, get_summary

st.set_page_config(page_title="Acadex AI PoC", page_icon="🎓", layout="wide")
st.title("🎓 Acadex AI")
st.caption("Agentic Academic Intelligence & Intervention System · Synthetic-data PoC")

try:
    df = load_data()
except Exception as exc:
    st.error(f"Could not load student data: {exc}")
    st.stop()

summary = get_summary()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Students", summary["total_students"])
c2.metric("High risk", summary["risk_counts"]["High"])
c3.metric("Moderate risk", summary["risk_counts"]["Moderate"])
c4.metric("Average attendance", f"{summary['average_attendance']}%")

st.subheader("Student explorer")
col1, col2, col3 = st.columns([1, 1, 2])
departments = ["All"] + summary["departments"]
department = col1.selectbox("Department", departments)
risk = col2.selectbox("Calculated risk", ["All", "High", "Moderate", "Low"])
search = col3.text_input("Search Student ID", placeholder="e.g. ST1001")

records = []
for _, row in df.iterrows():
    s = serialize_student(row)
    if department != "All" and s["department"] != department:
        continue
    if risk != "All" and s["calculated_risk"] != risk:
        continue
    if search.strip() and search.strip().lower() not in str(s["student_id"]).lower():
        continue
    records.append(s)

table = pd.DataFrame([{
    "Student ID": s["student_id"],
    "Department": s["department"],
    "Semester": s["semester"],
    "Attendance %": s["attendance_pct"],
    "Average": s["current_average"],
    "Assignments %": s["completion_pct"],
    "Risk (calculated)": s["calculated_risk"],
    "Reference risk": s["risk_category"],
} for s in records])
st.dataframe(table, use_container_width=True, hide_index=True)

st.subheader("Run the agent workflow")
ids = [str(x) for x in df["student_id"].tolist()]
default_id = search.strip() if search.strip() in ids else (ids[0] if ids else "")
selected = st.selectbox("Choose a student", ids, index=ids.index(default_id) if default_id in ids else 0)
if st.button("Run agent workflow", type="primary"):
    with st.spinner("Planner → Research → Executor → Critic → Business Automation; readiness checks in parallel..."):
        result = run_workflow(selected)
    final = result["final_result"]
    st.success(f"Workflow status: {result['workflow_status']}")
    a, b, c = st.columns(3)
    a.metric("Calculated risk", final["calculated_risk"])
    b.metric("Validation", "Passed" if final["validation_passed"] else "Review needed")
    c.metric("Human review", "Required" if final["human_review_required"] else "Not flagged")
    st.write("**Why this risk was assigned**")
    for reason in final["risk_reasons"]:
        st.write(f"- {reason}")
    st.write("**Recommended intervention**")
    st.info(final["recommended_intervention"])
    st.write("**Generated follow-up task**")
    st.json(final["follow_up_task"])
    st.write("**Report-ready summary**")
    st.write(final["report_summary"])
    st.write("**Agent trace**")
    for agent in result["agents"]:
        with st.expander(f"{agent['agent']} — {agent['status']}", expanded=False):
            st.write(agent.get("summary", ""))
            st.json(agent)

with st.expander("Risk rule and validation"):
    st.markdown("""
    - **High:** attendance < 65%, average < 50, performance change ≤ -10 points, or assignment completion < 50%.
    - **Moderate:** attendance < 80%, average < 65, performance change < -3 points, or completion < 80%.
    - **Low:** none of those thresholds triggered.
    - Critic Agent independently recomputes the risk, checks data ranges and assignment counts, and flags problems for human review.
    """)
st.caption("Synthetic data only. The risk score is an illustrative triage aid, not an official academic decision. Faculty approval is required before any real-world intervention.")
