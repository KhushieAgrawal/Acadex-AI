from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from backend.agents import load_data, get_summary, get_students, run_workflow, serialize_student

app = FastAPI(title="Acadex AI API", version="1.0.0", description="Synthetic academic intelligence and agentic intervention PoC")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class WorkflowRequest(BaseModel):
    student_id: str = Field(..., min_length=1, max_length=50)
    request: str = Field(default="Identify academic risk and recommend an intervention.", max_length=500)

@app.get("/api/health")
def health():
    try:
        df = load_data()
        return {"status": "ok", "service": "Acadex AI API", "rows": len(df)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/api/summary")
def summary():
    try:
        return get_summary()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/api/students")
def students(
    department: Optional[str] = Query(default=None),
    risk: Optional[str] = Query(default=None),
    search: Optional[int] = Query(default=None),
):
    try:
        return get_students(department=department, risk=risk, search=search)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/api/students/{student_id}")
def student_detail(student_id: str):
    try:
        df = load_data()
        matched = df[df["student_id"].astype(str) == student_id]
        if matched.empty:
            raise HTTPException(status_code=404, detail="Student not found")
        return serialize_student(matched.iloc[0])
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/api/workflow")
def workflow(body: WorkflowRequest):
    try:
        return run_workflow(body.student_id, body.request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
