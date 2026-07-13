"""RAG evaluation API endpoints.

Runs the automatic-metric evaluation over a test dataset, persists every run,
and exposes history + per-question detail for the admin panel evaluation page.
"""

import csv
import io
import json
import logging

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import List

from app.database.session import get_db
from app.security.auth import get_current_active_admin
from app.models.admin import Admin
from app.models.evaluation import EvaluationRun, EvaluationItem
from app.schemas.evaluation import (
    EvaluationRunCreate,
    EvaluationRunSummary,
    EvaluationRunDetail,
    DatasetInfo,
)
from app.services import evaluation_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/evaluation/datasets", response_model=List[DatasetInfo])
async def list_datasets(
    current_admin: Admin = Depends(get_current_active_admin),
):
    """List available evaluation datasets (JSON files in scripts/)."""
    return evaluation_service.list_datasets()


@router.post("/evaluation/datasets/upload", response_model=DatasetInfo)
async def upload_dataset(
    file: UploadFile = File(...),
    current_admin: Admin = Depends(get_current_active_admin),
):
    """Upload a new evaluation dataset JSON (validated and saved to scripts/)."""
    content = await file.read()
    try:
        return evaluation_service.save_uploaded_dataset(file.filename, content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal menyimpan dataset: {e}")


@router.get("/evaluation/runs", response_model=List[EvaluationRunSummary])
async def list_runs(
    db: Session = Depends(get_db),
    current_admin: Admin = Depends(get_current_active_admin),
):
    """List all evaluation runs (history), newest first."""
    return db.query(EvaluationRun).order_by(EvaluationRun.created_at.desc()).all()


@router.post("/evaluation/runs", response_model=EvaluationRunSummary)
async def create_run(
    payload: EvaluationRunCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_admin: Admin = Depends(get_current_active_admin),
):
    """Start a new evaluation run (executed in the background)."""
    # Validate dataset exists
    try:
        evaluation_service._load_dataset(payload.dataset_name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid dataset: {e}")

    run = EvaluationRun(
        run_name=payload.run_name or f"Run {payload.dataset_name}",
        dataset_name=payload.dataset_name,
        status="pending",
        division_id=payload.division_id,
        category_id=payload.category_id,
        config={
            "metrics": payload.metrics,
            "ragas_use_ground_truth": payload.ragas_use_ground_truth,
            "limit": payload.limit,
            "division_id": payload.division_id,
            "category_id": payload.category_id,
        },
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    background_tasks.add_task(evaluation_service.run_evaluation_task, run.id)
    return run


@router.get("/evaluation/runs/{run_id}", response_model=EvaluationRunDetail)
async def get_run(
    run_id: int,
    db: Session = Depends(get_db),
    current_admin: Admin = Depends(get_current_active_admin),
):
    """Get a run with full per-question detail (used for progress polling + drill-down)."""
    run = db.query(EvaluationRun).filter(EvaluationRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    return run


@router.delete("/evaluation/runs/{run_id}")
async def delete_run(
    run_id: int,
    db: Session = Depends(get_db),
    current_admin: Admin = Depends(get_current_active_admin),
):
    """Delete an evaluation run and its items."""
    run = db.query(EvaluationRun).filter(EvaluationRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    db.delete(run)
    db.commit()
    return {"success": True}


@router.get("/evaluation/runs/{run_id}/export")
async def export_run(
    run_id: int,
    format: str = "json",
    db: Session = Depends(get_db),
    current_admin: Admin = Depends(get_current_active_admin),
):
    """Export a run's per-question results as JSON or CSV."""
    run = db.query(EvaluationRun).filter(EvaluationRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Evaluation run not found")

    items = db.query(EvaluationItem).filter(EvaluationItem.run_id == run_id).order_by(EvaluationItem.id).all()

    if format == "csv":
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([
            "no", "id", "division_id", "question", "ground_truth", "answer",
            "bertscore_f1", "bleu", "rougeL",
            "ragas_faithfulness", "ragas_answer_relevancy", "ragas_context_precision",
            "total_time_ms",
        ])
        for i, it in enumerate(items, 1):
            writer.writerow([
                i, it.question_ref, it.division_id, it.question, it.ground_truth, it.answer,
                it.bertscore_f1, it.bleu, it.rougeL,
                it.ragas_faithfulness, it.ragas_answer_relevancy, it.ragas_context_precision,
                it.total_time_ms,
            ])
        buf.seek(0)
        return StreamingResponse(
            iter([buf.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="evaluation_run_{run_id}.csv"'},
        )

    # JSON
    payload = {
        "run": {
            "id": run.id,
            "run_name": run.run_name,
            "dataset_name": run.dataset_name,
            "status": run.status,
            "num_samples": run.num_samples,
            "config": run.config,
            "aggregate": {
                "bertscore_f1": run.bertscore_f1,
                "bertscore_precision": run.bertscore_precision,
                "bertscore_recall": run.bertscore_recall,
                "bleu": run.bleu,
                "rougeL": run.rougeL,
                "ragas_faithfulness": run.ragas_faithfulness,
                "ragas_answer_relevancy": run.ragas_answer_relevancy,
                "ragas_context_precision": run.ragas_context_precision,
                "avg_total_ms": run.avg_total_ms,
            },
            "created_at": run.created_at.isoformat() if run.created_at else None,
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
        },
        "items": [
            {
                "id": it.question_ref,
                "division_id": it.division_id,
                "category_id": it.category_id,
                "question": it.question,
                "ground_truth": it.ground_truth,
                "answer": it.answer,
                "contexts": it.contexts,
                "sources": it.sources,
                "scores": {
                    "bertscore_f1": it.bertscore_f1,
                    "bleu": it.bleu,
                    "rougeL": it.rougeL,
                    "ragas_faithfulness": it.ragas_faithfulness,
                    "ragas_answer_relevancy": it.ragas_answer_relevancy,
                    "ragas_context_precision": it.ragas_context_precision,
                },
                "total_time_ms": it.total_time_ms,
            }
            for it in items
        ],
    }
    data = json.dumps(payload, indent=2, ensure_ascii=False)
    return StreamingResponse(
        iter([data]),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="evaluation_run_{run_id}.json"'},
    )
