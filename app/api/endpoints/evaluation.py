"""Evaluation API endpoints"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from sqlalchemy.orm import Session
from typing import Optional, List
from pydantic import BaseModel, Field
from datetime import datetime
import logging

from app.database.session import get_db
from app.models.evaluation import EvaluationRun, EvaluationSampleResult
from app.rag.evaluation.evaluator import get_evaluator
from app.rag.evaluation.dataset import EvaluationDataset
from app.rag.evaluation.reporter import evaluation_reporter
from app.rag.evaluation.config import evaluation_config

router = APIRouter()
logger = logging.getLogger(__name__)


# Schemas
class EvaluateQueryRequest(BaseModel):
    question: str = Field(..., description="Question to evaluate")
    ground_truth: Optional[str] = Field(None, description="Expected answer")
    
class EvaluateDatasetRequest(BaseModel):
    dataset_path: str = Field(..., description="Path to dataset file (JSON or CSV)")
    run_name: Optional[str] = Field(None, description="Optional name for this run")
    save_to_db: bool = Field(True, description="Save results to database")

class CompareRunsRequest(BaseModel):
    run1_id: Optional[int] = Field(None, description="First run ID (from DB)")
    run2_id: Optional[int] = Field(None, description="Second run ID (from DB)")
    run1_file: Optional[str] = Field(None, description="First run file path")
    run2_file: Optional[str] = Field(None, description="Second run file path")


@router.post("/evaluation/query")
async def evaluate_query(
    request: EvaluateQueryRequest,
    db: Session = Depends(get_db)
):
    """
    Generate RAG response and evaluate it
    """
    try:
        evaluator = get_evaluator()
        result = evaluator.evaluate_rag_query(
            question=request.question,
            ground_truth=request.ground_truth
        )
        
        return {
            "status": "success",
            "result": result
        }
    except Exception as e:
        logger.error(f"Error evaluating query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluation/run")
async def run_evaluation(
    request: EvaluateDatasetRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Run evaluation on a dataset
    """
    try:
        # Create evaluation run record
        run = EvaluationRun(
            run_name=request.run_name or f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            dataset_name=request.dataset_path.split('/')[-1],
            dataset_size=0,
            status="running",
            ai_provider=evaluation_config.ai_provider,
            evaluation_model=evaluation_config.evaluation_model,
            enabled_metrics=evaluation_config.enabled_metrics
        )
        db.add(run)
        db.commit()
        db.refresh(run)
        
        # Run evaluation in background
        background_tasks.add_task(
            _run_evaluation_task,
            run_id=run.id,
            dataset_path=request.dataset_path,
            save_to_db=request.save_to_db
        )
        
        return {
            "status": "started",
            "run_id": run.id,
            "run_name": run.run_name,
            "message": "Evaluation started in background"
        }
        
    except Exception as e:
        logger.error(f"Error starting evaluation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/evaluation/results/{run_id}")
async def get_evaluation_results(
    run_id: int,
    include_samples: bool = Query(False, description="Include per-sample results"),
    db: Session = Depends(get_db)
):
    """
    Get evaluation results by run ID
    """
    run = db.query(EvaluationRun).filter(EvaluationRun.id == run_id).first()
    
    if not run:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    
    result = {
        "run_id": run.id,
        "run_name": run.run_name,
        "status": run.status,
        "dataset_name": run.dataset_name,
        "dataset_size": run.dataset_size,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
        "evaluation_time_seconds": run.evaluation_time_seconds,
        "aggregated_metrics": run.aggregated_metrics,
        "error_message": run.error_message
    }
    
    if include_samples:
        samples = db.query(EvaluationSampleResult).filter(
            EvaluationSampleResult.run_id == run_id
        ).all()
        
        result["samples"] = [
            {
                "id": s.id,
                "question": s.question,
                "answer": s.answer,
                "faithfulness": s.faithfulness,
                "answer_relevancy": s.answer_relevancy,
                "context_precision": s.context_precision,
                "context_recall": s.context_recall
            }
            for s in samples
        ]
    
    return result


@router.get("/evaluation/runs")
async def list_evaluation_runs(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """
    List evaluation runs
    """
    query = db.query(EvaluationRun)
    
    if status:
        query = query.filter(EvaluationRun.status == status)
    
    total = query.count()
    runs = query.order_by(EvaluationRun.started_at.desc()).offset(offset).limit(limit).all()
    
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "runs": [
            {
                "run_id": r.id,
                "run_name": r.run_name,
                "status": r.status,
                "dataset_size": r.dataset_size,
                "started_at": r.started_at.isoformat() if r.started_at else None,
                "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                "faithfulness_mean": r.faithfulness_mean,
                "answer_relevancy_mean": r.answer_relevancy_mean
            }
            for r in runs
        ]
    }


@router.post("/evaluation/compare")
async def compare_evaluation_runs(
    request: CompareRunsRequest,
    db: Session = Depends(get_db)
):
    """
    Compare two evaluation runs
    """
    try:
        # If using file paths
        if request.run1_file and request.run2_file:
            comparison = evaluation_reporter.compare_runs(
                request.run1_file,
                request.run2_file
            )
            return comparison
        
        # If using DB IDs
        if request.run1_id and request.run2_id:
            run1 = db.query(EvaluationRun).filter(EvaluationRun.id == request.run1_id).first()
            run2 = db.query(EvaluationRun).filter(EvaluationRun.id == request.run2_id).first()
            
            if not run1 or not run2:
                raise HTTPException(status_code=404, detail="One or both runs not found")
            
            return {
                "run1": {
                    "id": run1.id,
                    "name": run1.run_name,
                    "faithfulness_mean": run1.faithfulness_mean,
                    "answer_relevancy_mean": run1.answer_relevancy_mean,
                    "context_precision_mean": run1.context_precision_mean,
                    "context_recall_mean": run1.context_recall_mean
                },
                "run2": {
                    "id": run2.id,
                    "name": run2.run_name,
                    "faithfulness_mean": run2.faithfulness_mean,
                    "answer_relevancy_mean": run2.answer_relevancy_mean,
                    "context_precision_mean": run2.context_precision_mean,
                    "context_recall_mean": run2.context_recall_mean
                },
                "differences": {
                    "faithfulness": run2.faithfulness_mean - run1.faithfulness_mean if run1.faithfulness_mean and run2.faithfulness_mean else None,
                    "answer_relevancy": run2.answer_relevancy_mean - run1.answer_relevancy_mean if run1.answer_relevancy_mean and run2.answer_relevancy_mean else None,
                    "context_precision": run2.context_precision_mean - run1.context_precision_mean if run1.context_precision_mean and run2.context_precision_mean else None,
                    "context_recall": run2.context_recall_mean - run1.context_recall_mean if run1.context_recall_mean and run2.context_recall_mean else None
                }
            }
        
        raise HTTPException(status_code=400, detail="Must provide either run IDs or file paths")
        
    except Exception as e:
        logger.error(f"Error comparing runs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/evaluation/metrics")
async def get_metric_definitions():
    """
    Get definitions of evaluation metrics
    """
    return {
        "metrics": {
            "faithfulness": {
                "name": "Faithfulness",
                "description": "Measures if the answer is grounded in the provided context. Score ranges from 0 to 1.",
                "range": [0, 1],
                "threshold": evaluation_config.min_faithfulness
            },
            "answer_relevancy": {
                "name": "Answer Relevancy",
                "description": "Measures if the answer actually addresses the question. Score ranges from 0 to 1.",
                "range": [0, 1],
                "threshold": evaluation_config.min_answer_relevancy
            },
            "context_precision": {
                "name": "Context Precision",
                "description": "Measures if relevant chunks are ranked higher in retrieval. Score ranges from 0 to 1.",
                "range": [0, 1],
                "threshold": evaluation_config.min_context_precision
            },
            "context_recall": {
                "name": "Context Recall",
                "description": "Measures if all relevant information was retrieved. Requires ground truth. Score ranges from 0 to 1.",
                "range": [0, 1],
                "threshold": evaluation_config.min_context_recall
            }
        }
    }


# Background task function
def _run_evaluation_task(run_id: int, dataset_path: str, save_to_db: bool):
    """Background task to run evaluation"""
    from app.database.session import SessionLocal
    
    db = SessionLocal()
    try:
        run = db.query(EvaluationRun).filter(EvaluationRun.id == run_id).first()
        if not run:
            logger.error(f"Run {run_id} not found")
            return
        
        # Load evaluator
        evaluator = get_evaluator()
        
        # Run evaluation
        logger.info(f"Starting evaluation run {run_id}")
        result = evaluator.evaluate_from_file(dataset_path)
        
        # Update run record
        run.status = "completed"
        run.completed_at = datetime.utcnow()
        run.dataset_size = result.get('dataset_size', 0)
        run.evaluation_time_seconds = result.get('evaluation_time_seconds', 0)
        run.aggregated_metrics = result.get('aggregated_metrics', {})
        
        # Extract mean scores
        agg = result.get('aggregated_metrics', {})
        run.faithfulness_mean = agg.get('faithfulness_mean')
        run.answer_relevancy_mean = agg.get('answer_relevancy_mean')
        run.context_precision_mean = agg.get('context_precision_mean')
        run.context_recall_mean = agg.get('context_recall_mean')
        
        # Save results file
        file_path = evaluation_reporter.save_results(result, run.run_name)
        run.results_file_path = file_path
        
        # Save sample results to DB if requested
        if save_to_db and 'per_sample_scores' in result:
            for sample_data in result['per_sample_scores']:
                sample = EvaluationSampleResult(
                    run_id=run_id,
                    question=sample_data.get('question', ''),
                    answer=sample_data.get('answer', ''),
                    contexts=sample_data.get('contexts', []),
                    ground_truth=sample_data.get('ground_truth'),
                    faithfulness=sample_data.get('faithfulness'),
                    answer_relevancy=sample_data.get('answer_relevancy'),
                    context_precision=sample_data.get('context_precision'),
                    context_recall=sample_data.get('context_recall')
                )
                db.add(sample)
        
        db.commit()
        logger.info(f"Evaluation run {run_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Error in evaluation task: {e}", exc_info=True)
        run = db.query(EvaluationRun).filter(EvaluationRun.id == run_id).first()
        if run:
            run.status = "failed"
            run.error_message = str(e)
            run.completed_at = datetime.utcnow()
            db.commit()
    finally:
        db.close()
