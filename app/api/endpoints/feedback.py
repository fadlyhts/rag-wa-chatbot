"""User Feedback API endpoints"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime
import logging

from app.database.session import get_db
from app.models.evaluation import UserFeedback
from app.models.conversation import Conversation
from app.models.message import Message

router = APIRouter()
logger = logging.getLogger(__name__)


# Schemas
class SubmitFeedbackRequest(BaseModel):
    conversation_id: int = Field(..., description="Conversation ID")
    message_id: Optional[int] = Field(None, description="Specific message ID (optional)")
    question: str = Field(..., description="User question")
    answer: str = Field(..., description="AI answer")
    rating: Optional[int] = Field(None, ge=1, le=5, description="Rating 1-5 stars")
    feedback_type: Optional[str] = Field(None, description="Feedback type: helpful, not_helpful, incorrect, etc.")
    feedback_text: Optional[str] = Field(None, description="Additional feedback text")
    corrected_answer: Optional[str] = Field(None, description="User-provided correct answer")


class RateAnswerRequest(BaseModel):
    conversation_id: int
    message_id: Optional[int] = None
    question: str
    answer: str
    rating: int = Field(..., ge=1, le=5)


class ReportIncorrectRequest(BaseModel):
    conversation_id: int
    message_id: Optional[int] = None
    question: str
    answer: str
    corrected_answer: str
    feedback_text: Optional[str] = None


@router.post("/feedback/submit")
async def submit_feedback(
    request: SubmitFeedbackRequest,
    db: Session = Depends(get_db)
):
    """
    Submit general feedback on a RAG response
    """
    try:
        # Verify conversation exists
        conversation = db.query(Conversation).filter(
            Conversation.id == request.conversation_id
        ).first()
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Create feedback record
        feedback = UserFeedback(
            conversation_id=request.conversation_id,
            message_id=request.message_id,
            question=request.question,
            answer=request.answer,
            rating=request.rating,
            feedback_type=request.feedback_type,
            feedback_text=request.feedback_text,
            corrected_answer=request.corrected_answer
        )
        
        db.add(feedback)
        db.commit()
        db.refresh(feedback)
        
        logger.info(f"Feedback submitted: {feedback.id} (rating: {request.rating})")
        
        return {
            "status": "success",
            "feedback_id": feedback.id,
            "message": "Feedback submitted successfully"
        }
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback/rate")
async def rate_answer(
    request: RateAnswerRequest,
    db: Session = Depends(get_db)
):
    """
    Quick rating endpoint (1-5 stars)
    """
    try:
        feedback = UserFeedback(
            conversation_id=request.conversation_id,
            message_id=request.message_id,
            question=request.question,
            answer=request.answer,
            rating=request.rating,
            feedback_type="rating"
        )
        
        db.add(feedback)
        db.commit()
        
        logger.info(f"Answer rated: {request.rating} stars")
        
        return {
            "status": "success",
            "message": f"Rated {request.rating} stars"
        }
        
    except Exception as e:
        logger.error(f"Error rating answer: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback/report-incorrect")
async def report_incorrect(
    request: ReportIncorrectRequest,
    db: Session = Depends(get_db)
):
    """
    Report incorrect answer and provide correction
    """
    try:
        feedback = UserFeedback(
            conversation_id=request.conversation_id,
            message_id=request.message_id,
            question=request.question,
            answer=request.answer,
            corrected_answer=request.corrected_answer,
            feedback_text=request.feedback_text,
            feedback_type="incorrect",
            rating=1  # Incorrect answers get low rating
        )
        
        db.add(feedback)
        db.commit()
        
        logger.info(f"Incorrect answer reported for conversation {request.conversation_id}")
        
        return {
            "status": "success",
            "message": "Thank you for the correction. We'll use it to improve our system."
        }
        
    except Exception as e:
        logger.error(f"Error reporting incorrect answer: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feedback/stats")
async def get_feedback_stats(
    days: int = Query(7, ge=1, le=365, description="Number of days to analyze"),
    db: Session = Depends(get_db)
):
    """
    Get feedback statistics
    """
    try:
        from datetime import timedelta
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Get feedback in date range
        feedback_list = db.query(UserFeedback).filter(
            UserFeedback.created_at >= cutoff_date
        ).all()
        
        if not feedback_list:
            return {
                "total_feedback": 0,
                "average_rating": None,
                "rating_distribution": {},
                "feedback_types": {}
            }
        
        # Calculate stats
        ratings = [f.rating for f in feedback_list if f.rating]
        avg_rating = sum(ratings) / len(ratings) if ratings else None
        
        rating_dist = {}
        for i in range(1, 6):
            rating_dist[str(i)] = sum(1 for r in ratings if r == i)
        
        feedback_types = {}
        for f in feedback_list:
            if f.feedback_type:
                feedback_types[f.feedback_type] = feedback_types.get(f.feedback_type, 0) + 1
        
        return {
            "total_feedback": len(feedback_list),
            "average_rating": round(avg_rating, 2) if avg_rating else None,
            "rating_distribution": rating_dist,
            "feedback_types": feedback_types,
            "period_days": days
        }
        
    except Exception as e:
        logger.error(f"Error getting feedback stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feedback/recent")
async def get_recent_feedback(
    limit: int = Query(20, ge=1, le=100),
    feedback_type: Optional[str] = Query(None, description="Filter by feedback type"),
    db: Session = Depends(get_db)
):
    """
    Get recent feedback entries
    """
    try:
        query = db.query(UserFeedback)
        
        if feedback_type:
            query = query.filter(UserFeedback.feedback_type == feedback_type)
        
        feedback_list = query.order_by(
            UserFeedback.created_at.desc()
        ).limit(limit).all()
        
        return {
            "feedback": [
                {
                    "id": f.id,
                    "question": f.question[:100] + "..." if len(f.question) > 100 else f.question,
                    "rating": f.rating,
                    "feedback_type": f.feedback_type,
                    "feedback_text": f.feedback_text,
                    "has_correction": f.corrected_answer is not None,
                    "created_at": f.created_at.isoformat()
                }
                for f in feedback_list
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting recent feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feedback/ground-truth")
async def get_ground_truth_candidates(
    min_rating: int = Query(4, ge=1, le=5, description="Minimum rating"),
    with_corrections: bool = Query(True, description="Only include feedback with corrections"),
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db)
):
    """
    Get high-quality feedback suitable for ground truth dataset
    """
    try:
        query = db.query(UserFeedback).filter(
            UserFeedback.rating >= min_rating
        )
        
        if with_corrections:
            query = query.filter(UserFeedback.corrected_answer.isnot(None))
        
        feedback_list = query.order_by(
            UserFeedback.created_at.desc()
        ).limit(limit).all()
        
        return {
            "candidates": [
                {
                    "question": f.question,
                    "answer": f.answer,
                    "ground_truth": f.corrected_answer or f.answer,
                    "rating": f.rating,
                    "created_at": f.created_at.isoformat()
                }
                for f in feedback_list
            ],
            "total": len(feedback_list)
        }
        
    except Exception as e:
        logger.error(f"Error getting ground truth candidates: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
