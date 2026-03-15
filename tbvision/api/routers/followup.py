"""Follow-up chat endpoints that reuse saved report context."""

from fastapi import APIRouter, HTTPException, Request

from tbvision.api.schemas import (
    FollowUpRequest,
    FollowUpResponse,
    FollowUpHistoryEntry,
    FollowUpHistoryResponse,
)

router = APIRouter()


@router.post("/follow-up", response_model=FollowUpResponse)
async def follow_up(request: Request, payload: FollowUpRequest):
    report_store = request.app.state.report_store
    report = report_store.get_report(payload.report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    # Build answer using the generation service
    history = report_store.list_followups(payload.report_id)
    answer = await request.app.state.generation_service.generate_followup(
        report_context=report,
        question=payload.question,
        history=history,
    )

    report_store.append_followup(payload.report_id, payload.question, answer)
    updated_history = report_store.list_followups(payload.report_id)

    return FollowUpResponse(
        report_id=payload.report_id,
        question=payload.question,
        answer=answer,
        history=[FollowUpHistoryEntry(**entry) for entry in updated_history],
    )


@router.get("/follow-up/history", response_model=FollowUpHistoryResponse)
async def follow_up_history(report_id: str, request: Request):
    report_store = request.app.state.report_store
    report = report_store.get_report(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    history = report_store.list_followups(report_id)
    return FollowUpHistoryResponse(
        report_id=report_id,
        history=[FollowUpHistoryEntry(**entry) for entry in history],
    )
