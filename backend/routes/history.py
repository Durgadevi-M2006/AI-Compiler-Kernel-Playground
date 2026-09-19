"""
FastAPI router for session experiment history and CSV / JSON export
"""

from fastapi import APIRouter, Response
from backend.services.history_store import session_history

router = APIRouter(prefix="/api", tags=["Session History & Export"])


@router.get("/history")
async def get_history():
    """Returns all experiment and custom code runs in the current session."""
    return {"history": session_history.get_all()}


@router.get("/history/{item_id}")
async def get_history_item(item_id: int):
    """Retrieves a specific history run record by ID."""
    item = session_history.get_by_id(item_id)
    if not item:
        return {"status": "error", "message": f"History item #{item_id} not found"}, 404
    return {"status": "success", "item": item}


@router.delete("/history/{item_id}")
async def delete_history_item(item_id: int):
    """Deletes a specific history record by ID."""
    deleted = session_history.delete_by_id(item_id)
    if not deleted:
        return {"status": "error", "message": f"History item #{item_id} not found"}, 404
    return {"status": "success", "deleted_id": item_id}


@router.delete("/history")
async def clear_history():
    """Clears all session experiment history."""
    cleared_count = session_history.clear()
    return {"status": "success", "cleared_records": cleared_count}


@router.get("/export/csv")
async def export_csv_file():
    """Exports session history as a CSV download."""
    csv_data = session_history.export_csv()
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=experiment_results.csv"}
    )


@router.get("/export/json")
async def export_json_file():
    """Exports session history as a JSON download."""
    json_data = session_history.export_json()
    return Response(
        content=json_data,
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=experiment_results.json"}
    )
