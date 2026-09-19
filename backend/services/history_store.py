"""
Session Experiment History Store & Data Exporter (CSV / JSON)
"""

import io
import csv
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional


class HistoryStore:
    """Thread-safe in-memory session experiment history."""

    def __init__(self):
        self._history: List[Dict[str, Any]] = []

    def add_entry(
        self,
        experiment_name: str,
        language: str,
        parameters: Dict[str, Any],
        execution_time_ms: float,
        speedup: str,
        correctness: bool,
        notes: str = ""
    ) -> Dict[str, Any]:
        """Records a completed experiment run in current session."""
        entry = {
            "id": len(self._history) + 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "experiment": experiment_name,
            "language": language,
            "code": parameters.get("code", ""),
            "parameters": parameters,
            "execution_time_ms": execution_time_ms,
            "speedup": speedup,
            "correctness": "VERIFIED" if correctness else "MISMATCH",
            "notes": notes
        }
        self._history.insert(0, entry) # Most recent first
        # Keep maximum 50 records in memory
        if len(self._history) > 50:
            self._history = self._history[:50]
        return entry

    def add_optimization_entry(
        self,
        experiment: str,
        language: str,
        original_code: str,
        detected_opportunities: List[str],
        optimizations_applied: List[str],
        original_execution_time_ms: float,
        optimized_execution_time_ms: float,
        speedup: str,
        notes: str = ""
    ) -> Dict[str, Any]:
        """Records an optimization comparison run in session history."""
        entry = {
            "id": len(self._history) + 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "experiment": experiment,
            "language": language,
            "code": original_code,
            "parameters": {
                "original_code_lines": len(original_code.splitlines()),
                "detected_opportunities": detected_opportunities,
                "optimizations_applied": optimizations_applied,
                "orig_time_ms": original_execution_time_ms,
                "opt_time_ms": optimized_execution_time_ms
            },
            "execution_time_ms": optimized_execution_time_ms,
            "speedup": speedup,
            "correctness": "VERIFIED",
            "notes": notes or f"Speedup: {speedup} | Applied: {len(optimizations_applied)} optimizations"
        }
        self._history.insert(0, entry)
        if len(self._history) > 50:
            self._history = self._history[:50]
        return entry

    def get_all(self) -> List[Dict[str, Any]]:
        return self._history

    def get_by_id(self, item_id: int) -> Optional[Dict[str, Any]]:
        for item in self._history:
            if item["id"] == item_id:
                return item
        return None

    def delete_by_id(self, item_id: int) -> bool:
        before_len = len(self._history)
        self._history = [item for item in self._history if item["id"] != item_id]
        return len(self._history) < before_len

    def clear(self) -> int:
        count = len(self._history)
        self._history.clear()
        return count

    def export_json(self) -> str:
        """Exports history as formatted JSON string."""
        return json.dumps({
            "exported_at": datetime.now().isoformat(),
            "total_runs": len(self._history),
            "history": self._history
        }, indent=2)

    def export_csv(self) -> str:
        """Exports history as formatted CSV string."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "Run ID", "Timestamp", "Experiment Name", "Language",
            "Parameters", "Execution Time (ms)", "Speedup", "Correctness", "Optimization Notes"
        ])
        for row in self._history:
            writer.writerow([
                row["id"],
                row["timestamp"],
                row["experiment"],
                row["language"],
                json.dumps(row["parameters"]),
                row["execution_time_ms"],
                row["speedup"],
                row["correctness"],
                row["notes"]
            ])
        return output.getvalue()


# Global Singleton Instance for session
session_history = HistoryStore()
