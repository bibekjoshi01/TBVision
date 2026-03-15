import json
import sqlite3
from pathlib import Path
from typing import Any


class ReportStore:
    REQUIRED_COLUMNS = {
        "summary": "TEXT",
        "evidence_summary": "TEXT",
        "uncertainty_level": "TEXT",
        "uncertainty_std": "REAL",
        "gradcam_region": "TEXT",
    }

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reports (
                    report_id TEXT PRIMARY KEY,
                    metadata TEXT,
                    analysis TEXT,
                    prediction TEXT,
                    probability REAL,
                    explanation TEXT,
                    summary TEXT,
                    evidence_summary TEXT,
                    uncertainty_level TEXT,
                    uncertainty_std REAL,
                    gradcam_region TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS followups (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_id TEXT,
                    question TEXT,
                    answer TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(report_id) REFERENCES reports(report_id)
                )
                """
            )
            self._ensure_columns(conn)

    def _ensure_columns(self, conn: sqlite3.Connection) -> None:
        existing = {
            row[1]
            for row in conn.execute("PRAGMA table_info(reports)").fetchall()
        }
        for column, col_type in self.REQUIRED_COLUMNS.items():
            if column not in existing:
                conn.execute(
                    f"ALTER TABLE reports ADD COLUMN {column} {col_type}"
                )

    def save_report(
        self,
        report_id: str,
        metadata: dict[str, Any],
        analysis: dict[str, Any],
        *,
        summary: str | None = None,
        evidence_summary: str | None = None,
        uncertainty_level: str | None = None,
        uncertainty_std: float | None = None,
        gradcam_region: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO reports
                (report_id, metadata, analysis, prediction, probability, explanation, summary, evidence_summary, uncertainty_level, uncertainty_std, gradcam_region)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report_id,
                    json.dumps(metadata),
                    json.dumps(analysis),
                    analysis.get("prediction"),
                    analysis.get("probability"),
                    analysis.get("explanation"),
                    summary,
                    evidence_summary,
                    uncertainty_level,
                    uncertainty_std,
                    gradcam_region,
                ),
            )

    def get_report(self, report_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT metadata, analysis, summary, evidence_summary, uncertainty_level, uncertainty_std, gradcam_region
                FROM reports
                WHERE report_id = ?
                """,
                (report_id,),
            ).fetchone()
        if not row:
            return None
        metadata = json.loads(row[0]) if row[0] else {}
        analysis = json.loads(row[1]) if row[1] else {}
        return {
            "report_id": report_id,
            "metadata": metadata,
            "analysis": analysis,
            "summary": row[2],
            "evidence_summary": row[3],
            "uncertainty_level": row[4],
            "uncertainty_std": row[5],
            "gradcam_region": row[6],
        }

    def append_followup(
        self,
        report_id: str,
        question: str,
        answer: str,
    ) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO followups (report_id, question, answer)
                VALUES (?, ?, ?)
                """,
                (report_id, question, answer),
            )
            return cursor.lastrowid

    def list_followups(self, report_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT question, answer, created_at
                FROM followups
                WHERE report_id = ?
                ORDER BY id ASC
                """,
                (report_id,),
            ).fetchall()
        return [
            {"question": row[0], "answer": row[1], "created_at": row[2]}
            for row in rows
        ]
