"""SQLite storage for SupportPilot tickets (Work 2 - Ticket Ingestion)."""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "tickets.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tickets (
                ticket_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_name VARCHAR(100),
                email         VARCHAR(150),
                department    VARCHAR(100),
                title         VARCHAR(255),
                description   TEXT,
                category      VARCHAR(50),
                severity      VARCHAR(20),
                priority      VARCHAR(10),
                confidence    FLOAT,
                status        VARCHAR(30) DEFAULT 'Open',
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def insert_ticket(t):
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO tickets
              (employee_name, email, department, title, description,
               category, severity, priority, confidence, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (t["employee_name"], t["email"], t["department"], t["title"],
             t["description"], t["category"], t["severity"], t["priority"],
             t["confidence"], t.get("status", "Open")),
        )
        return cur.lastrowid


def get_ticket(ticket_id):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,)
        ).fetchone()
        return dict(row) if row else None


def list_tickets(limit=50):
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM tickets ORDER BY ticket_id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
