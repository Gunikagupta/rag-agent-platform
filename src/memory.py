# src/memory.py
import psycopg2
from src.db import get_conn

def save_conversation(session_id: str, question: str, answer: str, retrieved_chunks: list = None, chunk_ids: list = None):
    """Log interaction and retrieved paper IDs into CockroachDB."""
    try:
        # Extract paper_ids if retrieved_chunks list of dicts is passed
        final_ids = []
        if retrieved_chunks:
            final_ids = [c.get("paper_id", "") if isinstance(c, dict) else str(c) for c in retrieved_chunks]
        elif chunk_ids:
            final_ids = chunk_ids

        conn = get_conn()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO conversations (session_id, question, answer, retrieved_chunk_ids)
            VALUES (%s, %s, %s, %s);
        """, (session_id, question, answer, final_ids))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Database logging note: {e}")
def recall_memory(session_id: str, query_hint: str, limit: int = 3) -> list[dict]:
    """Agentic Memory Tool: Search past interactions in this session for historical context."""
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT question, answer, created_at FROM conversations
            WHERE session_id = %s AND (question ILIKE %s OR answer ILIKE %s)
            ORDER BY created_at DESC LIMIT %s;
        """, (session_id, f"%{query_hint}%", f"%{query_hint}%", limit))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [{"question": r[0], "answer": r[1], "timestamp": str(r[2])} for r in rows]
    except Exception as e:
        print(f"Memory recall note: {e}")
        return []