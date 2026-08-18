import json
import psycopg2
from datetime import datetime

CONN_STR = "postgresql://gunika:rpEVIqirX3d0gYZOjk-cTQ@joyous-runner-32378.j77.aws-ap-south-1.cockroachlabs.cloud:26257/defaultdb?sslmode=require"

def init_memory_db():
    """Creates the conversation memory table in CockroachDB."""
    conn = psycopg2.connect(CONN_STR)
    cur = conn.cursor()
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS conversation_memory (
            memory_id STRING PRIMARY KEY,
            session_id STRING,
            question STRING,
            retrieved_chunks JSONB,
            answer STRING,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    cur.close()
    conn.close()
    print("Memory table initialized in CockroachDB!")

def save_conversation(session_id: str, question: str, retrieved_chunks: list, answer: str):
    """Logs an agent interaction to CockroachDB memory."""
    conn = psycopg2.connect(CONN_STR)
    cur = conn.cursor()
    
    import uuid
    memory_id = str(uuid.uuid4())
    
    cur.execute("""
        INSERT INTO conversation_memory (memory_id, session_id, question, retrieved_chunks, answer)
        VALUES (%s, %s, %s, %s, %s);
    """, (
        memory_id,
        session_id,
        question,
        json.dumps(retrieved_chunks),
        answer
    ))
    
    conn.commit()
    cur.close()
    conn.close()

if __name__ == "__main__":
    init_memory_db()