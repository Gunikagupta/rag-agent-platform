# src/db.py
import os
import psycopg2

CONN_STR = os.environ.get(
    "COCKROACH_CONN_STRING",
    "postgresql://gunika:rpEVIqirX3d0gYZOjk-cTQ@joyous-runner-32378.j77.aws-ap-south-1.cockroachlabs.cloud:26257/defaultdb?sslmode=require"
)

def get_conn():
    return psycopg2.connect(CONN_STR)

def setup_tables():
    conn = get_conn()
    cur = conn.cursor()
    
    # Paper Vector Embeddings Table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS paper_embeddings (
            chunk_id STRING PRIMARY KEY,
            paper_id STRING,
            title STRING,
            text STRING,
            embedding VECTOR(384)
        );
    """)
    
    # Conversational Agentic Memory Table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            session_id STRING,
            question STRING,
            answer STRING,
            retrieved_chunk_ids STRING[],
            created_at TIMESTAMP DEFAULT now()
        );
    """)
    
    conn.commit()
    cur.close()
    conn.close()
    print("CockroachDB vector and conversation memory tables ready.")

if __name__ == "__main__":
    setup_tables()