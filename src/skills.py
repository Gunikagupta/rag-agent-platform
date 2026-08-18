# src/skills.py
import psycopg2

class RoachMindSkills:
    """Structured operational skills for CockroachDB vector search and state management."""
    
    def __init__(self, conn_str: str):
        self.conn_str = conn_str

    def search_episodic_memory(self, session_id: str):
        """Skill: Retrieve past interaction turns for active session state from CockroachDB."""
        try:
            with psycopg2.connect(self.conn_str) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT question, answer FROM conversation_memory WHERE session_id = %s ORDER BY created_at DESC LIMIT 3;",
                        (session_id,)
                    )
                    return cur.fetchall()
        except Exception:
            return []

    def log_procedural_state(self, session_id: str, prompt_type: str):
        """Skill: Log current agent state and active execution path to CockroachDB."""
        try:
            with psycopg2.connect(self.conn_str) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO agent_state (session_id, execution_type) VALUES (%s, %s) ON CONFLICT DO NOTHING;",
                        (session_id, prompt_type)
                    )
                    conn.commit()
        except Exception:
            pass