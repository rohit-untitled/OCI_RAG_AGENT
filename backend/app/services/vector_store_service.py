import os
import json
import logging
import time
import math
from typing import Dict, Any, List, Optional
import cx_Oracle

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Configuration
WALLET_PATH = os.getenv(
    "ORACLE_WALLET_PATH",
    r"C:\Users\shshrohi\Desktop\ai_rag_agent\backend\Wallet_POCSOLUTIONSATPDEV_Nov_2025"
)
os.environ["TNS_ADMIN"] = WALLET_PATH

DB_USER = os.getenv("ORACLE_DB_USER", "gsc_km_2")
DB_PASSWORD = os.getenv("ORACLE_DB_PASSWORD", "Pa$$word#234")
DB_TNS = os.getenv("ORACLE_DB_TNS", "pocsolutionsatpdev_high")

VECTOR_DIM = int(os.getenv("VECTOR_DIM", "1536"))

_pool: Optional[cx_Oracle.SessionPool] = None

# Connection Pool
def get_pool() -> cx_Oracle.SessionPool:
    global _pool
    if _pool is None:
        logger.info("Initializing Oracle SessionPool...")
        _pool = cx_Oracle.SessionPool(
            user=DB_USER,
            password=DB_PASSWORD,
            dsn=DB_TNS,
            min=1,
            max=10,
            increment=1,
            encoding="UTF-8",
            threaded=True,
            getmode=cx_Oracle.SPOOL_ATTRVAL_WAIT
        )
        logger.info("Oracle SessionPool created.")
    return _pool

def get_connection() -> cx_Oracle.Connection:
    pool = get_pool()
    return pool.acquire()

def close_pool():
    global _pool
    if _pool:
        try:
            logger.info("Closing Oracle SessionPool...")
            _pool.close()
        except Exception as e:
            logger.exception("Error closing pool: %s", e)
        finally:
            _pool = None

# Test Connection
def test_connection(timeout_seconds: int = 5) -> Dict[str, Any]:
    t0 = time.time()
    try:
        conn = cx_Oracle.connect(user=DB_USER, password=DB_PASSWORD, dsn=DB_TNS, encoding="UTF-8")
        cur = conn.cursor()
        cur.execute("SELECT USER FROM dual")
        row = cur.fetchone()
        cur.close()
        conn.close()
        elapsed = round(time.time() - t0, 2)
        return {"ok": True, "user": row[0] if row else None, "elapsed": elapsed}
    except Exception as e:
        logger.exception("test_connection failed")
        return {"ok": False, "error": str(e), "elapsed": round(time.time() - t0, 2)}

def insert_embedding_record(chunk_text: str, embedding_vector: List[float], metadata: Dict[str, Any]):
    conn = get_connection()
    cur = conn.cursor()

    metadata_json = json.dumps(metadata or {})

    try:
        vector_var = cur.arrayvar(cx_Oracle.NUMBER, embedding_vector)

        cur.execute("""
            INSERT INTO ai_vector_store (chunk, embedding, metadata)
            VALUES (:chunk, :embedding, :metadata)
        """, {
            "chunk": chunk_text,
            "embedding": vector_var,
            "metadata": metadata_json
        })

        conn.commit()
        logger.info("Inserted embedding chunk successfully.")

    finally:
        cur.close()
        try:
            pool = get_pool()
            pool.release(conn)
        except Exception:
            conn.close()

# Insert embeddings from JSON file

def insert_embeddings_from_json(json_file_path):
    conn = get_connection()
    cur = conn.cursor()

    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sql = """
        INSERT INTO ai_vector_store (chunk, embedding, metadata)
        VALUES (
            :chunk,
            TO_VECTOR(:embedding_string),
            :metadata
        )
    """

    for entry in data:
        chunk = entry["chunk"]
        embedding_list = entry["embedding"]
        metadata = json.dumps(entry.get("metadata", {}))

        embedding_string = "[" + ",".join(str(v) for v in embedding_list) + "]"

        cur.execute(sql, {
            "chunk": chunk,
            "embedding_string": embedding_string,
            "metadata": metadata
        })

    conn.commit()
    cur.close()
    conn.close()


def search_similar_chunks(query_embedding: list, top_k: int = 5, filters: dict = None):
    conn = get_connection()
    cur = conn.cursor()

    query_embedding = normalize(query_embedding)
    embedding_string = "[" + ",".join(str(v) for v in query_embedding) + "]"

    filter_sql = ""
    params = {"embedding_string": embedding_string, "top_k": top_k}

    if filters:
        for key, value in filters.items():
            filter_sql += f" AND JSON_VALUE(metadata, '$.{key}') = :{key} "
            params[key] = value

    sql = f"""
        SELECT chunk, metadata
        FROM ai_vector_store
        WHERE 1=1 {filter_sql}
        ORDER BY embedding <=> TO_VECTOR(:embedding_string)
        FETCH FIRST :top_k ROWS ONLY
    """

    cur.execute(sql, params)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    result = []
    for chunk, metadata_json in rows:
        result.append({
            "chunk": chunk.read() if hasattr(chunk, "read") else chunk,
            "metadata": json.loads(metadata_json.read() if hasattr(metadata_json, "read") else metadata_json)
        })

    return result

