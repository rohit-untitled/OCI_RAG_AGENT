import os
import json
import logging
import time
from typing import Dict, Any, List, Optional
import cx_Oracle

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ORACLE VECTOR DB CONFIG

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
            getmode=cx_Oracle.SPOOL_ATTRVAL_WAIT,
        )
        logger.info("Oracle SessionPool created.")
    return _pool


def get_connection() -> cx_Oracle.Connection:
    return get_pool().acquire()


def close_pool():
    global _pool
    if _pool:
        try:
            _pool.close()
        except Exception as e:
            logger.exception("Error closing pool: %s", e)
        finally:
            _pool = None


# Test connection
def test_connection() -> Dict[str, Any]:
    try:
        conn = cx_Oracle.connect(DB_USER, DB_PASSWORD, DB_TNS)
        cur = conn.cursor()
        cur.execute("SELECT USER FROM dual")
        row = cur.fetchone()
        cur.close()
        conn.close()
        return {"ok": True, "user": row[0]}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# Insert single embedding
def insert_embedding_record(chunk_text: str, embedding_vector: List[float], metadata: Dict[str, Any]):
    conn = get_connection()
    cur = conn.cursor()

    metadata_json = json.dumps(metadata)

    try:
        # Oracle requires JSON array-like string for TO_VECTOR()
        embedding_string = "[" + ",".join(map(str, embedding_vector)) + "]"

        cur.execute("""
            INSERT INTO ai_vector_store (chunk, embedding, metadata)
            VALUES (:chunk, TO_VECTOR(:embedding_string), :metadata)
        """, {
            "chunk": chunk_text,
            "embedding_string": embedding_string,
            "metadata": metadata_json
        })

        conn.commit()
        logger.info("Inserted embedding successfully.")

    finally:
        cur.close()
        get_pool().release(conn)

# Insert multiple embeddings from JSON
def insert_embeddings_from_json(json_file_path: str):
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
        embedding_string = "[" + ",".join(map(str, entry["embedding"])) + "]"
        cur.execute(sql, {
            "chunk": entry["chunk"],
            "embedding_string": embedding_string,
            "metadata": json.dumps(entry.get("metadata", {}))
        })

    conn.commit()
    cur.close()
    get_pool().release(conn)
    logger.info("Batch embedding insert completed.")

def search_similar_chunks(query_embedding: list, top_k: int = 5) -> list[dict]:
    conn = None
    cur = None

    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.arraysize = top_k
        cur.prefetchrows = top_k

        embedding_string = "[" + ",".join(str(v) for v in query_embedding) + "]"

        sql = f"""
            SELECT chunk, metadata
            FROM ai_vector_store
            ORDER BY embedding <=> TO_VECTOR(:embedding_string)
            FETCH FIRST :top_k ROWS ONLY
        """

        cur.execute(sql, embedding_string=embedding_string, top_k=top_k)

        hits = []

        for chunk, metadata_json in cur:
            # Convert chunk
            if hasattr(chunk, "read"):
                chunk_text = chunk.read()
            else:
                chunk_text = str(chunk)

            # Convert metadata
            if hasattr(metadata_json, "read"):
                metadata_text = metadata_json.read()
            else:
                metadata_text = str(metadata_json)

            hits.append({
                "chunk": chunk_text,
                "metadata": json.loads(metadata_text)
            })

        return hits

    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
