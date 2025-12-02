from app.db.connection import engine
from sqlalchemy import text

try:
    with engine.connect() as conn:
        result = conn.execute(text("SELECT '✅ Connection successful!' FROM dual"))
        print(result.scalar())
except Exception as e:
    print("❌ Connection failed:", e)
