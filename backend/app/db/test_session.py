from app.db.session import get_db

def main():
    db_gen = get_db()
    db = next(db_gen)
    try:
        result = db.execute("SELECT '✅ Session is working!' FROM dual")
        print(result.scalar())
    finally:
        db_gen.close()

if __name__ == "__main__":
    main()
