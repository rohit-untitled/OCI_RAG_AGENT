import os
from dotenv import load_dotenv
import oracledb
from sqlalchemy import create_engine

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_TNS_ALIAS = os.getenv("DB_TNS_ALIAS")
WALLET_PATH = os.getenv("WALLET_PATH")
INSTANT_CLIENT_DIR = os.getenv("INSTANT_CLIENT_DIR")

oracledb.init_oracle_client(lib_dir=INSTANT_CLIENT_DIR, config_dir=WALLET_PATH)

engine = create_engine(
    f"oracle+oracledb://{DB_USER}:{DB_PASSWORD}@{DB_TNS_ALIAS}",
    connect_args={
        "config_dir": WALLET_PATH,
        "wallet_location": WALLET_PATH,
        "wallet_password": None,
    },
    echo=False,
)
