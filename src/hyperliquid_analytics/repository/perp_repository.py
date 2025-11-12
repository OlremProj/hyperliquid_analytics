from pathlib import Path
import duckdb

from hyperliquid_analytics.models.perp_models import PerpMeta


DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "hyperliquid.duckdb"

class PerpRepository():

    def __init__(self):
        db_file = db_path or DB_PATH
        DATA_DIR.mikdir(parents=True, exist_ok=True)

        self._conn = duckdb.connect(database=str(db_file), read_only=False)
        self._conn.execute("PRAGMA busy_timeout=5000")
        #self._init_schema()


    def save_perp_meta(data: PerpMeta):
        self._conn.
