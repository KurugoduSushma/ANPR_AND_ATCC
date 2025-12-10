
import sqlite3
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "traffic.db")

ATCC_CSV = os.path.join(BASE_DIR, "data", "logs", "atcc_results.csv")
ANPR_CSV = os.path.join(BASE_DIR, "data", "logs", "anpr_results.csv")

conn = sqlite3.connect(DB_PATH)

atcc_df = pd.read_csv(ATCC_CSV)
anpr_df = pd.read_csv(ANPR_CSV)

atcc_df.to_sql("atcc", conn, if_exists="replace", index=False)
anpr_df.to_sql("anpr", conn, if_exists="replace", index=False)

conn.close()
print("✅ Database created successfully")
