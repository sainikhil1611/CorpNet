"""Quick script to inspect the indexed data."""
import sqlite3, json, os

db = "data/archrag.db"
vecs = "data/chnsw_vectors.json"

if not os.path.exists(db):
    print("No database found.")
else:
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    print(f"Tables: {tables}")
    for t in tables:
        count = cur.execute(f"SELECT count(*) FROM [{t}]").fetchone()[0]
        print(f"  {t}: {count} rows")
        if count > 0:
            row = cur.execute(f"SELECT * FROM [{t}] LIMIT 1").fetchone()
            cols = [d[0] for d in cur.description]
            print(f"    columns: {cols}")
            print(f"    sample:  {row[:3]}...")
    conn.close()

if os.path.exists(vecs):
    with open(vecs) as f:
        data = json.load(f)
    print(f"\nVector index layers: {list(data.keys())}")
    for layer, vecs_dict in data.items():
        print(f"  layer {layer}: {len(vecs_dict)} vectors")
