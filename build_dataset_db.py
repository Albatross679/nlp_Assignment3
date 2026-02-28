"""Build a consolidated SQLite database from all NL-SQL data files and the flight database."""

import re
import sqlite3


def main():
    db_path = "data/nl2sql_dataset.db"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # --- 1. NL-SQL pairs table ---
    cur.execute("DROP TABLE IF EXISTS nl_sql_pairs")
    cur.execute("""
    CREATE TABLE nl_sql_pairs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        split TEXT NOT NULL,
        nl_query TEXT NOT NULL,
        sql_query TEXT
    );
    """)

    # Train
    with open("data/train.nl") as f:
        train_nl = [line.strip() for line in f if line.strip()]
    with open("data/train.sql") as f:
        train_sql = [line.strip() for line in f if line.strip()]
    assert len(train_nl) == len(train_sql)
    for nl, sql in zip(train_nl, train_sql):
        cur.execute("INSERT INTO nl_sql_pairs (split, nl_query, sql_query) VALUES (?,?,?)",
                    ("train", nl, sql))

    # Dev
    with open("data/dev.nl") as f:
        dev_nl = [line.strip() for line in f if line.strip()]
    with open("data/dev.sql") as f:
        dev_sql = [line.strip() for line in f if line.strip()]
    assert len(dev_nl) == len(dev_sql)
    for nl, sql in zip(dev_nl, dev_sql):
        cur.execute("INSERT INTO nl_sql_pairs (split, nl_query, sql_query) VALUES (?,?,?)",
                    ("dev", nl, sql))

    # Test (NL only)
    with open("data/test.nl") as f:
        test_nl = [line.strip() for line in f if line.strip()]
    for nl in test_nl:
        cur.execute("INSERT INTO nl_sql_pairs (split, nl_query, sql_query) VALUES (?,?,?)",
                    ("test", nl, None))

    # --- 2. Stats view ---
    cur.execute("DROP VIEW IF EXISTS dataset_stats")
    cur.execute("""
    CREATE VIEW dataset_stats AS
    SELECT
        split,
        COUNT(*) AS num_examples,
        SUM(CASE WHEN sql_query IS NOT NULL THEN 1 ELSE 0 END) AS has_sql,
        ROUND(AVG(LENGTH(nl_query)), 1) AS avg_nl_chars,
        ROUND(AVG(LENGTH(sql_query)), 1) AS avg_sql_chars
    FROM nl_sql_pairs
    GROUP BY split
    ORDER BY CASE split WHEN 'train' THEN 1 WHEN 'dev' THEN 2 WHEN 'test' THEN 3 END;
    """)

    # --- 3. Flight DB schema metadata ---
    cur.execute("DROP TABLE IF EXISTS flight_db_schema")
    cur.execute("""
    CREATE TABLE flight_db_schema (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        table_name TEXT NOT NULL,
        column_name TEXT NOT NULL,
        column_type TEXT,
        is_primary_key INTEGER DEFAULT 0,
        column_position INTEGER
    );
    """)

    flight_db = sqlite3.connect("data/flight_database.db")
    flight_cur = flight_db.cursor()

    flight_cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    flight_tables = [row[0] for row in flight_cur.fetchall()]

    for tbl in flight_tables:
        flight_cur.execute(f'PRAGMA table_info("{tbl}")')
        for col in flight_cur.fetchall():
            cur.execute(
                "INSERT INTO flight_db_schema (table_name, column_name, column_type, is_primary_key, column_position) VALUES (?,?,?,?,?)",
                (tbl, col[1], col[2], col[5], col[0])
            )

    # --- 4. Copy flight DB content (prefixed tables) ---
    for tbl in flight_tables:
        prefixed = f"flight_{tbl}"
        cur.execute(f'DROP TABLE IF EXISTS "{prefixed}"')

        # Get original CREATE TABLE SQL
        flight_cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (tbl,))
        create_sql = flight_cur.fetchone()[0]

        # Replace the table name with the prefixed version
        create_sql = re.sub(
            r'CREATE\s+TABLE\s+"?' + re.escape(tbl) + r'"?',
            f'CREATE TABLE "{prefixed}"',
            create_sql,
            count=1,
            flags=re.IGNORECASE,
        )
        cur.execute(create_sql)

        # Copy rows
        flight_cur.execute(f'SELECT * FROM "{tbl}"')
        rows = flight_cur.fetchall()
        if rows:
            placeholders = ",".join(["?"] * len(rows[0]))
            cur.executemany(f'INSERT INTO "{prefixed}" VALUES ({placeholders})', rows)

    flight_db.close()
    conn.commit()

    # --- Print summary ---
    print(f"Database created: {db_path}\n")

    cur.execute("SELECT * FROM dataset_stats")
    print(f"{'Split':<8} {'Examples':>8} {'Has SQL':>8} {'Avg NL':>8} {'Avg SQL':>8}")
    print("-" * 44)
    for row in cur.fetchall():
        print(f"{row[0]:<8} {row[1]:>8} {row[2]:>8} {row[3]:>8} {str(row[4] or '-'):>8}")

    print(f"\nFlight DB tables copied ({len(flight_tables)}):")
    for tbl in flight_tables:
        cur.execute(f'SELECT COUNT(*) FROM "flight_{tbl}"')
        nrows = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM flight_db_schema WHERE table_name=?", (tbl,))
        ncols = cur.fetchone()[0]
        print(f"  {tbl:<25} {ncols:>3} cols, {nrows:>6} rows")

    cur.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
    ntables = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='view'")
    nviews = cur.fetchone()[0]
    print(f"\nTotal: {ntables} tables, {nviews} views")

    conn.close()


if __name__ == "__main__":
    main()
