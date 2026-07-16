"""
Text-to-SQL evaluation metrics, following the conventions used by standard
benchmarks like Spider and WikiSQL:

  - EM  (Exact Match)        : normalized string equality
  - EX  (Execution Accuracy) : do predicted & gold return the same result set
                                when run against real/synthetic data?
  - Valid SQL rate           : does the predicted query execute without error?
  - Component Match          : partial credit comparing SELECT/WHERE/etc.
                                clauses independently (softer than EM, catches
                                cases where clause order differs but meaning
                                is identical)

EX is generally considered the most reliable single metric, since it captures
semantic correctness regardless of surface phrasing -- this mirrors why
Spider's leaderboard reports execution accuracy as its primary metric rather
than exact string match.
"""

import random
import re
import sqlite3
import string
from collections import Counter

try:
    import sqlparse
    SQLPARSE_AVAILABLE = True
except ImportError:
    SQLPARSE_AVAILABLE = False


def normalize_sql(sql: str) -> str:
    sql = sql.strip().rstrip(";").lower()
    sql = re.sub(r"\s+", " ", sql)
    return sql


def exact_match(pred: str, gold: str) -> bool:
    return normalize_sql(pred) == normalize_sql(gold)


# ---------------------------------------------------------------------------
# Execution-based metrics
# ---------------------------------------------------------------------------
def _random_value(col_type: str):
    col_type = col_type.upper()
    if "INT" in col_type:
        return random.randint(1, 100)
    if any(t in col_type for t in ("REAL", "FLOAT", "DOUBLE", "DECIMAL")):
        return round(random.uniform(1, 100), 2)
    return "".join(random.choices(string.ascii_lowercase, k=6))


def _populate_synthetic_data(conn, context_sql: str, rows_per_table: int = 5):
    cursor = conn.cursor()
    table_defs = re.findall(
        r"CREATE TABLE\s+([`\"]?\w+[`\"]?)\s*\((.*?)\)",
        context_sql, re.IGNORECASE | re.DOTALL,
    )
    for table_name, columns_str in table_defs:
        table_name = table_name.strip("`\"")
        col_defs = [c.strip() for c in columns_str.split(",")]
        col_names, col_types = [], []
        for col in col_defs:
            parts = col.split()
            if len(parts) < 2:
                continue
            col_names.append(parts[0].strip("`\""))
            col_types.append(parts[1])
        if not col_names:
            continue
        placeholders = ", ".join(["?"] * len(col_names))
        insert_sql = f"INSERT INTO {table_name} ({', '.join(col_names)}) VALUES ({placeholders})"
        for _ in range(rows_per_table):
            row = [_random_value(t) for t in col_types]
            try:
                cursor.execute(insert_sql, row)
            except sqlite3.Error:
                pass
    conn.commit()


def _run_query(conn, sql: str):
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        return True, cursor.fetchall()
    except sqlite3.Error as e:
        return False, str(e)


def execution_metrics(context: str, pred_sql: str, gold_sql: str, rows_per_table: int = 5):
    """Returns (is_valid, results_match) for one example."""
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(context)
    except sqlite3.Error:
        conn.close()
        return False, False

    _populate_synthetic_data(conn, context, rows_per_table)

    pred_ok, pred_result = _run_query(conn, pred_sql)
    gold_ok, gold_result = _run_query(conn, gold_sql)
    conn.close()

    if not pred_ok:
        return False, False
    if not gold_ok:
        return True, False

    results_match = Counter(map(tuple, pred_result)) == Counter(map(tuple, gold_result))
    return True, results_match


# ---------------------------------------------------------------------------
# Component match (soft partial-credit metric, optional -- requires sqlparse)
# ---------------------------------------------------------------------------
def component_match(pred: str, gold: str) -> float:
    """
    Rough component-level overlap: compares SELECT / FROM / WHERE / GROUP BY /
    ORDER BY clauses independently and returns the fraction that match after
    normalization. This is a simplified stand-in for the kind of partial
    credit used in Spider's "component matching" metric -- not a full
    reimplementation, but useful for spotting *which* clause type a model
    struggles with most.
    """
    if not SQLPARSE_AVAILABLE:
        return float(exact_match(pred, gold))

    def extract_clauses(sql):
        sql_norm = normalize_sql(sql)
        clauses = {}
        for kw in ["select", "from", "where", "group by", "order by", "having", "join"]:
            pattern = rf"{kw}\s+(.*?)(?=\s+(?:select|from|where|group by|order by|having|join|limit)|$)"
            match = re.search(pattern, sql_norm)
            clauses[kw] = match.group(1).strip() if match else None
        return clauses

    pred_clauses = extract_clauses(pred)
    gold_clauses = extract_clauses(gold)

    present_in_gold = [k for k, v in gold_clauses.items() if v is not None]
    if not present_in_gold:
        return 1.0 if exact_match(pred, gold) else 0.0

    matches = sum(1 for k in present_in_gold if pred_clauses.get(k) == gold_clauses.get(k))
    return matches / len(present_in_gold)
