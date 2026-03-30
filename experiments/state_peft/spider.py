from __future__ import annotations

import re
import sqlite3
import subprocess
import sys
import tempfile
from typing import Any

from state_peft.common import apply_chat_template_no_thinking


def format_spider_prompt(question: str, db_id: str, schema: str, tokenizer: Any) -> str:
    return apply_chat_template_no_thinking(
        tokenizer,
        [{
            "role": "user",
            "content": (
                f"Given the following SQLite database schema for database '{db_id}':\n\n"
                f"{schema}\n\n"
                f"Write a SQL query to answer: {question}\n\n"
                "Output only the SQL query, no explanation."
            ),
        }],
    )


def extract_sql(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    m = re.search(r"```(?:sql)?\s*\n?(.*?)```", cleaned, re.DOTALL)
    if m:
        cleaned = m.group(1).strip()
    else:
        m = re.search(
            r"((?:SELECT|INSERT|UPDATE|DELETE|WITH|CREATE)\b.+)",
            cleaned, re.DOTALL | re.IGNORECASE,
        )
        if m:
            cleaned = m.group(1).strip()
    cleaned = cleaned.strip().rstrip(";") + ";"
    return cleaned


def execute_sql(sql: str, db_path: str, timeout: float = 10.0) -> list | None:
    script = (
        "import sqlite3, sys, json\n"
        "conn = sqlite3.connect(sys.argv[1])\n"
        "conn.execute('PRAGMA busy_timeout = 5000')\n"
        "try:\n"
        "    cur = conn.execute(sys.argv[2])\n"
        "    rows = cur.fetchall()\n"
        "    print(json.dumps(rows))\n"
        "except Exception as e:\n"
        "    print('ERROR:' + str(e), file=sys.stderr)\n"
        "    sys.exit(1)\n"
    )
    try:
        r = subprocess.run(
            [sys.executable, "-c", script, db_path, sql],
            capture_output=True, text=True, timeout=timeout,
        )
        if r.returncode != 0:
            return None
        import json
        return json.loads(r.stdout)
    except (subprocess.TimeoutExpired, Exception):
        return None


def check_sql_answer(predicted_sql: str, gold_sql: str, db_path: str) -> bool:
    pred_rows = execute_sql(predicted_sql, db_path)
    gold_rows = execute_sql(gold_sql, db_path)
    if pred_rows is None or gold_rows is None:
        return False
    pred_set = set(tuple(r) for r in pred_rows)
    gold_set = set(tuple(r) for r in gold_rows)
    return pred_set == gold_set


def get_schema(db_path: str) -> str:
    import os
    if not os.path.isfile(db_path):
        raise FileNotFoundError(
            f"Spider database not found: {db_path}. "
            "Did download_spider_dbs() run successfully?"
        )
    conn = sqlite3.connect(db_path)
    cur = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL")
    tables = [row[0] for row in cur.fetchall()]
    conn.close()
    if not tables:
        raise ValueError(
            f"No tables found in {db_path}. The database file may be empty or corrupt."
        )
    return "\n\n".join(tables)
