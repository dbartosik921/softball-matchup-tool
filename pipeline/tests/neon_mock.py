"""A tiny stand-in for Neon's SQL-over-HTTP endpoint, backed by a real Postgres.

Speaks the same request/response shape as https://api.<region>.neon.tech/sql (as used by
@neondatabase/serverless): Neon-Connection-String header, {query, params} or {queries: [...]},
text values in array mode. Lets the HTTP transport be tested without a Neon account.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import psycopg
from psycopg import pq


def _run(conn: psycopg.Connection, q: dict) -> dict:
    params = [None if p is None else str(p).encode() for p in q.get("params", [])]
    res = conn.pgconn.exec_params(q["query"].encode(), params)
    if res.status not in (pq.ExecStatus.COMMAND_OK, pq.ExecStatus.TUPLES_OK):
        raise RuntimeError(res.error_message.decode())
    rows = [
        [None if res.get_value(r, c) is None else res.get_value(r, c).decode() for c in range(res.nfields)]
        for r in range(res.ntuples)
    ]
    fields = [{"name": res.fname(c).decode(), "dataTypeID": res.ftype(c)} for c in range(res.nfields)]
    tag = (res.command_status or b"").decode()
    return {"command": tag.split(" ")[0], "rowCount": res.ntuples, "rows": rows, "fields": fields}


def start(backing_url: str):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert self.path == "/sql"
            assert self.headers.get("Neon-Connection-String")
            assert self.headers.get("Neon-Array-Mode") == "true"
            try:
                with psycopg.connect(backing_url, autocommit=True) as conn:
                    if "queries" in body:
                        conn.pgconn.exec_(b"BEGIN")
                        try:
                            out = {"results": [_run(conn, q) for q in body["queries"]]}
                            conn.pgconn.exec_(b"COMMIT")
                        except Exception:
                            conn.pgconn.exec_(b"ROLLBACK")
                            raise
                    else:
                        out = _run(conn, body)
                code, payload = 200, out
            except Exception as e:  # noqa: BLE001
                code, payload = 400, {"message": str(e)}
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv
