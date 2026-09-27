"""Relational realisation of the time-expanded property-graph view (DuckDB).

Base relations : node(nid), slot(k, nk), contact(u, v, k)
View (arity-2 node ids, arity-3 edge ids, cf. Rotschield & Peterfreund):
  vnode(nid, k)                = node x slot                      -- Q1
  vedge(sn, sk, tn, tk, lab)   = transmission edges (u,k)->(v,k)  -- Q2..Q5
                                 hold edges        (n,k)->(n,nk)
DuckDB has no SQL/PGQ GRAPH_TABLE in its core distribution, so the Kleene-star
reachability pattern (a)-[]->*(b) is executed as the equivalent recursive CTE.
"""
from __future__ import annotations
import time
import duckdb
import numpy as np

VIEW_SQL = """
CREATE OR REPLACE TABLE vnode AS
  SELECT n.nid, s.k FROM node n CROSS JOIN slot s;
CREATE OR REPLACE TABLE vedge AS
  SELECT c.u AS sn, c.k AS sk, c.v AS tn, c.k AS tk, 'TX' AS lab
    FROM contact c JOIN node a ON a.nid = c.u JOIN node b ON b.nid = c.v
  UNION ALL
  SELECT n.nid, s.k, n.nid, s.nk, 'HOLD'
    FROM node n CROSS JOIN slot s WHERE s.nk IS NOT NULL;
"""

# tau = 1: every transmission occupies one slot, TX edges (u,k)->(v,k+1) via Succ
VIEW_SQL_TAU1 = """
CREATE OR REPLACE TABLE vnode AS
  SELECT n.nid, s.k FROM node n CROSS JOIN slot s;
CREATE OR REPLACE TABLE vedge AS
  SELECT c.u AS sn, c.k AS sk, c.v AS tn, s.nk AS tk, 'TX' AS lab
    FROM contact c JOIN node a ON a.nid = c.u JOIN node b ON b.nid = c.v
    JOIN slot s ON s.k = c.k WHERE s.nk IS NOT NULL
  UNION ALL
  SELECT n.nid, s.k, n.nid, s.nk, 'HOLD'
    FROM node n CROSS JOIN slot s WHERE s.nk IS NOT NULL;
"""

EARLIEST_SQL = """
WITH RECURSIVE reach(n, k) AS (
    SELECT ?::INTEGER, ?::INTEGER
  UNION
    SELECT e.tn, e.tk
    FROM reach r JOIN vedge e ON e.sn = r.n AND e.sk = r.k
    WHERE e.tk <= ?
)
SELECT min(k) FROM reach WHERE n = ?;
"""


class SQLView:
    def __init__(self, cp, threads=1, tau=0):
        self.con = duckdb.connect()
        self.con.execute(f"SET threads={threads}")
        self.con.execute("CREATE TABLE node(nid INTEGER)")
        self.con.execute("CREATE TABLE slot(k INTEGER, nk INTEGER)")
        self.con.execute("CREATE TABLE contact(u INTEGER, v INTEGER, k INTEGER)")
        self.con.executemany("INSERT INTO node VALUES (?)", [(i,) for i in range(cp.N)])
        self.con.executemany("INSERT INTO slot VALUES (?, ?)",
                             [(k, k + 1 if k + 1 < cp.T else None) for k in range(cp.T)])
        import pandas as pd
        df = pd.DataFrame({"u": cp.u, "v": cp.v, "k": cp.k})
        self.con.register("cdf", df)
        self.con.execute("INSERT INTO contact SELECT u, v, k FROM cdf")
        t = time.perf_counter()
        self.con.execute(VIEW_SQL_TAU1 if tau else VIEW_SQL)
        self.materialise_s = time.perf_counter() - t
        self.n_vnode = self.con.execute("SELECT count(*) FROM vnode").fetchone()[0]
        self.n_vedge = self.con.execute("SELECT count(*) FROM vedge").fetchone()[0]
        try:
            # in-memory table storage of the whole database (base tables + view tables)
            self.mem_mb = self.con.execute(
                "SELECT memory_usage_bytes FROM duckdb_memory() WHERE tag = 'IN_MEMORY_TABLE'").fetchone()[0] / 2**20
        except Exception:
            self.mem_mb = float("nan")

    def earliest(self, s, d, k0, k1):
        r = self.con.execute(EARLIEST_SQL, [s, k0, k1, d]).fetchone()[0]
        return r
