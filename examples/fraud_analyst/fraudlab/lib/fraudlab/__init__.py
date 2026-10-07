"""Kestrel Pay's fraud-review backend, as it runs inside a case's sandbox.

Every tool the analyst calls goes through here and is appended to the case's ledger with its
cost: SQL and Chalk feature queries are free, deep identity verification costs $5 a call and the
consortium social-network search $2. The verifier grades the ledger.
"""
