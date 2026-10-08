"""Larkspur's helpdesk backend, as it runs inside a support scenario's sandbox.

Every support action an agent can take (refunds, exchange exceptions, escalations, follow-ups,
dispatches, customer messages) is validated here and appended to an action ledger. The ledger
is the ground truth the verifier grades and the cost-of-service scorer prices.
"""
