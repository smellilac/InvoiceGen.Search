"""Contract / snapshot tests for the search wire shapes.

This service is the *receiving* side of the C# backend → Python request
(Pair 2) and the *sending* side of the Python → C# backend response (Pair 3).
These tests pin down the exact JSON shape on both sides so an accidental model
change (a renamed field, a lost default, a flat-vs-nested regression) fails
loudly instead of silently breaking the integration.

The ``__snapshots__/*.ambr`` baselines are the contract. Do NOT regenerate them
with ``--snapshot-update`` to make a failing test pass without explicit user
review and approval — see CLAUDE.md, "Contract snapshot tests".
"""

from __future__ import annotations

from datetime import date

from syrupy.assertion import SnapshotAssertion

from app.search import SearchRequest, SearchResponse, SearchResult


def test_search_request_matches_csharp_payload(snapshot: SnapshotAssertion) -> None:
    """Pair 2: the flat JSON the C# backend posts fully populates SearchRequest.

    The payload mirrors what the C# backend actually sends — every filter is a
    top-level key, with no nested ``filters`` wrapper. Validating it through the
    real model and snapshotting the result proves each field is recognized and
    bound; this is exactly the check that catches a flat-vs-nested mismatch.
    """
    payload = {
        "query": "unpaid invoices for acme over 500 dollars",
        "customer_id": "3f1c2d4e-5a6b-7c8d-9e0f-1a2b3c4d5e6f",
        "status": "overdue",
        "min_amount": 500.0,
        "max_amount": 5000.0,
        "date_from": "2026-01-01",
        "date_to": "2026-03-31",
        "limit": 25,
    }

    result = SearchRequest.model_validate(payload)

    assert result.model_dump() == snapshot


def test_search_response_matches_wire_shape(snapshot: SnapshotAssertion) -> None:
    """Pair 3: the JSON sent back to the C# backend keeps its exact shape.

    The response is serialized the same way the ``/search`` endpoint emits it
    (``response_model=SearchResponse`` → ``model_dump(mode="json")``), so the
    snapshot pins the precise field names and value formats (e.g. ISO date
    strings) the C# side deserializes.
    """
    response = SearchResponse(
        results=[
            SearchResult(
                id="3f1c2d4e-5a6b-7c8d-9e0f-1a2b3c4d5e6f",
                number="INV-2026-0042",
                to="Acme Corporation",
                date=date(2026, 2, 14),
                status="Overdue",
                total=1234.56,
                type="Invoice",
                score=0.9231,
            )
        ]
    )

    assert response.model_dump(mode="json") == snapshot
