"""The predicate schema is CLOSED: out-of-schema predicates are rejected.

Membership is what the extractor enforces in code — a predicate outside the set is
dropped rather than trusted — so this guards the mechanism the whole closed-schema
argument rests on.
"""
from platform_core.graph.schema import PredicateSchema


def test_schema_membership_is_closed():
    schema = PredicateSchema(predicates=frozenset({"treats", "causes"}))
    assert schema.is_allowed("treats")
    assert not schema.is_allowed("cures")  # invented predicate must be rejected
