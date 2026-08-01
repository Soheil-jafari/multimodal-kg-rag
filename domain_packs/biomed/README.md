# Biomed domain pack

Domain-specific configuration for the scientific-paper corpus. The platform core
stays domain-agnostic; everything here is what makes it "biomed".

## Contents
- `predicates.yaml` — the **closed** knowledge-graph predicate schema, 8 predicates.
  The extractor is prompted with the list and the schema is enforced in code: a
  triple whose predicate falls outside the set is dropped rather than trusted.
  Loaded by `platform_core.graph.schema.PredicateSchema`.
- `gold/` — the hand-verified evaluation gold set (see `gold/README.md`).

## Adding a domain
A new domain is a directory `domain_packs/<name>/` carrying its own `predicates.yaml`
and `gold/`, selected by `domain_pack: <name>` in the config. The platform core needs
no change — it never references a predicate name or a corpus directly.
