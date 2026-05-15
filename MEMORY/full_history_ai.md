# Session History (AI-readable, append-only)

Schema: see .skills/portfolio-memory/SKILL.md

---
session: 2026-05-14T15:00:00Z
duration_min: 50
issue: 1
focus: pick_corpus_cpython_stdlib_and_ship_loader
delta:
  files_added: 8
  files_changed: 4
  tests_added: 15
  coverage_pct: 100
  corpus_chunks_measured: 12010
context_for_next_session:
  - corpus_is_cpython_stdlib_docstrings_psf_licensed_reproducible_from_inspect_not_committed_as_data
  - default_modules_curated_list_142_entries_yields_12010_chunks_on_cpython_314_clears_10k_acceptance_bar
  - chunk_shape_locked_chunk_id_text_module_qualname_kind_source_issue_2_model_sweep_will_consume_this_shape
  - all_treated_as_authoritative_reexport_list_handles_json_jsonencoder_etc_dir_only_members_filtered_by_module
  - corpus_test_clears_10k_is_the_gate_against_future_stdlib_shrinkage
decisions_made: [D-002, D-003]
followups: []
---

---
session: 2026-05-15T20:32Z
duration_min: 80
issue: 2
focus: model_sweep_harness_with_six_providers
delta:
  files_added: 9
  files_changed: 4
  tests_added: 23
  test_pass_rate: "38/38"
  hash_baseline_run_committed: "results/hash.json — recall@5=0.520 NDCG@10=0.449 on 12010 chunks × 50 queries"
context_for_next_session:
  - sweep_harness_shipped_protocol_plus_run_sweep_plus_six_providers
  - hash_provider_baseline_committed_real_provider_runs_pending_operator_api_keys
  - queries_derived_from_corpus_with_seed_d005_no_drift_between_corpus_and_queries
  - cost_per_million_tokens_recorded_per_run_d006_historical_comparisons_remain_interpretable
  - per_provider_json_files_aggregator_merges_d007_no_concurrent_collisions
  - operator_runs_emb_shootout_sweep_run_per_provider_then_aggregate_to_populate_docs_benchmarks_md
decisions_made: [D-004, D-005, D-006, D-007]
followups: []
---
