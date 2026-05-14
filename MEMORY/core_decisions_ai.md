# Core Decisions (AI-readable, YAML, append-only)
# Schema: see .skills/portfolio-memory/SKILL.md

- id: D-001
  date: 2026-05-10
  decision: scope_per_portfolio_handoff_section_2
  rationale: locked_scope_prevents_drift
  alternatives_rejected: []
  reversibility: expensive
  related_issues: []
  superseded_by: null

- id: D-002
  date: 2026-05-14
  decision: corpus_reproducible_from_source_not_committed_as_data
  rationale: same_python_version_plus_curated_module_list_yields_deterministic_corpus_keeps_repo_small_no_redistribution_licensing_ambiguity
  alternatives_rejected: [commit_corpus_jsonl_directly, fetch_from_remote_url_at_build_time]
  reversibility: cheap
  related_issues: [1, 2]
  superseded_by: null

- id: D-003
  date: 2026-05-14
  decision: chunk_shape_is_one_stdlib_member_equals_one_chunk
  rationale: retrieval_task_being_benchmarked_is_find_the_answer_to_a_question_and_the_unit_of_answer_is_one_symbols_documentation
  alternatives_rejected: [split_long_docstrings_by_paragraph, merge_module_overview_with_all_member_docs]
  reversibility: cheap
  related_issues: [1, 2]
  superseded_by: null
