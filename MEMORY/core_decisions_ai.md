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

- id: D-004
  date: 2026-05-15
  decision: embedder_is_single_method_protocol_parallel_to_portfolio_pattern
  rationale: same_seam_as_rag_kit_eval_harness_cost_optimizer_chunking_lab_swappable_providers_via_one_method
  alternatives_rejected: [hard_coded_openai_client, abstract_base_class, sklearn_style_estimator]
  reversibility: cheap
  related_issues: [2]
  superseded_by: null

- id: D-005
  date: 2026-05-15
  decision: queries_derived_from_corpus_at_sweep_time_deterministic_seed_not_pre_committed_fixture
  rationale: corpus_evolves_as_python_does_committed_query_set_drifts_seeded_derivation_keeps_them_in_sync
  alternatives_rejected: [pre_committed_query_jsonl, hand_curated_query_set_per_corpus_version]
  reversibility: cheap
  related_issues: [2]
  superseded_by: null

- id: D-006
  date: 2026-05-15
  decision: cost_per_million_tokens_is_operator_supplied_at_provider_construction_default_to_public_list_2026_05
  rationale: pricing_changes_record_the_price_used_alongside_quality_numbers_so_historical_comparisons_remain_interpretable
  alternatives_rejected: [hard_coded_in_provider_module, fetch_from_provider_pricing_api_at_runtime]
  reversibility: cheap
  related_issues: [2]
  superseded_by: null

- id: D-007
  date: 2026-05-15
  decision: per_provider_result_json_files_aggregator_merges_them_no_single_file_multiple_providers
  rationale: avoid_concurrent_run_collisions_each_operator_run_writes_one_file_aggregator_is_pure_read
  alternatives_rejected: [single_results_jsonl_appended_per_run, sqlite_results_db]
  reversibility: cheap
  related_issues: [2]
  superseded_by: null

- id: D-008
  date: 2026-05-16
  decision: pareto_axes_cost_per_million_tokens_x_recall_at_5_y_frontier_pure_python_renderer_behind_matplotlib_plot_extra
  rationale: acceptance_criteria_specifies_cost_vs_recall_at_5_pareto_computation_is_dep_free_pure_python_so_it_tests_in_standard_ci_matrix_without_extras_matplotlib_only_needed_for_rendering_so_keeps_base_install_dep_free_same_pattern_as_d_004_provider_extras
  alternatives_rejected: [latency_as_third_axis_not_in_acceptance_criteria, matplotlib_in_base_install_breaks_dep_free_default, hand_rolled_svg_renderer_avoids_extra_but_loses_quality_and_dev_velocity]
  reversibility: cheap
  related_issues: [3]
  superseded_by: null
