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

---
session: 2026-05-16T23:15Z
duration_min: 60
issue: 3
focus: pareto_frontier_plot_cost_vs_recall_at_5
delta:
  files_added: 4  # emb_shootout/pareto.py, emb_shootout/plot.py, tests/test_pareto.py, tests/test_plot.py
  files_changed: 3  # emb_shootout/cli.py, pyproject.toml, README.md
  files_committed: 2  # docs/pareto.png, docs/pareto.svg
  tests_added: 26  # 15 pareto unit + 11 plot integration
  test_pass_rate: "64/64"
  benchmarks:
    frontier_points_today: 1
    frontier_member: "hash-embedder-128d-ngram2"
    recall_at_5_hash_baseline: 0.520
    cost_per_million_tokens_hash_baseline: 0.0
context_for_next_session:
  - pareto_frontier_computation_is_pure_python_in_emb_shootout_pareto_dep_free
  - plot_renderer_in_emb_shootout_plot_lazy_imports_matplotlib_extra_named_plot
  - cli_subcommand_emb_shootout_sweep_plot_results_dir_out_png_out_svg
  - axes_locked_cost_per_million_tokens_x_recall_at_5_y_per_acceptance_criteria_d_008
  - frontier_polyline_only_drawn_when_distinct_frontier_points_at_least_two_otherwise_title_says_so
  - committed_docs_pareto_png_svg_show_single_hash_baseline_real_provider_runs_pending_operator
  - identical_points_both_kept_on_frontier_dropping_ties_would_hide_co_located_providers
  - tests_for_plot_use_pytest_importorskip_matplotlib_so_standard_ci_matrix_without_extras_still_passes
  - readme_pareto_subsection_under_benchmarks_results_with_real_plot_and_honest_single_point_caveat
decisions_made: [D-008]
followups: []
---

---
session: 2026-05-18T05:25Z
duration_min: 30
issue: 5
focus: reproduce_notebook_plus_verify_script_plus_shape_tests
delta:
  files_changed: 5
  tests_added: 5
context_for_next_session:
  - notebooks_reproduce_ipynb_runs_top_to_bottom_no_jupyter_dep_needed_for_ci
  - verify_py_is_the_executable_twin_pytest_runs_it_end_to_end
  - build_notebook_py_emits_the_ipynb_from_static_cell_spec_diffable_in_source_review
  - 5_shape_tests_pin_imports_no_cached_outputs_no_drift
  - notebook_reads_results_glob_so_future_provider_jsons_slot_in_without_changes
  - no_new_d_entry_pure_reproducibility_infrastructure
decisions_made: []
followups: []
---

---
session: 2026-05-18T15:32Z
duration_min: 10
issue: 5
focus: unblock_pr_9_lint_per_file_ignore_for_notebooks
delta:
  files_changed: 1  # pyproject.toml
  files_added: 0
  tests_added: 0
  test_pass_rate: "73/73"
context_for_next_session:
  - tool_ruff_lint_per_file_ignores_added_for_ipynb_with_e402_ignored
  - rationale_notebooks_start_with_path_shim_before_imports_is_standard_idiom_not_a_real_smell
  - production_py_code_still_gets_e402_check
  - this_branch_session_2026_05_18_issue_05_now_lint_clean_pending_ci_re_run
decisions_made: []
followups: []
---

---
session: 2026-05-19T03:30Z
duration_min: 45
issue: 4
focus: honest_narrative_takeaway_plus_readme_snapshot_test
delta:
  files_changed: 1   # README.md
  files_added: 1     # tests/test_readme_snapshot.py
  tests_added: 7
  test_pass_rate: "76/76"
context_for_next_session:
  - readme_what_this_is_rewritten_to_present_tense_four_shipped_pieces_drops_corpus_pr_framing
  - takeaways_section_grounded_in_results_hash_json_only_real_measurement_today
  - explicit_no_provider_winner_claim_until_operator_commits_real_provider_jsons
  - readme_snapshot_test_locks_recall_at_k_ndcg_corpus_count_query_count_query_p95_corpus_embed_time
  - same_snapshot_pattern_as_seven_sister_prs_across_portfolio_landed_2026_05_18
  - tamper_verified_recall_at_5_substitution_fires_snapshot
  - no_new_d_entry_pure_documentation_plus_drift_guard
decisions_made: []
followups: []
---

---
session: 2026-05-19T15:15Z
duration_min: 20
issue: 11
focus: snapshot_test_locks_docs_benchmarks_md_to_aggregate_markdown_output
delta:
  files_added: 1   # tests/test_benchmarks_md_snapshot.py
  files_changed: 0
  tests_added: 2
  test_pass_rate: "78/78"
context_for_next_session:
  - second_snapshot_test_in_this_repo_first_locks_readme_takeaways_numbers_second_locks_aggregator_format
  - test_loads_every_results_star_json_into_sweepresult_calls_aggregate_markdown_asserts_block_appears_verbatim_in_docs_benchmarks_md
  - second_test_is_header_signature_presence_guard_against_accidental_table_deletion
  - failure_message_includes_full_aggregator_output_plus_regen_command_emb_shootout_sweep_aggregate
  - tamper_verified_by_editing_recall_at_5_cell_0_520_to_0_999_fires_immediately_then_reverted
  - gap_closed_is_orthogonal_to_readme_snapshot_numbers_locked_by_results_hash_json_via_readme_test_aggregator_format_drift_locked_here
  - parallel_to_rag_kit_eval_bench_snapshot_landed_same_day_d_007_governs_aggregator_no_new_decision
decisions_made: []
followups: []
---

---
session: 2026-05-20T03:17Z
duration_min: 25
issue: 13
focus: public_surface_snapshot_test_locks_emb_shootout_top_level_init
delta:
  files_added: 1   # tests/test_public_surface.py
  files_changed: 1   # emb_shootout/__init__.py (+__version__)
  tests_added: 8   # 4 standalone + 4 parametrized submodule anchors
  test_pass_rate: "86/86"
context_for_next_session:
  - emb_shootout_now_publishes_dunder_version_str_0_0_1_mirror_of_pyproject
  - top_level_surface_locked_only_corpus_providers_queries_sweep_re_exported_cli_pareto_plot_dotted_path_only
  - readme_dotted_path_test_anchors_emb_shootout_pareto_pareto_frontier_line_210
  - ast_parser_filters_on_level_geq_1_for_relative_imports
  - tamper_verified_four_axes_bad_version_drop_chunk_rename_pareto_module_alias_rename_providers_anchor
  - portable_pattern_fifth_strike_after_eval_harness_cost_optimizer_prompt_regression_rag_kit_remaining_targets_chunking_strategies_lab_python_async_llm_pipelines_mcp_python_example
decisions_made: []
followups: []
---

---
session: 2026-05-21T19:10Z
duration_min: 30
issue: 15
focus: scripts_capture_demo_sh_three_surface_60s_driver_plus_smoke_test
delta:
  files_added: 2   # scripts/capture_demo.sh, tests/test_capture_demo_smoke.py
  files_changed: 1 # README.md (Demo section pending placeholder → real invocation)
  tests_added: 4
  test_pass_rate: "90/90"
context_for_next_session:
  - sixth_repo_to_land_capture_demo_sh_after_llm_eval_harness_prompt_regression_llm_cost_optimizer_rag_production_kit_today_and_this_one
  - three_surfaces_corpus_build_single_module_sweep_run_hash_aggregate_markdown_chosen_for_dep_free_no_network_no_api_key
  - single_module_corpus_deliberate_for_tempo_full_corpus_numbers_stay_locked_in_docs_benchmarks_md_and_readme_takeaways_via_existing_snapshot_tests
  - capture_pace_seconds_capture_demo_module_capture_demo_queries_env_knobs_smoke_test_uses_pace_0_and_defaults_for_module_queries
  - smoke_test_pins_aggregator_markdown_header_signature_belt_and_braces_with_test_benchmarks_md_snapshot
  - per_run_tempdir_via_mktemp_d_trap_exit_int_term_so_concurrent_recordings_or_smoke_runs_dont_collide
  - no_new_d_entry_d_002_d_003_d_004_d_007_already_govern_the_surfaces_this_glues
decisions_made: []
followups: []
---

---
session: 2026-05-22T03:14Z
duration_min: 25
issue: 17
focus: fix_readme_character_bigram_misdescription_lock_against_drift
delta:
  files_changed: 2   # README.md, emb_shootout/providers/hash_embedder.py
  files_added: 1     # tests/test_hash_baseline_description.py
  tests_added: 2
  test_pass_rate: "94/94"
context_for_next_session:
  - readme_called_hash_baseline_character_bigrams_three_places_l59_l64_l71_code_is_word_bigrams_str_lower_split_then_join_window
  - chose_prose_fix_not_logic_change_results_hash_json_unchanged_existing_snapshot_locks_recall_ndcg_latency_corpus_count_query_count_decision_ids_still_pass
  - added_self_tokenizer_word_literal_attr_to_hashembedderprovider_validates_kwarg_rejects_anything_else_so_char_variant_must_ship_as_separate_provider_per_d_007
  - new_test_hash_baseline_description_indexes_into_self_tokenizer_required_patterns_word_bigram_word_n_gram_overlap_word_overlap_forbidden_patterns_character_bigram_character_n_gram_overlap_character_overlap
  - test_slices_takeaways_section_only_not_whole_readme_so_a_d_007_mention_elsewhere_doesnt_false_positive
  - portfolio_pattern_first_post_v0_1_improvement_issue_filed_and_shipped_in_same_session_per_phase_a_step_5_when_zero_open_priority_high_existed
decisions_made: []
followups: []
---
