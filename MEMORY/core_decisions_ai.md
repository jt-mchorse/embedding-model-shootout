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

- id: D-009
  date: 2026-05-26
  decision: atomic_write_helpers_live_in_package_level_io_utils_module_following_portfolio_standard_emerged_from_2026_05_26_atomic_write_arc
  rationale: matches_rag_kit_io_utils_atomic_write_text_pr_44_45_eval_harness_io_utils_atomic_write_text_pr_51_d_015_one_helper_one_test_surface_per_repo_centralized_monkey_patch_target
  alternatives_rejected: [file_private_helper_per_module, separate_distribution_package, in_place_re_implementation_at_each_call_site]
  reversibility: cheap
  related_issues: [#37]
  superseded_by: null

- id: D-010
  date: 2026-08-25
  decision: an_unmeasured_aggregate_cell_is_reported_as_absent_em_dash_in_markdown_json_null_never_as_0_0
  rationale: d_123s_rule_do_not_invent_benchmark_numbers_reached_ONE_CELL_OF_EIGHT_recall_at_k_in_aggregate_markdown_only_leaving_recall_in_aggregate_json_and_ALL_THREE_embed_latency_ms_reads_in_BOTH_formats_publishing_a_fabricated_0_0_TWO_HARMS_first_the_two_formats_DISAGREED_about_the_same_cell_markdown_em_dash_vs_json_0_0_while_aggregate_jsons_own_docstring_promises_a_consumer_can_cross_check_the_two_formats_line_by_line_and_the_json_is_what_ci_parses_second_for_LATENCY_the_fabricated_0_0_is_the_BEST_POSSIBLE_VALUE_so_a_provider_that_reported_no_timings_WON_ANY_WHICH_IS_FASTEST_READ_of_the_published_benchmark_a_default_landing_at_an_extreme_of_a_comparison_does_not_abstain_it_RANKS
  json_spelling: null_not_omitted_key   # a missing key and a null key are different contracts; a consumer reading row["recall"]["5"] gets KeyError from omission but a readable None from null, and the column set is a property of the AGGREGATE (union of every result's k) not of the row
  markdown_spelling: em_dash            # markdown has no spelling for absent; ABSENT_RECALL_CELL was already the convention from #123
  alternatives_rejected: [em_dash_in_json_too_REJECTED_it_turns_a_number_column_into_a_string_column_for_a_typed_consumer_and_json_HAS_a_spelling_for_absent, omit_the_key_in_json_REJECTED_missing_and_null_are_different_contracts_and_the_column_set_belongs_to_the_aggregate, reject_a_partial_embed_latency_ms_at_from_dict_REJECTED_a_sweep_result_is_a_REPORTED_MEASUREMENT_and_a_provider_that_legitimately_cannot_time_its_calls_should_still_be_comparable_on_recall_and_cost_same_read_vs_render_call_123_made, leave_json_alone_since_no_committed_artifact_has_absent_cells_REJECTED_the_external_result_file_path_from_dict_is_exactly_the_reachability_123_cites_for_its_own_case]
  widens_public_json_field_type: true   # recall[k] and the three *_ms fields go from number to number|null; no committed artifact changes because results/hash.json has all three ks and all three latency keys
  reversibility: cheap
  related_issues: [#123, #127]
  superseded_by: null

- id: D-011
  date: 2026-09-11
  decision: a_generated_markdown_artifact_marks_the_region_the_generator_owns_with_html_comments_and_sweep_aggregate_replaces_only_that_region_when_the_destination_marks_one
  markers: ["<!-- emb-shootout:table:begin -->", "<!-- emb-shootout:table:end -->"]
  rationale: docs_benchmarks_md_opens_by_saying_it_IS_REGENERATED_by_emb_shootout_sweep_aggregate_and_DONT_HAND_EDIT_and_running_that_command_as_documented_took_the_file_from_44_LINES_TO_3_deleting_the_current_results_framing_the_interpretation_paragraph_the_whole_reproducing_section_the_apples_to_apples_note_and_the_sentence_PER_THE_NO_FABRICATED_BENCHMARKS_RULE_THIS_README_DOES_NOT_CARRY_PLACEHOLDER_NUMBERS_which_encodes_the_portfolios_FIRST_quality_rule_AND_ALL_873_TESTS_STAYED_GREEN_because_test_benchmarks_md_snapshot_locks_by_CONTAINMENT_and_a_file_truncated_TO_the_table_still_CONTAINS_the_table_a_containment_lock_cannot_see_a_deletion_and_a_green_suite_is_exactly_why_an_operator_would_believe_the_regeneration_had_gone_fine
  fallback: a_destination_with_no_markers_or_with_an_unclosed_marker_is_written_whole_exactly_as_before_so_a_scratch_out_path_is_unaffected_and_no_existing_caller_has_to_learn_about_markers
  why_not_append_only: the_generator_must_be_able_to_SHRINK_its_own_region_when_a_providers_json_is_removed_from_results_and_an_append_only_rule_would_accumulate_stale_tables_the_markers_are_what_make_replacement_and_preservation_the_same_operation
  alternatives_rejected: [keep_truncating_and_move_the_prose_into_README_REJECTED_the_prose_explains_the_table_and_belongs_beside_it_and_the_no_fabricated_benchmarks_disclosure_has_to_live_where_the_numbers_are, refuse_to_write_an_existing_file_without_a_force_flag_REJECTED_it_breaks_the_documented_one_liner_and_makes_the_honest_path_the_longer_one, stop_claiming_the_file_is_regenerated_and_hand_maintain_the_table_REJECTED_the_table_is_exactly_the_part_a_generator_should_own, equality_snapshot_lock_alone_without_markers_REJECTED_it_would_catch_the_deletion_AFTER_an_operator_had_already_destroyed_their_working_copy]
  reversibility: cheap
  related_issues: [#145]
  superseded_by: null
  # The precision half of #145 -- a MEASURED latency below half of 10**-places
  # rendering as `0.0`, indistinguishable from a genuine zero -- is recorded as
  # APPLYING D-010, not as a new decision. D-010's rule is "an unmeasured cell is
  # absent, never 0.0" and its rationale argues from the OBSERVABLE ("0.0 is the
  # best possible value... a default at an extreme does not abstain, it RANKS").
  # That argument does not depend on how the 0.0 arrived, and D-010 closed only
  # the path where it arrives as a default. Extending it to the rounding path is
  # the same decision reaching the rest of its own reason. Flagged here rather
  # than assumed, because it does widen what D-010 is understood to cover.
