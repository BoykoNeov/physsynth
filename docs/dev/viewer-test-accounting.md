# `tests/test_web_backend.py` → `crates/physsynth-viewer/tests/`: every test accounted for

Retirement plan §23.19. The Python file was the viewer's specification (§5): 338 test functions,
429 once parametrized. Before it was deleted, every function was mapped to the Rust test that
carries it. The mapping was proposed by name (the Rust tests keep the Python names minus the
`test_` and model prefixes) and every pair below a strong match, and every miss, was read by hand.
Parametrized cases are carried as a loop over the same cases inside the one Rust test (the case
counts were checked for every function with three or more).

**Result: 338 of 338 carried; none dies.** Two had not been carried by phases D1–D6 and were added
at §23.19 (marked below). "Merged" means one Rust test asserts what two or three Python tests did.

The Rust side also has tests with no Python counterpart — the dependency allowlist, the server's
routing, Python's number formats, the frozen fixture and its guards, and a handful of pins found
while porting (NumPy's percentile, the juari's float quirk, the disk's paired modes, …).

| line | Python test (×parametrized cases) | Rust test (`crates/physsynth-viewer/tests/…`) | note |
|---:|---|---|---|
| 225 | `test_lossless_drift_survives_wrapper` | `strings.rs::lossless_drift_survives_wrapper` |  |
| 237 | `test_all_three_models_build` ×3 | `strings.rs::all_three_models_build` |  |
| 251 | `test_frame_bookkeeping` | `strings.rs::frame_bookkeeping` |  |
| 265 | `test_frames_decode_to_field_values_and_boundary` | `strings.rs::frames_decode_to_field_values_and_boundary` |  |
| 283 | `test_audio_resampled_and_normalized` | `strings.rs::audio_resampled_and_normalized` |  |
| 295 | `test_high_N_audio_stays_in_browser_range` | `strings.rs::high_n_audio_stays_in_browser_range` |  |
| 306 | `test_lossy_reports_passivity_not_drift` | `strings.rs::lossy_reports_passivity_not_drift` |  |
| 322 | `test_cfl_violation_is_clean_error` | `strings.rs::cfl_violation_is_clean_error` |  |
| 331 | `test_stiff_admits_lambda_above_one` | `strings.rs::stiff_admits_lambda_above_one` |  |
| 351 | `test_bad_params_give_error_payload` ×8 | `strings.rs::bad_params_give_error_payload` |  |
| 358 | `test_none_params_does_not_crash` | `strings.rs::none_params_does_not_crash` |  |
| 371 | `test_tension_shift_matches_the_exact_duffing_oracle` | `tension.rs::shift_matches_the_exact_duffing_oracle` |  |
| 385 | `test_tension_shift_is_immune_to_loss` | `tension.rs::shift_is_immune_to_loss` |  |
| 402 | `test_tension_energy_survives_the_wrapper_and_the_nonlinearity_is_engaged` | `tension.rs::energy_survives_the_wrapper_and_the_nonlinearity_is_engaged` |  |
| 414 | `test_tension_ea_zero_collapses_to_the_linear_string` | `tension.rs::ea_zero_collapses_to_the_linear_string` |  |
| 424 | `test_tension_dt_over_t_matches_the_closed_form` | `tension.rs::dt_over_t_matches_the_closed_form` |  |
| 433 | `test_purity_gate_nulls_the_shift_when_the_mode_breaks_up` | `tension.rs::purity_gate_nulls_the_shift_when_the_mode_breaks_up` |  |
| 461 | `test_tension_dt_guard_is_not_an_amplitude_proxy` | `tension.rs::dt_guard_is_not_an_amplitude_proxy` |  |
| 472 | `test_tension_lossy_reports_passivity` | `tension.rs::lossy_reports_passivity` |  |
| 479 | `test_tension_frames_decode_to_a_mode1_sine_with_fixed_ends` | `tension.rs::frames_decode_to_a_mode1_sine_with_fixed_ends` |  |
| 508 | `test_tension_bad_params_give_error_payload` ×8 | `tension.rs::bad_params_give_error_payload` |  |
| 538 | `test_parametric_conserves_through_a_complete_disintegration` | `tension.rs::parametric_conserves_through_a_complete_disintegration` |  |
| 555 | `test_parametric_pair_straddles_and_the_stable_run_stays_on_its_seed` | `tension.rs::parametric_pair_straddles_and_the_stable_run_stays_on_its_seed` |  |
| 565 | `test_parametric_below_run_is_not_a_linear_control` | `tension.rs::parametric_below_run_is_not_a_linear_control` |  |
| 574 | `test_parametric_drift_is_scored_on_the_full_array_not_the_shipped_trace` | `tension.rs::parametric_drift_is_scored_on_the_full_array_not_the_shipped_trace` |  |
| 585 | `test_parametric_log_axis_traces_are_not_rounded_into_zero` | `tension.rs::parametric_log_axis_traces_are_not_rounded_into_zero` |  |
| 596 | `test_parametric_energy_lands_in_low_neighbours_not_at_the_grid_scale` | `tension.rs::parametric_energy_lands_in_low_neighbours_not_at_the_grid_scale` |  |
| 615 | `test_parametric_names_no_partner_below_the_tongue` | `tension.rs::parametric_names_no_partner_below_the_tongue` |  |
| 624 | `test_parametric_mode_one_is_the_robust_one` | `tension.rs::parametric_mode_one_is_the_robust_one` |  |
| 635 | `test_parametric_run_length_is_periods_of_mode_m_not_steps` | `tension.rs::parametric_run_length_is_periods_of_mode_m_not_steps` |  |
| 645 | `test_parametric_self_seeding_still_breaks_up_without_the_explicit_seed` | `tension.rs::parametric_self_seeding_still_breaks_up_without_the_explicit_seed` |  |
| 662 | `test_parametric_sweep_bins_stable_unstable_and_unsaturated_separately` | `tension.rs::parametric_sweep_bins_stable_unstable_and_unsaturated_separately` |  |
| 676 | `test_parametric_sweep_is_a_controlled_reference_curve` | `tension.rs::parametric_sweep_is_a_controlled_reference_curve` |  |
| 685 | `test_parametric_sweep_truncation_is_labelled_never_silent` | `tension.rs::parametric_sweep_truncation_is_labelled_never_silent` |  |
| 699 | `test_parametric_shipped_sweep_path_is_pinned_with_the_overrides_absent` | `tension.rs::parametric_shipped_sweep_path_is_pinned_with_the_overrides_absent` |  |
| 710 | `test_parametric_forces_sigma_to_zero` | `tension.rs::parametric_forces_sigma_to_zero` |  |
| 720 | `test_parametric_regime_leaves_the_duffing_path_bit_for_bit` | `tension.rs::parametric_regime_leaves_the_duffing_path_bit_for_bit` |  |
| 744 | `test_parametric_bad_params_give_error_payload` ×13 | `tension.rs::parametric_bad_params_give_error_payload` |  |
| 768 | `test_membrane_lossless_drift_survives_wrapper` | `membrane.rs::lossless_drift_survives_the_wrapper` |  |
| 782 | `test_membrane_frame_bookkeeping_2d` ×2 | `membrane.rs::frame_bookkeeping_is_2d_for_both_domains` |  |
| 796 | `test_membrane_spatial_decimation_shrinks_field` | `membrane.rs::spatial_decimation_shrinks_a_field_above_the_display_budget` |  |
| 806 | `test_membrane_frames_decode_to_field_values_and_mask` | `membrane.rs::frames_decode_to_field_values_aligned_with_the_mask` |  |
| 830 | `test_membrane_rectangle_aspect_uses_snapped_ly` | `membrane.rs::rectangle_extent_uses_the_snapped_ly` |  |
| 839 | `test_membrane_spectrum_block_fundamental_is_self_consistent` | `membrane.rs::the_spectrum_rings_at_the_discrete_fundamental` |  |
| 852 | `test_membrane_lossy_reports_passivity` | `membrane.rs::a_lossy_run_reports_passivity` |  |
| 877 | `test_membrane_bad_params_give_error_payload` ×8 | `membrane.rs::bad_params_give_an_error_payload` |  |
| 884 | `test_membrane_cfl_ceiling_is_the_2d_bar` | `membrane.rs::the_cfl_ceiling_is_the_2d_bar` |  |
| 893 | `test_membrane_thin_rectangle_rejected_by_nlive_guard` | `membrane.rs::a_thin_rectangle_is_rejected_by_the_live_node_guard` |  |
| 905 | `test_membrane_small_geometry_rejected_by_work_budget` | `membrane.rs::a_small_geometry_is_rejected_by_the_work_budget_but_not_banned` |  |
| 940 | `test_mallet_lossless_conserves_through_the_wrapper` | `mallet.rs::a_lossless_strike_conserves_through_the_wrapper` |  |
| 956 | `test_mallet_bounces_with_near_unity_restitution_and_head_barely_rings` | `mallet.rs::bounces_with_near_unity_restitution_and_the_head_barely_rings` |  |
| 971 | `test_mallet_strike_marker_reports_the_snapped_node_in_fractions` | `mallet.rs::the_strike_marker_reports_the_snapped_node_in_fractions` |  |
| 981 | `test_mallet_audio_is_the_ring_not_the_dimple` | `mallet.rs::the_audio_is_the_ring_not_the_dimple` |  |
| 996 | `test_mallet_lossy_reports_passivity_without_a_decay_oracle` | `mallet.rs::a_lossy_strike_reports_passivity_without_a_decay_oracle` |  |
| 1012 | `test_mallet_hysteresis_lowers_restitution` | `mallet.rs::hysteresis_lowers_restitution` |  |
| 1034 | `test_mallet_bad_params_give_error_payload` ×8 | `mallet.rs::bad_params_give_an_error_payload` |  |
| 1039 | `test_mallet_small_geometry_rejected_by_work_budget` | `mallet.rs::a_small_geometry_is_rejected_by_the_work_budget_but_short_audio_fits` |  |
| 1069 | `test_plate_lossless_drift_survives_wrapper` ×2 | `plate.rs::both_rectangle_boundaries_conserve_and_book_keep_in_2d` | merged: drift, bookkeeping and the rectangle's no-claim check, both boundaries in one loop |
| 1082 | `test_plate_frame_bookkeeping_2d` ×2 | `plate.rs::both_rectangle_boundaries_conserve_and_book_keep_in_2d` | merged (see above) |
| 1095 | `test_plate_supported_spectrum_is_tight_tier` | `plate.rs::the_supported_spectrum_is_the_tight_tier` |  |
| 1107 | `test_plate_free_spectrum_has_leissa_only_when_square` | `plate.rs::the_free_spectrum_has_the_leissa_anchor_only_when_square` |  |
| 1119 | `test_plate_lossy_reports_passivity` | `plate.rs::a_lossy_plate_reports_passivity` |  |
| 1142 | `test_plate_bad_params_give_error_payload` ×8 | `plate.rs::bad_params_give_an_error_payload` |  |
| 1149 | `test_plate_low_mu_rejected_by_work_budget` | `plate.rs::low_mu_is_rejected_by_the_work_budget` |  |
| 1193 | `test_guitar_waist_swaps_the_fundamental` | `plate.rs::the_waist_swaps_the_fundamental` |  |
| 1221 | `test_guitar_crossing_is_a_bracket_not_a_number` | `plate.rs::the_crossing_is_a_bracket_not_a_number` |  |
| 1243 | `test_guitar_strike_overlap_is_reported_because_it_can_hide_the_claim` | `plate.rs::the_strike_overlap_is_reported_because_it_can_hide_the_claim` |  |
| 1269 | `test_guitar_display_mask_is_connected_and_pooled_not_sampled` | `plate.rs::the_display_mask_is_connected_and_pooled` |  |
| 1286 | `test_a_reachable_guitar_that_point_sampling_would_draw_as_two_plates` | `plate.rs::a_reachable_narrow_deep_waisted_guitar_still_draws_one_plate` |  |
| 1313 | `test_pooling_keeps_a_one_node_isthmus_that_point_sampling_severs` | `plate.rs::pooling_keeps_a_one_node_isthmus_that_point_sampling_severs` |  |
| 1343 | `test_pooling_reduces_to_point_sampling_on_an_all_live_mask` | `plate.rs::pooling_reduces_to_point_sampling_on_an_all_live_mask` |  |
| 1365 | `test_rectangle_plate_still_reports_a_rectangle_and_no_claim` ×2 | `plate.rs::both_rectangle_boundaries_conserve_and_book_keep_in_2d` | merged (see above) |
| 1380 | `test_guitar_reports_its_own_outline_and_the_numbers_are_self_consistent` | `plate.rs::the_guitar_reports_its_own_outline_self_consistently` |  |
| 1401 | `test_guitar_energy_conserves_but_this_tier_is_NOT_evidence` | `plate.rs::the_guitar_energy_conserves_but_this_tier_is_not_evidence` |  |
| 1415 | `test_guitar_sweep_grid_is_capped_independently_of_the_audio_grid` | `plate.rs::the_sweep_grid_is_capped_independently_of_the_audio_grid` |  |
| 1427 | `test_a_curved_outline_gets_NO_continuum_reference_whatever_its_bounding_box` | `plate.rs::a_curved_outline_gets_no_continuum_reference_whatever_its_box` |  |
| 1453 | `test_a_short_render_still_serialises_and_reports_a_POSITIVE_fundamental` ×2 | `plate.rs::a_short_render_reports_a_positive_fundamental` |  |
| 1483 | `test_the_claim_panel_and_the_spectrum_panel_quote_the_SAME_hertz` ×2 | `plate.rs::the_claim_and_spectrum_panels_quote_the_same_hertz` |  |
| 1502 | `test_a_long_body_pushes_the_crossing_past_the_slider_and_the_panel_says_so` | `plate.rs::a_long_body_pushes_the_crossing_past_the_slider` |  |
| 1536 | `test_guitar_bad_params_give_error_payload` ×4 | `plate.rs::guitar_bad_params_give_an_error_payload_and_guitar_means_free` |  |
| 1542 | `test_guitar_domain_can_only_mean_a_free_guitar` | `plate.rs::guitar_bad_params_give_an_error_payload_and_guitar_means_free` | merged with the guitar's refusals |
| 1570 | `test_vk_supported_conserves_converges_and_hardens` | `vk.rs::the_supported_gong_conserves_converges_and_hardens` |  |
| 1587 | `test_vk_linear_toggle_has_no_convergence_block_and_no_shift` | `vk.rs::the_linear_toggle_has_no_convergence_block_and_no_shift` |  |
| 1597 | `test_vk_free_cymbal_conserves_without_a_fundamental` | `vk.rs::the_free_cymbal_conserves_without_a_fundamental` |  |
| 1611 | `test_vk_lossy_reports_passivity` | `vk.rs::a_lossy_plate_reports_passivity_and_keeps_its_convergence_gate` |  |
| 1636 | `test_vk_bad_params_give_error_payload` ×10 | `vk.rs::bad_params_give_an_error_payload` |  |
| 1656 | `test_bow_lossless_balance_is_the_money_number` | `bow.rs::lossless_balance_is_the_money_number` |  |
| 1669 | `test_bow_balance_replaces_both_older_verdicts_because_both_would_lie` | `bow.rs::balance_replaces_both_older_verdicts_because_both_would_lie` |  |
| 1698 | `test_bow_lossy_reports_inferred_dissipation_not_a_tautological_residual` | `bow.rs::lossy_reports_inferred_dissipation_not_a_tautological_residual` |  |
| 1711 | `test_bow_balance_curves_share_one_decimation` | `bow.rs::balance_curves_share_one_decimation` |  |
| 1722 | `test_bow_helmholtz_slip_fraction_matches_beta` | `bow.rs::helmholtz_slip_fraction_matches_beta` |  |
| 1734 | `test_bow_slip_fraction_tracks_beta_as_the_bow_moves` ×3 | `bow.rs::slip_fraction_tracks_beta_as_the_bow_moves` |  |
| 1744 | `test_bow_pitch_is_the_strings_not_the_bows` | `bow.rs::pitch_is_the_strings_not_the_bows` |  |
| 1754 | `test_bow_out_of_window_is_labelled_not_failed` | `bow.rs::out_of_window_is_labelled_not_failed` |  |
| 1776 | `test_bow_zero_force_leaves_the_string_at_rest` | `bow.rs::zero_force_leaves_the_string_at_rest` |  |
| 1785 | `test_bow_frames_decode_to_a_string_with_fixed_ends` | `bow.rs::frames_decode_to_a_string_with_fixed_ends` |  |
| 1797 | `test_bow_short_render_animates_the_attack_from_rest` | `bow.rs::short_render_animates_the_attack_from_rest` |  |
| 1822 | `test_bow_helmholtz_number_is_reported_never_asserted` | `bow.rs::helmholtz_number_is_reported_never_asserted` |  |
| 1846 | `test_bow_bad_params_give_error_payload` ×9 | `bow.rs::bad_params_give_error_payload` |  |
| 1853 | `test_bow_work_budget_is_its_own` | `bow.rs::work_budget_is_its_own` |  |
| 1885 | `test_geometric_planar_max_w_is_bit_exact_zero` | `geometric.rs::planar_max_w_is_bit_exact_zero` |  |
| 1900 | `test_geometric_planar_conserves_through_the_wrapper` | `geometric.rs::planar_conserves_through_the_wrapper` |  |
| 1908 | `test_geometric_rotating_wave_is_a_true_circle` | `geometric.rs::rotating_wave_is_a_true_circle` |  |
| 1926 | `test_geometric_rotating_wave_longitudinal_field_leans_but_does_not_move` | `geometric.rs::rotating_wave_longitudinal_field_leans_but_does_not_move` |  |
| 1937 | `test_geometric_frames_decode_to_three_fields_with_clamped_ends` | `geometric.rs::frames_decode_to_three_fields_with_clamped_ends` |  |
| 1956 | `test_geometric_orbit_trail_is_the_probe_node_and_indexable_by_frame` | `geometric.rs::orbit_trail_is_the_probe_node_and_indexable_by_frame` |  |
| 1968 | `test_geometric_whirl_grows_inside_the_tongue_and_conserves_through_it` | `geometric.rs::whirl_grows_inside_the_tongue_and_conserves_through_it` |  |
| 1985 | `test_geometric_whirl_needs_no_new_energy_verdict_unlike_the_bow` | `geometric.rs::whirl_needs_no_new_energy_verdict_unlike_the_bow` |  |
| 1998 | `test_geometric_whirl_is_dead_outside_the_tongue` | `geometric.rs::whirl_is_dead_outside_the_tongue` |  |
| 2012 | `test_geometric_degenerate_string_cannot_whirl` | `geometric.rs::degenerate_string_cannot_whirl` |  |
| 2025 | `test_geometric_velocity_seed_makes_the_degenerate_string_marginal_not_stable` | `geometric.rs::velocity_seed_makes_the_degenerate_string_marginal_not_stable` |  |
| 2044 | `test_geometric_unseeded_whirl_is_the_honesty_gate` | `geometric.rs::unseeded_whirl_is_the_honesty_gate` |  |
| 2058 | `test_geometric_whirl_rate_matches_the_mathieu_prediction_and_runs_low` | `geometric.rs::whirl_rate_matches_the_mathieu_prediction_and_runs_low` |  |
| 2072 | `test_geometric_has_no_audio_and_says_why` | `geometric.rs::has_no_audio_and_says_why` |  |
| 2084 | `test_geometric_lam_long_above_one_is_rejected` | `geometric.rs::lam_long_above_one_is_rejected` |  |
| 2093 | `test_geometric_lam_long_is_the_knob_and_lambda_is_derived` | `geometric.rs::lam_long_is_the_knob_and_lambda_is_derived` |  |
| 2103 | `test_geometric_work_budget_is_its_own` | `geometric.rs::work_budget_is_its_own` |  |
| 2123 | `test_geometric_bad_params_give_error_payload` ×7 | `geometric.rs::bad_params_give_error_payload` |  |
| 2165 | `test_phantom_peaks_are_the_quadratic_combinations_of_the_measured_partials` | `geometric.rs::phantom_peaks_are_the_quadratic_combinations_of_the_measured_partials` |  |
| 2200 | `test_phantom_no_longitudinal_peak_sits_on_a_transverse_partial` | `geometric.rs::phantom_no_longitudinal_peak_sits_on_a_transverse_partial` |  |
| 2218 | `test_phantom_headline_is_the_defect_measured_from_both_sides` | `geometric.rs::phantom_headline_is_the_defect_measured_from_both_sides` |  |
| 2242 | `test_phantom_conserves_through_the_wrapper` | `geometric.rs::phantom_conserves_through_the_wrapper` |  |
| 2251 | `test_phantom_window_is_fixed_physics_and_ignores_the_animation_slider` | `geometric.rs::phantom_window_is_fixed_physics_and_ignores_the_animation_slider` |  |
| 2266 | `test_phantom_display_grid_is_denser_than_the_measurement_grid` | `geometric.rs::phantom_display_grid_is_denser_than_the_measurement_grid` |  |
| 2291 | `test_phantom_ships_the_audio_that_batch_3_deferred` | `geometric.rs::phantom_ships_the_audio_that_the_viz_only_regimes_do_not` |  |
| 2310 | `test_phantom_linear_string_has_no_channel_to_put_a_phantom_in` | `geometric.rs::phantom_linear_string_has_no_channel_to_put_a_phantom_in` |  |
| 2333 | `test_phantom_labels_a_grid_too_coarse_to_show_the_stiffness` | `geometric.rs::phantom_labels_a_grid_too_coarse_to_show_the_stiffness` |  |
| 2355 | `test_phantom_defect_gate_is_one_sided_not_absolute` | `geometric.rs::phantom_defect_gate_is_one_sided_not_absolute` |  |
| 2381 | `test_phantom_budget_and_amplitude_guards_give_clean_error_payloads` ×2 | `geometric.rs::phantom_budget_and_amplitude_guards_give_clean_error_payloads` |  |
| 2388 | `test_phantom_has_its_own_work_budget_because_it_is_the_slowest_render` | `geometric.rs::phantom_has_its_own_work_budget_because_it_is_the_slowest_render` | **added at §23.19** — the one the phase had not carried |
| 2416 | `test_symp_antisymmetric_mode_keeps_the_bridge_bit_exact_still` | `sympathetic.rs::antisymmetric_mode_keeps_the_bridge_bit_exact_still` |  |
| 2431 | `test_symp_symmetric_mode_is_the_contrast_that_makes_the_zero_mean_something` | `sympathetic.rs::symmetric_mode_is_the_contrast_that_makes_the_zero_mean_something` |  |
| 2440 | `test_symp_detune_is_ignored_in_the_normal_regime` | `sympathetic.rs::detune_is_ignored_in_the_normal_regime` |  |
| 2447 | `test_symp_normal_conserves_through_the_wrapper_ordinary_drift` | `sympathetic.rs::normal_conserves_through_the_wrapper_ordinary_drift` |  |
| 2458 | `test_symp_frames_decode_to_two_mirror_strings_with_clamped_nuts` | `sympathetic.rs::frames_decode_to_two_mirror_strings_with_clamped_nuts` |  |
| 2471 | `test_symp_audio_is_the_plucked_string_pickup_not_silence` | `sympathetic.rs::audio_is_the_plucked_string_pickup_not_silence` |  |
| 2481 | `test_symp_transfer_tuned_unison_drains_most_of_the_energy` | `sympathetic.rs::transfer_tuned_unison_drains_most_of_the_energy` |  |
| 2492 | `test_symp_transfer_detuned_neighbour_stays_quiet` | `sympathetic.rs::transfer_detuned_neighbour_stays_quiet` |  |
| 2501 | `test_symp_transfer_conserves_and_the_fractions_stay_physical` | `sympathetic.rs::transfer_conserves_and_the_fractions_stay_physical` |  |
| 2511 | `test_symp_lambda_must_be_below_one` | `sympathetic.rs::lambda_must_be_below_one` |  |
| 2518 | `test_symp_over_stiff_bridge_is_rejected_by_the_core_guard` | `sympathetic.rs::over_stiff_bridge_is_rejected_by_the_core_guard` |  |
| 2530 | `test_symp_bad_params_give_error_payload` ×7 | `sympathetic.rs::bad_params_give_error_payload` |  |
| 2535 | `test_symp_normal_work_budget_counts_both_runs` | `sympathetic.rs::normal_work_budget_counts_both_runs` |  |
| 2561 | `test_symp_weinreich_two_stage_knee_prompt_faster_than_aftersound` | `sympathetic.rs::weinreich_two_stage_knee_prompt_faster_than_aftersound` |  |
| 2571 | `test_symp_weinreich_unison_aftersound_is_lossless_rising_with_detune` | `sympathetic.rs::weinreich_unison_aftersound_is_lossless_rising_with_detune` |  |
| 2584 | `test_symp_weinreich_strike_both_decays_away_no_aftersound` | `sympathetic.rs::weinreich_strike_both_decays_away_no_aftersound` |  |
| 2594 | `test_symp_weinreich_lossy_body_reports_passivity_without_a_decay_oracle` | `sympathetic.rs::weinreich_lossy_body_reports_passivity_without_a_decay_oracle` |  |
| 2605 | `test_symp_weinreich_zero_body_loss_flips_to_the_drift_check` | `sympathetic.rs::weinreich_zero_body_loss_flips_to_the_drift_check` |  |
| 2618 | `test_symp_weinreich_envelopes_are_finite_and_normalized` | `sympathetic.rs::weinreich_envelopes_are_finite_and_normalized` |  |
| 2629 | `test_symp_weinreich_audio_is_the_struck_string_pickup` | `sympathetic.rs::weinreich_audio_is_the_struck_string_pickup` |  |
| 2640 | `test_symp_weinreich_work_budget_counts_both_runs` | `sympathetic.rs::weinreich_work_budget_counts_both_runs` |  |
| 2647 | `test_symp_weinreich_detune_range_is_fine_not_semitones` | `sympathetic.rs::weinreich_detune_range_is_fine_not_semitones` |  |
| 2669 | `test_jawari_reproduces_the_suites_shimmer_and_wrap_numbers` | `contact.rs::jawari_reproduces_the_suites_shimmer_and_wrap_numbers` |  |
| 2684 | `test_jawari_energy_keeps_the_flat_loss_oracle_unlike_the_mallet` | `contact.rs::jawari_energy_keeps_the_flat_loss_oracle_unlike_the_mallet` |  |
| 2696 | `test_jawari_sigma0_gates_the_verdict_and_conserves_through_the_curved_wrap` | `contact.rs::jawari_sigma0_gates_the_verdict_and_conserves_through_the_curved_wrap` |  |
| 2707 | `test_jawari_grazing_config_is_labelled_not_failed` | `contact.rs::jawari_grazing_config_is_labelled_not_failed` |  |
| 2719 | `test_jawari_ratio_is_the_control_and_amplitude_moves_it_as_hard_as_depth` | `contact.rs::jawari_ratio_is_the_control_and_amplitude_moves_it_as_hard_as_depth` |  |
| 2729 | `test_jawari_ignores_the_mallet_alpha_and_the_sympathetic_K` | `contact.rs::jawari_ignores_the_mallet_alpha_and_the_sympathetic_k` |  |
| 2742 | `test_jawari_payload_carries_the_bridge_profile_and_the_wrap_marker` | `contact.rs::jawari_payload_carries_the_bridge_profile_and_the_wrap_marker` |  |
| 2760 | `test_jawari_late_spectra_share_one_scale` | `contact.rs::jawari_late_spectra_share_one_scale` |  |
| 2771 | `test_jawari_audio_is_real_and_finite` | `contact.rs::jawari_audio_is_real_and_finite` |  |
| 2779 | `test_jawari_work_budget_counts_both_runs_and_the_guards_are_reachable` | `contact.rs::jawari_work_budget_counts_both_runs_and_the_guards_are_reachable` |  |
| 2790 | `test_jawari_sustain_ratio_is_reported_but_never_gates` | `contact.rs::jawari_sustain_ratio_is_reported_but_never_gates` |  |
| 2817 | `test_juari_tuning_curve_is_position_selective` | `contact.rs::juari_tuning_curve_is_position_selective` |  |
| 2830 | `test_juari_thread_marker_sits_on_the_drawn_curve` | `contact.rs::juari_thread_marker_sits_on_the_drawn_curve` |  |
| 2840 | `test_juari_energy_keeps_the_flat_loss_oracle_like_the_jawari` | `contact.rs::juari_energy_keeps_the_flat_loss_oracle_like_the_jawari` |  |
| 2850 | `test_juari_sigma0_gates_and_conserves_through_the_point_contact` | `contact.rs::juari_sigma0_gates_and_conserves_through_the_point_contact` |  |
| 2862 | `test_juari_thread_snaps_to_a_grid_node_and_reports_the_resolution` | `contact.rs::juari_thread_snaps_to_a_grid_node_and_reports_the_resolution` |  |
| 2876 | `test_juari_reference_lines_are_clean_1x_and_a_flat_jawari` | `contact.rs::juari_reference_lines_are_clean_1x_and_a_flat_jawari` |  |
| 2885 | `test_juari_tuning_curve_is_decoupled_from_the_audio_length` | `contact.rs::juari_tuning_curve_is_decoupled_from_the_audio_length` |  |
| 2894 | `test_juari_ignores_the_mallet_alpha_and_the_sympathetic_K` | `contact.rs::juari_ignores_the_mallet_alpha_and_the_sympathetic_k` |  |
| 2905 | `test_juari_audio_is_real_and_finite` | `contact.rs::juari_audio_is_real_and_finite` |  |
| 2912 | `test_juari_below_signal_is_labelled_not_failed` | `contact.rs::juari_below_signal_is_labelled_not_failed` |  |
| 2923 | `test_juari_guards_are_clean_error_payloads` | `contact.rs::juari_guards_are_clean_error_payloads` |  |
| 2935 | `test_juari_sweet_spot_sits_near_the_nut_settled` | `contact.rs::juari_sweet_spot_sits_near_the_nut_settled` |  |
| 2966 | `test_bore_conserves_with_the_bell_radiating_and_the_split_actually_moves` | `bore.rs::conserves_with_the_bell_radiating_and_the_split_actually_moves` |  |
| 2986 | `test_bore_a_lightly_radiating_clarinet_sheds_a_little_and_still_conserves` | `bore.rs::a_lightly_radiating_clarinet_sheds_a_little_and_still_conserves` |  |
| 2996 | `test_bore_reflection_matches_the_closed_form_including_the_anechoic_null` | `bore.rs::the_reflection_matches_the_closed_form_including_the_anechoic_null` |  |
| 3014 | `test_bore_r_over_z0_is_the_control_and_it_monotonically_buys_loss` | `bore.rs::r_over_z0_is_the_control_and_it_monotonically_buys_loss` |  |
| 3023 | `test_bore_ideal_open_end_is_the_lossless_contrast_with_no_bell_to_score` | `bore.rs::the_ideal_open_end_is_the_lossless_contrast_with_no_bell_to_score` |  |
| 3037 | `test_bore_is_a_clarinet_odd_harmonics_only_at_the_shortest_allowed_render` | `bore.rs::it_is_a_clarinet_odd_harmonics_only_at_the_shortest_allowed_render` |  |
| 3048 | `test_bore_partials_land_on_the_eigenvalue_oracle_which_is_exact_at_lambda_one` | `bore.rs::the_partials_land_on_the_eigenvalue_oracle_which_is_exact_at_lambda_one` |  |
| 3060 | `test_bore_heavily_absorbing_bell_is_labelled_not_failed` | `bore.rs::a_heavily_absorbing_bell_is_labelled_not_failed` |  |
| 3072 | `test_bore_dispersion_is_an_eigenvalue_computation_showing_second_order_departure` | `bore.rs::the_dispersion_is_an_eigenvalue_computation_showing_second_order_departure` |  |
| 3085 | `test_bore_animation_is_paced_on_the_transit_not_the_fundamental` | `bore.rs::the_animation_is_paced_on_the_transit_not_the_fundamental` |  |
| 3096 | `test_bore_frame_and_grid_bookkeeping_line_up` | `bore.rs::frame_and_grid_bookkeeping_line_up` |  |
| 3116 | `test_bore_radiated_frames_are_cumulative_and_normalized_for_the_mouth_glow` | `bore.rs::radiated_frames_are_cumulative_and_normalized_for_the_mouth_glow` |  |
| 3125 | `test_bore_audio_is_real_finite_and_normalized` | `bore.rs::the_audio_is_real_finite_and_normalized` |  |
| 3133 | `test_bore_work_budget_counts_the_reflection_run_and_the_guards_are_reachable` | `bore.rs::the_work_budget_counts_the_reflection_run_and_the_guards_are_reachable` |  |
| 3146 | `test_bore_animation_window_has_its_own_cap_not_the_shared_one` | `bore.rs::the_animation_window_has_its_own_cap_not_the_shared_one` |  |
| 3156 | `test_bore_ignores_params_that_belong_to_other_models` | `bore.rs::it_ignores_params_that_belong_to_other_models` |  |
| 3185 | `test_reed_balance_is_a_measured_residual_and_the_channels_are_load_bearing` | `reed.rs::balance_is_a_measured_residual_and_the_channels_are_load_bearing` |  |
| 3200 | `test_reed_every_balance_channel_is_non_trivially_populated` | `reed.rs::every_balance_channel_is_non_trivially_populated` |  |
| 3214 | `test_reed_balance_has_no_sigma_gate_and_that_is_not_an_oversight` | `reed.rs::balance_has_no_sigma_gate_and_that_is_not_an_oversight` |  |
| 3227 | `test_reed_closes_the_balance_with_the_bell_radiating` | `reed.rs::closes_the_balance_with_the_bell_radiating` |  |
| 3238 | `test_reed_actually_speaks_above_threshold_and_is_silent_below` | `reed.rs::actually_speaks_above_threshold_and_is_silent_below` |  |
| 3253 | `test_reed_blowing_threshold_brackets_one_third` | `reed.rs::blowing_threshold_brackets_one_third` |  |
| 3271 | `test_reed_sweep_is_pinned_off_the_render_grid` | `reed.rs::sweep_is_pinned_off_the_render_grid` |  |
| 3282 | `test_reed_sweep_memo_key_carries_everything_that_moves_p_closing` | `reed.rs::the_sweep_moves_with_everything_that_moves_the_reed_or_the_bell` |  |
| 3294 | `test_reed_pitch_is_set_by_the_air_column_not_the_reed` | `reed.rs::pitch_is_set_by_the_air_column_not_the_reed` |  |
| 3306 | `test_reed_is_a_clarinet_odd_harmonics_dominate` | `reed.rs::is_a_clarinet_odd_harmonics_dominate` |  |
| 3314 | `test_reed_beating_is_debounced_to_one_slam_per_period` | `reed.rs::beating_is_debounced_to_one_slam_per_period` |  |
| 3325 | `test_reed_ships_the_far_field_caveat_beside_the_mouthpiece_audio` | `reed.rs::ships_the_far_field_caveat_beside_the_mouthpiece_audio` |  |
| 3344 | `test_reed_below_threshold_withdraws_the_spectrum_claims` | `reed.rs::below_threshold_withdraws_the_spectrum_claims` |  |
| 3354 | `test_reed_announces_its_mouth_end_for_the_viz_without_touching_the_bore_boundary` | `reed.rs::announces_its_mouth_end_for_the_viz_without_touching_the_bore_boundary` |  |
| 3362 | `test_reed_animation_is_paced_on_the_transit_and_captured_from_the_settled_tail` | `reed.rs::animation_is_paced_on_the_transit_and_captured_from_the_settled_tail` |  |
| 3375 | `test_reed_guards_reject_out_of_range_configurations_cleanly` | `reed.rs::guards_reject_out_of_range_configurations_cleanly` |  |
| 3392 | `test_reed_ignores_params_that_belong_to_other_models` | `reed.rs::ignores_params_that_belong_to_other_models` |  |
| 3419 | `test_fret_reproduces_the_probes_intermittency_numbers` | `contact.rs::fret_reproduces_the_probes_intermittency_numbers` |  |
| 3431 | `test_fret_active_set_is_a_vector_and_the_newton_is_cheap` | `contact.rs::fret_active_set_is_a_vector_and_the_newton_is_cheap` |  |
| 3444 | `test_fret_sigma0_gates_the_verdict_and_conserves_through_genuine_contact` | `contact.rs::fret_sigma0_gates_the_verdict_and_conserves_through_genuine_contact` |  |
| 3454 | `test_fret_drops_the_jawari_decay_oracle_and_ships_the_triple_instead` | `contact.rs::fret_drops_the_jawari_decay_oracle_and_ships_the_triple_instead` |  |
| 3474 | `test_fret_equipartition_correction_tracks_the_rate_across_clearances` | `contact.rs::fret_equipartition_correction_tracks_the_rate_across_clearances` |  |
| 3487 | `test_fret_out_of_reach_rail_is_labelled_not_failed` | `contact.rs::fret_out_of_reach_rail_is_labelled_not_failed` |  |
| 3497 | `test_fret_rail_frac_floor_is_enforced_server_side` | `contact.rs::fret_rail_frac_floor_is_enforced_server_side` |  |
| 3506 | `test_fret_intermittency_is_structural_not_a_tuned_accident` | `contact.rs::fret_intermittency_is_structural_not_a_tuned_accident` |  |
| 3522 | `test_fret_scalars_are_computed_at_full_rate_not_read_off_the_raster` | `contact.rs::fret_scalars_are_computed_at_full_rate_not_read_off_the_raster` |  |
| 3535 | `test_fret_raster_resolves_at_least_ten_columns_per_period` | `contact.rs::fret_raster_resolves_at_least_ten_columns_per_period` |  |
| 3546 | `test_fret_raster_decodes_to_the_grid_and_greys_without_losing_contacts` | `contact.rs::fret_raster_decodes_to_the_grid_and_greys_without_losing_contacts` |  |
| 3564 | `test_fret_brightness_is_reported_with_its_non_monotonicity_named` | `contact.rs::fret_brightness_is_reported_with_its_non_monotonicity_named` |  |
| 3580 | `test_fret_brightness_peaks_at_an_intermediate_clearance` ×2 | `contact.rs::fret_brightness_peaks_at_an_intermediate_clearance` |  |
| 3600 | `test_fret_signature_block_carries_its_dispatch_kind` | `contact.rs::fret_reproduces_the_probes_intermittency_numbers + fret_crossing_rate_is_never_called_pitch` | the two `kind` asserts, one in each |
| 3610 | `test_fret_crossing_rate_is_never_called_pitch` | `contact.rs::fret_crossing_rate_is_never_called_pitch` |  |
| 3624 | `test_fret_control_window_is_independent_of_the_fret_window` | `contact.rs::fret_control_window_is_independent_of_the_fret_window` |  |
| 3638 | `test_fret_ignores_the_names_that_belong_to_other_models` | `contact.rs::fret_ignores_the_names_that_belong_to_other_models` |  |
| 3652 | `test_fret_work_budget_counts_both_runs_and_the_guards_are_reachable` | `contact.rs::fret_work_budget_counts_both_runs_and_the_guards_are_reachable` |  |
| 3668 | `test_fret_control_is_bounded_and_the_animation_is_a_stride_of_one_run` | `contact.rs::fret_control_is_bounded_and_the_animation_is_a_stride_of_one_run` |  |
| 3684 | `test_fret_audio_is_a_near_termination_pickup_and_is_real` | `contact.rs::fret_audio_is_a_near_termination_pickup_and_is_real` |  |
| 3698 | `test_fret_episode_debounce_merges_chatter_without_inventing_it` | `contact.rs::fret_episode_debounce_merges_chatter_without_inventing_it` |  |
| 3724 | `test_body_conserves_through_the_coupling_while_the_string_alone_does_not` | `body.rs::conserves_through_the_coupling_while_the_string_alone_does_not` |  |
| 3741 | `test_body_k_zero_decouples_the_string_bit_for_bit` | `body.rs::k_zero_decouples_the_string_bit_for_bit` |  |
| 3767 | `test_body_exchange_fractions_carry_e_conn_and_must_not_be_stacked` | `body.rs::exchange_fractions_carry_e_conn_and_must_not_be_stacked` |  |
| 3786 | `test_body_exchange_slosh_is_prompt_and_windowed` | `body.rs::exchange_slosh_is_prompt_and_windowed` |  |
| 3795 | `test_body_terminus_glides_from_free_toward_clamped_as_the_bridge_stiffens` | `body.rs::terminus_glides_from_free_toward_clamped_as_the_bridge_stiffens` |  |
| 3807 | `test_body_monopole_omega2_is_a_consistency_check_near_one_not_an_oracle` | `body.rs::monopole_omega2_is_a_consistency_check_near_one_not_an_oracle` |  |
| 3816 | `test_body_one_over_r_scales_level_and_latency_only_never_the_spectrum_shape` | `body.rs::one_over_r_scales_level_and_latency_only_never_the_spectrum_shape` |  |
| 3828 | `test_body_sigma_body_gates_the_verdict_and_drops_the_decay_oracle` | `body.rs::sigma_body_gates_the_verdict_and_drops_the_decay_oracle` |  |
| 3840 | `test_body_guard_is_the_exact_bound_surfaced_as_a_clean_error` | `body.rs::guard_is_the_exact_bound_surfaced_as_a_clean_error` |  |
| 3853 | `test_body_frame_and_grid_bookkeeping_line_up_and_the_nut_stays_clamped` | `body.rs::frame_and_grid_bookkeeping_line_up_and_the_nut_stays_clamped` |  |
| 3866 | `test_body_audio_is_the_far_field_pressure_real_and_normalized` | `body.rs::audio_is_the_far_field_pressure_real_and_normalized` |  |
| 3877 | `test_body_work_budget_and_the_n_ceiling_are_reachable` | `body.rs::work_budget_and_the_n_ceiling_are_reachable` |  |
| 3896 | `test_body_ignores_params_that_belong_to_other_models` | `body.rs::ignores_params_that_belong_to_other_models` |  |
| 3925 | `test_platebody_conserves_through_the_coupling_on_both_boundaries` | `platebody.rs::conserves_through_the_coupling_on_both_boundaries` |  |
| 3942 | `test_platebody_terminus_is_the_OPPOSITE_story_per_boundary` | `platebody.rs::the_terminus_is_the_opposite_story_per_boundary` |  |
| 3969 | `test_platebody_k_zero_decouples_the_string_bit_for_bit` | `platebody.rs::k_zero_decouples_the_string_bit_for_bit` |  |
| 3994 | `test_platebody_heatmap_is_a_real_2d_field_that_rings_and_stays_masked` | `platebody.rs::the_heatmap_is_a_real_2d_field_that_rings_and_stays_masked` |  |
| 4019 | `test_platebody_exchange_fractions_carry_e_conn_and_must_not_be_stacked` | `platebody.rs::the_exchange_fractions_carry_e_conn_and_must_not_be_stacked` |  |
| 4038 | `test_platebody_monopole_omega2_uses_volume_displacement_not_the_driving_point` | `platebody.rs::the_monopole_omega2_uses_the_volume_displacement` |  |
| 4050 | `test_platebody_one_over_r_scales_level_and_latency_only_never_the_spectrum_shape` | `platebody.rs::one_over_r_scales_level_and_latency_only_never_the_spectrum_shape` |  |
| 4061 | `test_platebody_sigma_plate_gates_the_verdict_and_drops_the_decay_oracle` | `platebody.rs::sigma_plate_gates_the_verdict_and_drops_the_decay_oracle` |  |
| 4072 | `test_platebody_guard_is_the_exact_bound_surfaced_as_a_clean_error_on_both_boundaries` | `platebody.rs::the_guard_is_the_exact_bound_surfaced_as_a_clean_error_on_both_boundaries` |  |
| 4093 | `test_platebody_audio_is_the_far_field_pressure_real_and_normalized` | `platebody.rs::the_audio_is_the_far_field_pressure_real_and_normalized` |  |
| 4104 | `test_platebody_work_budget_and_the_ceilings_are_reachable` | `platebody.rs::the_work_budget_and_the_ceilings_are_reachable` |  |
| 4126 | `test_platebody_ignores_params_that_belong_to_other_models` | `platebody.rs::it_ignores_params_that_belong_to_other_models` |  |
| 4153 | `test_radbody_conserves_while_the_air_carries_the_energy_away` | `radbody.rs::conserves_while_the_air_carries_the_energy_away` |  |
| 4169 | `test_radbody_r_zero_is_bit_identical_to_the_readout_only_body` | `body.rs::radbody_at_r_zero_is_bit_identical_to_the_readout_only_body` |  |
| 4187 | `test_radbody_four_channels_sum_to_the_flat_reference_and_e_conn_stays_signed` | `radbody.rs::four_channels_sum_to_the_flat_reference_and_e_conn_stays_signed` |  |
| 4206 | `test_radbody_t50_optimum_is_physics_not_the_scheme_timestep` | `radbody.rs::t50_optimum_is_physics_not_the_scheme_timestep` |  |
| 4220 | `test_radbody_sweep_is_a_controlled_reference_curve_not_the_render` | `radbody.rs::sweep_is_a_controlled_reference_curve_not_the_render` |  |
| 4237 | `test_radbody_radiated_fraction_is_amplitude_invariant_bit_exactly` | `radbody.rs::radiated_fraction_is_amplitude_invariant_bit_exactly` |  |
| 4248 | `test_radbody_more_air_is_worse_the_optimum_is_a_broad_basin` | `radbody.rs::more_air_is_worse_the_optimum_is_a_broad_basin` |  |
| 4264 | `test_radbody_k_zero_skips_the_sweep_instead_of_drawing_a_row_of_nans` | `radbody.rs::k_zero_skips_the_sweep_instead_of_drawing_a_row_of_nans` |  |
| 4278 | `test_radbody_sigma_body_gates_the_verdict_and_radiation_never_does` | `radbody.rs::sigma_body_gates_the_verdict_and_radiation_never_does` |  |
| 4292 | `test_radbody_the_load_turns_the_reservoir_into_a_conduit` | `radbody.rs::the_load_turns_the_reservoir_into_a_conduit` |  |
| 4304 | `test_radbody_f_match_names_the_one_frequency_where_load_and_readout_agree` | `radbody.rs::f_match_names_the_one_frequency_where_load_and_readout_agree` |  |
| 4320 | `test_radbody_shipped_sweep_settings_are_the_measured_ones` | `radbody.rs::shipped_sweep_settings_are_the_measured_ones` |  |
| 4346 | `test_radbody_sweep_truncation_censors_the_tail_rather_than_hanging` | `radbody.rs::sweep_truncation_censors_the_tail_rather_than_hanging` |  |
| 4362 | `test_radbody_frame_and_grid_bookkeeping_line_up_and_the_nut_stays_clamped` | `radbody.rs::frame_and_grid_bookkeeping_line_up_and_the_nut_stays_clamped` |  |
| 4374 | `test_radbody_audio_is_the_far_field_pressure_real_and_normalized` | `radbody.rs::audio_is_the_far_field_pressure_real_and_normalized` |  |
| 4387 | `test_radbody_guards_and_budget_are_clean_error_payloads` | `radbody.rs::guards_and_budget_are_clean_error_payloads` |  |
| 4407 | `test_radbody_ignores_params_that_belong_to_other_models` | `radbody.rs::ignores_params_that_belong_to_other_models` |  |
| 4433 | `test_airload_conserves_while_the_air_both_stores_and_radiates` | `airload.rs::conserves_while_the_air_both_stores_and_radiates` |  |
| 4449 | `test_airload_a_zero_corner_reproduces_batch_15s_shipped_picture_bit_identically` | `airload.rs::a_zero_corner_reproduces_the_constant_r_body_bit_identically` |  |
| 4468 | `test_airload_five_channels_sum_to_the_flat_reference_and_e_conn_stays_signed` | `airload.rs::five_channels_sum_to_the_flat_reference_and_e_conn_stays_signed` |  |
| 4494 | `test_airload_the_ledger_residual_catches_what_the_drift_cannot` | `airload.rs::the_ledger_residual_catches_what_the_drift_cannot` |  |
| 4511 | `test_airload_alpha_climbs_with_frequency_and_the_constant_r_impostor_cannot` | `airload.rs::alpha_climbs_with_frequency_and_the_constant_r_impostor_cannot` |  |
| 4524 | `test_airload_the_oracle_residual_is_second_order_in_the_loading_ratio` | `airload.rs::the_oracle_residual_is_second_order_in_the_loading_ratio` |  |
| 4543 | `test_airload_the_pitch_drop_is_the_reactance_and_a_flat_load_has_none` | `airload.rs::the_pitch_drop_is_the_reactance_and_a_flat_load_has_none` |  |
| 4562 | `test_airload_the_pitch_shift_needs_the_schemes_own_unloaded_reference` | `airload.rs::the_pitch_shift_needs_the_schemes_own_unloaded_reference` |  |
| 4578 | `test_airload_overdamped_points_are_censored_not_guessed` | `airload.rs::overdamped_points_are_censored_not_guessed` |  |
| 4589 | `test_airload_r_zero_skips_the_sweep_with_a_label_instead_of_a_row_of_nulls` | `airload.rs::r_zero_skips_the_sweep_with_a_label_instead_of_a_row_of_nulls` |  |
| 4602 | `test_airload_sweep_is_pinned_at_48k_and_does_not_follow_the_renders_rate` | `airload.rs::sweep_is_pinned_at_48k_and_does_not_follow_the_renders_rate` |  |
| 4615 | `test_airload_sigma_body_gates_the_verdict_and_radiation_never_does` | `airload.rs::sigma_body_gates_the_verdict_and_radiation_never_does` |  |
| 4625 | `test_airload_the_radiation_weight_is_not_a_volume_control_and_the_peak_says_so` | `airload.rs::the_radiation_weight_is_not_a_volume_control_and_the_peak_says_so` |  |
| 4636 | `test_airload_the_sphere_readout_says_which_radius_each_coefficient_implies` | `airload.rs::the_sphere_readout_says_which_radius_each_coefficient_implies` |  |
| 4650 | `test_airload_the_exact_load_tracks_batch_15s_compact_law_over_the_band` | `airload.rs::the_exact_load_tracks_the_compact_law_over_the_band` |  |
| 4664 | `test_airload_shipped_sweep_settings_are_the_measured_ones` | `airload.rs::shipped_sweep_settings_are_the_measured_ones` |  |
| 4683 | `test_airload_sweep_truncation_censors_the_tail_rather_than_hanging` | `airload.rs::sweep_truncation_censors_the_tail_rather_than_hanging` |  |
| 4694 | `test_airload_the_coupled_k_guard_is_the_bare_bodys_and_the_load_does_not_move_it` | `airload.rs::the_coupled_k_guard_is_the_bare_bodys_and_the_load_does_not_move_it` |  |
| 4708 | `test_airload_frame_and_grid_bookkeeping_line_up_and_the_nut_stays_clamped` | `airload.rs::frame_and_grid_bookkeeping_line_up_and_the_nut_stays_clamped` |  |
| 4719 | `test_airload_guards_and_budget_are_clean_error_payloads` | `airload.rs::guards_and_budget_are_clean_error_payloads` |  |
| 4733 | `test_airload_ignores_params_that_belong_to_other_models` | `airload.rs::ignores_params_that_belong_to_other_models` |  |
| 4772 | `test_airbox_the_cone_is_the_manhattan_cell_count_exactly` | `airbox.rs::the_cone_is_the_manhattan_cell_count_exactly` |  |
| 4783 | `test_airbox_the_cone_tracks_the_mic_and_never_stops_matching` | `airbox.rs::the_cone_tracks_the_mic_and_never_stops_matching` |  |
| 4793 | `test_airbox_the_cone_is_lambda_independent_while_the_amplitude_arrivals_are_not` | `airbox.rs::the_cone_is_lambda_independent_while_the_amplitude_arrivals_are_not` |  |
| 4811 | `test_airbox_the_money_test_is_the_cross_ledger_residual` | `airbox.rs::the_money_test_is_the_cross_ledger_residual` |  |
| 4821 | `test_airbox_the_room_sets_fs_and_the_strings_lambda_is_derived` | `airbox.rs::the_room_sets_fs_and_the_strings_lambda_is_derived` |  |
| 4838 | `test_airbox_ships_three_named_slices_not_a_volume` | `airbox.rs::it_ships_three_named_slices_not_a_volume` |  |
| 4855 | `test_airbox_slices_decimate_and_report_their_own_stride` | `airbox.rs::the_slices_decimate_and_report_their_own_stride` |  |
| 4871 | `test_airbox_the_colour_mapping_travels_with_the_values` | `airbox.rs::the_colour_mapping_travels_with_the_values` |  |
| 4881 | `test_airbox_grid_snap_is_reported_never_silently_resampled` | `airbox.rs::the_grid_snap_is_reported_never_silently_resampled` |  |
| 4889 | `test_airbox_open_walls_are_lossless_and_only_absorbing_is_not` | `airbox.rs::open_walls_are_lossless_and_only_absorbing_is_not` |  |
| 4904 | `test_airbox_payload_survives_the_servers_strict_json` | `airbox.rs::the_payload_survives_strict_json` |  |
| 4914 | `test_airbox_guards_and_budget_are_clean_error_payloads` | `airbox.rs::guards_and_budget_are_clean_error_payloads` |  |
| 4934 | `test_airbox_ignores_params_that_belong_to_other_models` | `airbox.rs::it_ignores_params_that_belong_to_other_models` |  |
| 4980 | `test_vkroom_the_struck_plates_pattern_moves_and_the_linear_ones_does_not` | `vkroom.rs::the_struck_plates_pattern_moves_and_the_linear_ones_does_not` |  |
| 5001 | `test_vkroom_the_honesty_line_is_convergence_because_neither_ledger_is` | `vkroom.rs::the_honesty_line_is_convergence_because_neither_ledger_is` |  |
| 5018 | `test_vkroom_both_ledgers_are_green_and_the_scene_is_conserved` | `vkroom.rs::both_ledgers_are_green_and_the_scene_is_conserved` |  |
| 5028 | `test_vkroom_the_compact_monopole_is_shipped_beside_the_truth_not_as_it` | `vkroom.rs::the_compact_monopole_is_shipped_beside_the_truth_not_as_it` |  |
| 5042 | `test_vkroom_ships_the_plate_field_and_the_room_slices_on_TWO_clocks` | `vkroom.rs::it_ships_the_plate_field_and_the_room_slices_on_two_clocks` |  |
| 5060 | `test_vkroom_reports_the_snapped_room_and_the_derived_rate` | `vkroom.rs::it_reports_the_snapped_room_and_the_derived_rate` |  |
| 5073 | `test_vkroom_payload_survives_the_servers_strict_json` | `vkroom.rs::it_reports_the_snapped_room_and_the_derived_rate` | `has_nonfinite` — the strict-JSON check |
| 5078 | `test_vkroom_the_flag_off_makes_the_run_its_own_control` | `vkroom.rs::the_flag_off_makes_the_run_its_own_control` |  |
| 5088 | `test_vkroom_the_claim_survives_the_COARSEST_legal_plate` | `vkroom.rs::the_claim_survives_the_coarsest_legal_plate` |  |
| 5124 | `test_vkroom_both_tiers_run_and_the_suspended_one_radiates_from_both_faces` | `vkroom.rs::both_tiers_run` |  |
| 5131 | `test_vkroom_guards_are_clean_error_payloads_never_a_500` | `vkroom.rs::guards_are_clean_error_payloads` |  |
| 5150 | `test_vkroom_the_two_term_budget_refuses_what_neither_shipped_cap_would` | `vkroom.rs::the_two_term_budget_refuses_what_neither_shipped_cap_would` |  |
| 5161 | `test_vkroom_ignores_params_that_belong_to_other_models` | `vkroom.rs::it_ignores_params_that_belong_to_other_models` |  |
| 5202 | `test_horizon_every_model_the_viewer_OFFERS_is_classified` | `horizon.rs::every_model_the_viewer_offers_is_classified` |  |
| 5226 | `test_horizon_the_canvas_MARK_vocabulary_is_the_same_word_in_both_files` | `front_end.rs::the_canvas_mark_vocabulary_is_the_same_word_in_both_files` | now against the Rust harness, `examples/verify_headless.rs` |
| 5254 | `test_horizon_a_prefix_block_carries_every_field_the_frontend_reads` | `horizon.rs::a_prefix_block_carries_every_field_the_frontend_reads` |  |
| 5271 | `test_horizon_is_absent_from_an_error_payload` | `horizon.rs::is_absent_from_an_error_payload` |  |
| 5277 | `test_horizon_the_explicit_string_at_lambda_one_is_in_tune_across_the_WHOLE_grid` | `horizon.rs::the_explicit_string_at_lambda_one_is_in_tune_across_the_whole_grid` |  |
| 5291 | `test_horizon_refining_the_explicit_timestep_makes_the_read_out_WORSE` | `horizon.rs::refining_the_explicit_timestep_makes_the_read_out_worse` |  |
| 5304 | `test_horizon_the_theta_string_cannot_pass_its_own_space_floor` | `horizon.rs::the_theta_string_cannot_pass_its_own_space_floor` |  |
| 5320 | `test_horizon_is_built_from_the_SCHEME_and_not_from_the_display_arrays` | `horizon.rs::is_built_from_the_scheme_and_not_from_the_display_arrays` |  |
| 5335 | `test_horizon_a_tighter_bound_can_only_shorten_the_claim` | `horizon.rs::a_tighter_bound_can_only_shorten_the_claim` |  |
| 5355 | `test_horizon_the_hertz_ceiling_stops_BELOW_the_first_mode_that_is_out_of_tune` | `horizon.rs::the_hertz_ceiling_stops_below_the_first_mode_that_is_out_of_tune` |  |
| 5371 | `test_horizon_the_1d_prefix_and_index_readings_are_the_SAME_list` | `horizon.rs::the_1d_prefix_and_index_readings_are_the_same_list` |  |
| 5389 | `test_horizon_the_2d_readings_differ_and_the_index_one_is_the_conservative_one` | `horizon.rs::the_2d_readings_differ_and_the_index_one_is_the_conservative_one` |  |
| 5405 | `test_horizon_a_membranes_worst_corner_is_AXIAL_at_the_courant_ceiling` | `horizon.rs::a_membranes_worst_corner_is_axial_at_the_courant_ceiling` |  |
| 5424 | `test_horizon_a_zero_horizon_ships_None_rather_than_a_NUMBER` | `horizon.rs::a_zero_horizon_ships_null_rather_than_a_number` |  |
| 5450 | `test_horizon_every_refusal_names_its_MECHANISM_not_just_its_absence` | `horizon.rs::every_refusal_names_its_mechanism_not_just_its_absence` |  |
| 5465 | `test_horizon_a_bridge_coupled_string_is_REFUSED_rather_than_quoted` | `horizon.rs::a_bridge_coupled_string_is_refused_rather_than_quoted` | **added at §23.19** — the table was asserted, the payload path was not |
| 5478 | `test_horizon_the_plate_key_covers_three_plates_and_only_one_has_a_horizon` | `horizon.rs::the_plate_key_covers_three_plates_and_only_one_has_a_horizon` |  |
| 5491 | `test_horizon_the_nonlinear_plate_is_refused_and_its_LINEAR_twin_is_not` | `horizon.rs::the_nonlinear_plate_is_refused_and_its_linear_twin_is_not` |  |
| 5505 | `test_horizon_an_exciter_inherits_the_horizon_of_what_it_DRIVES` | `bow.rs::the_horizon_is_the_strings_it_drives + mallet.rs::a_struck_rectangle_reads_exactly_like_the_membrane_on_its_own` | split by scene |
| 5519 | `test_horizon_survives_the_servers_strict_json` | `horizon.rs::survives_strict_json` |  |
