# Branch coverage checklist

- [x] `farkle.utils.random` — torch import failure and tensorflow seed suppression branches — `test_seed_everything_ignores_torch_import_error`, `test_seed_everything_suppresses_tensorflow_set_seed_error`.
- [x] `farkle.orchestration.run_contexts` — missing combine stage, preserved tiering seeds, and interseed layout fallbacks (`None`/unknown layout) — `test_interseed_context_raises_when_combine_missing`, `test_interseed_context_preserves_existing_tiering_seeds`, `test_run_context_config_interseed_folder_mapping_missing_key`.
- [x] `farkle.utils.mdd` — zero-division fallback without Jeffreys prior and missing-games default branch — `test_ensure_winrate_without_jeffreys_fills_nan_to_zero`, `test_prepare_cell_means_defaults_games_when_column_missing`.
- [x] `farkle.utils.manifest` — empty append-many early return, windows binary-flag path, ensure-dir false path, and windows lock retry/unlock branches — `test_append_manifest_many_empty_records_no_file_created`, `test_open_append_fd_adds_binary_flag_on_windows`, `test_append_manifest_many_without_ensure_dir_writes_when_parent_exists`, `test_windows_lock_helpers_retry_and_unlock`.
