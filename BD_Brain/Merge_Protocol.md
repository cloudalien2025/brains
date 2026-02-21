# Merge Protocol

1. Validate extracted records against `Schemas/brain_core.schema.json`.
2. Deduplicate by semantic assertion and source overlap.
3. If contradictory, mark as `disputed` and append to `Conflict_Log.md`.
4. Append accepted records to `BD_Brain_Core.jsonl` only after reviewer sign-off.
5. Record pack summary in `Assertions_Index.md`.
