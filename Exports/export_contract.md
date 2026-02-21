# BD_Brain Export Contract

## Export Artifact

- **Filename:** `bd_brain_export.json`

## Structure

```json
{
  "brain": "BD_Brain",
  "version": "vX.X",
  "generated_at": "2026-02-21T00:00:00Z",
  "records": [],
  "index": {
    "by_topic": {},
    "by_type": {},
    "by_status": {},
    "by_tag": {}
  }
}
```

## Inclusion Rules

- Include records with `status` in: `active`, `experimental`.
- Exclude `deprecated` records by default.
- Include `disputed` records only when explicitly requested by consumer configuration.
- Consumers must version-pin (`version`) to avoid non-deterministic behavior across updates.
