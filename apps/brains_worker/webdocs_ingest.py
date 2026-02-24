from __future__ import annotations

from typing import Any

from apps.brains_worker.ingest_types import DocCandidate, ItemResult, RunContext


def process_doc(doc: DocCandidate, run_ctx: RunContext, config: dict[str, Any]) -> ItemResult:
    return ItemResult(
        item_id=doc.doc_id,
        success=False,
        error_code="doc_ingest_not_implemented",
        error_message="webdocs ingestion not implemented",
    )
