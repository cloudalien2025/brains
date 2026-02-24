from __future__ import annotations

from brains.status_adapter import normalize_run_for_display


def test_normalize_run_for_display_with_selected_lists():
    run_json = {
        "selected_youtube": [
            {"video_id": "abc123", "title": "A", "uploader": "Chan", "duration": 90, "url": "https://youtu.be/abc123"},
        ],
        "selected_webdocs": [
            {"doc_id": "doc1", "title": "Doc", "url": "https://example.com", "domain": "example.com", "provider": "ddg"},
        ],
    }
    display = normalize_run_for_display(run_json)
    assert display["selected_youtube"][0]["video_id"] == "abc123"
    assert display["selected_webdocs"][0]["doc_id"] == "doc1"


def test_normalize_run_for_display_fallback_to_files():
    run_json = {"transcripts_succeeded": 1}
    files_json = {
        "transcript_files_txt": [{"name": "transcripts/yt_xyz.txt", "bytes": 12}],
        "doc_text_files": ["docs/text/doc_42.txt"],
    }
    display = normalize_run_for_display(run_json, files_json)
    assert display["selected_youtube"][0]["video_id"] == "xyz"
    assert display["selected_webdocs"][0]["doc_id"] == "doc_42"
    assert display["succeeded_youtube"][0]["video_id"] == "xyz"
    assert display["succeeded_webdocs"][0]["doc_id"] == "doc_42"


def test_normalize_run_for_display_failures():
    run_json = {
        "sample_failures": [
            {"item": "vid1", "stage": "audio", "error_code": "audio_download_failed", "error": "boom"},
            {"item": "doc7", "stage": "webdocs", "error_code": "doc_failed", "error": "bad"},
        ]
    }
    display = normalize_run_for_display(run_json)
    assert display["failed_youtube"][0]["video_id"] == "vid1"
    assert display["failed_webdocs"][0]["doc_id"] == "doc7"


def test_normalize_run_for_display_missing_keys():
    display = normalize_run_for_display(None, None)
    assert display["selected_youtube"] == []
    assert display["selected_webdocs"] == []
