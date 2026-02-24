from __future__ import annotations

from brains.status_adapter import report_counters, report_not_ready


def test_report_not_ready_detects_detail():
    assert report_not_ready({"detail": "report_not_ready"}) is True
    assert report_not_ready({"detail": "REPORT_NOT_READY"}) is True
    assert report_not_ready({"detail": "ok"}) is False
    assert report_not_ready(None) is False


def test_report_counters_fallbacks_to_run_data():
    run_data = {
        "transcripts_succeeded": 4,
        "transcripts_failed": 1,
        "total_audio_minutes": 12.5,
    }
    counters = report_counters({}, run_data)
    assert counters["items_succeeded_total"] == 4
    assert counters["items_failed_total"] == 1
    assert counters["total_audio_minutes"] == 12.5

    report = {
        "items_succeeded_total": 7,
        "items_failed_total": 2,
        "total_audio_minutes": 1.5,
    }
    counters = report_counters(report, run_data)
    assert counters["items_succeeded_total"] == 7
    assert counters["items_failed_total"] == 2
    assert counters["total_audio_minutes"] == 1.5
