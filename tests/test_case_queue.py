from pathlib import Path
import csv
import tempfile

from globalhab_demo.case_manager import (
    archive_case,
    bulk_create_cases,
    cancel_case,
    case_status_code,
    case_status_counts,
    delete_case,
    get_case,
    mark_case_in_progress,
    next_review_case,
    queue_cases,
    restore_case,
)


def _research(region: str, risk: float = 70.0):
    return {
        "source": "test",
        "candidate_region": region,
        "risk_score": risk,
        "issue_date": "2026-09-16",
        "forecast_window": "2026-09-20 至 2026-09-26",
        "horizon_days": 7,
        "route": "downstream",
        "lag_days": 14,
        "scenario": {"mhw_intensity_c": 2.0},
    }


def test_bulk_case_queue_status_and_deduplication():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        created, skipped = bulk_create_cases([_research("A"), _research("B")], root)
        assert len(created) == 2 and not skipped
        created2, skipped2 = bulk_create_cases([_research("A")], root)
        assert not created2 and len(skipped2) == 1
        first = created[0]["case_id"]
        second = created[1]["case_id"]
        mark_case_in_progress(first, root)
        assert case_status_code(get_case(first, root)) == "in_progress"
        nxt = next_review_case(first, root)
        assert nxt and nxt["case_id"] == second
        cancel_case(second, root=root)
        assert case_status_code(get_case(second, root)) == "cancelled"
        assert second not in [c["case_id"] for c in queue_cases(root)]
        restore_case(second, root)
        assert case_status_code(get_case(second, root)) == "pending_review"
        archive_case(second, root=root)
        assert case_status_code(get_case(second, root)) == "archived"
        counts = case_status_counts(root)
        assert counts["in_progress"] == 1 and counts["archived"] == 1


def test_delete_case_can_remove_linked_sample_from_future_training():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        created, _ = bulk_create_cases([_research("C")], root)
        cid = created[0]["case_id"]
        lib = root / "data" / "field_visual" / "user_library"
        lib.mkdir(parents=True, exist_ok=True)
        path = lib / "records.csv"
        fields = ["sample_id", "case_id", "include_in_training"]
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerow({"sample_id": "S1", "case_id": cid, "include_in_training": "True"})
        usage = delete_case(cid, root, remove_from_future_training=True)
        assert usage["future_training_samples"] == 1
        assert usage["removed_from_future_training"] == 1
        assert get_case(cid, root) is None
        text = path.read_text(encoding="utf-8")
        assert "False" in text


def test_delete_case_can_remove_library_rows_and_images():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        created, _ = bulk_create_cases([_research("D")], root)
        cid = created[0]["case_id"]
        lib = root / "data" / "field_visual" / "user_library"
        images = lib / "images"
        images.mkdir(parents=True, exist_ok=True)
        img = images / "sample.jpg"
        img.write_bytes(b"fake")
        path = lib / "records.csv"
        fields = ["sample_id", "case_id", "include_in_training", "image_relpath"]
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerow({"sample_id": "S2", "case_id": cid, "include_in_training": "False", "image_relpath": "data/field_visual/user_library/images/sample.jpg"})
        usage = delete_case(cid, root, delete_library_samples=True)
        assert usage["deleted_library_samples"] == 1
        assert not img.exists()
        assert cid not in path.read_text(encoding="utf-8")
