"""sync_storage mirrors the archetype model ahead of the catalog (DEC-TB-0007)."""
import importlib.util
import sys
from pathlib import Path

SYNC = Path(__file__).resolve().parents[1] / "src" / "sources" / "sync_storage.py"


def _load():
    spec = importlib.util.spec_from_file_location("sync_storage_t", SYNC)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sync_storage_t"] = mod
    spec.loader.exec_module(mod)
    return mod


class FakeS3:
    def __init__(self):
        self.order = []

    def head_bucket(self, Bucket):
        return {}

    def get_paginator(self, _):
        class P:
            def paginate(self, **kw):
                return [{"Contents": []}]
        return P()

    def list_objects_v2(self, **kw):
        return {"Contents": [], "IsTruncated": False}

    def upload_file(self, path, bucket, key):
        self.order.append(key)


def test_model_uploads_before_catalog_before_data(tmp_path, monkeypatch):
    m = _load()
    (tmp_path / "sources.yaml").write_text("- id: x\n")
    d = tmp_path / "data" / "x"
    d.mkdir(parents=True)
    (d / "2026-11-03.parquet").write_bytes(b"pq")
    model = tmp_path / "archetypes.json"
    model.write_text('{"version": 1}')
    monkeypatch.setattr(m, "ARCHETYPE_MODEL", model)
    monkeypatch.setattr(m, "_remote_sizes", lambda s3, b: {})
    s3 = FakeS3()
    assert m._upload(s3, "bucket", tmp_path, dry_run=False) == 0
    assert s3.order[:2] == ["archetypes.json", "sources.yaml"]
    assert "data/x/2026-11-03.parquet" in s3.order


def test_missing_model_is_not_an_error(tmp_path, monkeypatch):
    m = _load()
    (tmp_path / "sources.yaml").write_text("- id: x\n")
    monkeypatch.setattr(m, "ARCHETYPE_MODEL", tmp_path / "absent.json")
    monkeypatch.setattr(m, "_remote_sizes", lambda s3, b: {})
    s3 = FakeS3()
    assert m._upload(s3, "bucket", tmp_path, dry_run=False) == 0
    assert s3.order == ["sources.yaml"]
