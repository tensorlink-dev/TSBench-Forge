"""Resolving the Cascade king checkpoint from the public receipts (network mocked)."""

import cascade_king as ck


def test_resolve_king_picks_the_latest_scored_round_king_checkpoint(monkeypatch):
    index = {"rounds": [
        {"status": "scored", "receipt_key": "receipts/v/old.json", "published_at": "2026-10-05T01:00:00+00:00",
         "round_id": "1"},
        {"status": "scored", "receipt_key": "receipts/v/new.json", "published_at": "2026-10-06T01:00:00+00:00",
         "round_id": "2", "post_round_king_uid": 147, "post_round_king_hotkey": "5E..."},
        {"status": "skipped", "receipt_key": "receipts/v/skip.json", "published_at": "2026-10-07T01:00:00+00:00"},
    ]}
    receipt = {"manifest": {"entries": [
        {"trained_pointer": "metro-v1:trained:hippius:cascade/ckpt-r2-challenger-toto2-4m-u163@sha256:aa"},
        {"trained_pointer": "metro-v1:trained:hippius:cascade/ckpt-r2-king-toto2-4m@sha256:bb"},
    ]}}
    urls = []

    def fake_get(url, timeout=60):
        urls.append(url)
        return index if url.endswith("index.json") else receipt

    monkeypatch.setattr(ck, "_get_json", fake_get)
    king = ck.resolve_king()
    assert king.ref == "cascade/ckpt-r2-king-toto2-4m@sha256:bb"
    assert (king.round_id, king.king_uid) == ("2", 147)
    assert urls[-1].endswith("receipts/v/new.json")
