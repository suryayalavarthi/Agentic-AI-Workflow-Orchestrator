from __future__ import annotations

from src.tools.memory import VectorDB


def test_chunk_text_basic(mock_settings):
    db = VectorDB.__new__(VectorDB)
    text = "a" * 2500
    chunks = db.chunk_text(text, chunk_size=1000, chunk_overlap=200)
    assert len(chunks) == 3


def test_chunk_text_overlap(mock_settings):
    db = VectorDB.__new__(VectorDB)
    text = "ABCDEFGHIJ" * 100  # 1000 chars
    chunks = db.chunk_text(text, chunk_size=500, chunk_overlap=100)
    assert len(chunks) >= 2
    # Verify overlap: end of chunk 0 should appear at start of chunk 1
    overlap = chunks[0][-100:]
    assert chunks[1].startswith(overlap)


def test_chunk_text_empty(mock_settings):
    db = VectorDB.__new__(VectorDB)
    assert db.chunk_text("", chunk_size=1000, chunk_overlap=200) == []


def test_chunk_text_smaller_than_chunk_size(mock_settings):
    db = VectorDB.__new__(VectorDB)
    text = "short text"
    chunks = db.chunk_text(text, chunk_size=1000, chunk_overlap=200)
    assert len(chunks) == 1
    assert chunks[0] == "short text"


def test_chunk_text_zero_chunk_size(mock_settings):
    db = VectorDB.__new__(VectorDB)
    text = "some text"
    chunks = db.chunk_text(text, chunk_size=0, chunk_overlap=0)
    assert chunks == ["some text"]
