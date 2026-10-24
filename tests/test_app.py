import pytest
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient

import answer as answer_module
import app as app_module
from app import app
from ingest import _hard_split, _strip_running_headers, extract_chunks


def _escape(text: str) -> bytes:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)").encode(
        "latin-1"
    )


def make_pdf(pages: list[list[str]]) -> bytes:
    """Build a minimal, valid multi-page PDF with one text line per entry."""
    font_number = 3 + 2 * len(pages)
    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: (
            f"<< /Type /Pages /Kids [{' '.join(f'{3 + 2 * i} 0 R' for i in range(len(pages)))}] "
            f"/Count {len(pages)} >>"
        ).encode(),
        font_number: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
        b"/Encoding /WinAnsiEncoding >>",
    }

    for number, lines in enumerate(pages):
        page_obj = 3 + 2 * number
        content_obj = page_obj + 1
        stream = b"BT /F1 11 Tf 72 760 Td 16 TL\n" + b"".join(
            b"(" + _escape(line) + b") Tj T*\n" for line in lines
        ) + b"ET"
        objects[page_obj] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {content_obj} 0 R /Resources << /Font << /F1 {font_number} 0 R >> >> >>"
        ).encode()
        objects[content_obj] = (
            f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream"
        )

    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for number in sorted(objects):
        offsets[number] = len(out)
        out += f"{number} 0 obj\n".encode() + objects[number] + b"\nendobj\n"

    xref_offset = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for number in sorted(objects):
        out += f"{offsets[number]:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode()
    return bytes(out)


PAGES = [
    [
        "Gewährleistung und Garantie",
        "Die Gewährleistung für Neufahrzeuge beträgt 24 Monate ab Auslieferung.",
        "Die Garantie beginnt mit der Übergabe des Fahrzeugs an den Endkunden.",
        "30.04.2025 Volkswagen AG Nachdruck nicht gestattet intern 1",
    ],
    [
        "Mobilitätsgarantie",
        "Die Mobilitätsgarantie umfasst Pannenhilfe und Abschleppdienst europaweit.",
        "Sie gilt für alle Fahrzeuge innerhalb der Garantiezeit.",
        "30.04.2025 Volkswagen AG Nachdruck nicht gestattet intern 2",
    ],
]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def document(client: TestClient) -> dict:
    response = client.post(
        "/documents",
        files={"file": ("garantie.pdf", make_pdf(PAGES), "application/pdf")},
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_upload_reports_pages_and_chunks(document: dict) -> None:
    assert document["filename"] == "garantie.pdf"
    assert document["page_count"] == 2
    assert document["chunk_count"] > 0


def test_search_returns_most_relevant_chunk_first(
    client: TestClient, document: dict
) -> None:
    response = client.post(
        "/search",
        json={"query": "Wie lange dauert die Gewährleistung für Neuwagen?", "top_k": 2},
    )
    assert response.status_code == 200, response.text
    results = response.json()["results"]
    assert results, "expected at least one hit"
    top = results[0]
    assert top["page"] == 1
    assert "24 Monate" in top["text"]
    assert top["document_id"] == document["document_id"]
    assert 0.0 <= top["score"] <= 1.0 + 1e-6


def test_search_can_be_scoped_to_document(client: TestClient, document: dict) -> None:
    response = client.post(
        "/search",
        json={
            "query": "Pannenhilfe",
            "top_k": 1,
            "document_id": document["document_id"],
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["results"][0]["page"] == 2


def test_search_rejects_blank_query(client: TestClient) -> None:
    response = client.post("/search", json={"query": "   "})
    assert response.status_code == 422


def test_unknown_document_returns_404(client: TestClient) -> None:
    response = client.post(
        "/search", json={"query": "Garantie", "document_id": "does-not-exist"}
    )
    assert response.status_code == 404


def test_empty_query_is_rejected(client: TestClient) -> None:
    response = client.post("/search", json={"query": ""})
    assert response.status_code == 422


def test_non_pdf_upload_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/documents", files={"file": ("notes.txt", b"hello", "text/plain")}
    )
    assert response.status_code == 415


def test_fake_pdf_upload_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/documents",
        files={"file": ("fake.pdf", b"not a pdf at all", "application/pdf")},
    )
    assert response.status_code == 415


# Optional answer layer.


def test_ask_returns_answer_with_sources(
    client: TestClient, document: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        answer_module,
        "_generate_answer",
        lambda query, hits: "Die Gewährleistung beträgt 24 Monate (Seite 1).",
    )
    response = client.post(
        "/ask", json={"query": "Wie lange dauert die Gewährleistung?", "top_k": 2}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["answer"] == "Die Gewährleistung beträgt 24 Monate (Seite 1)."
    assert body["sources"][0]["page"] == 1
    assert body["sources"][0]["document_id"] == document["document_id"]


# Browser UI, an optional layer over the same retrieval functions.


def test_ui_index_is_served(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "PDF RAG" in response.text
    assert "Upload a PDF to get started" in response.text
    assert 'name="mode" value="search"' in response.text
    assert "Search returns matching passages" in response.text
    assert "downloads about 2.4 GB" in response.text
    assert response.headers["cache-control"] == "no-store"


def test_health_reports_models_and_index(client: TestClient, document: dict) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["embedding_model"] == "intfloat/multilingual-e5-small"
    assert body["ask_model"] == "LiquidAI/LFM2-1.2B-RAG"
    assert body["ask_model_loaded"] is False
    assert body["document_count"] >= 1


def test_ui_delete_removes_document(client: TestClient) -> None:
    created = client.post(
        "/documents",
        files={
            "file": ("removable.pdf", make_pdf([["Removable page"]]), "application/pdf")
        },
    )
    assert created.status_code == 201, created.text
    document_id = created.json()["document_id"]

    response = client.delete(f"/ui/documents/{document_id}")
    assert response.status_code == 200
    assert "removable.pdf" not in response.text
    assert 'data-count="' in response.text

    search = client.post(
        "/search", json={"query": "Removable", "document_id": document_id}
    )
    assert search.status_code == 404

    missing = client.delete(f"/ui/documents/{document_id}")
    assert missing.status_code == 404
    assert "not found" in missing.text


def test_ui_search_returns_html(client: TestClient, document: dict) -> None:
    response = client.post(
        "/ui/query", data={"query": "Gewährleistung Neuwagen", "mode": "search"}
    )
    assert response.status_code == 200
    assert "24 Monate" in response.text


def test_ui_ask_returns_answer(
    client: TestClient, document: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        answer_module, "_generate_answer", lambda query, hits: "24 Monate (p. 1)."
    )
    response = client.post(
        "/ui/query", data={"query": "Wie lange gilt die Gewährleistung?", "mode": "ask"}
    )
    assert response.status_code == 200
    assert "24 Monate (p. 1)." in response.text
    assert "Sources" in response.text


def test_ui_query_validates_input(client: TestClient) -> None:
    response = client.post("/ui/query", data={"query": "x", "mode": "bogus"})
    assert response.status_code == 422
    assert "mode must be" in response.text

    response = client.post("/ui/query", data={"query": "x"})
    assert response.status_code == 422
    assert "mode must be" in response.text

    response = client.post("/ui/query", data={"query": ""})
    assert response.status_code == 422
    assert "query is required" in response.text

    response = client.post("/ui/query", data={"query": "x" * 2001, "mode": "search"})
    assert response.status_code == 422
    assert "query exceeds" in response.text


def test_ui_rejects_cross_origin(client: TestClient) -> None:
    response = client.post(
        "/ui/query",
        data={"query": "x", "mode": "search"},
        headers={"origin": "https://evil.example"},
    )
    assert response.status_code == 403
    assert "cross-origin" in response.text


def test_ui_upload_error_is_html(client: TestClient) -> None:
    response = client.post(
        "/ui/upload", files={"files": ("notes.txt", b"hello", "text/plain")}
    )
    assert response.status_code == 422
    assert "only .pdf" in response.text


def test_ui_upload_handles_multiple_files(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []

    def fake_upload(file: UploadFile) -> app_module.DocumentInfo:
        seen.append(file.filename or "")
        if file.filename == "bad.pdf":
            raise HTTPException(status_code=415, detail="not a PDF")
        if file.filename == "crash.pdf":
            raise ValueError("boom")
        return app_module.DocumentInfo(
            document_id="fake",
            filename=file.filename or "",
            page_count=1,
            chunk_count=2,
        )

    monkeypatch.setattr(app_module, "upload_document", fake_upload)
    response = client.post(
        "/ui/upload",
        files=[
            ("files", ("a.pdf", b"%PDF-1.4", "application/pdf")),
            ("files", ("bad.pdf", b"x", "application/pdf")),
            ("files", ("crash.pdf", b"x", "application/pdf")),
        ],
    )
    assert response.status_code == 200
    assert seen == ["a.pdf", "bad.pdf", "crash.pdf"]
    assert "Indexed 1 document" in response.text
    assert "bad.pdf: not a PDF" in response.text
    assert "crash.pdf: ValueError: boom" in response.text


def test_running_headers_are_stripped() -> None:
    long_line = "Ein langer wiederholter Absatz " + "x" * 200
    pages = [
        "\n".join(
            [
                f"Abschnitt {word}",
                long_line,
                "Wert 10" if number <= 3 else "Anderer Eintrag",
                f"30.04.2025 Volkswagen AG intern {number}",
            ]
        )
        for number, word in enumerate(
            ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"], start=1
        )
    ]
    cleaned = _strip_running_headers(pages)
    # The footer appears on all five pages and is removed.
    assert all("Volkswagen AG" not in page for page in cleaned)
    # A long repeated line exceeds the length limit and is kept.
    assert all(long_line in page for page in cleaned)
    # "Wert 10" appears on three of five pages (60%) and is kept.
    assert all("Wert 10" in page for page in cleaned[:3])
    assert all(
        word in page
        for page, word in zip(
            cleaned,
            ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"],
            strict=True,
        )
    )


def test_short_page_is_indexed() -> None:
    page_count, chunks = extract_chunks(make_pdf([["Kurzer Text"]]))
    assert page_count == 1
    assert len(chunks) == 1
    assert chunks[0].text == "Kurzer Text"


def test_page_without_text_is_rejected() -> None:
    with pytest.raises(HTTPException):
        extract_chunks(make_pdf([["42"]]))


def test_hard_split_avoids_tiny_tail() -> None:
    parts = _hard_split("x" * 1001, 1000)
    assert "".join(parts) == "x" * 1001
    assert max(map(len, parts)) <= 1000
    assert min(map(len, parts)) >= 500


def test_answer_model_warm_up_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loads: list[int] = []

    def fake_load() -> tuple[str, str]:
        loads.append(1)
        return ("tokenizer", "model")

    monkeypatch.setattr(answer_module, "_llm", None)
    monkeypatch.setattr(answer_module, "_load_llm", fake_load)

    answer_module.warm_up()
    answer_module.warm_up()

    assert len(loads) == 1
    assert answer_module.is_loaded() is True
