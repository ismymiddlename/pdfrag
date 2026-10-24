# PDF RAG

A local FastAPI service for PDF retrieval. It uploads and processes PDF
documents, then answers natural-language queries by semantic search over their
text.

Upload and search are the core service. Question answering and the browser UI
are optional layers built on the same retrieval pipeline; the core does not
depend on them.

## Project layout

```
app.py              core API, PDF extraction, chunking, embeddings, search,
                    optional answers
ui/                 browser UI: index.html, styles.css, vendored htmx
tests/test_app.py   end-to-end and unit tests
requirements.txt    pinned dependencies
```

## Setup

Python 3.12 and about 5 GB of free disk space for the embedding model and
PyTorch.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

The service listens on <http://127.0.0.1:8000>. OpenAPI documentation is
available at <http://127.0.0.1:8000/docs>.

The embedding model `intfloat/multilingual-e5-small` downloads on first use and
is cached by Hugging Face. The optional answer model
`LiquidAI/LFM2-1.2B-RAG` is downloaded on the first `/ask` request.

## API

### Upload and process: `POST /documents`

Accepts one PDF (`.pdf`, 25 MB limit), extracts its text, splits it into chunks,
embeds the chunks, and adds them to the in-memory index.

```bash
curl -X POST http://127.0.0.1:8000/documents \
  -F "file=@path/to/document.pdf;type=application/pdf"
```

`201 Created`:

```json
{"document_id":"9b2c8f6a4d1e4c3b8a7f0e5d2c1b6a93","filename":"document.pdf","page_count":12,"chunk_count":34}
```

### Semantic search: `POST /search`

Embeds the query and returns the highest-scoring chunks by cosine similarity.

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "content-type: application/json" \
  -d '{"query":"your question","top_k":5}'
```

Request fields: `query` (required, non-blank, at most 2000 characters),
`top_k` (1–20, default 5), and `document_id` (optional; restricts search to one
document).

`200 OK`:

```json
{"query":"your question","results":[{"document_id":"9b2c8f6a4d1e4c3b8a7f0e5d2c1b6a93","filename":"document.pdf","page":3,"chunk_index":7,"text":"...","score":0.89}]}
```

### Answer generation: `POST /ask` (optional)

Retrieves the top chunks and generates an answer from them with LFM2. Requires
the optional answer model.

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "content-type: application/json" \
  -d '{"query":"your question","top_k":4}'
```

`200 OK`:

```json
{"query":"your question","answer":"...","sources":[{"document_id":"9b2c8f6a4d1e4c3b8a7f0e5d2c1b6a93","filename":"document.pdf","page":3,"chunk_index":7,"text":"...","score":0.89}]}
```

### Other endpoints

- `GET /documents`: indexed documents with page and chunk counts.
- `GET /health`: model names and document count.

## Browser UI (optional)

<http://127.0.0.1:8000/> serves a two-pane interface over the same retrieval
functions. The left pane uploads PDFs and lists the indexed documents, each with
a Remove button. The right pane searches or asks questions. Before the first
document is indexed, the query controls are hidden and an upload prompt is
displayed.

The UI posts to `/ui/upload`, `/ui/query`, and `/ui/documents/{id}`, which
return HTML fragments. htmx is vendored in `ui/`; the page makes no third-party
network requests.

## How it works

1. Text is extracted page by page with `pypdf`.
2. Short lines repeated on at least 80% of pages (running headers and footers)
   are dropped. Digits are masked when comparing, so page numbers still match.
3. Sentences are packed into chunks of about 1000 characters with one sentence
   of overlap; each chunk keeps its page number.
4. Chunks are embedded with the E5 `passage: ` prefix and queries with
   `query: `, then L2-normalized.
5. Search uses cosine similarity (a dot product on normalized vectors); the
   top-k chunks across all documents, or one selected document, are returned.
6. For `/ask`, the top-k chunks are placed in the LFM2 prompt with page
   markers, and a greedy answer is generated, instructed to use only that
   context.

## Tests

```bash
pytest
```

The suite builds small PDFs in memory, uploads them through the API, and checks
retrieval, validation, and the UI routes. The `/ask` tests replace generation
with a stub, so the answer model is not required.

## Limitations

- The index is held in memory; restarting the server clears uploaded documents.
- PDFs without a text layer (scans) are rejected. There is no OCR.
- Indexing is synchronous during upload, so large PDFs delay the response until
  processing finishes.
- Answer quality depends on retrieval quality; the returned sources allow each
  answer to be checked against the retrieved passages.
- Generation with the 1.2B model takes several seconds on CPU or Apple MPS.
  The model is loaded once and reused.
- Licenses: E5 is MIT, LFM2 uses the LFM Open License v1.0.
