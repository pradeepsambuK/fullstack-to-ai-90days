# RAG in a Box for .NET

Drop in your docs, ask questions, get answers with sources. An ASP.NET Core
API plus Semantic Kernel, with an in-memory vector store. No database, no
search server, no infra to stand up. Built to be read: the whole RAG loop is
visible code, not hidden behind a framework call.

## How it works

1. **Chunk** — documents are split into passages of about 1000 characters,
   with a small overlap so ideas split across a boundary survive.
2. **Embed** — each chunk becomes a vector with an embedding model
   (Azure OpenAI or OpenAI).
3. **Retrieve** — your question is embedded the same way, and the closest
   chunks are found with cosine similarity.
4. **Answer** — the chat model answers using only the retrieved context,
   and says it doesn't know when the answer isn't there. Every fact comes
   back with a citation.

## Run it (about 5 minutes)

Prereqs: .NET 8 SDK, and an Azure OpenAI or OpenAI API key.

```bash
# Azure OpenAI (default)
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-key"
export AZURE_OPENAI_CHAT_DEPLOYMENT="gpt-4o-mini"
export AZURE_OPENAI_EMBEDDING_DEPLOYMENT="text-embedding-3-small"

dotnet run --project src/RagInABox
```

Open http://localhost:5000. Upload the sample docs from `docs/sample-docs`,
then ask "What is RAG?" or "What does Semantic Kernel do?"

Prefer OpenAI instead? Set `"Provider": "openai"` in
`src/RagInABox/appsettings.json` and export `OPENAI_API_KEY`.

With Docker:

```bash
docker build -t rag-in-a-box .
docker run -p 8080:8080 \
  -e AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/" \
  -e AZURE_OPENAI_API_KEY="your-key" \
  -e AZURE_OPENAI_CHAT_DEPLOYMENT="gpt-4o-mini" \
  -e AZURE_OPENAI_EMBEDDING_DEPLOYMENT="text-embedding-3-small" \
  rag-in-a-box
```

## API

| Method | Route | What it does |
| ------ | ----- | ------------ |
| GET | `/health` | Status and whether AI is configured. Works without a key. |
| POST | `/ingest-text` | `{"title": "...", "text": "..."}` — chunk, embed, store. |
| POST | `/ingest` | Multipart upload of `.md` / `.txt` files. |
| GET | `/docs` | Documents currently in the store, with chunk counts. |
| DELETE | `/docs` | Clear the store. |
| POST | `/ask` | `{"question": "...", "topK": 5}` — answer plus citations. |

Asking without a key configured returns a 503 with a plain message telling
you which variables to set.

## Project layout

```
src/RagInABox/
  Program.cs                 Minimal API endpoints
  Services/AiEngine.cs       Semantic Kernel setup (lazy, key errors are clear)
  Services/Chunker.cs        Sentence-aware chunking with overlap
  Services/InMemoryVectorStore.cs   The "vector DB": a list plus cosine similarity
  Services/RagService.cs     The RAG loop: ingest and ask
  wwwroot/index.html         Demo page: upload docs, ask questions
docs/sample-docs/            Three docs so the demo works out of the box
```

## Make it yours

- **Bigger collections:** swap `InMemoryVectorStore` for Azure AI Search or
  Qdrant. `RagService` only uses its `Upsert` and `Search` methods, so the
  swap is one class.
- **Better retrieval:** try a larger chunk size for long prose, a smaller one
  for dense docs, or add a re-ranking pass over the top hits.
- **Streaming answers:** `IChatCompletionService` has a streaming overload;
  wire it to a `text/event-stream` endpoint.
- **Persistent docs:** the in-memory store resets on restart by design. Add a
  startup step that ingests a folder so the demo is ready on boot.

## License

MIT. Use it, fork it, break it.
