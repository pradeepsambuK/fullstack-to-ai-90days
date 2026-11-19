namespace RagInABox.Services;

public sealed record Citation(int Index, string DocumentTitle, double Score, string Snippet);

public sealed record AskResponse(string Answer, Citation[] Citations);

/// <summary>
/// The RAG loop, in order: chunk the document, embed each chunk, store it,
/// then at ask time embed the question, retrieve the closest chunks, and let
/// the model answer using only that retrieved context, with citations.
/// </summary>
public sealed class RagService
{
    private readonly AiEngine _ai;
    private readonly InMemoryVectorStore _store;
    private readonly int _chunkSize;
    private readonly int _overlap;
    private readonly int _defaultTopK;

    public RagService(AiEngine ai, InMemoryVectorStore store, IConfiguration config)
    {
        _ai = ai;
        _store = store;
        _chunkSize = config.GetValue("Rag:ChunkSize", 1000);
        _overlap = config.GetValue("Rag:Overlap", 150);
        _defaultTopK = config.GetValue("Rag:TopK", 5);
    }

    public async Task<object> IngestAsync(string title, string text)
    {
        var chunks = Chunker.Chunk(text, _chunkSize, _overlap);
        var documentId = Guid.NewGuid().ToString("N");

        for (var i = 0; i < chunks.Count; i++)
        {
            var embedding = await _ai.EmbedAsync(chunks[i]);
            _store.Upsert(new StoredChunk($"{documentId}-{i}", title, chunks[i], embedding.ToArray()));
        }

        return new { documentId, title, chunks = chunks.Count };
    }

    public async Task<AskResponse> AskAsync(string question, int? topK)
    {
        var queryEmbedding = await _ai.EmbedAsync(question);
        var hits = _store.Search(queryEmbedding, topK ?? _defaultTopK);

        if (hits.Count == 0)
        {
            return new AskResponse(
                "I don't know. Nothing in your documents covers this.",
                Array.Empty<Citation>());
        }

        var context = string.Join("\n\n", hits.Select((hit, i) =>
            $"[{i + 1}] ({hit.DocumentTitle})\n{hit.Text}"));

        const string systemPrompt =
            "You answer questions using ONLY the context below. " +
            "If the answer is not in the context, say you don't know. " +
            "Add a citation like [1] after each fact you use.";

        var userMessage = $"Context:\n{context}\n\nQuestion: {question}";
        var answer = await _ai.ChatAsync(systemPrompt, userMessage);

        var citations = hits.Select((hit, i) => new Citation(
            i + 1,
            hit.DocumentTitle,
            Math.Round(hit.Score, 3),
            hit.Text.Length > 200 ? hit.Text[..200] + "..." : hit.Text)).ToArray();

        return new AskResponse(answer, citations);
    }
}
