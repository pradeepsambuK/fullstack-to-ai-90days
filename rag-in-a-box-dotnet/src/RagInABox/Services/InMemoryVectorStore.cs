namespace RagInABox.Services;

public sealed record StoredChunk(string Id, string DocumentTitle, string Text, float[] Embedding);

public sealed record SearchHit(string Id, string DocumentTitle, string Text, double Score);

/// <summary>
/// The whole "vector database" for this starter: a list in memory plus
/// cosine similarity. No server to install, nothing to configure, and the
/// retrieval step is visible code instead of a black box. When the collection
/// outgrows memory, swap this class for Azure AI Search or Qdrant and keep
/// the rest of the app unchanged.
/// </summary>
public sealed class InMemoryVectorStore
{
    private readonly List<StoredChunk> _chunks = new();
    private readonly object _lock = new();

    public void Upsert(StoredChunk chunk)
    {
        lock (_lock)
        {
            _chunks.RemoveAll(c => c.Id == chunk.Id);
            _chunks.Add(chunk);
        }
    }

    public IReadOnlyList<SearchHit> Search(ReadOnlyMemory<float> query, int topK)
    {
        var queryVector = query.ToArray();
        List<SearchHit> scored;

        lock (_lock)
        {
            scored = new List<SearchHit>(_chunks.Count);
            foreach (var chunk in _chunks)
            {
                scored.Add(new SearchHit(chunk.Id, chunk.DocumentTitle, chunk.Text, Cosine(queryVector, chunk.Embedding)));
            }
        }

        return scored
            .Where(s => s.Score > 0)
            .OrderByDescending(s => s.Score)
            .Take(Math.Max(1, topK))
            .ToList();
    }

    public object ListDocuments()
    {
        lock (_lock)
        {
            return _chunks
                .GroupBy(c => c.DocumentTitle)
                .Select(g => new { title = g.Key, chunks = g.Count() })
                .ToList();
        }
    }

    public void Clear()
    {
        lock (_lock)
        {
            _chunks.Clear();
        }
    }

    private static double Cosine(float[] a, float[] b)
    {
        if (a.Length != b.Length)
        {
            return 0;
        }

        double dot = 0, normA = 0, normB = 0;

        for (var i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }

        if (normA == 0 || normB == 0)
        {
            return 0;
        }

        return dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
    }
}
