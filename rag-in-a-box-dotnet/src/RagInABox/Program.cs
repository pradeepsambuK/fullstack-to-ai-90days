using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Embeddings;
using RagInABox.Services;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddSingleton<InMemoryVectorStore>();
builder.Services.AddSingleton<AiEngine>();
builder.Services.AddSingleton<RagService>();

var app = builder.Build();

app.UseDefaultFiles();
app.UseStaticFiles();

app.MapGet("/health", (AiEngine ai) => Results.Ok(new
{
    status = "ok",
    aiConfigured = ai.IsConfigured,
    provider = ai.Provider
}));

app.MapPost("/ingest-text", async (IngestTextRequest req, RagService rag) =>
{
    if (string.IsNullOrWhiteSpace(req.Text))
    {
        return Results.BadRequest(new { error = "Text is required." });
    }

    var result = await rag.IngestAsync(req.Title ?? "untitled", req.Text);
    return Results.Ok(result);
});

app.MapPost("/ingest", async (HttpRequest request, RagService rag) =>
{
    if (!request.HasFormContentType)
    {
        return Results.BadRequest(new { error = "Upload files as multipart form data." });
    }

    var form = await request.ReadFormAsync();
    var ingested = new List<object>();

    foreach (var file in form.Files)
    {
        if (file.Length == 0)
        {
            continue;
        }

        using var reader = new StreamReader(file.OpenReadStream());
        var text = await reader.ReadToEndAsync();
        ingested.Add(await rag.IngestAsync(file.FileName, text));
    }

    return Results.Ok(new { ingested });
});

app.MapGet("/docs", (InMemoryVectorStore store) => Results.Ok(store.ListDocuments()));

app.MapDelete("/docs", (InMemoryVectorStore store) =>
{
    store.Clear();
    return Results.Ok(new { cleared = true });
});

app.MapPost("/ask", async (AskRequest req, RagService rag) =>
{
    if (string.IsNullOrWhiteSpace(req.Question))
    {
        return Results.BadRequest(new { error = "Question is required." });
    }

    try
    {
        var answer = await rag.AskAsync(req.Question, req.TopK ?? 5);
        return Results.Ok(answer);
    }
    catch (InvalidOperationException ex)
    {
        return Results.Problem(ex.Message, statusCode: 503);
    }
});

app.Run();

public record IngestTextRequest(string? Title, string Text);

public record AskRequest(string Question, int? TopK);
