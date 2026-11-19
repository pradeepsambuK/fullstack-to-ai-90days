using Microsoft.Extensions.AI;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;

namespace RagInABox.Services;

/// <summary>
/// Builds the Semantic Kernel client lazily, only when an AI call is made.
/// That way /health and the demo page work without any API key configured,
/// and a missing key produces a clear error instead of a crash at startup.
/// </summary>
public sealed class AiEngine
{
    private readonly IConfiguration _config;
    private readonly Lazy<Kernel> _kernel;

    public AiEngine(IConfiguration config)
    {
        _config = config;
        _kernel = new Lazy<Kernel>(BuildKernel);
    }

    public string Provider => _config["Ai:Provider"] ?? "azure";

    public bool IsConfigured
    {
        get
        {
            if (Provider.Equals("openai", StringComparison.OrdinalIgnoreCase))
            {
                return !string.IsNullOrWhiteSpace(OpenAiKey);
            }

            return !string.IsNullOrWhiteSpace(AzureEndpoint)
                && !string.IsNullOrWhiteSpace(AzureKey)
                && !string.IsNullOrWhiteSpace(AzureChatDeployment)
                && !string.IsNullOrWhiteSpace(AzureEmbeddingDeployment);
        }
    }

    private string? AzureEndpoint =>
        _config["Ai:Azure:Endpoint"] ?? Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT");

    private string? AzureKey =>
        _config["Ai:Azure:ApiKey"] ?? Environment.GetEnvironmentVariable("AZURE_OPENAI_API_KEY");

    private string? AzureChatDeployment =>
        _config["Ai:Azure:ChatDeployment"] ?? Environment.GetEnvironmentVariable("AZURE_OPENAI_CHAT_DEPLOYMENT");

    private string? AzureEmbeddingDeployment =>
        _config["Ai:Azure:EmbeddingDeployment"] ?? Environment.GetEnvironmentVariable("AZURE_OPENAI_EMBEDDING_DEPLOYMENT");

    private string? OpenAiKey =>
        _config["Ai:OpenAI:ApiKey"] ?? Environment.GetEnvironmentVariable("OPENAI_API_KEY");

    private string OpenAiChatModel =>
        _config["Ai:OpenAI:ChatModel"] ?? "gpt-4o-mini";

    private string OpenAiEmbeddingModel =>
        _config["Ai:OpenAI:EmbeddingModel"] ?? "text-embedding-3-small";

    private Kernel BuildKernel()
    {
        var kernelBuilder = Kernel.CreateBuilder();

        if (Provider.Equals("openai", StringComparison.OrdinalIgnoreCase))
        {
            if (string.IsNullOrWhiteSpace(OpenAiKey))
            {
                throw new InvalidOperationException(
                    "OpenAI is not configured. Set OPENAI_API_KEY (or Ai:OpenAI:ApiKey).");
            }

            kernelBuilder.AddOpenAIChatCompletion(OpenAiChatModel, OpenAiKey);
            kernelBuilder.AddOpenAIEmbeddingGenerator(OpenAiEmbeddingModel, OpenAiKey);
        }
        else
        {
            if (!IsConfigured)
            {
                throw new InvalidOperationException(
                    "Azure OpenAI is not configured. Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, " +
                    "AZURE_OPENAI_CHAT_DEPLOYMENT and AZURE_OPENAI_EMBEDDING_DEPLOYMENT.");
            }

            kernelBuilder.AddAzureOpenAIChatCompletion(AzureChatDeployment!, AzureEndpoint!, AzureKey!);
            kernelBuilder.AddAzureOpenAIEmbeddingGenerator(AzureEmbeddingDeployment!, AzureEndpoint!, AzureKey!);
        }

        return kernelBuilder.Build();
    }

    public async Task<ReadOnlyMemory<float>> EmbedAsync(string text)
    {
        var generator = _kernel.Value.GetRequiredService<IEmbeddingGenerator<string, Embedding<float>>>();
        return await generator.GenerateVectorAsync(text);
    }

    public async Task<string> ChatAsync(string systemPrompt, string userMessage)
    {
        var chat = _kernel.Value.GetRequiredService<IChatCompletionService>();
        var history = new ChatHistory(systemPrompt);
        history.AddUserMessage(userMessage);

        var result = await chat.GetChatMessageContentAsync(history);
        return result.Content ?? string.Empty;
    }
}
