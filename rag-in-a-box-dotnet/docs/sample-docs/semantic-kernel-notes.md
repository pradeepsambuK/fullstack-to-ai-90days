# Semantic Kernel notes

Semantic Kernel is Microsoft's open source SDK for building AI agents and
applications in .NET, Python, and Java. The .NET version is the natural fit
for teams already running ASP.NET Core services, because the same dependency
injection, configuration, and hosting patterns apply.

The two services this starter uses are chat completion and text embedding
generation. Chat completion takes a conversation history and returns the
model's reply. Text embedding generation takes a string and returns a vector
of floats. Both are available through Azure OpenAI and through OpenAI's API,
and Semantic Kernel exposes them behind the same interfaces, so switching
providers is a configuration change, not a code change.

A kernel is the container that holds the AI services and any plugins you add.
You build it once at startup with Kernel.CreateBuilder, add your chat and
embedding services, and resolve IChatCompletionService and
ITextEmbeddingGenerationService where you need them.

Plugins are plain C# classes whose methods are marked with
[KernelFunction]. The model can call them as tools. This starter does not
need plugins, because the RAG loop is ordinary code: chunk, embed, retrieve,
then call chat completion with the retrieved context. Keeping that loop
visible is the point. When you outgrow it, plugins and planners are the
next step, not the first one.
