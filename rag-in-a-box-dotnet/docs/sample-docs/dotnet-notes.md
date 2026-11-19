# .NET notes

ASP.NET Core minimal APIs let you define an HTTP endpoint in a few lines,
which is why this starter uses them. A full controller setup would add files
without adding anything the demo needs. The endpoints here are plain
functions: ingest text, upload files, list documents, ask a question.

Configuration in .NET comes from appsettings.json, environment variables,
and other providers layered together. Environment variables override the
JSON file, which is how secrets stay out of the repository. A setting like
Ai:Azure:Endpoint can be set with the environment variable
Ai__Azure__Endpoint, using double underscores for the colons.

The vector store in this starter is a list in memory with cosine
similarity. It holds a few thousand chunks comfortably and needs no server.
For production, swap it for Azure AI Search or Qdrant and keep the rest of
the app unchanged. The RagService only depends on the store through its
Upsert and Search methods, so the swap is one class.

The project targets .NET 8, the current long term support release. It runs
with dotnet run, or in a container with the included Dockerfile. The API
serves a small demo page at the root URL, so you can try the whole loop
from a browser without writing a client.
