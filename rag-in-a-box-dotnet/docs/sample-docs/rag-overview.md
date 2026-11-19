# What RAG is

Retrieval-Augmented Generation (RAG) is a pattern for letting a language model
answer questions about documents it was never trained on. Instead of trusting
the model to know everything, you give it the relevant passages at ask time
and let it answer from those.

The loop has four steps. First, chunking: split each document into small
passages, usually a few hundred to a thousand characters each, with a small
overlap so an idea split across a boundary is not lost. Second, embedding:
turn every chunk into a vector, a list of numbers that captures what the text
means, using an embedding model such as text-embedding-3-small. Third,
retrieval: embed the user's question the same way and find the chunks whose
vectors are closest to it, measured with cosine similarity. Fourth,
generation: put those chunks into the prompt and ask the chat model to answer
using only that context.

The most common beginner mistake is skipping the retrieval quality work and
blaming the chat model. If the chunks are bad, the answers are bad, no matter
which model you use. Good chunking and good retrieval beat a bigger model
almost every time.

RAG fits anywhere the answers live in your own documents: internal wikis,
support articles, product manuals, policy documents, and codebases. It does
not need training data or fine tuning, and you can add or remove documents
without retraining anything.
