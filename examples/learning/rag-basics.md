# RAG basics

Retrieval-Augmented Generation (RAG) answers questions by first looking up relevant
passages in your own documents and then giving those passages to a language model.
The model writes the answer from the evidence instead of from memory alone, which
reduces made-up facts and lets you cite sources.

## The six stages

1. **Ingest** reads files (PDF, Markdown) into plain text and remembers where each piece came from.
2. **Chunking** cuts the text into passages small enough to embed and to fit in a prompt.
3. **Embedding** turns each chunk into a vector, a list of numbers that places similar meanings close together.
4. **Vector store** keeps the chunks and vectors so they can be searched quickly. This demo uses a single SQLite file with the sqlite-vec extension.
5. **Retrieval** embeds the question and returns the k nearest chunks.
6. **Generation** sends the question plus the retrieved chunks to an LLM with instructions to answer only from that context.

## Why a vector database is not required

For thousands of chunks, a SQLite file is enough. The sqlite-vec extension adds a
`vec0` virtual table that supports nearest-neighbour search with cosine distance, so
the whole index is one portable file you can copy, back up or delete.
