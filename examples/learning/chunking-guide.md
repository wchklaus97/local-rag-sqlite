# Chunking guide

Chunking is the decision about what one retrievable unit looks like. It happens in
your ingest pipeline, before embedding. An embedding API accepts a string and returns
a vector; it has no chunk size or overlap setting and it will not split a long document
for you. If a string is longer than the model's token limit it is usually truncated or
rejected, so text beyond the limit silently never gets searched.

## Chunk size

Small chunks (200 to 400 characters) give precise matches but lose surrounding context.
Large chunks (1500 characters or more) keep context but blur several topics into one
vector, so the match gets vague and the prompt fills up with irrelevant text.
A few hundred to about a thousand characters is a common starting point.

## Chunk overlap

Overlap repeats the last part of one chunk at the start of the next. If a sentence that
answers a question is cut exactly at a boundary, overlap makes sure it still appears
whole in at least one chunk. Ten to twenty percent of the chunk size is typical. Too much
overlap wastes storage and returns near-duplicate results.

## Boundaries

Prefer to cut at paragraph breaks, then line breaks, then sentence ends, then spaces.
Cutting in the middle of a word or a table row hurts retrieval quality.
