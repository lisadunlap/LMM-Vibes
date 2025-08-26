CANONICAL_SPEC_MD = """
### Canonical target format (OpenAI-style messages)

- A conversation is a list of message objects
- Each message has:
  - `role`: one of `user`, `assistant`, `system`, `tool`
  - `content`: string OR object
    - If object, may include keys like `text`, `image`, or `tool_calls`
    - If `tool_calls` exists, it must be a list of objects with at least `name` and `arguments`
- Extra keys should be preserved; do not discard structured content

Single-model normalization result: `[ {"role": ..., "content": ...}, ... ]`

Side-by-side use cases can be represented externally as two separate normalized lists.
"""


