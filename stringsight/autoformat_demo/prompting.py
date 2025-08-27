from typing import Any


SYSTEM_PROMPT = (
    "You are an expert data wrangler. Write small, deterministic Python functions\n"
    "to transform heterogeneous agent trace records into a canonical OpenAI\n"
    "messages list. Do not add commentary; only return the requested code."
)


def build_parser_prompt(sample_json_str: str, canonical_spec_md: str) -> str:
    """Compose a single-string prompt for parser synthesis.

    The model must output ONLY a python code block with a single function:
    def parse_to_oai(input_obj):
        ...
        return messages_list

    Requirements for the function:
      - No imports, no file/network I/O, deterministic
      - Accepts any JSON-like object (dict/list/str)
      - Returns a list of dicts with keys 'role' and 'content'
      - If 'messages' exists and is a list of message dicts, convert/normalize\n"
      "        them to role+content pairs and return
      - If 'fncall_messages' exists, append their role+content
      - If a final assistant message exists at response.choices[0].message,\n"
      "        append it
      - Preserve strings as-is; if content is a dict, keep it as dict
      - Do not fabricate tool_calls; only pass through if present on input
      - Handle None/missing fields gracefully by skipping them
    """

    return (
        "Desired canonical format (summary):\n" +
        canonical_spec_md +
        "\n\nCurrent input example (JSON):\n" +
        "```json\n" + sample_json_str.strip() + "\n```\n\n" +
        "Write a minimal Python function named parse_to_oai(input_obj) that implements\n"
        "the transformation described above.\n"
        "Rules:\n"
        "- No imports.\n"
        "- No I/O.\n"
        "- Deterministic.\n"
        "- Return value must be a list of message dicts with 'role' and 'content'.\n\n"
        "Return ONLY the python code block with the function definition."
    )


