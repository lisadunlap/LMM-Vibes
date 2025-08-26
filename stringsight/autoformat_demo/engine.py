"""Engine stubs for auto-format demo.

These functions are simple placeholders to enable an isolated demo UI.
They will later be replaced by real LLM-based synthesis and sandboxed
execution.
"""

from typing import Any, Dict, List, Tuple


def synthesize_parser_stub(sample: Dict[str, Any]) -> str:
    """Return a human-readable parser description (stub).

    For now, we detect common wrappers and describe a trivial transform to
    get to the OpenAI-like {"messages": [...] } shape or just pass-through
    if already present.
    """
    if isinstance(sample, dict) and "messages" in sample:
        return "Parser: input already in OpenAI chat format; emit sample['messages'] as-is."
    return (
        "Parser: if input contains a top-level 'messages' list, use it; "
        "else wrap the entire object as a single assistant message with stringified content."
    )


def apply_parser_safely_stub(parser_text: str, obj: Any) -> Any:
    """Apply the stub parser without any code execution.

    - If obj has 'messages' key and is list-like, return obj['messages']
    - Else return a single-message list in assistant role with str(obj) content
    """
    if isinstance(obj, dict) and "messages" in obj and isinstance(obj["messages"], list):
        return obj["messages"]
    # Fallback: single assistant message with string content
    return [
        {
            "role": "assistant",
            "content": str(obj),
        }
    ]


def validate_oai_conversation(conv: Any) -> Tuple[bool, List[str]]:
    """Minimal validator for OpenAI-style messages list.

    Rules:
      - conv must be a list
      - each item must be a dict with 'role' (str) and 'content' (str or dict)
      - if content is dict and has 'tool_calls', it must be a list
    """
    errors: List[str] = []
    if not isinstance(conv, list):
        return False, ["Conversation must be a list"]
    for i, msg in enumerate(conv):
        if not isinstance(msg, dict):
            errors.append(f"Message {i} must be a dict")
            continue
        role = msg.get("role")
        content = msg.get("content")
        if not isinstance(role, str):
            errors.append(f"Message {i}: role must be a string")
        if not (isinstance(content, str) or isinstance(content, dict)):
            errors.append(f"Message {i}: content must be str or dict")
        if isinstance(content, dict):
            tc = content.get("tool_calls")
            if tc is not None and not isinstance(tc, list):
                errors.append(f"Message {i}: content.tool_calls must be a list if present")
    return (len(errors) == 0), errors


