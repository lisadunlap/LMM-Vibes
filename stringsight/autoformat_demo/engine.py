"""Engine stubs for auto-format demo.

These functions are simple placeholders to enable an isolated demo UI.
They will later be replaced by real LLM-based synthesis and sandboxed
execution.
"""

from typing import Any, Dict, List, Tuple, Optional
import json
import re


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

    Incorporates all available conversation info commonly found in agent traces:
      - Top-level 'messages' (preferred if present)
      - 'fncall_messages' (additional system/user context)
      - Final assistant reply from 'response.choices[0].message'

    Returns a single OpenAI-style messages list.
    """
    if not isinstance(obj, dict):
        return [{"role": "assistant", "content": str(obj)}]

    result: List[Dict[str, Any]] = []

    # 1) Base messages
    base_msgs = obj.get("messages")
    if isinstance(base_msgs, list):
        for m in base_msgs:
            if isinstance(m, dict) and "role" in m and "content" in m:
                result.append({"role": m["role"], "content": m["content"]})

    # 2) fncall_messages (often contains extra system/user context)
    fn_msgs = obj.get("fncall_messages")
    if isinstance(fn_msgs, list):
        for m in fn_msgs:
            if isinstance(m, dict) and "role" in m and "content" in m:
                result.append({"role": m["role"], "content": m["content"]})

    # 3) Final assistant message from response.choices[0].message
    resp = obj.get("response")
    if isinstance(resp, dict):
        choices = resp.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                msg = first.get("message")
                if isinstance(msg, dict):
                    role = msg.get("role") or "assistant"
                    content = msg.get("content")
                    if content is not None:
                        result.append({"role": role, "content": content})

    # If nothing was extracted, fallback to a single assistant message
    if not result:
        return [{"role": "assistant", "content": str(obj)}]
    return result


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


# ----------------------- LLM-backed synthesis -----------------------

from stringsight.core.llm_utils import parallel_completions
from .prompting import SYSTEM_PROMPT, build_parser_prompt
from .format_spec import CANONICAL_SPEC_MD


def extract_code_block(text: str) -> Optional[str]:
    """Extract the first python code block from text."""
    m = re.search(r"```python\n([\s\S]*?)```", text)
    if m:
        return m.group(1)
    m = re.search(r"```\n([\s\S]*?)```", text)
    if m:
        return m.group(1)
    return None


def synthesize_parser_via_llm(sample_obj: Any, model: str = "gpt-4.1") -> str:
    """Call LLM to produce a parse_to_oai function source code.

    Returns the code string; caller compiles it in a sandbox.
    """
    sample_json_str = json.dumps(sample_obj, ensure_ascii=False, indent=2)
    user_prompt = build_parser_prompt(sample_json_str, CANONICAL_SPEC_MD)
    resp = parallel_completions(
        [user_prompt],
        model=model,
        system_prompt=SYSTEM_PROMPT,
        temperature=0.0,
        top_p=1.0,
        max_tokens=1200,
        show_progress=False,
    )[0]
    code = extract_code_block(resp) or resp
    return code


def compile_parser_function(code: str) -> Optional[Any]:
    """Compile a user-provided function named parse_to_oai in a restricted namespace."""
    # Very restricted builtins: only allow harmless constructors
    safe_builtins = {
        "len": len,
        "range": range,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "min": min,
        "max": max,
        "sum": sum,
        "enumerate": enumerate,
        "isinstance": isinstance,
        "getattr": getattr,
        "hasattr": hasattr,
    }
    globals_ns = {"__builtins__": safe_builtins}
    locals_ns: Dict[str, Any] = {}
    try:
        exec(code, globals_ns, locals_ns)
    except Exception:
        return None
    fn = locals_ns.get("parse_to_oai") or globals_ns.get("parse_to_oai")
    return fn if callable(fn) else None



