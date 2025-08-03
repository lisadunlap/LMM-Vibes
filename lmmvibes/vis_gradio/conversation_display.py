from __future__ import annotations

"""Conversation display helpers for vis_gradio.

This module encapsulates everything related to:
• safely parsing model responses (lists / dicts / JSON strings)
• pretty-printing embedded dictionaries for readability
• converting multiple conversation formats to the OpenAI chat list format
• rendering that list as HTML (including accordion grouping + raw JSON viewer).

Moving this logic out of utils.py keeps the latter lean and focussed on general
analytics utilities.
"""

from typing import List, Dict, Any
import ast
import json
import html
import markdown

__all__: List[str] = [
    "convert_to_openai_format",
    "display_openai_conversation_html",
    "pretty_print_embedded_dicts",
]

# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------

def _find_balanced_spans(text: str):
    """Return (start, end) spans of balanced {...} or [...] regions in *text*."""
    spans, stack = [], []
    for i, ch in enumerate(text):
        if ch in "{[":
            stack.append((ch, i))
        elif ch in "}]" and stack:
            opener, start = stack.pop()
            if (opener, ch) in {("{", "}"), ("[", "]")} and not stack:
                spans.append((start, i + 1))
    return spans


def _try_parse_slice(slice_: str):
    """Attempt to parse *slice_* into a Python object; return None on failure."""
    try:
        return ast.literal_eval(slice_)
    except Exception:
        try:
            return json.loads(slice_)
        except Exception:
            return None


def pretty_print_embedded_dicts(text: str) -> str:
    """Replace any dicts or list-of-dicts inside *text* with a `<pre>` block."""
    if not text:
        return text

    new_parts, last_idx = [], 0
    for start, end in _find_balanced_spans(text):
        candidate = text[start:end]
        parsed = _try_parse_slice(candidate)
        is_good = isinstance(parsed, dict) or (
            isinstance(parsed, list) and parsed and all(isinstance(d, dict) for d in parsed)
        )
        if is_good:
            new_parts.append(html.escape(text[last_idx:start]))
            pretty = json.dumps(parsed, indent=2, ensure_ascii=False)
            new_parts.append(
                f"<pre style='background:#f8f9fa;padding:10px;border-radius:4px;overflow-x:auto;'>{pretty}</pre>"
            )
            last_idx = end
    new_parts.append(html.escape(text[last_idx:]))
    return "".join(new_parts)

# ---------------------------------------------------------------------------
# Format conversion
# ---------------------------------------------------------------------------

def convert_to_openai_format(response_data: Any):
    """Convert various response payloads into the OpenAI chat format list."""
    if isinstance(response_data, list):
        return response_data
    if isinstance(response_data, str):
        # Try Python literal first (handles single quotes)
        try:
            parsed = ast.literal_eval(response_data)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass
        # Try JSON
        try:
            parsed = json.loads(response_data)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
        # Fallback plain-text assistant message
        return [{"role": "assistant", "content": response_data}]
    # Fallback for any other type
    return [{"role": "assistant", "content": str(response_data)}]

# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

def _markdown(text: str) -> str:
    """Render markdown after pretty-printing any embedded dicts."""
    return markdown.markdown(pretty_print_embedded_dicts(text), extensions=["nl2br", "fenced_code"])


def display_openai_conversation_html(conversation_data: List[Dict[str, Any]], *, use_accordion: bool = True) -> str:
    """Convert an OpenAI-style conversation list into styled HTML for Gradio."""

    if not conversation_data:
        return "<p>No conversation data available</p>"

    # Collapsed raw JSON section for debugging
    raw_json = json.dumps(conversation_data, indent=2, ensure_ascii=False)
    html_out = f"""
    <details style="margin: 8px 0; border: 1px solid #ddd; border-radius: 8px; overflow: hidden;">
        <summary style="padding: 12px 15px; background: linear-gradient(135deg, #757575 0%, #616161 100%); color: white; cursor: pointer; font-weight: 600; font-size: 14px; user-select: none; list-style: none;">
            📄 Raw Response (JSON)
        </summary>
        <div style="padding: 15px; background: #f8f9fa; max-height: 400px; overflow-y: auto;">
            <pre style="white-space: pre-wrap; word-wrap: break-word;">{raw_json}</pre>
        </div>
    </details>
    """

    role_colors = {
        "system": "#ff6b6b",
        "info": "#4ecdc4",
        "assistant": "#45b7d1",
        "tool": "#96ceb4",
        "user": "#feca57",
    }

    def _format_msg(role: str, content: Any) -> str:
        if isinstance(content, dict):
            content_html = (
                f"<pre style='background: #f8f9fa; padding: 10px; border-radius: 4px; overflow-x: auto;'>{json.dumps(content, indent=2)}</pre>"
            )
        elif isinstance(content, str):
            content_html = _markdown(content)
        elif content is None:
            content_html = "<em>(No content)</em>"
        else:
            content_html = str(content)
        color = role_colors.get(role.lower(), "#95a5a6")
        return (
            f"<div style='border-left: 4px solid {color}; margin: 8px 0; background-color: #f8f9fa; padding: 12px; border-radius: 0 8px 8px 0;'>"
            f"<div style='font-weight: 600; color: {color}; margin-bottom: 8px; text-transform: capitalize; font-size: 14px;'>{role}</div>"
            f"<div style='color: #333; line-height: 1.6; font-family: \"Segoe UI\", Tahoma, Geneva, Verdana, sans-serif;'>{content_html}</div>"
            "</div>"
        )

    if use_accordion:
        system_msgs, info_msgs, other_msgs = [], [], []
        for m in conversation_data:
            if not isinstance(m, dict):
                continue
            role = m.get("role", "unknown").lower()
            content = m.get("content", "")
            if isinstance(content, dict) and "text" in content:
                content = content["text"]
            if role == "system":
                system_msgs.append((role, content))
            elif role == "info":
                info_msgs.append((role, content))
            else:
                other_msgs.append((role, content))

        def _accordion(title: str, col: str, items: List):
            if not items:
                return ""
            inner = "".join(_format_msg(r, c) for r, c in items)
            return (
                f"<details style='margin: 8px 0; border: 1px solid #ddd; border-radius: 8px; overflow: hidden;'>"
                f"<summary style='padding: 12px 15px; background: {col}; color: white; cursor: pointer; font-weight: 600; font-size: 14px; user-select: none; list-style: none;'>{title} ({len(items)})</summary>"
                f"<div style='padding: 15px; background: #f8f9fa;'>{inner}</div></details>"
            )

        html_out += _accordion("🔧 System Messages", "linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%)", system_msgs)
        html_out += _accordion("ℹ️ Info Messages", "linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%)", info_msgs)
        for r, c in other_msgs:
            html_out += _format_msg(r, c)
    else:
        # No accordion: just render everything
        for m in conversation_data:
            if not isinstance(m, dict):
                continue
            role = m.get("role", "unknown")
            content = m.get("content", "")
            if isinstance(content, dict) and "text" in content:
                content = content["text"]
            html_out += _format_msg(role, content)

    # CSS for summary hover effects when accordion is enabled
    if use_accordion:
        html_out = (
            "<style>details > summary {transition: all 0.3s ease;}"  # noqa: E501
            "details > summary:hover {transform: translateY(-1px); box-shadow: 0 2px 4px rgba(0,0,0,0.1);}"
            "details > summary::-webkit-details-marker, details > summary::marker {display:none;}"
            "</style>" + html_out
        )

    return html_out 