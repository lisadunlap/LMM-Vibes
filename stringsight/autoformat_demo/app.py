"""Gradio app for isolated auto-formatting demo.

Loads a default test JSON, synthesizes a parser (stub), applies it with
validation, and displays results. No edits to core pipeline.
"""

import json
from typing import Any, Dict, List, Tuple

import gradio as gr

# Local demo helpers
from .format_spec import CANONICAL_SPEC_MD
from .engine import (
    synthesize_parser_stub,
    apply_parser_safely_stub,
    validate_oai_conversation,
)

DEFAULT_TEST_PATH = (
    "/home/lisabdunlap/LMM-Vibes/data/agentic_traces/swe-traces/March12/"
    "QwQ-32B_maxiter_50_N_v0.25.0-no-hint-run_1/llm_completions/"
    "astropy__astropy-7166/openai__Qwen__QwQ-32B-1741763968.9436522.json"
)


def load_json_file(path: str) -> Tuple[str, Dict[str, Any] | None]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return "Loaded.", data
    except Exception as e:
        return f"Failed to load: {e}", None


def run_autoformat(raw_json_text: str) -> Tuple[str, str, str]:
    """End-to-end demo: synthesize parser, apply, validate.

    Returns (parser_text, parsed_preview, validation_report)
    """
    try:
        raw_obj = json.loads(raw_json_text)
    except Exception as e:
        return "", "", f"Invalid JSON input: {e}"

    # 1) Synthesize parser (stubbed for now)
    parser_text = synthesize_parser_stub(raw_obj)

    # 2) Apply parser (stubbed) and 3) Validate
    parsed = apply_parser_safely_stub(parser_text, raw_obj)
    ok, errors = validate_oai_conversation(parsed)

    parsed_preview = json.dumps(parsed, ensure_ascii=False, indent=2) if parsed is not None else ""
    validation_report = (
        "Validation: PASS" if ok else f"Validation: FAIL\nErrors: {errors}"
    )

    return parser_text, parsed_preview, validation_report


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Auto-format Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Auto-format Demo (Isolated Prototype)")
        with gr.Row():
            input_path = gr.Textbox(value=DEFAULT_TEST_PATH, label="Input JSON path")
            load_btn = gr.Button("Load file")
        raw_json = gr.Textbox(label="Raw JSON", lines=20)

        with gr.Accordion("Canonical Target Spec", open=False):
            gr.Markdown(CANONICAL_SPEC_MD)

        run_btn = gr.Button("Synthesize + Apply + Validate")
        parser_out = gr.Code(label="Generated parser (stub)")
        parsed_out = gr.Code(label="Parsed output (preview)")
        report_out = gr.Textbox(label="Validation report", lines=6)

        def _on_load(path: str):
            msg, obj = load_json_file(path)
            text = json.dumps(obj, ensure_ascii=False, indent=2) if obj is not None else ""
            return gr.update(value=text), gr.update(value=f"{msg}")

        load_status = gr.Textbox(label="Load status", interactive=False)
        load_btn.click(_on_load, inputs=[input_path], outputs=[raw_json, load_status])

        run_btn.click(run_autoformat, inputs=[raw_json], outputs=[parser_out, parsed_out, report_out])

    return demo


if __name__ == "__main__":
    build_app().launch()


