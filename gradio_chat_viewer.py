import gradio as gr
import json
import pandas as pd
import markdown
from typing import List, Dict, Any

def load_data():
    """Load the JSONL data file"""
    try:
        df = pd.read_json("data/koala/koala_results_oai_format.jsonl", lines=True)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def format_message_html(role: str, content: Any) -> str:
    """Format a single message as HTML"""
    
    # Define colors for different roles
    role_colors = {
        "system": "#ff6b6b",      # Red
        "user": "#4ecdc4",        # Teal 
        "assistant": "#45b7d1",   # Blue
        "tool": "#96ceb4",        # Green
        "info": "#feca57"         # Yellow
    }
    
    # Get color for this role, default to gray
    color = role_colors.get(role.lower(), "#95a5a6")
    
    # Format content for HTML display
    if isinstance(content, dict):
        content_html = f"<pre style='background: #f8f9fa; padding: 10px; border-radius: 4px; overflow-x: auto;'>{json.dumps(content, indent=2)}</pre>"
    elif isinstance(content, str):
        # Convert markdown to HTML with proper extensions
        try:
            content_html = markdown.markdown(
                content, 
                extensions=[
                    'nl2br',           # Convert newlines to <br>
                    'fenced_code',     # Support ```code blocks```
                    'codehilite',      # Syntax highlighting for code blocks
                    'tables',          # Support tables
                    'toc',             # Table of contents
                    'sane_lists',      # Better list handling
                    'def_list',        # Definition lists
                    'attr_list',       # Attribute lists
                    'footnotes',       # Footnotes
                    'abbr',            # Abbreviations
                    'md_in_html'       # Allow HTML in markdown
                ],
                output_format='html5'
            )
        except Exception as e:
            # Fallback to basic markdown if advanced features fail
            content_html = markdown.markdown(content, extensions=['nl2br', 'fenced_code'])
    elif content is None:
        content_html = "<em>(No content)</em>"
    else:
        content_html = markdown.markdown(str(content), extensions=['nl2br', 'fenced_code'])
    
    return f"""
    <div style="
        border-left: 4px solid {color};
        margin: 8px 0;
        background-color: white;
        padding: 12px;
        border-radius: 0 8px 8px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    ">
        <div style="
            color: #666;
            font-size: 14px;
            font-weight: bold;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 1px;
        ">{role}</div>
        <div style="
            color: #333;
            line-height: 1.6;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        ">
            {content_html}
        </div>
    </div>
    """

def display_trajectory(trajectory_index: int, df: pd.DataFrame) -> tuple:
    """Display a trajectory at the given index"""
    if df is None or trajectory_index >= len(df):
        return "", "", "", ""
    
    current_row = df.iloc[trajectory_index]
    
    # Get metadata
    question_id = current_row.get('question_id', 'N/A')
    model = current_row.get('model', 'N/A')
    trace = current_row.get('trace', [])
    message_count = len(trace) if isinstance(trace, list) else 0
    
    # Format metadata
    metadata_html = f"""
    <div style="display: flex; justify-content: space-between; margin-bottom: 20px; font-size: 14px;">
        <div><strong>Question ID:</strong> {question_id}</div>
        <div><strong>Model:</strong> {model}</div>
        <div><strong>Messages:</strong> {message_count}</div>
    </div>
    """
    
    # Format conversation trajectory
    conversation_html = ""
    if isinstance(trace, list):
        for message in trace:
            if isinstance(message, dict):
                role = message.get('role', 'unknown')
                content = message.get('content')
                conversation_html += format_message_html(role, content)
                
                # Handle tool calls if present
                if 'tool_calls' in message:
                    for tool_call in message['tool_calls']:
                        conversation_html += f"""
                        <div style="
                            border-left: 4px solid #e67e22;
                            padding-left: 20px;
                            margin: 5px 0 5px 20px;
                            background-color: #fdf6e3;
                            padding: 10px;
                            border-radius: 0 5px 5px 0;
                        ">
                            <h5 style="
                                color: #666;
                                font-size: 12px;
                                font-weight: bold;
                                margin-bottom: 5px;
                                text-transform: uppercase;
                            ">TOOL CALL</h5>
                            <pre>{json.dumps(tool_call, indent=2)}</pre>
                        </div>
                        """
                
                # Handle tool call responses
                if role == 'tool':
                    tool_call_id = message.get('tool_call_id', 'Unknown')
                    name = message.get('name', 'Unknown')
                    conversation_html += f"<p style='font-size: 12px; color: #666;'>Tool: {name} | Call ID: {tool_call_id}</p>"
    
    # Format prompt
    prompt_html = ""
    if 'prompt' in current_row:
        prompt_html = f"""
        <div style="margin-top: 20px; padding: 15px; background-color: #f8f9fa; border-radius: 8px;">
            <h4>Prompt</h4>
            <p><strong>{current_row['prompt']}</strong></p>
        </div>
        """
    
    # Format scores
    scores_html = ""
    if 'score' in current_row and current_row['score']:
        score_data = current_row['score']
        if isinstance(score_data, dict):
            scores_html = "<div style='margin-top: 20px; padding: 15px; background-color: #f8f9fa; border-radius: 8px;'><h4>Scores</h4>"
            for metric, value in score_data.items():
                scores_html += f"<p><strong>{metric}:</strong> {value}</p>"
            scores_html += "</div>"
    
    return metadata_html, conversation_html, prompt_html, scores_html

def create_trajectory_options(df: pd.DataFrame) -> List[str]:
    """Create trajectory options for the dropdown"""
    if df is None:
        return []
    
    options = []
    for i in range(len(df)):
        question_id = df.iloc[i].get('question_id', 'N/A')
        model = df.iloc[i].get('model', 'N/A')
        options.append(f"Trajectory {i} (ID: {question_id}, Model: {model})")
    
    return options

def main():
    # Load data
    df = load_data()
    if df is None:
        return gr.Interface(lambda: "Error loading data", title="Chat Trajectory Viewer")
    
    # Create trajectory options
    trajectory_options = create_trajectory_options(df)
    
    def update_trajectory(trajectory_choice: str):
        """Update the display when trajectory selection changes"""
        if not trajectory_choice:
            return "", "", "", ""
        
        # Extract index from choice string
        try:
            index = int(trajectory_choice.split()[1])
        except (ValueError, IndexError):
            return "", "", "", ""
        
        return display_trajectory(index, df)
    
    # Create the interface
    with gr.Blocks(title="Chat Trajectory Viewer", theme=gr.themes.Soft()) as demo:
        # Add custom CSS for better markdown rendering
        gr.HTML("""
        <style>
        /* Code block styling */
        pre {
            background-color: #f6f8fa !important;
            border: 1px solid #e1e4e8 !important;
            border-radius: 6px !important;
            padding: 16px !important;
            overflow-x: auto !important;
            font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace !important;
            font-size: 14px !important;
            line-height: 1.45 !important;
        }
        
        code {
            background-color: #f6f8fa !important;
            border-radius: 3px !important;
            padding: 2px 4px !important;
            font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace !important;
            font-size: 13px !important;
        }
        
        /* Table styling */
        table {
            border-collapse: collapse !important;
            width: 100% !important;
            margin: 16px 0 !important;
        }
        
        th, td {
            border: 1px solid #e1e4e8 !important;
            padding: 8px 12px !important;
            text-align: left !important;
        }
        
        th {
            background-color: #f6f8fa !important;
            font-weight: 600 !important;
        }
        
        /* List styling */
        ul, ol {
            padding-left: 20px !important;
            margin: 8px 0 !important;
        }
        
        li {
            margin: 4px 0 !important;
        }
        
        /* Blockquote styling */
        blockquote {
            border-left: 4px solid #dfe2e5 !important;
            margin: 16px 0 !important;
            padding-left: 16px !important;
            color: #6a737d !important;
        }
        
        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            margin: 16px 0 8px 0 !important;
            font-weight: 600 !important;
        }
        
        /* Links */
        a {
            color: #0366d6 !important;
            text-decoration: none !important;
        }
        
        a:hover {
            text-decoration: underline !important;
        }
        
        /* Horizontal rule */
        hr {
            border: none !important;
            border-top: 1px solid #e1e4e8 !important;
            margin: 16px 0 !important;
        }
        </style>
        """)
        
        gr.Markdown("# Chat Trajectory Viewer")
        gr.Markdown("Browse through conversation trajectories")
        
        with gr.Row():
            trajectory_dropdown = gr.Dropdown(
                choices=trajectory_options,
                label="Select Trajectory",
                value=trajectory_options[0] if trajectory_options else None
            )
        
        with gr.Row():
            metadata_output = gr.HTML(label="Metadata")
        
        with gr.Row():
            conversation_output = gr.HTML(label="Conversation Trajectory")
        
        with gr.Row():
            prompt_output = gr.HTML(label="Prompt")
            scores_output = gr.HTML(label="Scores")
        
        # Set up the event handler
        trajectory_dropdown.change(
            fn=update_trajectory,
            inputs=[trajectory_dropdown],
            outputs=[metadata_output, conversation_output, prompt_output, scores_output]
        )
        
        # Initialize with first trajectory
        if trajectory_options:
            initial_metadata, initial_conversation, initial_prompt, initial_scores = display_trajectory(0, df)
            metadata_output.value = initial_metadata
            conversation_output.value = initial_conversation
            prompt_output.value = initial_prompt
            scores_output.value = initial_scores
    
    return demo

if __name__ == "__main__":
    demo = main()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860) 