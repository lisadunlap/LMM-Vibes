"""
Conversation display utilities for pipeline results app.

This module contains functions for displaying conversations in OpenAI format
with proper styling and formatting.
"""

import streamlit as st
import json
import markdown
from typing import Any, List, Dict

# Import conversation conversion function
try:
    from lmmvibes.extractors.conv_to_str import conv_to_str
except ImportError:
    def conv_to_str(conv):
        """Fallback function if conv_to_str is not available."""
        if isinstance(conv, str):
            return conv
        elif isinstance(conv, list):
            return "\n".join(str(item) for item in conv)
        else:
            return str(conv)


def display_openai_message(role, content, name=None, message_id=None):
    """Display a single OpenAI format message with role-specific styling"""
    
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
        # Convert markdown to HTML properly
        content_html = markdown.markdown(content, extensions=['nl2br', 'fenced_code'])
    elif content is None:
        content_html = "<em>(No content)</em>"
    else:
        content_html = markdown.markdown(str(content), extensions=['nl2br'])
    
    # Build role display text
    role_display = role.upper()
    if name:
        role_display += f" ({name})"
    if message_id:
        role_display += f" [ID: {message_id}]"
    
    # Special handling for system messages - make them collapsible
    if role.lower() == "system":
        with st.expander(f"🔧 {role_display}", expanded=False):
            st.markdown(f"""
            <div style="
                border-left: 4px solid {color};
                margin: 8px 0;
                background-color: #f8f9fa;
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
                ">{role_display}</div>
                <div style="
                    color: #333;
                    line-height: 1.6;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                ">
                    {content_html}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        # Regular message display for non-system messages
        st.markdown(f"""
        <div style="
            border-left: 4px solid {color};
            margin: 8px 0;
            background-color: #f8f9fa;
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
            ">{role_display}</div>
            <div style="
                color: #333;
                line-height: 1.6;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            ">
                {content_html}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Handle special cases that need separate display
    if isinstance(content, dict) and len(json.dumps(content, indent=2)) > 500:
        # For very large JSON, also show expandable version
        with st.expander("View JSON in expandable format"):
            st.json(content)


def display_openai_conversation(conversation_data):
    """Display a full OpenAI format conversation"""
    
    if isinstance(conversation_data, list):
        # Handle list of messages (OpenAI format)
        for i, message in enumerate(conversation_data):
            if not isinstance(message, dict):
                continue
                
            role = message.get('role', 'unknown')
            content = message.get('content')
            name = message.get('name')
            message_id = message.get('id')
            
            display_openai_message(role, content, name, message_id)
            
            # Handle tool calls if present
            if 'tool_calls' in message:
                for j, tool_call in enumerate(message['tool_calls']):
                    st.markdown(f"""
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
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display tool call details
                    tool_info = {
                        "function": tool_call.get('function', {}),
                        "id": tool_call.get('id', ''),
                        "type": tool_call.get('type', '')
                    }
                    st.json(tool_info)
            
            # Handle tool call responses
            if role == 'tool':
                tool_call_id = message.get('tool_call_id', 'Unknown')
                name = message.get('name', 'Unknown')
                st.caption(f"Tool: {name} | Call ID: {tool_call_id}")
    
    elif isinstance(conversation_data, str):
        # Handle string format - try to parse as JSON
        try:
            parsed = json.loads(conversation_data)
            display_openai_conversation(parsed)
        except json.JSONDecodeError:
            # Fallback to raw string display
            st.markdown(f"""
            <div style="
                border-left: 4px solid #95a5a6;
                margin: 8px 0;
                background-color: #f8f9fa;
                padding: 12px;
                border-radius: 0 8px 8px 0;
            ">
                <div style="
                    color: #333;
                    line-height: 1.6;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                ">
                    {markdown.markdown(conversation_data, extensions=['nl2br'])}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        # Fallback for other formats
        st.text(str(conversation_data))


def convert_to_openai_format(response_data):
    """Convert various response formats to OpenAI format"""
    
    if isinstance(response_data, list):
        # Already in OpenAI format
        return response_data
    elif isinstance(response_data, str):
        # Try to parse as Python literal first (handles single quotes)
        try:
            import ast
            parsed = ast.literal_eval(response_data)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass
        
        # Try to parse as JSON
        try:
            parsed = json.loads(response_data)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
        
        # If it's a string, try to convert using conv_to_str
        try:
            converted = conv_to_str(response_data)
            return converted
        except:
            pass
        
        # Fallback: treat as plain text
        return [{"role": "assistant", "content": response_data}]
    else:
        # Fallback for other types
        return [{"role": "assistant", "content": str(response_data)}] 