# This contains functions to convert OAI conversation objects to strings
# This format is a list of dictionaries with the following keys:
# - role: "user", "assistant", "system", "tool", etc
# - content: the content of the message (can be a string or a dictionary with values "text", "image", or "tool_calls")
# - the tool_calls key (optional) in the content dictionary: a list of tool calls, each tool call is a dict with the following keys:
#   - name: the name of the tool
#   - arguments: the arguments to the tool
#   - tool_call_id: the id of the tool call
#   - anything else that is present in the tool call
# - name (optional): the name of the model or tool that sent the message (this should persist for the entire conversation)
# - id (optional): the id of the specific model or tool, this is meant to only be used once (e.g. the id of a specifically function call)
# - any other keys that are present in the conversation object (not shown in the string right now)

import pprint
import ast
import json

def pretty_print_dict(val):
    """
    Pretty print a dictionary or a string that can be parsed as a dictionary.
    """
    if isinstance(val, dict):
        return pprint.pformat(val, width=80, compact=False)
    if isinstance(val, str):
        try:
            # Try to parse as dict
            parsed = ast.literal_eval(val)
            if isinstance(parsed, dict):
                return pprint.pformat(parsed, width=80, compact=False)
        except Exception:
            pass
    return str(val)

def convert_tool_calls_to_str(tool_calls):
    """
    Convert a list of tool calls to a string.
    """
    # create name and arguments string for each tool call
    tool_calls_str = []
    for tool_call in tool_calls:
        name = tool_call.get('name', '<no_name>')
        args = tool_call.get('arguments', {})
        # Pretty print arguments if dict or dict-string
        args_str = pretty_print_dict(args)
        base = f"call {name} with args {args_str} (id: {tool_call.get('id', '<no_id>')})"
        extra_keys = [key for key in tool_call.keys() if key not in ['name', 'arguments', 'id']]
        if extra_keys:
            # Show extra key-value pairs, pretty-printed if dict
            extras = {k: tool_call[k] for k in extra_keys}
            base += f"\nadditional info: {pretty_print_dict(extras)}"
        tool_calls_str.append(base)
    return "\n".join(tool_calls_str)

def convert_content_to_str(content):
    """
    Convert the content of a message to a string.
    Pretty print any dictionary values or stringified dictionaries.
    """
    if isinstance(content, str):
        # Try to pretty print if it's a dict string
        try:
            parsed = ast.literal_eval(content)
            if isinstance(parsed, dict):
                return pretty_print_dict(parsed)
        except Exception:
            pass
        return content
    elif isinstance(content, dict):
        ret = []
        for key, value in content.items():
            if key == 'text':
                # Pretty print if text is a dict or dict-string
                ret.append(pretty_print_dict(value))
            elif key == 'image':
                ret.append(f"image: {value}")
            elif key == 'tool_calls':
                ret.append(convert_tool_calls_to_str(value))
            else:
                # Pretty print for any other dict or dict-string
                ret.append(f"{key}: {pretty_print_dict(value)}")
        return "\n".join(ret)
    else:
        # Fallback: pretty print if possible
        return pretty_print_dict(content)

def conv_to_str(conv):
    """
    Convert an OAI conversation object to a string.
    """
    ret = []
    for msg in conv:
        if msg['role'] == 'tool':
            assert msg.get('name') is not None, "Tool call must have a name"
            ret.append(f"**output of tool {msg['name']}**\n{convert_content_to_str(msg['content'])}")
        else:
            if 'name' in msg:
                if 'id' in msg:
                    ret.append(f"\n**{msg['role']} {msg['name']} (id: {msg['id']}):**\n{convert_content_to_str(msg['content'])}")
                else:
                    ret.append(f"\n**{msg['role']} {msg['name']}**\n{convert_content_to_str(msg['content'])}")
            else:
                ret.append(f"\n**{msg['role']}:**\n{convert_content_to_str(msg['content'])}")
    return "\n\n".join(ret)

def simple_to_oai_format(prompt: str, response: str) -> list:
    """
    Convert a simple prompt-response pair to OAI format.
    
    Args:
        prompt: The user's prompt/question
        response: The model's response
        
    Returns:
        List of dictionaries in OAI conversation format
    """
    return [
        {
            "role": "user",
            "content": prompt
        },
        {
            "role": "assistant", 
            "content": response
        }
    ]

def check_and_convert_to_oai_format(prompt: str, response: str) -> tuple[list, bool]:
    """
    Check if response is a string and convert to OAI format if needed.
    
    Args:
        prompt: The user's prompt/question
        response: The model's response (could be string or already OAI format)
        
    Returns:
        Tuple of (conversation_in_oai_format, was_converted)
    """
    # If response is already a list (OAI format), return as is
    if isinstance(response, list):
        return response, False
    
    # If response is a string, convert to OAI format
    if isinstance(response, str):
        return simple_to_oai_format(prompt, response), True
    
    # For other types, try to convert to string first
    try:
        response_str = str(response)
        return simple_to_oai_format(prompt, response_str), True
    except Exception:
        # If conversion fails, return as is
        return response, False