import json
from typing import Dict, Any, Optional
import pandas as pd

def traj_to_string(record: dict, show_roles: bool = True) -> str:
    """
    Convert the `traj` portion of an agentic interaction record into a
    single readable string.

    Parameters
    ----------
    record : dict
        A dictionary like `data[0]` containing a `traj` key.
    show_roles : bool, optional
        If True (default), each line is prefixed with the speaker/role.

    Returns
    -------
    str
        Newline-delimited string representation of the trajectory.
    """
    if "traj" not in record or not isinstance(record["traj"], list):
        raise ValueError("Input record must contain a list under the key 'traj'.")

    lines = []
    for turn in record["traj"]:
        # Skip None entries
        if turn is None:
            continue
            
        role = turn.get("role", "unknown")
        content = turn.get("content")

        # 1) Normal message content
        if content:
            lines.append(f"**{role}:** {content}" if show_roles else content)

        # 2) Tool calls - handle None values explicitly
        tool_calls = turn.get("tool_calls")
        if tool_calls is not None:
            for call in tool_calls:
                if call is None:
                    continue
                func = call.get("function", {})
                func_name = func.get("name", "<unknown_function>")
                func_args = func.get("arguments", "")
                lines.append(
                    f"**{role}->tool:** CALL {func_name}({func_args})"
                    if show_roles
                    else f"CALL {func_name}({func_args})"
                )

        # 3) Direct tool responses (role == "tool"); avoid double-printing content
        if role == "tool" and content:
            lines.append(f"**tool_response:** {content}" if show_roles else content)

    return "\n\n".join(lines)

def agentic_record_to_string(record: dict, show_roles: bool = True, include_context: bool = True) -> str:
    """
    Convert an agentic interaction record into a comprehensive string for LLM analysis.
    Includes task context, user profile data, and conversation trajectory.

    Parameters
    ----------
    record : dict
        A dictionary containing the full agentic record with 'traj', 'info', etc.
    show_roles : bool, optional
        If True (default), each line is prefixed with the speaker/role.
    include_context : bool, optional
        If True (default), includes task instruction, user profile, and actions context.

    Returns
    -------
    str
        Comprehensive string representation including context and trajectory.
    """
    if "traj" not in record or not isinstance(record["traj"], list):
        raise ValueError("Input record must contain a list under the key 'traj'.")

    sections = []

    if include_context:
        # 1. Task Information
        task_info = record.get("info", {}).get("task", {})
        if task_info:
            sections.append("=" * 50)
            sections.append("TASK CONTEXT")
            sections.append("=" * 50)
            
            # User ID
            user_id = task_info.get("user_id")
            if user_id:
                sections.append(f"**User ID:** {user_id}")
            
            # Task Instruction
            instruction = task_info.get("instruction")
            if instruction:
                sections.append(f"**Task Instruction:** {instruction}")
            
            sections.append("")

        # 2. Extract User Profile Data from Tool Responses
        user_profile = _extract_user_profile_from_traj(record["traj"])
        if user_profile:
            sections.append("=" * 50)
            sections.append("USER PROFILE DATA")
            sections.append("=" * 50)
            sections.append(_format_user_profile(user_profile, show_roles))
            sections.append("")

        # 3. Final Actions Taken
        actions = task_info.get("actions", [])
        if actions:
            sections.append("=" * 50)
            sections.append("ACTIONS TAKEN")
            sections.append("=" * 50)
            for i, action in enumerate(actions, 1):
                action_name = action.get("name", "unknown_action")
                action_kwargs = action.get("kwargs", {})
                sections.append(f"**Action {i}:** {action_name}")
                sections.append(f"**Parameters:** {json.dumps(action_kwargs, indent=2)}")
            sections.append("")

    # 4. Conversation Trajectory
    sections.append("=" * 50)
    sections.append("CONVERSATION TRAJECTORY")
    sections.append("=" * 50)
    
    # Use existing traj_to_string logic with format handling
    traj_lines = []
    for turn in record["traj"]:
        # Skip None entries
        if turn is None:
            continue
            
        role = turn.get("role", "unknown")
        content = turn.get("content")

        # Normal message content
        if content:
            traj_lines.append(f"**{role}:** {content}" if show_roles else content)

        # Tool calls - handle both GPT and Claude formats, including None values
        tool_calls = turn.get("tool_calls")
        if tool_calls is not None:  # Explicitly check for None
            for call in tool_calls:
                if call is None:  # Skip None calls within the list
                    continue
                func = call.get("function", {})
                func_name = func.get("name", "<unknown_function>")
                func_args = func.get("arguments", "")
                
                # Handle different argument formats (GPT: string, Claude: dict)
                if isinstance(func_args, dict):
                    # Claude format - already a dict
                    args_str = json.dumps(func_args)
                else:
                    # GPT format - string
                    args_str = str(func_args)
                
                traj_lines.append(
                    f"**{role}->tool:** CALL {func_name}({args_str})"
                    if show_roles
                    else f"CALL {func_name}({args_str})"
                )

        # Tool responses
        if role == "tool" and content:
            traj_lines.append(f"**tool_response:** {content}" if show_roles else content)

    sections.append("\n\n".join(traj_lines))

    return "\n".join(sections)

def _extract_user_profile_from_traj(traj: list) -> Optional[Dict[str, Any]]:
    """Extract user profile data from tool responses in the trajectory."""
    for turn in traj:
        # Skip None entries
        if turn is None:
            continue
            
        if turn.get("role") == "tool":
            tool_call_id = turn.get("tool_call_id", "")
            content = turn.get("content", "")
            
            # Look for get_user_details responses
            if content and ("name" in content or "email" in content or "payment_methods" in content):
                try:
                    # Try to parse as JSON
                    profile_data = json.loads(content)
                    if isinstance(profile_data, dict):
                        return profile_data
                except (json.JSONDecodeError, TypeError):
                    # If not JSON, might be a string representation
                    if "payment_methods" in content or "membership" in content:
                        return {"raw_profile": content}
    return None

def _format_user_profile(profile: Dict[str, Any], show_roles: bool = True) -> str:
    """Format user profile data for display."""
    if "raw_profile" in profile:
        return profile["raw_profile"]
    
    lines = []
    
    # Name
    name_info = profile.get("name", {})
    if name_info:
        first_name = name_info.get("first_name", "")
        last_name = name_info.get("last_name", "")
        if first_name and last_name:
            lines.append(f"**Name:** {first_name} {last_name}")
    
    # Email
    email = profile.get("email")
    if email:
        lines.append(f"**Email:** {email}")
    
    # Date of Birth
    dob = profile.get("dob")
    if dob:
        lines.append(f"**Date of Birth:** {dob}")
    
    # Membership
    membership = profile.get("membership")
    if membership:
        lines.append(f"**Membership Tier:** {membership}")
    
    # Address
    address = profile.get("address", {})
    if address:
        addr_parts = []
        for key in ["address1", "address2", "city", "province", "zip", "country"]:
            value = address.get(key)
            if value:
                addr_parts.append(value)
        if addr_parts:
            lines.append(f"**Address:** {', '.join(addr_parts)}")
    
    # Payment Methods
    payment_methods = profile.get("payment_methods", {})
    if payment_methods:
        lines.append("**Payment Methods:**")
        for payment_id, details in payment_methods.items():
            source = details.get("source", "unknown")
            if source == "credit_card":
                brand = details.get("brand", "").upper()
                last_four = details.get("last_four", "")
                lines.append(f"  - {brand} ending in {last_four} (ID: {payment_id})")
            elif source == "certificate":
                amount = details.get("amount", "")
                lines.append(f"  - Certificate ${amount} (ID: {payment_id})")
            else:
                lines.append(f"  - {source} (ID: {payment_id})")
    
    # Saved Passengers
    saved_passengers = profile.get("saved_passengers", [])
    if saved_passengers:
        lines.append("**Saved Passengers:**")
        for passenger in saved_passengers:
            first_name = passenger.get("first_name", "")
            last_name = passenger.get("last_name", "")
            passenger_dob = passenger.get("dob", "")
            lines.append(f"  - {first_name} {last_name} (DOB: {passenger_dob})")
    
    # Existing Reservations
    reservations = profile.get("reservations", [])
    if reservations:
        lines.append(f"**Existing Reservations:** {', '.join(reservations)}")
    
    return "\n".join(lines)

def process_data(file: str, incorrect_only: bool = False) -> list[dict]:
    """
    Process the airline data to extract the relevant information.
    """
    data_gpt = pd.read_json(file)
    
    # Track successful conversions and errors
    conversion_errors = 0
    
    def safe_agentic_record_to_string(record):
        try:
            return agentic_record_to_string(record)
        except Exception as e:
            nonlocal conversion_errors
            conversion_errors += 1
            print(f"Error converting record {record.get('task_id', 'unknown')}-{record.get('trial', 'unknown')}: {str(e)}")
            return None
    
    # Apply conversion with error handling
    data_gpt["model_response"] = data_gpt.apply(safe_agentic_record_to_string, axis=1)
    
    # Drop rows where conversion failed
    initial_count = len(data_gpt)
    data_gpt = data_gpt.dropna(subset=['model_response'])
    final_count = len(data_gpt)
    
    print(f"Conversion errors: {conversion_errors}")
    print(f"Rows dropped due to conversion errors: {initial_count - final_count}")
    print(f"Successfully processed: {final_count}/{initial_count} rows")
    
    # Only proceed if we have data left
    if final_count == 0:
        print("No valid records found - returning empty DataFrame")
        return pd.DataFrame()
    
    data_gpt["prompt"] = data_gpt.apply(lambda x: x["info"]["task"]["instruction"], axis=1)
    data_gpt["question_id"] = data_gpt.apply(lambda x: f"{x['task_id']}-{x['trial']}", axis=1)
    
    # Handle different reward_info structures
    def get_reward_score(row):
        try:
            # Try GPT format first: info.reward_info.reward
            if 'info' in row and 'reward_info' in row['info'] and row['info']['reward_info'] is not None:
                return {"reward": row['info']['reward_info']['reward']}
            # Try Claude format: reward_info.reward  
            elif 'reward_info' in row and row['reward_info'] is not None:
                return {"reward": row['reward_info']['reward']}
            else:
                return {"reward": 0.0}
        except:
            return {"reward": 0.0}
    
    data_gpt["score"] = data_gpt.apply(get_reward_score, axis=1)
    if incorrect_only:
        data_gpt = data_gpt[data_gpt["score"].apply(lambda x: int(x["reward"])) == 0]
    
    return data_gpt

def process_airline_data(incorrect_only: bool = False) -> list[dict]:
    df1 = process_data("./data/taubench/gpt-4o-airline.json", incorrect_only)
    df1["model"] = "gpt-4o"
    df2 = process_data("./data/taubench/sonnet-35-new-airline.json", incorrect_only)
    df2["model"] = "claude-sonnet-35"
    return pd.concat([df1, df2])

def process_retail_data(incorrect_only: bool = False) -> list[dict]:
    df1 = process_data("./data/taubench/gpt-4o-retail.json", incorrect_only)
    df1["model"] = "gpt-4o"
    df2 = process_data("./data/taubench/sonnet-35-new-retail.json", incorrect_only)
    df2["model"] = "claude-sonnet-35"
    return pd.concat([df1, df2])

airline_data = process_airline_data()
print(f"Airline data: {len(airline_data)}")
airline_data.to_json("./data/taubench/airline_data.jsonl", orient="records", lines=True)
retail_data = process_retail_data()
print(f"Retail data: {len(retail_data)}")
retail_data.to_json("./data/taubench/retail_data.jsonl", orient="records", lines=True)

airline_data = process_airline_data(incorrect_only=True)
print(f"Airline data incorrect: {len(airline_data)}")
airline_data.to_json("./data/taubench/airline_data_incorrect.jsonl", orient="records", lines=True)
retail_data = process_retail_data(incorrect_only=True)
print(f"Retail data incorrect: {len(retail_data)}")
retail_data.to_json("./data/taubench/retail_data_incorrect.jsonl", orient="records", lines=True)
