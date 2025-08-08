fixed_axis_prompt = """You are an expert model behavior analyst. Your task is to meticulously analyze the trace of a large language model to identify whether it contains any of the following behaviors:

{fixed_axes}

If the trace contains any of the behaviors, return a list of objects with the following structure. If a trace has more than one behavior, return a list of objects with the structure below for each behavior. It the trace contains none of the behaviors, return an empty list.

**JSON Output Structure**
```json
[
  {
    "property_description": which behavior is present in the trace, select one of {fixed_axes_names},
    "reason": a explanation of the exact behaviors in the trace that fall under the property_description,
    "evidence": "Direct quote(s) or evidence from the model response of the behavior. Format as a list of quotes, comma separated and wrapped in double quotes."
  }
]
```"""