def format_oai_simple(row):
  return [{
    "role": "user", "content": row["prompt"]
  }, {
    "role": "assistant", "content": row["model_response"]
  }]