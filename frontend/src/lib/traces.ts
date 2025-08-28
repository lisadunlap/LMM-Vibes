export type Role = "user" | "assistant";
export interface Message { role: Role; content: string }

export interface SingleTrace { questionId: string; prompt: string; messages: Message[] }
export interface SideBySideTrace {
  questionId: string;
  prompt: string;
  modelA: string;
  modelB: string;
  messagesA: Message[];
  messagesB: Message[];
}

export function detectMethodFromColumns(cols: string[]): "single_model" | "side_by_side" | "unknown" {
  const single = ["prompt", "model", "model_response"].every((c) => cols.includes(c));
  const sbs = ["prompt", "model_a", "model_b", "model_a_response", "model_b_response"].every((c) => cols.includes(c));
  if (sbs) return "side_by_side";
  if (single) return "single_model";
  return "unknown";
}

export function formatSingleTraceFromRow(row: Record<string, any>): SingleTrace {
  const prompt = String(row["prompt"] ?? "");
  const response = row["model_response"];
  const messages = ensureOpenAIFormat(prompt, response);
  return { questionId: String(row["question_id"] ?? ""), prompt, messages };
}

export function formatSideBySideTraceFromRow(row: Record<string, any>): SideBySideTrace {
  const prompt = String(row["prompt"] ?? "");
  const messagesA = ensureOpenAIFormat(prompt, row["model_a_response"]);
  const messagesB = ensureOpenAIFormat(prompt, row["model_b_response"]);
  return {
    questionId: String(row["question_id"] ?? ""),
    prompt,
    modelA: String(row["model_a"] ?? "Model A"),
    modelB: String(row["model_b"] ?? "Model B"),
    messagesA,
    messagesB,
  };
}

export function ensureOpenAIFormat(prompt: string, response: any): Message[] {
  if (Array.isArray(response)) return response as Message[];
  const text = typeof response === "string" ? response : String(response ?? "");
  return [
    { role: "user", content: prompt },
    { role: "assistant", content: text },
  ];
}


