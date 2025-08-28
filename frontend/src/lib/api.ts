export type Method = "single_model" | "side_by_side" | null;

export interface DetectResponse {
  method: Method;
  valid: boolean;
  missing: string[];
  row_count: number;
  columns: string[];
  preview: Record<string, any>[];
}

const API_BASE = (import.meta as any).env?.VITE_API_BASE || "http://localhost:8000";

export async function detectAndValidate(file: File): Promise<DetectResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/detect-and-validate`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`detect-and-validate failed: ${res.status} ${text}`);
  }
  return res.json();
}

export async function readPath(path: string, limit?: number, method?: "single_model" | "side_by_side") {
  const res = await fetch(`${API_BASE}/read-path`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path, limit, method }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`read-path failed: ${res.status} ${text}`);
  }
  return res.json();
}

export async function listPath(path: string, exts?: string[]) {
  const res = await fetch(`${API_BASE}/list-path`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path, exts }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`list-path failed: ${res.status} ${text}`);
  }
  return res.json();
}

// DataFrame ops
export async function dfSelect(body: { rows: any[]; include?: Record<string, any[]>; exclude?: Record<string, any[]>; }) {
  const res = await fetch(`${API_BASE}/df/select`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function dfGroupPreview(body: { rows: any[]; by: string; numeric_cols?: string[]; }) {
  const url = `${API_BASE}/df/groupby/preview`;
  console.log('🔴 dfGroupPreview: Making request to:', url);
  console.log('🔴 dfGroupPreview: Request body:', { by: body.by, rows_count: body.rows.length, numeric_cols: body.numeric_cols });
  
  const res = await fetch(url, { 
    method: 'POST', 
    headers: { 'Content-Type': 'application/json' }, 
    body: JSON.stringify(body) 
  });
  
  console.log('🔴 dfGroupPreview: Response status:', res.status, res.statusText);
  
  if (!res.ok) {
    const errorText = await res.text();
    console.log('🔴 dfGroupPreview: Error response:', errorText);
    throw new Error(errorText);
  }
  
  const result = await res.json();
  console.log('🔴 dfGroupPreview: Success response:', result);
  return result;
}

export async function dfGroupRows(body: { rows: any[]; by: string; value: any; page?: number; page_size?: number; }) {
  const res = await fetch(`${API_BASE}/df/groupby/rows`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function dfCustom(body: { rows: any[]; code: string; }) {
  const res = await fetch(`${API_BASE}/df/custom`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}


