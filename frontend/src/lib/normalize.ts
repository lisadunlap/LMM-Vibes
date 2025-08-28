export type Method = "single_model" | "side_by_side" | "unknown";

// Flatten score dictionaries into scalar columns: score_* (single), score_a_*, score_b_* (sbs)
export function flattenScores(rows: Record<string, any>[], method: Method) {
  const out = rows.map((r) => ({ ...r }));

  function flattenField(field: string, prefix: string) {
    const keySet = new Set<string>();
    for (const row of out) {
      const val = row[field];
      if (val && typeof val === "object" && !Array.isArray(val)) {
        for (const k of Object.keys(val)) keySet.add(k);
      }
    }
    if (keySet.size === 0) return;
    for (const row of out) {
      const val = row[field] || {};
      for (const k of keySet) {
        const col = `${prefix}_${k}`;
        const v = val && typeof val === "object" ? val[k] : undefined;
        (row as any)[col] = v;
      }
      delete (row as any)[field];
    }
  }

  if (method === "single_model") {
    if (out.some((r) => typeof r?.score === "object")) {
      flattenField("score", "score");
    }
  } else if (method === "side_by_side") {
    if (out.some((r) => typeof r?.score_a === "object")) {
      flattenField("score_a", "score_a");
    }
    if (out.some((r) => typeof r?.score_b === "object")) {
      flattenField("score_b", "score_b");
    }
  }

  const columns = Array.from(
    out.reduce((set, r) => {
      Object.keys(r).forEach((k) => set.add(k));
      return set;
    }, new Set<string>())
  );

  return { rows: out, columns };
}


