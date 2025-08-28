import React, { useState, useCallback, useMemo } from "react";
import { Box, AppBar, Toolbar, Typography, Container, Button, Drawer, Stack, Divider, TextField, Autocomplete, Chip, FormControlLabel, Switch, Accordion, AccordionSummary, AccordionDetails, Pagination } from "@mui/material";
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { detectAndValidate, dfSelect, dfGroupPreview, dfGroupRows, dfCustom } from "./lib/api";
import { flattenScores } from "./lib/normalize";
import { parseFile } from "./lib/parse";
import { detectMethodFromColumns, ensureOpenAIFormat } from "./lib/traces";
import DataTable from "./components/DataTable";
import ConversationTrace from "./components/ConversationTrace";
import SideBySideTrace from "./components/SideBySideTrace";

// Memoized component to prevent re-renders affecting TextField responsiveness
const CustomCodeInput = React.memo(function CustomCodeInput({
  value,
  onChange,
  onRun,
  onReset,
  error,
}: {
  value: string;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onRun: () => void;
  onReset: () => void;
  error: string | null;
}) {
  return (
    <>
      <Stack direction="row" spacing={1} alignItems="center" sx={{ flex: 1 }}>
        <TextField 
          size="small" 
          fullWidth 
          placeholder={`pandas expression (returns DataFrame), e.g., df.query("model==\"gpt-4\"")`} 
          value={value} 
          onChange={onChange} 
        />
        <Button variant="outlined" onClick={onRun}>Run</Button>
        <Button variant="text" onClick={onReset}>Reset</Button>
      </Stack>
      {error && <Box sx={{ color: '#b91c1c', mt: 1, fontSize: 12 }}>{error}</Box>}
    </>
  );
});

function App() {
  const [rows, setRows] = useState<Record<string, any>[]>([]);
  const [originalRows, setOriginalRows] = useState<Record<string, any>[]>([]);
  const [columns, setColumns] = useState<string[]>([]);
  const [method, setMethod] = useState<"single_model" | "side_by_side" | "unknown">("unknown");
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [selectedTrace, setSelectedTrace] = useState<any>(null);
  // Remote-path loading removed for now

  async function onFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    const { rows, columns } = await parseFile(file);
    const detected = detectMethodFromColumns(columns);
    setMethod(detected);
    // Flatten scores on the client (defensive in case backend preview doesn't)
    const { rows: normRows, columns: normCols } = flattenScores(rows, detected);
    setRows(normRows);
    setOriginalRows(normRows);
    setColumns(normCols);
    try {
      await detectAndValidate(file); // optional backend validation
    } catch (_) {}
  }

  const onView = useCallback((row: Record<string, any>) => {
    if (method === "single_model") {
      const messages = ensureOpenAIFormat(String(row?.["prompt"] ?? ""), row?.["model_response"]);
      setSelectedTrace({ type: "single", messages });
    } else if (method === "side_by_side") {
      const prompt = String(row?.["prompt"] ?? "");
      const messagesA = ensureOpenAIFormat(prompt, row?.["model_a_response"]);
      const messagesB = ensureOpenAIFormat(prompt, row?.["model_b_response"]);
      setSelectedTrace({
        type: "sbs",
        messagesA,
        messagesB,
        modelA: String(row?.["model_a"] ?? "Model A"),
        modelB: String(row?.["model_b"] ?? "Model B"),
      });
    }
    setDrawerOpen(true);
  }, [method]);

  const responseKeys = useMemo(() => 
    method === "single_model"
      ? ["model_response"]
      : method === "side_by_side"
        ? ["model_a_response", "model_b_response"]
        : [],
    [method]
  );

  const allowedColumns = useMemo(() => {
    const scoreCols = method === 'single_model'
      ? columns.filter(c => c.startsWith('score_'))
      : method === 'side_by_side'
        ? columns.filter(c => c.startsWith('score_a_') || c.startsWith('score_b_'))
        : [];
    return method === 'single_model'
      ? ['prompt', 'model', ...scoreCols, 'model_response']
      : method === 'side_by_side'
        ? ['prompt', 'model_a', 'model_b', ...scoreCols, 'model_a_response', 'model_b_response']
        : [];
  }, [method, columns]);

  // Memoize expensive data overview calculations
  const dataOverview = useMemo(() => {
    if (rows.length === 0) return null;
    const uniquePrompts = new Set(rows.map(r => r?.prompt)).size;
    let uniqueModels = 0;
    if (method === 'single_model') {
      uniqueModels = new Set(rows.map(r => r?.model)).size;
    } else if (method === 'side_by_side') {
      uniqueModels = new Set([
        ...rows.map(r => r?.model_a || ''),
        ...rows.map(r => r?.model_b || '')
      ]).size;
    }
    return {
      rowCount: rows.length.toLocaleString(),
      uniquePrompts: uniquePrompts.toLocaleString(),
      uniqueModels: uniqueModels.toLocaleString(),
    };
  }, [rows, method]);

  // -------- Data Ops State ---------
  type Filter = { column: string; values: string[]; negated: boolean };
  const [filters, setFilters] = useState<Filter[]>([]);
  const [pendingColumn, setPendingColumn] = useState<string | null>(null);
  const [pendingValues, setPendingValues] = useState<string[]>([]);
  const [pendingNegated, setPendingNegated] = useState<boolean>(false);

  const categoricalColumns = useMemo(() => {
    if (rows.length === 0) return [] as string[];
    const cols = new Set<string>();
    for (const c of columns) {
      const uniq = new Set(rows.slice(0, 500).map(r => r?.[c])).size;
      if (uniq > 0 && uniq <= 50) cols.add(c);
    }
    return Array.from(cols);
  }, [rows, columns]);

  const uniqueValuesFor = useMemo(() => {
    const cache = new Map<string, string[]>();
    return (col: string) => {
      if (cache.has(col)) return cache.get(col)!;
      const s = new Set<string>();
      rows.forEach(r => { const v = r?.[col]; if (v !== undefined && v !== null) s.add(String(v)); });
      const result = Array.from(s).sort();
      cache.set(col, result);
      return result;
    };
  }, [rows]);

  const applyFilters = useCallback(async (newFilters: Filter[]) => {
    setFilters(newFilters);
    const include = Object.fromEntries(newFilters.filter(f => !f.negated && f.values.length).map(f => [f.column, f.values]));
    const exclude = Object.fromEntries(newFilters.filter(f => f.negated && f.values.length).map(f => [f.column, f.values]));
    // Fast path: do it locally first to keep UI responsive
    const locallyFiltered = originalRows.filter(r => {
      for (const [col, vals] of Object.entries(include)) {
        if (!vals.includes(String(r[col]))) return false;
      }
      for (const [col, vals] of Object.entries(exclude)) {
        if (vals.includes(String(r[col]))) return false;
      }
      return true;
    });
    setRows(locallyFiltered);
    // Fire-and-forget server-side (for consistency with backend semantics)
    try {
      const res = await dfSelect({ rows: originalRows, include, exclude });
      if (Array.isArray(res.rows)) setRows(res.rows);
    } catch (e) { /* ignore to keep UI snappy */ }
  }, [originalRows]);

  const resetAll = useCallback(() => {
    setRows(originalRows);
    setFilters([]);
    setGroupBy(null);
    setGroupPreview([]);
    setExpandedGroup(null);
    setCustomCode("");
  }, [originalRows]);

  // -------- GroupBy State ---------
  const [groupBy, setGroupBy] = useState<string | null>(null);
  const [groupPreview, setGroupPreview] = useState<{ value: any; count: number; means: Record<string, number> }[]>([]);
  const [expandedGroup, setExpandedGroup] = useState<any | null>(null);
  const [groupPage, setGroupPage] = useState<number>(1);
  const [groupRows, setGroupRows] = useState<Record<string, any>[]>([]);
  const [groupTotal, setGroupTotal] = useState<number>(0);

  const numericCols = useMemo(() => {
    if (rows.length === 0) return [] as string[];
    const first = rows[0];
    return columns.filter(c => typeof first?.[c] === 'number');
  }, [rows, columns]);

  async function refreshGroupPreview(by: string) {
    try {
      const res = await dfGroupPreview({ rows, by, numeric_cols: numericCols });
      setGroupPreview(res.groups || []);
    } catch (e) { console.error(e); }
  }

  async function loadGroupRows(value: any, page = 1) {
    if (!groupBy) return;
    try {
      const res = await dfGroupRows({ rows, by: groupBy, value, page, page_size: 10 });
      setGroupRows(res.rows || []);
      setGroupTotal(res.total || 0);
      setGroupPage(page);
    } catch (e) { console.error(e); }
  }

  // -------- Custom Code ---------
  const [customCode, setCustomCode] = useState<string>("");
  const [customError, setCustomError] = useState<string | null>(null);
  
  const handleCustomCodeChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setCustomCode(e.target.value);
  }, []);
  
  const runCustom = useCallback(async () => {
    try {
      const res = await dfCustom({ rows, code: customCode });
      if (res.error) { setCustomError(res.error); return; }
      setCustomError(null);
      setRows(res.rows || []);
    } catch (e: any) {
      setCustomError(String(e?.message || e));
    }
  }, [rows, customCode]);

  return (
    <Box>
      <AppBar position="fixed">
        <Toolbar sx={{ gap: 2 }}>
          <Typography variant="h6" sx={{ flexGrow: 1 }}>StringSight · Evaluation Console</Typography>
          <Stack direction="row" spacing={1} alignItems="center">
            <Button component="label" variant="contained" color="primary">
              Load File
              <input type="file" hidden accept=".jsonl,.json,.csv" onChange={onFileChange} />
            </Button>
          </Stack>
        </Toolbar>
      </AppBar>
      {/* offset for fixed AppBar */}
      <Box sx={{ height: (theme) => theme.mixins.toolbar.minHeight }} />
      <Container maxWidth={false} sx={{ py: 2 }}>
        {dataOverview && (
          <Box sx={{ mb: 2, color: 'text.secondary' }}>
            <strong>{dataOverview.rowCount}</strong> rows ·{' '}
            <strong>{dataOverview.uniquePrompts}</strong> unique prompts ·{' '}
            <strong>{dataOverview.uniqueModels}</strong> unique models
          </Box>
        )}

        {/* Data Ops bar */}
        {rows.length > 0 && (
          <Box sx={{ p: 1.5, border: '1px solid #E5E7EB', borderRadius: 2, background: '#FFFFFF', mb: 2 }}>
            <Stack direction={{ xs: 'column', md: 'row' }} spacing={1} alignItems={{ xs: 'stretch', md: 'center' }}>
              {/* Filters */}
              <Stack direction="row" spacing={1} alignItems="center" sx={{ flexWrap: 'wrap' }}>
                <Autocomplete
                  size="small"
                  sx={{ minWidth: 220 }}
                  options={categoricalColumns}
                  value={pendingColumn}
                  onChange={(_, v) => { setPendingColumn(v); setPendingValues([]); setPendingNegated(false); }}
                  renderInput={(params) => <TextField {...params} label="Add filter (column)" />}
                />
                {pendingColumn && (
                  <Autocomplete
                    multiple size="small"
                    sx={{ minWidth: 260 }}
                    options={uniqueValuesFor(pendingColumn)}
                    value={pendingValues}
                    onChange={(_, v) => setPendingValues(v)}
                    renderTags={(value, getTagProps) => value.map((option, index) => (
                      <Chip {...getTagProps({ index })} key={option} label={option} />
                    ))}
                    renderInput={(params) => <TextField {...params} label="Values" />}
                  />
                )}
                {pendingColumn && (
                  <FormControlLabel control={<Switch checked={pendingNegated} onChange={(_, c) => setPendingNegated(c)} />} label="NOT" />
                )}
                <Button
                  variant="outlined"
                  disabled={!pendingColumn || pendingValues.length === 0}
                  onClick={() => {
                    if (!pendingColumn || pendingValues.length === 0) return;
                    const next = [...filters, { column: pendingColumn, values: pendingValues, negated: pendingNegated }];
                    setPendingColumn(null); setPendingValues([]); setPendingNegated(false);
                    void applyFilters(next);
                  }}
                >Add Filter</Button>
                {filters.map((f, i) => (
                  <Chip key={`${f.column}-${i}`} label={`${f.column}: ${f.negated ? 'NOT ' : ''}${f.values.join(', ')}`} onDelete={() => void applyFilters(filters.filter((_, idx) => idx !== i))} />
                ))}
              </Stack>

              <Divider orientation="vertical" flexItem sx={{ display: { xs: 'none', md: 'block' } }} />

              {/* Groupby */}
              <Stack direction="row" spacing={1} alignItems="center">
                <Autocomplete
                  size="small"
                  sx={{ minWidth: 220 }}
                  options={columns}
                  value={groupBy}
                  onChange={(_, v) => { setGroupBy(v); setExpandedGroup(null); setGroupRows([]); if (v) refreshGroupPreview(v); else setGroupPreview([]); }}
                  renderInput={(params) => <TextField {...params} label="Group by" />}
                />
              </Stack>

              <Divider orientation="vertical" flexItem sx={{ display: { xs: 'none', md: 'block' } }} />

              {/* Custom */}
              <CustomCodeInput
                value={customCode}
                onChange={handleCustomCodeChange}
                onRun={runCustom}
                onReset={resetAll}
                error={customError}
              />
            </Stack>
          </Box>
        )}

        {/* Group Preview */}
        {groupBy && groupPreview.length > 0 && (
          <Box sx={{ mb: 2 }}>
            {groupPreview.map((g) => (
              <Accordion key={String(g.value)} expanded={expandedGroup === g.value} onChange={(_, isExp) => { setExpandedGroup(isExp ? g.value : null); if (isExp) loadGroupRows(g.value, 1); }}>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Stack direction={{ xs: 'column', md: 'row' }} spacing={2} sx={{ width: '100%', alignItems: { md: 'center' } }}>
                    <Box sx={{ fontWeight: 600 }}>{String(g.value)}</Box>
                    <Box sx={{ color: 'text.secondary' }}>count: {g.count}</Box>
                    <Box sx={{ color: 'text.secondary', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {Object.entries(g.means).slice(0, 6).map(([k, v]) => `${k}=${(v as number).toFixed(3)}`).join(' · ')}
                    </Box>
                  </Stack>
                </AccordionSummary>
                <AccordionDetails>
                  <Box sx={{ mb: 1 }}>
                    <Pagination count={Math.max(1, Math.ceil(groupTotal / 10))} page={groupPage} onChange={(_, p) => loadGroupRows(g.value, p)} size="small" />
                  </Box>
                  <DataTable
                    rows={groupRows}
                    columns={columns}
                    responseKeys={responseKeys}
                    onView={onView}
                    allowedColumns={allowedColumns}
                  />
                </AccordionDetails>
              </Accordion>
            ))}
          </Box>
        )}
        {useMemo(() => 
          rows.length > 0 ? (
            <DataTable
              rows={rows}
              columns={columns}
              responseKeys={responseKeys}
              onView={onView}
              allowedColumns={allowedColumns}
            />
          ) : null,
          [rows, columns, responseKeys, onView, allowedColumns]
        )}
      </Container>
      <Drawer anchor="right" open={drawerOpen} onClose={() => setDrawerOpen(false)} sx={{ '& .MuiDrawer-paper': { width: '50vw', maxWidth: 900, p: 2 } }} ModalProps={{ keepMounted: true, disableRestoreFocus: true }}>
        {selectedTrace?.type === "single" && (
          <ConversationTrace messages={selectedTrace.messages} />
        )}
        {selectedTrace?.type === "sbs" && (
          <SideBySideTrace messagesA={selectedTrace.messagesA} messagesB={selectedTrace.messagesB} modelA={selectedTrace.modelA} modelB={selectedTrace.modelB} />
        )}
      </Drawer>
    </Box>
  );
}

export default App
