import React, { useState, useCallback, useMemo, Component } from "react";
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

class ErrorBoundary extends Component<{children: React.ReactNode}, {hasError: boolean, error?: Error}> {
  constructor(props: {children: React.ReactNode}) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: any) {
    console.error('Error caught by boundary:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <Box sx={{ p: 4, textAlign: 'center' }}>
          <Typography variant="h4" color="error" gutterBottom>
            Something went wrong
          </Typography>
          <Typography variant="body1" sx={{ mb: 2 }}>
            {this.state.error?.message || 'An unexpected error occurred'}
          </Typography>
          <Button variant="contained" onClick={() => window.location.reload()}>
            Reload Page
          </Button>
        </Box>
      );
    }

    return <>{this.props.children}</>;
  }
}

function App() {
  // Data management layers as suggested
  const [originalRows, setOriginalRows] = useState<Record<string, any>[]>([]); // Raw uploaded data
  const [operationalRows, setOperationalRows] = useState<Record<string, any>[]>([]); // Cleaned, filtered columns
  const [currentRows, setCurrentRows] = useState<Record<string, any>[]>([]); // With filters applied
  
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
    
    // Set data layers
    setOriginalRows(normRows); // Raw uploaded data
    
    // Create operational data (only allowed columns)
    const scoreCols = detected === 'single_model'
      ? normCols.filter(c => c.startsWith('score_'))
      : detected === 'side_by_side'
        ? normCols.filter(c => c.startsWith('score_a_') || c.startsWith('score_b_'))
        : [];
    const allowedCols = detected === 'single_model'
      ? ['prompt', 'model', ...scoreCols, 'model_response']
      : detected === 'side_by_side'
        ? ['prompt', 'model_a', 'model_b', ...scoreCols, 'model_a_response', 'model_b_response']
        : normCols;
    
    const operationalData = normRows.map(row => 
      Object.fromEntries(allowedCols.filter(col => col in row).map(col => [col, row[col]]))
    );
    
    setOperationalRows(operationalData);
    setCurrentRows(operationalData); // Start with operational data
    
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

  // Get allowed columns from operational data
  const allowedColumns = useMemo(() => {
    if (operationalRows.length === 0) return [];
    return Object.keys(operationalRows[0]);
  }, [operationalRows]);

  // Memoize expensive data overview calculations using current data
  const dataOverview = useMemo(() => {
    if (currentRows.length === 0) return null;
    const uniquePrompts = new Set(currentRows.map(r => r?.prompt)).size;
    let uniqueModels = 0;
    if (method === 'single_model') {
      uniqueModels = new Set(currentRows.map(r => r?.model)).size;
    } else if (method === 'side_by_side') {
      uniqueModels = new Set([
        ...currentRows.map(r => r?.model_a || ''),
        ...currentRows.map(r => r?.model_b || '')
      ]).size;
    }
    return {
      rowCount: currentRows.length.toLocaleString(),
      uniquePrompts: uniquePrompts.toLocaleString(),
      uniqueModels: uniqueModels.toLocaleString(),
    };
  }, [currentRows, method]);

  // -------- Data Ops State ---------
  type Filter = { column: string; values: string[]; negated: boolean };
  const [filters, setFilters] = useState<Filter[]>([]);
  const [pendingColumn, setPendingColumn] = useState<string | null>(null);
  const [pendingValues, setPendingValues] = useState<string[]>([]);
  const [pendingNegated, setPendingNegated] = useState<boolean>(false);

  const categoricalColumns = useMemo(() => {
    if (operationalRows.length === 0) return [] as string[];
    const cols = new Set<string>();
    for (const c of allowedColumns) {
      const uniq = new Set(operationalRows.slice(0, 500).map(r => r?.[c])).size;
      if (uniq > 0 && uniq <= 50) cols.add(c);
    }
    return Array.from(cols);
  }, [operationalRows, allowedColumns]);

  const numericCols = useMemo(() => {
    if (operationalRows.length === 0) return [] as string[];
    const first = operationalRows[0];
    return allowedColumns.filter(c => typeof first?.[c] === 'number');
  }, [operationalRows, allowedColumns]);

  const uniqueValuesFor = useMemo(() => {
    const cache = new Map<string, string[]>();
    return (col: string) => {
      if (cache.has(col)) return cache.get(col)!;
      const s = new Set<string>();
      operationalRows.forEach(r => { const v = r?.[col]; if (v !== undefined && v !== null) s.add(String(v)); });
      const result = Array.from(s).sort();
      cache.set(col, result);
      return result;
    };
  }, [operationalRows]);

  const applyFilters = useCallback(async (newFilters: Filter[]) => {
    setFilters(newFilters);
    const include = Object.fromEntries(newFilters.filter(f => !f.negated && f.values.length).map(f => [f.column, f.values]));
    const exclude = Object.fromEntries(newFilters.filter(f => f.negated && f.values.length).map(f => [f.column, f.values]));
    // Fast path: filter operational data locally first
    const locallyFiltered = operationalRows.filter(r => {
      for (const [col, vals] of Object.entries(include)) {
        if (!vals.includes(String(r[col]))) return false;
      }
      for (const [col, vals] of Object.entries(exclude)) {
        if (vals.includes(String(r[col]))) return false;
      }
      return true;
    });
    setCurrentRows(locallyFiltered);
    // Optional backend validation (using smaller operational dataset)
    try {
      const res = await dfSelect({ rows: operationalRows, include, exclude });
      if (Array.isArray(res.rows)) setCurrentRows(res.rows);
    } catch (e) { /* ignore to keep UI snappy */ }
  }, [operationalRows]);

  const resetAll = useCallback(() => {
    setCurrentRows(operationalRows); // Reset to operational data, not original
    setFilters([]);
    setGroupBy(null);
    setGroupPreview([]);
    setExpandedGroup(null);
    setCustomCode("");
  }, [operationalRows]);

  // -------- GroupBy State ---------
  const [groupBy, setGroupBy] = useState<string | null>(null);
  const [groupPreview, setGroupPreview] = useState<{ value: any; count: number; means: Record<string, number> }[]>([]);
  const [expandedGroup, setExpandedGroup] = useState<any | null>(null);
  const [groupPage, setGroupPage] = useState<number>(1);
  const [groupRows, setGroupRows] = useState<Record<string, any>[]>([]);
  const [groupTotal, setGroupTotal] = useState<number>(0);

  const refreshGroupPreview = useCallback(async (by: string) => {
    console.log('🟡 refreshGroupPreview called with:', by);
    console.log('🟡 operational rows length:', operationalRows.length, 'numericCols:', numericCols);
    
    // Local-first groupby (like filters)
    const grouped = new Map<any, any[]>();
    operationalRows.forEach(row => {
      const key = row[by];
      if (!grouped.has(key)) grouped.set(key, []);
      grouped.get(key)!.push(row);
    });
    
    const localGroups = Array.from(grouped.entries()).map(([value, rows]) => {
      const count = rows.length;
      const means: Record<string, number> = {};
      numericCols.forEach(col => {
        const nums = rows.map(r => Number(r[col])).filter(n => !isNaN(n));
        if (nums.length > 0) {
          means[col] = nums.reduce((sum, n) => sum + n, 0) / nums.length;
        }
      });
      return { value, count, means };
    });
    
    console.log('🟡 Local groupby result:', localGroups);
    setGroupPreview(localGroups);
    
    // Optional backend validation (fire-and-forget)
    try {
      console.log('🟡 Making API call to dfGroupPreview...');
      const res = await dfGroupPreview({ rows: operationalRows, by, numeric_cols: numericCols });
      console.log('🟡 dfGroupPreview response:', res);
      // Only update if backend gives different result
      if (JSON.stringify(res.groups) !== JSON.stringify(localGroups)) {
        setGroupPreview(res.groups || localGroups);
      }
    } catch (e) { 
      console.log('🟡 Backend failed, using local groupby result'); 
    }
  }, [operationalRows, numericCols]);

  const loadGroupRows = useCallback(async (value: any, page = 1) => {
    if (!groupBy) return;
    try {
      const res = await dfGroupRows({ rows: operationalRows, by: groupBy, value, page, page_size: 10 });
      setGroupRows(res.rows || []);
      setGroupTotal(res.total || 0);
      setGroupPage(page);
    } catch (e) { console.error(e); }
  }, [operationalRows, groupBy]);

  // -------- Custom Code ---------
  const [customCode, setCustomCode] = useState<string>("");
  const [customError, setCustomError] = useState<string | null>(null);
  
  const handleCustomCodeChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setCustomCode(e.target.value);
  }, []);
  
  const runCustom = useCallback(async () => {
    try {
      const res = await dfCustom({ rows: currentRows, code: customCode });
      if (res.error) { setCustomError(res.error); return; }
      setCustomError(null);
      setCurrentRows(res.rows || []);
    } catch (e: any) {
      console.error('runCustom error:', e);
      setCustomError(String(e?.message || e));
    }
  }, [currentRows, customCode]);

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
        {currentRows.length > 0 && (
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
                  options={allowedColumns}
                  value={groupBy}
                  onChange={(_, v) => { 
                    console.log('🔵 GroupBy onChange triggered with value:', v);
                    setGroupBy(v); 
                    setExpandedGroup(null); 
                    setGroupRows([]); 
                    if (v) {
                      console.log('🔵 Calling refreshGroupPreview with:', v);
                      refreshGroupPreview(v);
                    } else {
                      console.log('🔵 Clearing group preview');
                      setGroupPreview([]);
                    }
                  }}
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
                    columns={allowedColumns}
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
          currentRows.length > 0 ? (
            <DataTable
              rows={currentRows}
              columns={allowedColumns}
              responseKeys={responseKeys}
              onView={onView}
              allowedColumns={allowedColumns}
            />
          ) : null,
          [currentRows, allowedColumns, responseKeys, onView]
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

export default function AppWithErrorBoundary() {
  return (
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  );
}
