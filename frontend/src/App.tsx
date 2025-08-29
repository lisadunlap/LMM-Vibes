import React, { useState, useCallback, useMemo, Component } from "react";
import { Box, AppBar, Toolbar, Typography, Container, Button, Drawer, Stack, Divider, TextField, Autocomplete, Chip, FormControlLabel, Switch, Accordion, AccordionSummary, AccordionDetails, Pagination, Table, TableBody, TableCell, TableContainer, TableHead, TableRow } from "@mui/material";
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import VisibilityOutlinedIcon from '@mui/icons-material/VisibilityOutlined';
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
    
    const operationalData = normRows.map((row, index) => ({
      __index: index, // Add original dataframe index
      ...Object.fromEntries(allowedCols.filter(col => col in row).map(col => [col, row[col]]))
    }));
    
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

  // Get allowed columns from operational data with proper ordering
  const allowedColumns = useMemo(() => {
    if (operationalRows.length === 0) return [];
    const allColumns = Object.keys(operationalRows[0]);
    
    // Order: index → prompt → response columns → remaining
    const indexCol = allColumns.filter((c) => c === '__index');
    const promptFirst = allColumns.filter((c) => c === 'prompt');
    const resp = allColumns.filter((c) => responseKeys.includes(c));
    const remaining = allColumns.filter((c) => c !== '__index' && c !== 'prompt' && !responseKeys.includes(c));
    
    return [...indexCol, ...promptFirst, ...resp, ...remaining];
  }, [operationalRows, responseKeys]);

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
      // Skip index column - it's not categorical
      if (c === '__index') continue;
      const uniq = new Set(operationalRows.slice(0, 500).map(r => r?.[c])).size;
      if (uniq > 0 && uniq <= 50) cols.add(c);
    }
    return Array.from(cols);
  }, [operationalRows, allowedColumns]);

  const numericCols = useMemo(() => {
    if (operationalRows.length === 0) return [] as string[];
    const first = operationalRows[0];
    return allowedColumns.filter(c => typeof first?.[c] === 'number' || c === '__index');
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
  const [groupPagination, setGroupPagination] = useState<Map<string, number>>(new Map());

  // Truncated Cell component for grouped view
  const TruncatedCell = React.memo(function TruncatedCell({ text }: { text: string }) {
    const [expanded, setExpanded] = React.useState(false);
    const MAX_LEN = 200;
    if (!expanded && text.length > MAX_LEN) {
      return (
        <span>
          {text.slice(0, MAX_LEN)}…{' '}
          <Button size="small" variant="text" onClick={() => setExpanded(true)}>Expand</Button>
        </span>
      );
    }
    if (expanded && text.length > MAX_LEN) {
      return (
        <span>
          {text}{' '}
          <Button size="small" variant="text" onClick={() => setExpanded(false)}>Collapse</Button>
        </span>
      );
    }
    return <span>{text}</span>;
  });

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
    <Box sx={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
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
      <Container maxWidth={false} sx={{ py: 2, flexGrow: 1, display: 'flex', flexDirection: 'column', alignItems: 'stretch' }}>
        {dataOverview && (
          <Box sx={{ mb: 2, color: 'text.secondary' }}>
            <strong>{dataOverview.rowCount}</strong> rows ·{' '}
            <strong>{dataOverview.uniquePrompts}</strong> unique prompts ·{' '}
            <strong>{dataOverview.uniqueModels}</strong> unique models
          </Box>
        )}

        {/* Data Ops bar - fixed width, no horizontal scroll */}
        {currentRows.length > 0 && (
          <Box sx={{ 
            p: 1.5, 
            border: '1px solid #E5E7EB', 
            borderRadius: 2, 
            background: '#FFFFFF', 
            mb: 1,
            width: '100%',
            maxWidth: '100vw',
            overflow: 'hidden'
          }}>
            <Stack direction={{ xs: 'column', lg: 'row' }} spacing={1} alignItems={{ xs: 'stretch', lg: 'center' }}>
              {/* Filters */}
              <Stack direction="row" spacing={1} alignItems="center" sx={{ flexWrap: 'wrap', flex: 1, minWidth: 0 }}>
                <Autocomplete
                  size="small"
                  sx={{ minWidth: 180, maxWidth: 220, flex: '0 1 auto' }}
                  options={categoricalColumns}
                  value={pendingColumn}
                  onChange={(_, v) => { setPendingColumn(v); setPendingValues([]); setPendingNegated(false); }}
                  renderInput={(params) => <TextField {...params} label="Add filter (column)" />}
                />
                {pendingColumn && (
                  <Autocomplete
                    multiple size="small"
                    sx={{ minWidth: 200, maxWidth: 300, flex: '0 1 auto' }}
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
                  sx={{ minWidth: 160, maxWidth: 220, flex: '0 1 auto' }}
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

        {useMemo(() => {
          if (currentRows.length === 0) return null;
          
          // If groupBy is active, show accordion view that looks like table rows
          if (groupBy && groupPreview.length > 0) {
            // Group the current rows by the selected column
            const groupedRows = new Map<any, any[]>();
            currentRows.forEach(row => {
              const key = row[groupBy];
              if (!groupedRows.has(key)) groupedRows.set(key, []);
              groupedRows.get(key)!.push(row);
            });
            
            return (
              <Box sx={{ border: '1px solid #E5E7EB', borderRadius: 2, overflow: 'auto', backgroundColor: '#FFFFFF' }}>
                {/* Table Header - only shown once at the top */}
                <Box sx={{ backgroundColor: '#F3F4F6', p: 2, borderBottom: '1px solid #E5E7EB' }}>
                  <Box sx={{ display: 'grid', gridTemplateColumns: `auto repeat(${allowedColumns.length}, 1fr)`, gap: 2, alignItems: 'center' }}>
                    <Box sx={{ width: 24 }} /> {/* Space for arrow */}
                    {allowedColumns.map(col => (
                      <Typography key={col} variant="subtitle2" sx={{ color: '#374151', fontWeight: 700, fontSize: 12, letterSpacing: 0.4, textTransform: 'uppercase' }}>
                        {col === '__index' ? 'INDEX' :
                         col === 'prompt' ? 'PROMPT' : 
                         col === 'model' ? 'MODEL' : 
                         col === 'model_response' ? 'RESPONSE' :
                         col === 'model_a' ? 'MODEL A' :
                         col === 'model_b' ? 'MODEL B' :
                         col === 'model_a_response' ? 'RESPONSE A' :
                         col === 'model_b_response' ? 'RESPONSE B' :
                         col.toUpperCase()}
                      </Typography>
                    ))}
                  </Box>
                </Box>
                
                {/* Grouped Rows */}
                {Array.from(groupedRows.entries()).map(([groupValue, rows]) => {
                  const groupKey = String(groupValue);
                  const currentPage = groupPagination.get(groupKey) || 1;
                  const pageSize = 10;
                  const paginatedRows = rows.slice((currentPage - 1) * pageSize, currentPage * pageSize);
                  const totalPages = Math.ceil(rows.length / pageSize);
                  
                  const handlePageChange = (page: number) => {
                    setGroupPagination(prev => new Map(prev).set(groupKey, page));
                  };
                  
                  return (
                    <Accordion key={String(groupValue)} sx={{ '&:before': { display: 'none' }, boxShadow: 'none', border: 'none' }}>
                      <AccordionSummary 
                        expandIcon={null}
                        sx={{ 
                          backgroundColor: '#F9FAFB',
                          borderBottom: '1px solid #E5E7EB',
                          minHeight: 48,
                          '&.Mui-expanded': { minHeight: 48 },
                          '& .MuiAccordionSummary-content': { margin: '12px 0' },
                          cursor: 'pointer',
                          '&:hover': { backgroundColor: '#F3F4F6' }
                        }}
                      >
                        <Box sx={{ display: 'grid', gridTemplateColumns: `auto repeat(${allowedColumns.length}, 1fr)`, gap: 2, alignItems: 'center', width: '100%' }}>
                            <ExpandMoreIcon sx={{ fontSize: 20, color: '#6B7280' }} />
                            
                            {allowedColumns.map((col, idx) => {
                              if (idx === 0) {
                                // First column: Show group value with count bubble
                                return (
                                  <Box key={col} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                    <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                                      {String(groupValue).length > 50 ? String(groupValue).slice(0, 50) + '...' : String(groupValue)}
                                    </Typography>
                                    <Typography variant="body2" sx={{ 
                                      backgroundColor: '#E0E7FF', 
                                      color: '#3730A3', 
                                      px: 1.5, 
                                      py: 0.25, 
                                      borderRadius: 9999,
                                      fontSize: 11,
                                      fontWeight: 500,
                                      textAlign: 'center',
                                      minWidth: 20
                                    }}>
                                      {rows.length}
                                    </Typography>
                                  </Box>
                                );
                              } else if (responseKeys.includes(col)) {
                                // Response columns: Empty
                                return <Box key={col} />;
                              } else {
                                // Other columns: Show mean if it's numeric
                                const groupStats = groupPreview.find(g => g.value === groupValue);
                                const mean = groupStats?.means[col];
                                if (typeof mean === 'number') {
                                  return (
                                    <Typography key={col} variant="body2" sx={{ 
                                      color: '#6B7280', 
                                      fontStyle: 'italic',
                                      fontSize: 13
                                    }}>
                                      avg: {mean.toFixed(2)}
                                    </Typography>
                                  );
                                }
                                return <Box key={col} />;
                              }
                            })}
                          </Box>
                      </AccordionSummary>
                      <AccordionDetails sx={{ p: 0 }}>
                        {/* Show paginated rows without header */}
                        <Box>
                          {paginatedRows.map((row, idx) => (
                            <Box key={idx} sx={{ display: 'grid', gridTemplateColumns: `auto repeat(${allowedColumns.length}, 1fr)`, gap: 2, alignItems: 'center', p: 2, borderBottom: idx < paginatedRows.length - 1 ? '1px solid #E5E7EB' : 'none' }}>
                              <Box sx={{ width: 24 }} /> {/* Space for arrow alignment */}
                              {allowedColumns.map(col => (
                                <Box key={col}>
                                  {responseKeys.includes(col) ? (
                                    <Button
                                      size="small"
                                      variant="text"
                                      color="secondary"
                                      startIcon={<VisibilityOutlinedIcon />}
                                      onClick={() => onView(row)}
                                      sx={{ fontWeight: 600 }}
                                    >
                                      View
                                    </Button>
                                  ) : (
                                    <Typography variant="body2" sx={{ 
                                      maxWidth: 200
                                    }}>
                                      <TruncatedCell text={String(row[col] || '')} />
                                    </Typography>
                                  )}
                                </Box>
                              ))}
                            </Box>
                          ))}
                          
                          {/* Pagination for this group */}
                          {totalPages > 1 && (
                            <Box sx={{ p: 2, display: 'flex', justifyContent: 'center', borderTop: '1px solid #E5E7EB' }}>
                              <Pagination 
                                count={totalPages} 
                                page={currentPage} 
                                onChange={(_, page) => handlePageChange(page)} 
                                size="small" 
                              />
                            </Box>
                          )}
                        </Box>
                      </AccordionDetails>
                    </Accordion>
                  );
                })}
              </Box>
            );
          }
          
          // Normal flat table view when no groupBy
          return (
            <DataTable
              rows={currentRows}
              columns={allowedColumns}
              responseKeys={responseKeys}
              onView={onView}
              allowedColumns={allowedColumns}
            />
          );
        }, [currentRows, allowedColumns, responseKeys, onView, groupBy, groupPreview, groupPagination])}
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
