import React, { useState, useCallback, useMemo, Component } from "react";
import { Box, AppBar, Toolbar, Typography, Container, Button, Drawer, Stack, Divider, TextField, Autocomplete, Chip, FormControlLabel, Switch, Accordion, AccordionSummary, AccordionDetails, Pagination, Table, TableBody, TableCell, TableContainer, TableHead, TableRow } from "@mui/material";
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import VisibilityOutlinedIcon from '@mui/icons-material/VisibilityOutlined';
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward';
import ArrowDownwardIcon from '@mui/icons-material/ArrowDownward';
import { detectAndValidate, dfGroupPreview, dfGroupRows, dfCustom } from "./lib/api";
import { flattenScores } from "./lib/normalize";
import { parseFile } from "./lib/parse";
import { detectMethodFromColumns, ensureOpenAIFormat } from "./lib/traces";
import DataTable from "./components/DataTable";
import ConversationTrace from "./components/ConversationTrace";
import SideBySideTrace from "./components/SideBySideTrace";
import FormattedCell from "./components/FormattedCell";
import FilterSummary from "./components/FilterSummary";
import { ColumnSelector, type ColumnMapping } from "./components/ColumnSelector";
import type { DataOperation } from "./types/operations";
import { createFilterOperation, createCustomCodeOperation, createSortOperation } from "./types/operations";

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
  
  // Flexible column mapping state
  const [showColumnSelector, setShowColumnSelector] = useState(false);
  const [availableColumns, setAvailableColumns] = useState<string[]>([]);
  const [autoDetectedMapping, setAutoDetectedMapping] = useState<ColumnMapping | null>(null);
  const [columnMapping, setColumnMapping] = useState<ColumnMapping | null>(null);
  const [mappingValid, setMappingValid] = useState(false);
  const [mappingErrors, setMappingErrors] = useState<string[]>([]);
  const [filterNotice, setFilterNotice] = useState<string | null>(null);
  // Remote-path loading removed for now

  async function onFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    
    // Parse the file
    const { rows, columns } = await parseFile(file);
    
    // Store raw data and columns
    setOriginalRows(rows);
    setAvailableColumns(columns);
    setFilterNotice(null);
    
    // Try to auto-detect column mapping using legacy method as fallback
    const legacyDetected = detectMethodFromColumns(columns);
    
    // Create auto-detected mapping based on legacy detection and available columns
    const autoMapping: ColumnMapping = {
      promptCol: columns.find(c => c.toLowerCase() === 'prompt') || '',
      responseCols: legacyDetected === 'side_by_side' 
        ? columns.filter(c => c.includes('model_a_response') || c.includes('model_b_response'))
        : columns.filter(c => c.includes('model_response')),
      modelCols: legacyDetected === 'side_by_side'
        ? columns.filter(c => (c.includes('model_a') || c.includes('model_b')) && !c.includes('response'))
        : columns.filter(c => c.toLowerCase() === 'model'),
      scoreCols: columns.filter(c => c.toLowerCase().includes('score')),
      method: legacyDetected === 'unknown' ? 'single_model' : legacyDetected
    };
    
    setAutoDetectedMapping(autoMapping);
    
    // Always show the column selector after upload (clear, explicit flow for new users)
    setShowColumnSelector(true);
    setMethod('unknown');
    setOperationalRows([]);
    setCurrentRows([]);
    
    try {
      await detectAndValidate(file); // optional backend validation
    } catch (_) {}
  }

  // New function to process data with flexible column mapping
  function processDataWithMapping(rows: Record<string, any>[], mapping: ColumnMapping) {
    setMethod(mapping.method);
    setColumnMapping(mapping);
    setFilterNotice(null);

    // Create operational data using user-specified columns (first build standardized + score dicts)
    const mappedRows = rows.map((row, index) => {
      const opRow: Record<string, any> = { __index: index };
      
      // Map prompt column
      if (mapping.promptCol && row[mapping.promptCol] !== undefined) {
        opRow.prompt = row[mapping.promptCol];
      }
      
      // Map response columns
      if (mapping.method === 'single_model' && mapping.responseCols[0]) {
        opRow.model_response = row[mapping.responseCols[0]];
      } else if (mapping.method === 'side_by_side') {
        if (mapping.responseCols[0]) opRow.model_a_response = row[mapping.responseCols[0]];
        if (mapping.responseCols[1]) opRow.model_b_response = row[mapping.responseCols[1]];
      }
      
      // Map model columns
      if (mapping.method === 'single_model' && mapping.modelCols[0]) {
        opRow.model = row[mapping.modelCols[0]];
      } else if (mapping.method === 'side_by_side') {
        if (mapping.modelCols[0]) opRow.model_a = row[mapping.modelCols[0]];
        if (mapping.modelCols[1]) opRow.model_b = row[mapping.modelCols[1]];
      }
      
      // Build score dictionaries per selected columns (backend contract)
      if (mapping.scoreCols.length > 0) {
        const toNumber = (v: any): number | undefined => {
          if (typeof v === 'number') return Number.isFinite(v) ? v : undefined;
          if (typeof v === 'string' && v.trim() !== '') {
            const n = Number(v);
            return Number.isFinite(n) ? n : undefined;
          }
          return undefined;
        };
        
        const parseMaybeJsonDict = (v: any): Record<string, any> | null => {
          if (v && typeof v === 'object' && !Array.isArray(v)) return v as Record<string, any>;
          if (typeof v === 'string') {
            const s = v.trim();
            if (s.startsWith('{') && s.endsWith('}')) {
              try { return JSON.parse(s); } catch (_) { return null; }
            }
          }
          return null;
        };

        if (mapping.method === 'single_model') {
          const scoreDict: Record<string, number> = {};
          for (const col of mapping.scoreCols) {
            const raw = row[col];
            const asDict = parseMaybeJsonDict(raw);
            if (asDict) {
              for (const [k, v] of Object.entries(asDict)) {
                const num = toNumber(v);
                if (num !== undefined) scoreDict[k] = num;
              }
            } else {
              const key = col.replace(/^(score_)?/i, '').replace(/_?score$/i, '') || 'value';
              const num = toNumber(raw);
              if (num !== undefined) scoreDict[key] = num;
            }
          }
          if (Object.keys(scoreDict).length > 0) (opRow as any).score = scoreDict;
        } else {
          const scoreADict: Record<string, number> = {};
          const scoreBDict: Record<string, number> = {};
          for (const col of mapping.scoreCols) {
            const raw = row[col];
            const asDict = parseMaybeJsonDict(raw);
            const target = col.toLowerCase().includes('_b') ? scoreBDict
                          : col.toLowerCase().includes('_a') ? scoreADict
                          : null; // apply to both if null
            if (asDict) {
              for (const [k, v] of Object.entries(asDict)) {
                const num = toNumber(v);
                if (num !== undefined) {
                  if (target === scoreADict) scoreADict[k] = num;
                  else if (target === scoreBDict) scoreBDict[k] = num;
                  else { scoreADict[k] = num; scoreBDict[k] = num; }
                }
              }
            } else {
              const base = col.replace(/^(score_)?/i, '').replace(/_?score$/i, '').replace(/_a$/i, '').replace(/_b$/i, '') || 'value';
              const num = toNumber(raw);
              if (num !== undefined) {
                if (target === scoreADict) scoreADict[base] = num;
                else if (target === scoreBDict) scoreBDict[base] = num;
                else { scoreADict[base] = num; scoreBDict[base] = num; }
              }
            }
          }
          if (Object.keys(scoreADict).length > 0) (opRow as any).score_a = scoreADict;
          if (Object.keys(scoreBDict).length > 0) (opRow as any).score_b = scoreBDict;
        }
      }

      // Do not include any other columns to keep the operational data clean
      return opRow;
    });

    // Filter out rows where any selected score is missing
    let filteredCount = 0;
    const scoreCols = mapping.scoreCols || [];
    const hasScoresSelected = scoreCols.length > 0;
    const isMissing = (v: any) => v === undefined || v === null || (typeof v === 'string' && v.trim() === '') || (typeof v === 'number' && Number.isNaN(v)) || (v && typeof v === 'object' && !Array.isArray(v) && Object.keys(v).length === 0);
    const rowsAfterFilter = !hasScoresSelected ? mappedRows : mappedRows.filter((_, idx) => {
      const original = rows[idx];
      const missingAny = scoreCols.some(col => isMissing(original[col]));
      if (missingAny) filteredCount += 1;
      return !missingAny;
    });

    if (hasScoresSelected && filteredCount > 0) {
      setFilterNotice(`Filtered out ${filteredCount} row(s) due to missing values in selected score columns: ${scoreCols.join(', ')}`);
    }

    // Now flatten score dicts for display
    const { rows: flattenedRows } = flattenScores(rowsAfterFilter, mapping.method);

    setOperationalRows(flattenedRows);
    setCurrentRows(flattenedRows);
  }

  // Handle column mapping changes from the selector
  const handleMappingChange = useCallback((mapping: ColumnMapping) => {
    if (mappingValid && originalRows.length > 0) {
      processDataWithMapping(originalRows, mapping);
      setShowColumnSelector(false);
    }
  }, [mappingValid, originalRows]);

  // Handle validation changes from the selector
  const handleValidationChange = useCallback((isValid: boolean, errors: string[]) => {
    setMappingValid(isValid);
    setMappingErrors(errors);
  }, []);

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

  // -------- Data Operations Chain ---------
  const [operationChain, setOperationChain] = useState<DataOperation[]>([]);
  const [pendingColumn, setPendingColumn] = useState<string | null>(null);
  const [pendingValues, setPendingValues] = useState<string[]>([]);
  const [pendingNegated, setPendingNegated] = useState<boolean>(false);
  
  // Legacy filter interface for compatibility
  type Filter = { column: string; values: string[]; negated: boolean };
  const filters: Filter[] = operationChain
    .filter(op => op.type === 'filter')
    .map(op => op as any);

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

  // Apply the entire operation chain to operational data
  const applyOperationChain = useCallback(async (operations: DataOperation[]) => {
    console.log('🔄 Applying operation chain:', operations);
    
    if (operations.length === 0) {
      setCurrentRows(operationalRows);
      return;
    }
    
    let currentData = [...operationalRows];
    
    // Apply operations in sequence
    for (const operation of operations) {
      switch (operation.type) {
        case 'filter': {
          const filterOp = operation as any;
          currentData = currentData.filter(row => {
            const rowValue = String(row[filterOp.column] || '');
            const matchesValues = filterOp.values.includes(rowValue);
            return filterOp.negated ? !matchesValues : matchesValues;
          });
          console.log(`🔍 Filter ${filterOp.column}: ${currentData.length} rows`);
          break;
        }
        case 'custom': {
          const customOp = operation as any;
          try {
            const res = await dfCustom({ rows: currentData, code: customOp.code });
            if (res.error) {
              console.error('Custom operation failed:', res.error);
              break;
            }
            currentData = res.rows || currentData;
            console.log(`🐍 Custom code: ${currentData.length} rows`);
          } catch (e) {
            console.error('Custom operation error:', e);
          }
          break;
        }
        case 'sort': {
          const sortOp = operation as any;
          currentData = [...currentData].sort((a, b) => {
            let aVal = a[sortOp.column];
            let bVal = b[sortOp.column];
            
            if (aVal == null && bVal == null) return 0;
            if (aVal == null) return sortOp.direction === 'asc' ? 1 : -1;
            if (bVal == null) return sortOp.direction === 'asc' ? -1 : 1;
            
            const aNum = Number(aVal);
            const bNum = Number(bVal);
            if (!isNaN(aNum) && !isNaN(bNum)) {
              const diff = aNum - bNum;
              return sortOp.direction === 'asc' ? diff : -diff;
            } else {
              const comp = String(aVal).toLowerCase().localeCompare(String(bVal).toLowerCase());
              return sortOp.direction === 'asc' ? comp : -comp;
            }
          });
          console.log(`🔄 Sort ${sortOp.column} ${sortOp.direction}: ${currentData.length} rows`);
          break;
        }
      }
    }
    
    console.log(`🎯 Final result: ${currentData.length} rows`);
    setCurrentRows(currentData);
  }, [operationalRows]);

  // Legacy wrapper for backward compatibility
  const applyFilters = useCallback(async (newFilters: Filter[]) => {
    const filterOps = newFilters.map(f => createFilterOperation(f.column, f.values, f.negated));
    const nonFilterOps = operationChain.filter(op => op.type !== 'filter');
    const newChain = [...filterOps, ...nonFilterOps];
    setOperationChain(newChain);
    await applyOperationChain(newChain);
  }, [operationChain, applyOperationChain]);

  const resetAll = useCallback(() => {
    setCurrentRows(operationalRows);
    setOperationChain([]);
    setGroupBy(null);
    setGroupPreview([]);
    setExpandedGroup(null);
    setCustomCode("");
    setCustomError(null);
    setSortColumn(null);
    setSortDirection(null);
  }, [operationalRows]);

  // -------- GroupBy State ---------
  const [groupBy, setGroupBy] = useState<string | null>(null);
  const [groupPreview, setGroupPreview] = useState<{ value: any; count: number; means: Record<string, number> }[]>([]);
  const [expandedGroup, setExpandedGroup] = useState<any | null>(null);
  const [groupPage, setGroupPage] = useState<number>(1);
  const [groupRows, setGroupRows] = useState<Record<string, any>[]>([]);
  const [groupTotal, setGroupTotal] = useState<number>(0);
  const [groupPagination, setGroupPagination] = useState<Map<string, number>>(new Map());

  // -------- Sorting State ---------
  const [sortColumn, setSortColumn] = useState<string | null>(null);
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc' | null>(null);

  // Sort function
  const handleSort = useCallback((column: string) => {
    let newDirection: 'asc' | 'desc' | null = 'asc';
    
    if (sortColumn === column) {
      if (sortDirection === 'asc') {
        newDirection = 'desc';
      } else if (sortDirection === 'desc') {
        newDirection = null;
      }
    }
    
    setSortColumn(newDirection ? column : null);
    setSortDirection(newDirection);
    
    // Update operation chain
    const nonSortOps = operationChain.filter(op => op.type !== 'sort');
    const newChain = newDirection 
      ? [...nonSortOps, createSortOperation(column, newDirection)]
      : nonSortOps;
    
    setOperationChain(newChain);
    void applyOperationChain(newChain);
  }, [sortColumn, sortDirection, operationChain, applyOperationChain]);

  // Apply sorting to currentRows with performance optimization
  const sortedRows = useMemo(() => {
    if (!sortColumn || !sortDirection) return currentRows;
    
    // Use faster array copy and optimize comparison
    const result = currentRows.slice();
    
    // Pre-determine if column is numeric for better performance
    const isNumericColumn = currentRows.length > 0 && 
      currentRows.slice(0, 10).every(row => {
        const val = row[sortColumn];
        return val == null || !isNaN(Number(val));
      });
    
    result.sort((a, b) => {
      let aVal = a[sortColumn];
      let bVal = b[sortColumn];
      
      // Handle null/undefined values
      if (aVal == null && bVal == null) return 0;
      if (aVal == null) return sortDirection === 'asc' ? 1 : -1;
      if (bVal == null) return sortDirection === 'asc' ? -1 : 1;
      
      if (isNumericColumn) {
        // Numeric comparison
        aVal = Number(aVal);
        bVal = Number(bVal);
        const diff = aVal - bVal;
        return sortDirection === 'asc' ? diff : -diff;
      } else {
        // String comparison with cached lowercase
        const aStr = String(aVal).toLowerCase();
        const bStr = String(bVal).toLowerCase();
        const comp = aStr.localeCompare(bStr);
        return sortDirection === 'asc' ? comp : -comp;
      }
    });
    
    return result;
  }, [currentRows, sortColumn, sortDirection]);

  // Memoize expensive data overview calculations using sorted data
  const dataOverview = useMemo(() => {
    if (sortedRows.length === 0) return null;
    const uniquePrompts = new Set(sortedRows.map(r => r?.prompt)).size;
    let uniqueModels = 0;
    if (method === 'single_model') {
      uniqueModels = new Set(sortedRows.map(r => r?.model)).size;
    } else if (method === 'side_by_side') {
      uniqueModels = new Set([
        ...sortedRows.map(r => r?.model_a || ''),
        ...sortedRows.map(r => r?.model_b || '')
      ]).size;
    }
    return {
      rowCount: sortedRows.length.toLocaleString(),
      uniquePrompts: uniquePrompts.toLocaleString(),
      uniqueModels: uniqueModels.toLocaleString(),
    };
  }, [sortedRows, method]);

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
    if (!customCode.trim()) return;
    
    try {
      // Add custom operation to chain
      const customOp = createCustomCodeOperation(customCode);
      const newChain = [...operationChain, customOp];
      setOperationChain(newChain);
      setCustomError(null);
      await applyOperationChain(newChain);
      setCustomCode(""); // Clear after successful application
    } catch (e: any) {
      console.error('runCustom error:', e);
      setCustomError(String(e?.message || e));
    }
  }, [customCode, operationChain, applyOperationChain]);

  // Operation management callbacks
  const removeOperation = useCallback((operationId: string) => {
    const newChain = operationChain.filter(op => op.id !== operationId);
    setOperationChain(newChain);
    
    // Update UI state for removed operations
    const removedOp = operationChain.find(op => op.id === operationId);
    if (removedOp?.type === 'sort') {
      setSortColumn(null);
      setSortDirection(null);
    }
    
    void applyOperationChain(newChain);
  }, [operationChain, applyOperationChain]);

  // Legacy filter removal for backward compatibility
  const removeFilter = useCallback((index: number) => {
    const filterOps = operationChain.filter(op => op.type === 'filter');
    if (index < filterOps.length) {
      removeOperation(filterOps[index].id);
    }
  }, [operationChain, removeOperation]);

  const clearCustomCode = useCallback(() => {
    setCustomCode("");
    setCustomError(null);
  }, []);

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
            {availableColumns.length > 0 && !showColumnSelector && (
              <Button 
                variant="outlined" 
                onClick={() => setShowColumnSelector(true)}
                size="small"
              >
                Configure Columns
              </Button>
            )}
          </Stack>
        </Toolbar>
      </AppBar>
      {/* offset for fixed AppBar */}
      <Box sx={{ height: (theme) => theme.mixins.toolbar.minHeight }} />
      <Container maxWidth={false} sx={{ py: 2, flexGrow: 1, display: 'flex', flexDirection: 'column', alignItems: 'stretch' }}>
        {/* Getting started helper - shown before any upload */}
        {originalRows.length === 0 && !showColumnSelector && (
          <Box sx={{ 
            mb: 2, py: 3, px: 2, borderRadius: 2,
            background: '#F8FAFC', color: 'text.secondary'
          }}>
            <Box sx={{ textAlign: 'left', maxWidth: 760 }}>
              <Typography variant="h6" sx={{ mb: 1, color: 'primary.dark' }}>Easily Visualize and Analyze your Model Outputs</Typography>
              <Typography variant="body2" sx={{ color: 'primary.dark' }}>1) Upload your dataset (.jsonl, .json, or .csv)</Typography>
              <Typography variant="body2" sx={{ color: 'primary.dark' }}>2) Select which columns correspond to your prompts, responses, models, and scores</Typography>
              <Typography variant="body2" sx={{ color: 'primary.dark' }}>3) Click Done to load your table and explore</Typography>
            </Box>
          </Box>
        )}
        
        {/* Column Selector - shown when user needs to specify column mapping */}
        {showColumnSelector && (
          <ColumnSelector
            columns={availableColumns}
            onMappingChange={handleMappingChange}
            onValidationChange={handleValidationChange}
            autoDetectedMapping={autoDetectedMapping || undefined}
          />
        )}

        {/* Show filter notice if any rows were dropped due to missing scores */}
        {filterNotice && (
          <Box sx={{ mb: 1, p: 1.5, border: '1px solid #F59E0B', background: '#FFFBEB', color: '#92400E', borderRadius: 1 }}>
            {filterNotice}
          </Box>
        )}

        {dataOverview && (
          <Box sx={{ 
            mb: 2, 
            display: 'flex', 
            justifyContent: 'space-between', 
            alignItems: 'center',
            flexWrap: 'wrap',
            gap: 3
          }}>
            <Box sx={{ color: 'text.secondary' }}>
              <strong>{dataOverview.rowCount}</strong> rows ·{' '}
              <strong>{dataOverview.uniquePrompts}</strong> unique prompts ·{' '}
              <strong>{dataOverview.uniqueModels}</strong> unique models
            </Box>
            <Box sx={{ 
              color: 'primary.main', 
              fontSize: 14, 
              fontWeight: 500,
              backgroundColor: '#F0F9FF',
              px: 2,
              py: 0.5,
              borderRadius: 1,
              border: '1px solid #0EA5E9'
            }}>
              Click headers to sort • Use filters to narrow results
            </Box>
          </Box>
        )}

        {/* Data Ops bar - fixed width, no horizontal scroll */}
        {sortedRows.length > 0 && (
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
                  <Chip key={`${f.column}-${i}`} label={`${f.column}: ${f.negated ? 'NOT ' : ''}${f.values.join(', ')}`} onDelete={() => removeFilter(i)} />
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

        {/* Operation Chain Summary */}
        <FilterSummary
          operations={operationChain}
          onRemoveOperation={removeOperation}
        />

        {useMemo(() => {
          // Always show table structure, even when empty
          if (operationalRows.length === 0) return null;
          
          // If groupBy is active, show accordion view that looks like table rows
          if (groupBy && groupPreview.length > 0) {
            // Group the sorted rows by the selected column
            const groupedRows = new Map<any, any[]>();
            sortedRows.forEach(row => {
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
                      <Box key={col} sx={{ display: 'flex', alignItems: 'center', gap: 0.5, cursor: 'pointer' }} onClick={() => handleSort(col)}>
                        <Typography variant="subtitle2" sx={{ color: '#374151', fontWeight: 700, fontSize: 12, letterSpacing: 0.4, textTransform: 'uppercase' }}>
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
                        {sortColumn === col && sortDirection === 'asc' && <ArrowUpwardIcon sx={{ fontSize: 12, color: '#374151' }} />}
                        {sortColumn === col && sortDirection === 'desc' && <ArrowDownwardIcon sx={{ fontSize: 12, color: '#374151' }} />}
                      </Box>
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
                                    (() => {
                                      const value = row[col];
                                      const isNumeric = col === '__index' || (value !== null && value !== undefined && !isNaN(Number(value)) && value !== '');
                                      const isPrompt = col === 'prompt';
                                      return (
                                        <Typography variant="body2" sx={{ 
                                          maxWidth: 200,
                                          textAlign: isNumeric ? 'center' : 'left'
                                        }}>
                                          {isPrompt ? (
                                            <FormattedCell text={String(value || '')} isPrompt={true} />
                                          ) : (
                                            <TruncatedCell text={String(value || '')} />
                                          )}
                                        </Typography>
                                      );
                                    })()
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
              rows={sortedRows}
              columns={allowedColumns}
              responseKeys={responseKeys}
              onView={onView}
              allowedColumns={allowedColumns}
              sortColumn={sortColumn}
              sortDirection={sortDirection}
              onSort={handleSort}
            />
          );
        }, [sortedRows, allowedColumns, responseKeys, onView, groupBy, groupPreview, groupPagination, sortColumn, sortDirection, handleSort])}
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
