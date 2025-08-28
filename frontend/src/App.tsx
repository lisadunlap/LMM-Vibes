import { useState, useCallback } from "react";
import { Box, AppBar, Toolbar, Typography, Container, Button, Drawer, Stack } from "@mui/material";
import { detectAndValidate } from "./lib/api";
import { flattenScores } from "./lib/normalize";
import { parseFile } from "./lib/parse";
import { detectMethodFromColumns, ensureOpenAIFormat } from "./lib/traces";
import DataTable from "./components/DataTable";
import ConversationTrace from "./components/ConversationTrace";
import SideBySideTrace from "./components/SideBySideTrace";

function App() {
  const [rows, setRows] = useState<Record<string, any>[]>([]);
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
    setColumns(normCols);
    try {
      await detectAndValidate(file); // optional backend validation
    } catch (_) {}
  }

  const onView = useCallback((rowIndex: number, key?: string) => {
    const row = rows[rowIndex];
    if (method === "single_model") {
      const messages = ensureOpenAIFormat(String(row["prompt"] ?? ""), row["model_response"]);
      setSelectedTrace({ type: "single", messages });
    } else if (method === "side_by_side") {
      const prompt = String(row["prompt"] ?? "");
      const messagesA = ensureOpenAIFormat(prompt, row["model_a_response"]);
      const messagesB = ensureOpenAIFormat(prompt, row["model_b_response"]);
      setSelectedTrace({
        type: "sbs",
        messagesA,
        messagesB,
        modelA: String(row["model_a"] ?? "Model A"),
        modelB: String(row["model_b"] ?? "Model B"),
      });
    }
    setDrawerOpen(true);
  }, [rows, method]);

  const responseKeys = method === "single_model"
    ? ["model_response"]
    : method === "side_by_side"
      ? ["model_a_response", "model_b_response"]
      : [];

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
        {rows.length > 0 && (
          <Box sx={{ mb: 2, color: 'text.secondary' }}>
            <strong>{rows.length.toLocaleString()}</strong> rows ·{' '}
            <strong>{new Set(rows.map(r => r?.prompt)).size.toLocaleString()}</strong> unique prompts ·{' '}
            <strong>{(() => {
              if (method === 'single_model') return new Set(rows.map(r => r?.model)).size;
              if (method === 'side_by_side') return new Set([...(rows.map(r => r?.model_a || '')), ...(rows.map(r => r?.model_b || ''))]).size;
              return 0;
            })().toLocaleString()}</strong> unique models
          </Box>
        )}
        {rows.length > 0 && (
          // Build allowed columns dynamically so flattened score_* columns are included
          (() => {
            const scoreCols = method === 'single_model'
              ? columns.filter(c => c.startsWith('score_'))
              : method === 'side_by_side'
                ? columns.filter(c => c.startsWith('score_a_') || c.startsWith('score_b_'))
                : [];
            const allowed = method === 'single_model'
              ? ['prompt', 'model', ...scoreCols, 'model_response']
              : method === 'side_by_side'
                ? ['prompt', 'model_a', 'model_b', ...scoreCols, 'model_a_response', 'model_b_response']
                : [];
            return (
          <DataTable
            rows={rows}
            columns={columns}
            responseKeys={responseKeys}
            onView={onView}
            method={method}
            allowedColumns={allowed}
          />
            );
          })()
        )}
      </Container>
      <Drawer anchor="right" open={drawerOpen} onClose={() => setDrawerOpen(false)} sx={{ '& .MuiDrawer-paper': { width: 600, p: 2 } }} ModalProps={{ keepMounted: true }}>
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
