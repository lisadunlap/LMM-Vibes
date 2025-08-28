import React, { useMemo } from "react";
import { useReactTable, getCoreRowModel, flexRender, createColumnHelper } from "@tanstack/react-table";
import { Box, Button, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Tooltip, Fade } from "@mui/material";
import VisibilityOutlinedIcon from "@mui/icons-material/VisibilityOutlined";

export function DataTable({
  rows,
  columns,
  responseKeys,
  onView,
  method,
  allowedColumns,
}: {
  rows: Record<string, any>[];
  columns: string[];
  responseKeys: string[]; // keys where an eye icon should appear
  onView: (rowIndex: number, key?: string) => void;
  method: "single_model" | "side_by_side" | "unknown";
  allowedColumns?: string[]; // limit visible columns
}) {
  const columnHelper = createColumnHelper<Record<string, any>>();

  const MAX_LEN = 200;
  function TruncatedCell({ text }: { text: string }) {
    const [expanded, setExpanded] = React.useState(false);
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
  }

  const displayColumns = useMemo(() => {
    const human: Record<string, string> = {
      prompt: "PROMPT",
      model: "MODEL",
      model_response: "RESPONSE",
      model_a: "MODEL A",
      model_b: "MODEL B",
      model_a_response: "RESPONSE A",
      model_b_response: "RESPONSE B",
      score: "SCORE",
      score_a: "SCORE A",
      score_b: "SCORE B",
    };

    const baseRaw = allowedColumns && allowedColumns.length > 0
      ? allowedColumns.filter((c) => columns.includes(c))
      : columns;

    // Order: prompt → response columns → remaining
    const promptFirst = baseRaw.filter((c) => c === 'prompt');
    const resp = baseRaw.filter((c) => responseKeys.includes(c));
    const remaining = baseRaw.filter((c) => c !== 'prompt' && !responseKeys.includes(c));
    const base = [...promptFirst, ...resp, ...remaining];

    return base.map((col) =>
      columnHelper.accessor((row) => row[col], {
        id: col,
        header: () => human[col] ?? col.toUpperCase(),
        cell: (info) => {
          if (responseKeys.includes(col)) {
            return (
              <Tooltip title="View full response">
                <Button
                  size="small"
                  variant="text"
                  color="secondary"
                  startIcon={<VisibilityOutlinedIcon />}
                  onClick={() => onView(info.row.index, col)}
                  sx={{ fontWeight: 600 }}
                >
                  View
                </Button>
              </Tooltip>
            );
          }
          const value = info.getValue();
          // Render score dictionaries compactly
          if (typeof value === "object" && value !== null) {
            const text = JSON.stringify(value);
            return <span style={{ fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace', fontSize: 12 }}>{text}</span>;
          }
          const str = String(value ?? "");
          return <TruncatedCell text={str} />;
        },
      })
    );
  }, [columns, allowedColumns, responseKeys, columnHelper, onView]);

  const table = useReactTable({
    data: rows,
    columns: displayColumns,
    getCoreRowModel: getCoreRowModel(),
  });

  return (
    <TableContainer sx={{ border: '1px solid #E5E7EB', borderRadius: 2, overflow: 'auto', backgroundColor: '#FFFFFF' }}>
      <Table size="small">
        <TableHead sx={{ backgroundColor: '#F3F4F6' }}>
          {table.getHeaderGroups().map((hg) => (
            <TableRow key={hg.id}>
              {hg.headers.map((h) => (
                <TableCell key={h.id} sx={{ color: '#374151', fontWeight: 700, fontSize: 12, letterSpacing: 0.4 }}>
                  {h.isPlaceholder ? null : flexRender(h.column.columnDef.header, h.getContext())}
                </TableCell>
              ))}
            </TableRow>
          ))}
        </TableHead>
        <TableBody>
          {table.getRowModel().rows.map((r, idx) => (
            <Fade in timeout={Math.min(250 + idx * 90, 2000)} key={r.id}>
              <TableRow hover>
                {r.getVisibleCells().map((c) => (
                  <TableCell key={c.id} sx={{ borderBottom: '1px solid #E5E7EB' }}>
                    {flexRender(c.column.columnDef.cell, c.getContext())}
                  </TableCell>
                ))}
              </TableRow>
            </Fade>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
}

export default DataTable;


