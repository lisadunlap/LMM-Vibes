import React, { useMemo } from "react";
import { useReactTable, getCoreRowModel, flexRender, createColumnHelper } from "@tanstack/react-table";
import { Box, Button, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Fade } from "@mui/material";
import VisibilityOutlinedIcon from "@mui/icons-material/VisibilityOutlined";

const DataTable = React.memo(function DataTable({
  rows,
  columns,
  responseKeys,
  onView,
  allowedColumns,
}: {
  rows: Record<string, any>[];
  columns: string[];
  responseKeys: string[]; // keys where an eye icon should appear
  onView: (row: Record<string, any>) => void;
  allowedColumns?: string[]; // limit visible columns
}) {
  const columnHelper = createColumnHelper<Record<string, any>>();

  const MAX_LEN = 200;
  const TruncatedCell = React.memo(function TruncatedCell({ text }: { text: string }) {
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
  });

  // Animate only on initial mount (first paint) for the first 20 rows
  const animateOnMountRef = React.useRef(true);
  React.useEffect(() => {
    const id = requestAnimationFrame(() => {
      animateOnMountRef.current = false;
    });
    return () => cancelAnimationFrame(id);
  }, []);

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

    return base.map((col) => {
      const isResponse = responseKeys.includes(col);
      return columnHelper.accessor((row) => row[col], {
        id: col,
        header: human[col] ?? col.toUpperCase(),
        cell: (info) => {
          if (isResponse) {
            return (
              <Button
                size="small"
                variant="text"
                color="secondary"
                startIcon={<VisibilityOutlinedIcon />}
                onClick={() => onView(info.row.original)}
                sx={{ fontWeight: 600 }}
              >
                View
              </Button>
            );
          }
          const value = info.getValue();
          // Treat nested objects as simple strings (scores should be flattened already)
          if (typeof value === "object" && value !== null) {
            return <span>[object]</span>;
          }
          const str = String(value ?? "");
          return <TruncatedCell text={str} />;
        },
      });
    });
  }, [columns, allowedColumns, responseKeys, onView]);

  // Limit rendering for large datasets
  const displayRows = useMemo(() => {
    // Only render first 1000 rows to prevent UI lag
    return rows.length > 1000 ? rows.slice(0, 1000) : rows;
  }, [rows]);

  const table = useReactTable({
    data: displayRows,
    columns: displayColumns,
    getCoreRowModel: getCoreRowModel(),
  });

  return (
    <>
      {rows.length > 1000 && (
        <Box sx={{ mb: 1, p: 1, backgroundColor: '#FEF3C7', border: '1px solid #F59E0B', borderRadius: 1, fontSize: 14 }}>
          Showing first 1,000 of {rows.length.toLocaleString()} rows for performance. Use filters to narrow results.
        </Box>
      )}
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
          {table.getRowModel().rows.map((r, idx) => {
            const rowEl = (
              <TableRow hover key={r.id}>
                {r.getVisibleCells().map((c) => (
                  <TableCell key={c.id} sx={{ borderBottom: '1px solid #E5E7EB' }}>
                    {flexRender(c.column.columnDef.cell, c.getContext())}
                  </TableCell>
                ))}
              </TableRow>
            );
            if (animateOnMountRef.current && idx < 20) {
              return (
                <Fade in timeout={Math.min(250 + idx * 90, 2000)} key={`fade-${r.id}`}>
                  {rowEl}
                </Fade>
              );
            }
            return rowEl;
          })}
        </TableBody>
        </Table>
      </TableContainer>
    </>
  );
});

export default DataTable;
