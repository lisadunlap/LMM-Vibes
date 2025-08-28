# StringSight React Preview

Minimal React (Vite + TypeScript + MUI) UI for loading a dataset and previewing rows with a side drawer conversation view.

## Prerequisites

- Node 20+
- Python 3.8+

## Backend API

From the repo root:

```bash
python3 -m pip install "uvicorn[standard]" fastapi pandas python-multipart
python3 -m uvicorn stringsight.api:app --reload --host 127.0.0.1 --port 8000
```

Check: `curl http://127.0.0.1:8000/health` → `{ "ok": true }`.

## Frontend

From `frontend/`:

```bash
# Point the UI to the API
printf "VITE_API_BASE=http://127.0.0.1:8000\n" > .env.local

# Install & run
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

Open `http://127.0.0.1:5173`.

## Usage

- Click "Load File" and choose a local `.jsonl`, `.json`, or `.csv` file in one of these formats:
  - single model: `prompt`, `model`, `model_response` (+ optional `score`)
  - side-by-side: `prompt`, `model_a`, `model_b`, `model_a_response`, `model_b_response` (+ optional `score_a`, `score_b`)
- The table shows only the relevant columns. Response columns are rightmost.
- Click the eye icon to open the right drawer with the full conversation trace.

## Notes

- Remote path browsing is not enabled yet (intended for a future iteration).
- Styling uses a light, neutral theme with indigo accents to match StringSight.

## Architecture & Abstractions

- UI stack: Vite + React + TypeScript + MUI.
- Table: TanStack Table (headless) rendered with MUI `Table`.
- Components:
  - `src/App.tsx`: Shell, fixed header, file loader, data wiring.
  - `src/components/DataTable.tsx`: Table rendering; restricted columns; truncation with per-cell expand/collapse; “View” buttons.
  - `src/components/ConversationTrace.tsx`: Renders a single OpenAI-style message list.
  - `src/components/SideBySideTrace.tsx`: Two-column trace view built from `ConversationTrace`.
  - `src/lib/parse.ts`: Client-side JSONL/JSON/CSV parsing for local files.
  - `src/lib/traces.ts`: Client-format helpers (detect method, ensure OpenAI messages).
  - `src/lib/api.ts`: Fetch helpers (currently only `detectAndValidate`; remote listing/loading disabled for now).
  - `src/theme.ts`: Theme to keep styling consistent.

## Input Assumptions

- Single model dataset must include columns:
  - Required: `prompt`, `model`, `model_response`
  - Optional: `score` (number or object)
- Side-by-side dataset must include columns:
  - Required: `prompt`, `model_a`, `model_b`, `model_a_response`, `model_b_response`
  - Optional: `score_a`, `score_b` (numbers or objects)
- Parsing behavior:
  - JSONL: newline-delimited JSON objects
  - JSON: array of objects or single object
  - CSV: header row required
- UI shows only a subset of columns:
  - Single model: `prompt`, `model`, `score`, `model_response` (response is rightmost)
  - Side-by-side: `prompt`, `model_a`, `model_b`, `score_a`, `score_b`, `model_a_response`, `model_b_response` (responses are rightmost)

## Behaviors & UX

- Long text cells (>200 chars) render truncated with an “Expand/Collapse” toggle per-cell.
- “View” buttons in response columns open a right drawer with a conversation trace:
  - Single model: messages list
  - Side-by-side: two columns (`model_a`, `model_b`)
- Header is fixed; content uses toolbar height as offset.

## Extending

- To add server-side dataset loading:
  - Re-enable `readPath` in `src/lib/api.ts` and add a path browser component.
  - Ensure CORS and auth (if needed) are configured server-side.
- To add more columns, update `allowedColumns` in `App.tsx` and header labels in `DataTable.tsx`.
- For clustering/metrics views, fetch saved artifacts from the Python pipeline and render plots via Plotly or MUI charts.

# React + TypeScript + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Babel](https://babeljs.io/) for Fast Refresh
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh

## Expanding the ESLint configuration

If you are developing a production application, we recommend updating the configuration to enable type-aware lint rules:

```js
export default tseslint.config([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...

      // Remove tseslint.configs.recommended and replace with this
      ...tseslint.configs.recommendedTypeChecked,
      // Alternatively, use this for stricter rules
      ...tseslint.configs.strictTypeChecked,
      // Optionally, add this for stylistic rules
      ...tseslint.configs.stylisticTypeChecked,

      // Other configs...
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])
```

You can also install [eslint-plugin-react-x](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-x) and [eslint-plugin-react-dom](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-dom) for React-specific lint rules:

```js
// eslint.config.js
import reactX from 'eslint-plugin-react-x'
import reactDom from 'eslint-plugin-react-dom'

export default tseslint.config([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...
      // Enable lint rules for React
      reactX.configs['recommended-typescript'],
      // Enable lint rules for React DOM
      reactDom.configs.recommended,
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])
```
