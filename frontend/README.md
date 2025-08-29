# StringSight React Frontend

Modern React evaluation console for loading, filtering, sorting, and analyzing evaluation datasets with conversation traces.

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
cd frontend
npm run dev -- --host 127.0.0.1 --port 5180
```

Open `http://127.0.0.1:5173`.

## Core Features

### 📊 **Data Loading & Management**
- **File Upload**: Supports `.jsonl`, `.json`, `.csv` formats
- **Format Detection**: Auto-detects single-model vs side-by-side evaluation formats
- **Index Preservation**: Original dataframe indices maintained through all operations
- **Performance Optimization**: Smart loading with 1000+ row performance warnings

### 🔍 **Filtering & Search**
- **Multi-Column Filters**: Add/remove filters on categorical columns
- **Negation Support**: Include/exclude value sets with NOT toggle
- **Real-Time Updates**: Instant filter application with backend validation
- **Custom Pandas Code**: Execute arbitrary pandas expressions for complex filtering

### 📈 **Sorting & Organization**
- **Click-to-Sort**: Click any column header to sort (asc → desc → none cycle)
- **Visual Indicators**: Arrow icons show current sort direction
- **Smart Type Detection**: Automatic numeric vs string sorting
- **Performance Optimized**: Efficient sorting for large datasets

### 📊 **Groupby Analysis**
- **Dynamic Grouping**: Group by any column with summary statistics
- **Accordion View**: Expandable groups with individual row pagination
- **Statistical Previews**: Average scores displayed for numeric columns
- **Pagination**: Page through examples within each group

### 💬 **Conversation Traces**
- **Side Drawer**: Full conversation view with OpenAI message format
- **Dual Views**: Single model or side-by-side comparison modes
- **Responsive Layout**: Adapts to different screen sizes

## Data Formats

### Single Model Evaluation
```json
{
  "prompt": "What is the capital of France?",
  "model": "gpt-4",
  "model_response": "The capital of France is Paris.",
  "score": 4.5
}
```

**Required columns**: `prompt`, `model`, `model_response`  
**Optional columns**: `score` (number or nested object)

### Side-by-Side Evaluation
```json
{
  "prompt": "What is the capital of France?",
  "model_a": "gpt-4",
  "model_b": "claude-3",
  "model_a_response": "The capital of France is Paris.",
  "model_b_response": "Paris is the capital city of France.",
  "score_a": 4.5,
  "score_b": 4.2
}
```

**Required columns**: `prompt`, `model_a`, `model_b`, `model_a_response`, `model_b_response`  
**Optional columns**: `score_a`, `score_b` (numbers or nested objects)

## Architecture

### 🏗️ **Component Structure**

```
src/
├── App.tsx                     # Main shell with data management
├── components/
│   ├── DataTable.tsx          # Sortable table with truncation
│   ├── ConversationTrace.tsx  # Single conversation view
│   └── SideBySideTrace.tsx    # Dual conversation comparison
├── lib/
│   ├── api.ts                 # Backend API calls
│   ├── parse.ts               # Client-side file parsing
│   ├── traces.ts              # Message format utilities
│   └── normalize.ts           # Score flattening
└── theme.ts                   # MUI theme configuration
```

### 🗃️ **Data Layer Architecture**

The app uses a three-layer data management system:

1. **Original Rows** (`originalRows`): Raw uploaded data, never modified
2. **Operational Rows** (`operationalRows`): Cleaned data with allowed columns + index
3. **Current Rows** (`currentRows`): Filtered/transformed data for display

### 🎯 **State Management**

**Data States:**
- `originalRows` - Immutable uploaded data
- `operationalRows` - Processed data with index and allowed columns
- `currentRows` - Filtered data ready for display
- `sortedRows` - Final sorted data for rendering

**Filter States:**
- `filters` - Active column filters
- `pendingColumn/Values/Negated` - UI state for building filters

**Sort States:**
- `sortColumn` - Currently sorted column
- `sortDirection` - 'asc' | 'desc' | null

**Group States:**
- `groupBy` - Column to group by
- `groupPreview` - Summary statistics per group
- `groupPagination` - Page state for each group

### ⚡ **Performance Optimizations**

- **Memoized Components**: React.memo on expensive renders
- **Optimized Sorting**: Pre-computed type detection and efficient comparisons
- **Smart Re-renders**: Careful dependency arrays in useMemo/useCallback
- **Local-First Operations**: Client-side filtering/sorting with optional backend validation
- **Lazy Loading**: 1000-row display limit with performance warnings

## Extending the Frontend

### 🔌 **Adding New Column Types**

1. Update column detection logic in `App.tsx`:
```typescript
const allowedCols = [...existing, 'new_column_type'];
```

2. Add human-readable labels in `DataTable.tsx`:
```typescript
const human: Record<string, string> = {
  // existing...
  new_column_type: "NEW COLUMN"
};
```

### 📈 **Adding New Views**

1. Create component in `src/components/`
2. Add routing/state management in `App.tsx`
3. Integrate with existing data layers

### 🔍 **Custom Analysis Features**

The pandas expression feature provides a foundation for advanced analysis:

```typescript
// Example: Add clustering results
const clusteringResults = await dfCustom({
  rows: currentRows,
  code: "df.assign(cluster=kmeans_predict(df[score_cols]))"
});
```

### 🌐 **API Integration**

Current API endpoints:
- `POST /detect_and_validate` - File validation
- `POST /df_select` - Filtering
- `POST /df_group_preview` - Group statistics
- `POST /df_group_rows` - Group pagination
- `POST /df_custom` - Custom pandas code

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
