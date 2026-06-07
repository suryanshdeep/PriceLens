# Frontend Documentation

This document tracks the React application, page structure, API client, and frontend setup.

## Purpose

The frontend is the user-facing dashboard for PriceLens. The first working screen is the prediction
workflow, not a landing page. It connects to the FastAPI backend and supports optional image upload.

## Technology

- React
- TypeScript
- Vite
- Tailwind CSS
- React Router
- Recharts
- lucide-react icons
- Fetch API for backend calls

The project intentionally does not use Next.js, SSR, App Router, or Next.js API routes.

## Directory Map

```text
frontend/
  package.json
  package-lock.json
  vite.config.ts
  tailwind.config.js
  postcss.config.js
  index.html
  src/
    main.tsx
    App.tsx
    index.css
    api/
      priceLensApi.ts
    components/
      ImageUpload.tsx
      Navbar.tsx
      PredictionResultCard.tsx
      ProductInputForm.tsx
    pages/
      Predict.tsx
      Analysis.tsx
      SimilarSearch.tsx
      Home.tsx
      About.tsx
    types/
      index.ts
    utils/
      formatters.ts
```

## Local Setup

Use npm from inside `frontend/`. `node_modules` will be created on the D: drive.

```powershell
cd D:\PriceLens\frontend
npm install --cache D:\PriceLens\.npm-cache
```

Create a local environment file:

```powershell
Copy-Item .env.example .env.local
```

The expected value is:

```text
VITE_API_BASE_URL=http://localhost:8000
```

## Run Frontend

```powershell
cd D:\PriceLens\frontend
npm run dev
```

Open:

```text
http://localhost:5173
```

## Routes

- `/predict`: main working prediction UI
- `/analysis`: mock metrics dashboard with Recharts
- `/similar-search`: placeholder for FAISS-based retrieval
- `/home`: project overview
- `/about`: architecture, dataset, and limitations

The root route `/` redirects to `/predict`.

## API Client

API calls live in `frontend/src/api/priceLensApi.ts`.

Functions:

- `healthCheck()`
- `getModelInfo()`
- `predictPrice(payload)`

`predictPrice` chooses the backend endpoint based on whether an image file exists:

- No image: `POST /predict` with JSON
- With image: `POST /predict-with-image` with multipart form data

## UI Behavior

The Predict page includes:

- catalog content textarea
- optional brand input
- optional category input
- optional image upload
- submit button
- prediction result card
- confidence band
- model version
- features used

`explanation_tokens` are returned by the backend but are intentionally not displayed in the frontend
yet. They are for developer visibility and model inspection.

## Build Check

```powershell
cd D:\PriceLens\frontend
npm run build
```

## Development Rules

- Keep reusable UI pieces in `src/components`.
- Keep page-level composition in `src/pages`.
- Keep shared request/response types in `src/types`.
- Use `VITE_API_BASE_URL` for backend URL.
- Do not hardcode localhost in page components.
- Keep visual design utilitarian and dashboard-focused.
- Use lucide icons for button/icon UI when possible.

## Next Frontend Milestones

- Add backend health indicator in the navbar or Predict page.
- Add loading and retry states for model metadata.
- Display real metrics when the backend exposes them.
- Add image preview for uploaded product images.
- Add similar product result cards backed by FAISS later.
