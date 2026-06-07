# PriceLens Frontend

React + TypeScript + Vite frontend for the PriceLens dashboard.

Full frontend documentation lives in [../docs/frontend.md](../docs/frontend.md).

## Run

```powershell
cd D:\PriceLens\frontend
npm install --cache D:\PriceLens\.npm-cache
Copy-Item .env.example .env.local
npm run dev
```

The app expects the backend at `http://localhost:8000` by default. Override it with:

```text
VITE_API_BASE_URL=http://localhost:8000
```

## Pages

- `/predict`
- `/analysis`
- `/similar-search`
- `/home`
- `/about`
