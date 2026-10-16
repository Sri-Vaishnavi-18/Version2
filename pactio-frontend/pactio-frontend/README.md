# Pactio frontend

Pactio is the React + Vite frontend for the privacy-policy analysis project. It uses the supplied Pactio cat logo and images, with a responsive ivory-and-teal design.

## Features

- Home, Review, Insights, How it works, About, and evidence/results screens.
- Paste policy text, upload PDF/TXT, or submit a public policy URL.
- Evidence-linked topic groups and JSON/TXT report export.
- Responsive layout, keyboard focus styles, reduced-motion support, and loading/error states.
- No sample analysis: the frontend calls the project's FastAPI backend, which requires the trained BiLSTM + Attention model and tokenizer.

## Development

For local frontend development, use Node.js 20.19+ and Python 3.10+. From this folder:

```powershell
npm install
Copy-Item .env.example .env
npm run dev
```

In another terminal, from the repository root:

```powershell
python -m pip install -e .
python -m uvicorn api:app --reload --port 8000
```

The Vite development server calls `http://localhost:8000/api/analyze` and `/api/analyze-file`. The API reads the trained artifacts from `artifacts/` by default; set `CONTRACT_LENS_ARTIFACTS` to use another folder.

## Production

For Render deployment, no local npm installation is required. The repository-root Dockerfile uses a Node build stage to compile this frontend and packages it with FastAPI. In Render, deploy the repository's `render.yaml` Blueprint and add `bilstm_attention.pt` and `tokenizer.json` as secret files. See the repository [README](../../README.md) for complete deployment steps.
