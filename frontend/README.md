# TBVision Frontend

Next.js App Router UI for TBVision. It lets users upload a chest X-ray, enter clinical metadata, view the diagnostic report, and ask follow‑up questions in an embedded chat.

**Stack**
- Next.js 15 (App Router, React 18)
- HeroUI components
- Tailwind CSS v4
- React Hook Form + Zod validation
- Markdown rendering with math + code highlighting

**Requirements**
- Node.js `22.13.0+` or `20.19.0+`
- Yarn (recommended for this repo)

**Environment**
- `NEXT_PUBLIC_API_BASE` (optional)
  - Default: `http://127.0.0.1:8000`
  - Example: `NEXT_PUBLIC_API_BASE=http://localhost:8000`

**Install**
```bash
yarn
```

**Run**
```bash
yarn dev
```

**Build**
```bash
yarn build
yarn start
```

## API Contracts

The frontend expects these backend routes:

**Prediction**
- `POST /api/predict`
- Body: `multipart/form-data`
  - `image`: X‑ray image file
  - `metadata`: JSON string (patient info, symptoms, risk factors, etc.)
- Response includes `report_id`, `prediction`, `probabilities`, `explanation`, `evidence`, `gradcam_image`

**Follow‑up Chat**
- `POST /api/follow-up`
  - JSON: `{ "report_id": "string", "question": "string" }`
  - Response: answer + full history
- `GET /api/follow-up/history?report_id=...`
  - Response: full history

## Key Paths

- `app/analysis`: report + chat experience
- `app/analysis/components/DiagnosticReport.tsx`: report layout
- `app/analysis/components/ChatSection.tsx`: embedded chat UI
- `services/`: API wrappers
- `lib/schema.ts`: form validation schema
- `lib/constants.ts`: API base URL
- `lib/db.ts`: IndexedDB helpers for persisting the uploaded image
- `components/markdown/markdown-renderer.tsx`: markdown + math renderer

## Scripts

- `yarn dev` — start local dev server
- `yarn build` — production build
- `yarn start` — run production server
- `yarn lint` — lint
- `yarn typecheck` — TypeScript checks

## Notes

- The chat is embedded below the diagnostic report. There is no dedicated `/chat` page.
- If you refresh the report view, chat history is restored via `/api/follow-up/history`.
