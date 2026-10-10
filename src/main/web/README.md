# The pipeline service's browser app

React + TypeScript, built with Vite, served by the pipeline service at `/ui`. The design is
[`design/web-app.md`](../../../design/web-app.md); the interface it calls is
[`docs/schema/service.openapi.json`](../../../docs/schema/service.openapi.json).

Needs Node 24 (`.nvmrc`; `nvm use`).

```sh
cd src/main/web
npm ci
npm run dev        # http://localhost:5173/ui/, proxying /v1 to a service on BEQ_SERVICE_URL (default http://127.0.0.1:8080)
npm test           # Vitest + Testing Library, the service mocked with MSW
npm run typecheck
npm run lint
npm run build      # dist/, which the service serves with --ui-dir src/main/web/dist (or BEQ_SERVICE_UI)
```

A local service to develop against:

```sh
BEQ_SERVICE_TOKEN=dev PYTHONPATH=src/main/python uv run python -m pipeline.service --profile my-profile.yaml \
    --host 127.0.0.1 --ui-dir src/main/web/dist
```

## The API types

`src/api/schema.d.ts` is generated from the published OpenAPI document and committed. After changing the service's
interface (and regenerating `docs/schema/service.openapi.json`), run `npm run gen:api` and commit both; CI fails if the
types are not the document's.

## Layout

| Path | |
|---|---|
| `src/api/` | the typed client (`client.ts`), the token's storage (`token.ts`), following a job's events (`events.ts`) |
| `src/auth/` | sign-in and the auth context: a 401 anywhere signs out |
| `src/pages/` | one component per screen |
| `src/test/` | the MSW server, typed fixtures and `renderApp()` |

Versions are pinned exactly, and Dependabot proposes the bumps. TypeScript stays on 6.x until `typescript-eslint` and
`openapi-typescript` support 7 (design/web-app.md §4).
