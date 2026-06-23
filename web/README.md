# NanoGPT optimizer comparison UI

Astro static site for comparing NanoGPT optimizer traces fetched on demand from
a JSON manifest.

```bash
npm ci
npm run dev
```

`npm run build` generates the production site in `dist/`.

Set `PUBLIC_TRACE_MANIFEST_URL` to the public R2 manifest URL at build time. If
it is unset, the app uses `public/traces-manifest.json`. The generated manifest
contains a `traces` array:

```json
{
  "traces": [
    {
      "id": "muon",
      "title": "Muon baseline",
      "description": "Baseline optimizer run",
      "trace_url": "./traces/muon.json"
    }
  ]
}
```

Relative `trace_url` values resolve against the final manifest response URL.
Legacy array manifests remain supported. The R2 bucket must allow browser CORS
requests from the UI's origin.

To regenerate the manifest from the repository's `traces/` directory:

```bash
npm run manifest
```

This writes `public/traces-manifest.json`. Trace links such as
`./traces/example.json` assume the manifest is at the bucket root and the R2
object keys mirror the local `traces/` directory.

To serve the manifest locally while loading trace files from R2, generate it
with absolute URLs instead:

```bash
TRACE_URL_PREFIX="https://YOUR-R2-URL/traces/" npm run manifest
```
