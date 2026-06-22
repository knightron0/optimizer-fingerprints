# NanoGPT optimizer traces

This repository records per-parameter optimizer behavior from NanoGPT training
runs and renders those traces in a static web UI.

## Structure

```text
nanogpt/wrapper.py      # optimizer-step trace collector
nanogpt/examples/      # NanoGPT records instrumented with the collector
nanogpt/import_records.py
                        # imports and instruments record scripts
traces/                 # committed nanogpt_optimizer_trace JSON files
web/                    # trace and comparison UI
```

`OptimizerFingerprint.attach(...)` registers optimizer pre/post-step hooks.
Calling `finish()` writes a `nanogpt_optimizer_trace` JSON file containing run
metadata and sampled per-parameter metrics. The web build reads committed JSON
files directly from `traces/`.

## Web UI

```bash
cd web
npm ci
npm run dev
```

Use `npm run build` to verify the production site. GitHub Pages builds the same
UI from the committed traces.
