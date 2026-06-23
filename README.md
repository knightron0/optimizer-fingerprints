# NanoGPT optimizer traces

This repository records per-parameter optimizer behavior from NanoGPT training
runs and renders those traces in a static web UI using a remote manifest.

## Structure

```text
nanogpt/wrapper.py      # optimizer-step trace collector
nanogpt/examples/      # NanoGPT records instrumented with the collector
nanogpt/import_records.py
                        # imports and instruments record scripts
traces/                 # local nanogpt_optimizer_trace JSON files
web/                    # manifest-driven optimizer comparison UI
```

`OptimizerFingerprint.attach(...)` registers optimizer pre/post-step hooks.
Calling `finish()` writes a `nanogpt_optimizer_trace` JSON file containing run
metadata and sampled per-parameter metrics. The web UI discovers remote traces
from its configured manifest and fetches selected trace JSON files on demand.

## Web UI

```bash
cd web
npm ci
npm run dev
```

Use `npm run build` to verify the production site. GitHub Pages builds the same
UI using the configured trace manifest.
