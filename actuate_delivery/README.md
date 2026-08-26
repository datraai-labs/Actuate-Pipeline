# Actuate

Actuate turns raw multimodal camera recordings into verified, reviewable customer datasets. Panoculon Trinet is the first supported device adapter.

Current status: CP-01 through Bundle C are accepted. The CLI processes, reviews, and packages a delivery end to end. The browser application supports isolated uploads, processing progress, episode review and delivery download. Direct Google Drive ingestion remains optional future work.

## Install for development

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

## Run

```bash
actuate run SOURCE RUN_DIR --output DELIVERY_DIR
```

For recordings from the confirmed calibrated camera setup, select its calibration explicitly:

```bash
actuate run SOURCE RUN_DIR --output DELIVERY_DIR --calibration CALIBRATION.json
```

The command runs one fixed stage at a time: inventory, sensors, video, timing, QC, human review, and delivery. After each stage it prints measured results, elapsed time, exact failures, and issues requiring attention. It then waits for explicit approval before continuing. Choosing no exits safely; the same command and `RUN_DIR` resume without repeating intact work.

QC automatically writes `RUN_DIR/review.csv`. The CLI then collects include/exclude decisions and any required supplier limitation. The CSV remains available for detailed inspection. Its rules are:

- leave `decision` blank to keep the run pending, or enter `include` or `exclude`;
- enter a name or team identifier in `decided_by` for every decision;
- for an included row with `material_checks`, enter a JSON list of exact supplier limitations in `limitations_json`;
- do not edit measured fact columns or `decided_at`.

The final confirmation creates and validates both `DELIVERY_DIR` and `DELIVERY_DIR.zip`. Excluded and incomplete captures stay in the internal run but not the supplier delivery.

The customer package uses stable readable episode identities:

```text
DELIVERY_DIR/
├── README.md
├── episodes.csv
├── calibration/
│   └── calibration_000001.json
└── episodes/
    └── episode_000001/
        ├── meta.json
        ├── raw/
        ├── derived/
        └── previews/
```

`episode_id` is the customer-facing identity. `capture_id` remains the content-derived internal identity. Both appear in `episodes.csv` and `meta.json`, alongside the original source directory and source group. Once assigned in a dataset, an episode ID is never renumbered; later captures receive the next number. Customer files expose measured coverage and pairing facts, not the pipeline's internal review verdicts.

Repeating the command with the same source and run directory resumes the same SQLite-backed run and reuses intact artifacts. A different source is rejected without changing the stored run. A newly observed capture invalidates the affected approval and downstream approvals.

Inspect persisted progress without changing it:

```bash
actuate status RUN_DIR
```

## Browser application

```bash
ACTUATE_UI_HOST=0.0.0.0 PORT=8000 \
actuate run SOURCE RUN_DIR --ui --output DELIVERY_DIR
```

The browser groups work as Batch -> Episodes. Every upload creates an isolated batch and streams large files in 8 MB chunks. Upload stops before processing. Each stage shows its exact result and waits for approval before the next stage can run. CLI and browser approvals share the same run ledger. A completed batch is locked; a fresh upload never writes into it. The final confirmation creates the validated folder and ZIP.

`SOURCE`, `RUN_DIR` and `DELIVERY_DIR` provide the initial batch. Browser-created batches are stored beside `RUN_DIR`, each with separate source, run and delivery directories. Use persistent storage when hosting the service.

When the initial `RUN_DIR` contains `calibration.json`, a new browser batch remains calibration-neutral by default. The upload confirmation offers an explicit unchecked calibration choice with factual method and baseline information. A selected calibration is bound only to the included episode IDs when delivery is built.

The application is host-ready but is not deployed by this repository command. AWS deployment and direct Google Drive ingestion remain separate approval-gated work.
