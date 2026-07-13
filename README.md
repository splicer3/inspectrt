# InspectRT

InspectRT is an early-stage engineering project for building a portable runtime for industrial visual inspection, focused on reproducible anomaly-detection pipelines and measured behavior across software and hardware backends.

`RT` means **Runtime**. InspectRT isn't made with real-time guarantees in mind.

## Development

InspectRT is in early development.

Set up the environment with:

```bash
uv sync
```

Run the tests and lint checks with:

```bash
uv run pytest
uv run ruff check .
```
