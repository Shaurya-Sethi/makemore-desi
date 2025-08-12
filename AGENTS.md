# Repository Guidelines

## Project Structure & Module Organization
- `src/`: Core code — `wavenet.py` (model) and `train_wavenet.py` (trainer + sampling).
- `data/`: Datasets — `raw/` (scraped CSV/JSON) → `processed/` (cleaned CSV, `preprocess.py`).
- `models/`: Checkpoints — `wavenet_indian_names.pt` produced by training.
- `web scrapers/`: Data collection — `scraper.py` (A–Z crawl), `enhance.py` (merge JSON firecrawl output).
- `samples.txt`: Example generated names for reference.

## Build, Test, and Development Commands
- Create env and install: `python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt` (Windows: `.venv\Scripts\activate`).
- Preprocess data: `python data/processed/preprocess.py` (reads `data/raw/indian_first_names.csv`, writes cleaned CSV).
- Scrape (optional): `python "web scrapers/scraper.py"` then re-run preprocess.
- Train + sample: `python -m src.train_wavenet` (saves to `models/` and prints sample names).

## Coding Style & Naming Conventions
- Python, 4‑space indentation, PEP 8; type hints where practical.
- Files and functions: `snake_case`; classes: `CamelCase`; constants: `UPPER_SNAKE_CASE`.
- Prefer small, pure functions; keep I/O paths under `data/` and `models/`.
- Formatting: if available, run Black (`black src web\ scrapers data/processed`). No enforced linter in repo.

## Testing Guidelines
- No formal tests yet. For changes touching tokenization or sampling, verify by:
  - Short run: reduce `STEPS` in `src/train_wavenet.py` and confirm loss decreases.
  - Sampling sanity: ensure printed names terminate at `'.'` and look plausible.
- If adding tests, use `pytest` with files named `tests/test_*.py` and avoid network calls.

## Commit & Pull Request Guidelines
- Commits: imperative, concise subject (e.g., `feat: add gated residual block`, `fix: correct causal padding`).
- PRs: include summary, rationale, before/after behavior, and run steps. Link issues, attach sample output or logs (loss and a few names). Keep changes scoped and documented.

## Security & Configuration Tips
- Be polite when scraping (rate limits already present). Don’t commit large raw data; prefer small samples. Use Git LFS for big checkpoints if needed.
- Keep paths relative; avoid hard‑coding absolute directories. Never include secrets in code or data.

