# mlproject

Short description: small ML utilities and experiments by Bhagya Lakshmi.

**Contents:**
- Project packaging metadata in `setup.py`
- Python dependency hints in `requirements.txt`

**Dependencies:**
- pandas
- numpy
- seaborn

**Install (development):**
1. Create and activate a virtual environment (Python 3.8+ recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
# or if `requirements.txt` is empty:
pip install pandas numpy seaborn
```

3. (Optional) Install package in editable mode:

```bash
pip install -e .
```

**Usage:**
- Add notebooks or scripts that import the project packages and utilities.

**Notes / Next steps:**
- `setup.py` contains a small syntax/typo issue that should be fixed before publishing:

- See [setup.py](setup.py) for the current content — fix `nmae` -> `name` and add the missing comma after `author_email`.
- Populate `requirements.txt` with the library list above to make installs reproducible.

**Author:** Bhagya Lakshmi

---
Generated: 2026-06-23 (README updated by assistant)
## End to End ML Project
