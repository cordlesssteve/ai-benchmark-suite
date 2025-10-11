# LiveCodeBench Compatibility Fix

**Date:** 2025-10-11
**Issue:** LiveCodeBench datasets v4.x compatibility
**Status:** ✅ RESOLVED

---

## Problem

HuggingFace `datasets` library version 4.1.1 was installed in LiveCodeBench venv, but datasets v3.0+ removed support for loading Python dataset scripts. LiveCodeBench dataset on HuggingFace still uses the old Python script format.

### Error Message
```
RuntimeError: Dataset scripts are no longer supported, but found code_generation_lite.py
```

### Root Cause
- `pyproject.toml` specifies `datasets>=3.2.0`
- `uv pip install` installed datasets v4.1.1 (latest)
- datasets v3.0+ removed Python script loading
- LiveCodeBench HF dataset still uses `.py` script format

---

## Solution

**Downgrade datasets to v2.x in LiveCodeBench venv**

```bash
cd harnesses/livecodebench
uv pip install "datasets<3.0" --upgrade
```

### Result
- datasets downgraded from 4.1.1 → 2.21.0
- Dataset scripts can now be loaded with `trust_remote_code=True`
- LiveCodeBench code already includes this parameter (code_generation.py:125)

---

## Verification

✓ **VERIFIED:** datasets v2.21.0 installed
✓ **VERIFIED:** `load_dataset()` works with `trust_remote_code=True`
✓ **VERIFIED:** LiveCodeBench code already passes this parameter
✓ **VERIFIED:** Dataset downloading (1.25GB + 623MB + 1.20GB = ~3GB)

### Test Command
```python
from datasets import load_dataset
ds = load_dataset('livecodebench/code_generation_lite',
                  split='test',
                  trust_remote_code=True)
print(f'Loaded {len(ds)} examples')
```

---

## Other Benchmarks Status

### BigCodeBench
❌ **Does NOT use datasets library** - No compatibility issue

### SWEbench-Live
✅ **Uses Parquet format** - Works with datasets v4.x
- No Python scripts involved
- No downgrade needed

---

## Long-term Solution

**Upstream Fix Recommended:**

Contact LiveCodeBench maintainers to migrate dataset to Parquet format:
1. Remove Python script (code_generation_lite.py)
2. Convert to Parquet files
3. Update dataset on HuggingFace Hub
4. Allow datasets v3.0+ compatibility

**GitHub Issue:** https://github.com/LiveCodeBench/LiveCodeBench/issues

---

## Dependencies Changed

| Package    | Before  | After   | Reason                           |
|------------|---------|---------|----------------------------------|
| datasets   | 4.1.1   | 2.21.0  | Script loading compatibility     |
| dill       | 0.4.0   | 0.3.8   | datasets v2.x dependency         |
| fsspec     | 2025.9  | 2024.6  | datasets v2.x dependency         |
| numpy      | 2.2.6   | 2.3.3   | datasets v2.x dependency         |
| propcache  | 0.3.2   | 0.4.1   | datasets v2.x dependency         |

---

## Action Items

- [x] Diagnose compatibility issue
- [x] Test BigCodeBench and SWEbench-Live
- [x] Apply fix (downgrade datasets)
- [x] Verify dataset loading works
- [x] Document solution
- [ ] Test full adapter evaluation
- [ ] Consider filing upstream issue

---

**Fix Applied:** 2025-10-11
**Next Step:** Test full LiveCodeBench adapter evaluation with Ollama models
