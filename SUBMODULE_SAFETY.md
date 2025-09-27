# Submodule Safety Guidelines

## 🚨 CRITICAL: Read-Only Submodules

The harnesses are included as **read-only submodules** to capture upstream updates without risking accidental PRs.

### Safety Rules

1. **NEVER edit files in `harnesses/` directories**
2. **NEVER commit changes to submodule content**
3. **NEVER push from within submodule directories**

### Safe Operations

```bash
# ✅ Update submodules to latest upstream
git submodule update --remote

# ✅ Check submodule status
git submodule status

# ✅ Initialize submodules after cloning
git submodule init
git submodule update
```

### Unsafe Operations (DON'T DO)

```bash
# ❌ DON'T edit submodule files
cd harnesses/bigcode-evaluation-harness
vim main.py  # This could lead to accidental PRs

# ❌ DON'T commit from submodule directories
cd harnesses/lm-evaluation-harness
git add . && git commit  # This creates commits in upstream repo

# ❌ DON'T push from submodules
git push  # This could create unwanted PRs
```

### How Updates Work

- Submodules are pinned to specific commits
- `git submodule update --remote` pulls latest from upstream
- This updates the pointer to the latest commit
- Your main repo tracks which commit of each submodule to use

### If You Accidentally Edit Submodules

```bash
# Reset submodule to clean state
cd harnesses/bigcode-evaluation-harness
git reset --hard HEAD
git clean -fd

# Or reinitialize completely
cd ../..
git submodule deinit harnesses/bigcode-evaluation-harness
git submodule update --init harnesses/bigcode-evaluation-harness
```

## Wrapper Integration

All customizations should go in `/src/` - never modify submodule content directly.