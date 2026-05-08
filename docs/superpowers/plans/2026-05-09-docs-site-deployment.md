# Docs Site Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and deploy a public documentation site for Bayesian-Agent using MkDocs Material and GitHub Pages.

**Architecture:** The documentation site is generated from Markdown files under `docs/` with a root `mkdocs.yml` configuration. GitHub Actions builds the static site on every push to `main`, uploads the generated `site/` directory as a GitHub Pages artifact, and deploys it through GitHub Pages Actions.

**Tech Stack:** MkDocs, Material for MkDocs, GitHub Pages, GitHub Actions, Python packaging optional extras.

---

### Task 1: Add MkDocs Project Configuration

**Files:**
- Create: `mkdocs.yml`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add MkDocs Material config**

Create `mkdocs.yml` with site metadata, navigation, theme configuration, markdown extensions, and asset references.

- [ ] **Step 2: Add docs optional dependency group**

Modify `pyproject.toml` with:

```toml
[project.optional-dependencies]
docs = [
  "mkdocs-material>=9.6,<10",
]
```

- [ ] **Step 3: Verify config is parseable**

Run:

```bash
python3 -m pip install -e ".[docs]"
mkdocs build --strict
```

Expected: MkDocs exits with code 0.

### Task 2: Write Documentation Pages

**Files:**
- Create: `docs/index.md`
- Create: `docs/quick-start.md`
- Create: `docs/core-concepts.md`
- Create: `docs/architecture.md`
- Create: `docs/cli.md`
- Create: `docs/python-api.md`
- Create: `docs/adapters.md`
- Create: `docs/schemas.md`
- Create: `docs/experiments/index.md`
- Create: `docs/roadmap.md`
- Create: `docs/citation.md`
- Preserve: `docs/method.md`
- Preserve: `docs/experiments.md`

- [ ] **Step 1: Write landing page**

Use `docs/index.md` as the docs homepage with the banner, overview image, core value proposition, installation snippet, and links to primary docs.

- [ ] **Step 2: Write user-facing guides**

Write quick start, core concepts, architecture, CLI, API, adapters, schemas, experiments, roadmap, and citation pages using existing README content and project files.

- [ ] **Step 3: Preserve existing technical notes**

Keep `docs/method.md` and `docs/experiments.md` available in the navigation as reference notes.

### Task 3: Add GitHub Pages Workflow

**Files:**
- Create: `.github/workflows/docs.yml`

- [ ] **Step 1: Add Pages workflow**

Create a workflow triggered by pushes to `main` and manual dispatch. The build job installs `.[docs]`, runs `mkdocs build --strict`, and uploads `site/` with `actions/upload-pages-artifact@v4`.

- [ ] **Step 2: Add deploy job**

Deploy with `actions/deploy-pages@v4`, using `pages: write` and `id-token: write` permissions and the `github-pages` environment.

### Task 4: Link Docs From Repository Entry Points

**Files:**
- Modify: `README.md`
- Modify: `README_ZH.md`

- [ ] **Step 1: Add docs links**

Add links to the documentation site in the centered header once the deployment URL is known or inferable as `https://dataarctech.github.io/Bayesian-Agent/`.

- [ ] **Step 2: Preserve existing README edits**

Keep the local README citation author change to `Xiaojun Wu` and keep the removed English Status section removed.

### Task 5: Verify, Commit, Push, and Deploy

**Files:**
- All files from Tasks 1-4

- [ ] **Step 1: Run local verification**

Run:

```bash
mkdocs build --strict
python3 -m unittest discover -v
python3 -m compileall bayesian_agent
git diff --check
```

Expected: all commands exit with code 0.

- [ ] **Step 2: Commit and push**

Run:

```bash
git add mkdocs.yml pyproject.toml .github/workflows/docs.yml docs README.md README_ZH.md
git commit -m "docs: add mkdocs site deployment"
git pull --rebase origin main
git push origin main
```

- [ ] **Step 3: Check deployment**

Run:

```bash
gh run list --workflow docs.yml --limit 1
gh run watch <run-id> --exit-status
```

Expected: the workflow completes successfully. If Pages is not enabled for GitHub Actions, configure the repository Pages source to GitHub Actions and rerun the workflow.
