# Rob-LAPI(S)^2: Language to Action Planning via Iterative Schema Synthesis

A neuro-symbolic planning pipeline that auto-regressively elicits and refines PDDL planning models from natural language descriptions. Symbolic validation (VAL, CPDDL, FastDownward) closes the loop on LLM generation, ensuring that elicited domains are grounded in formal constraints.

## Paper Results

| Benchmark | VAL (%) | GT (%) | Avg Time (s) |
|-----------|---------|--------|--------------|
| IPC (7 domains, 20 problems each) | 96 | 73 | 8.9 |
| VirtualHome (20 tasks) | 90 | 90 | 41 |
| AlfWorld (20 tasks) | 55 | 55 | 149 |

- **VAL%**: Symbolic self-consistency (valid PDDL domain+problem pair)
- **GT%**: Ground-truth runnability in simulator

## Requirements

- Python 3.10+
- Linux (tested on Ubuntu 22.04)
- ~2GB disk space for dependencies

## Environment Setup

### 1. Clone Repository

```bash
git clone git@github.com:YOUR_ORG/Rob-LAPIS.git
cd Rob-LAPIS
```

### 2. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure API Keys

Copy the example environment file and add your API key:

```bash
cp .env.example .env
```

Edit `.env` and set:

```bash
# Required - Claude Sonnet 4.6 (default model)
ANTHROPIC_API_KEY=sk-ant-...

# Optional - for OpenAI models (gpt-4o, etc.)
OPENAI_API_KEY=sk-...

# Optional - for Gemini models
GEMINI_API_KEY=...
```

### 4. Third-Party Dependencies

The pipeline requires external tools for PDDL validation and planning. Set them up in the `third-party/` directory:

```bash
mkdir -p third-party
```

#### 4.1 VAL - PDDL Validator (Required)

VAL validates generated PDDL plans against domain/problem specifications.

```bash
# Clone and build VAL
git clone https://github.com/KCL-Planning/VAL.git third-party/VAL
cd third-party/VAL
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ../../..

# Verify installation
./third-party/VAL/build/Validate --help
```

**Build dependencies** (Ubuntu/Debian):
```bash
sudo apt-get install build-essential cmake flex bison
```

#### 4.2 LLM-PDDL - IPC Benchmark Data (Required for IPC benchmarks)

Contains natural language descriptions and ground-truth PDDL for 7 IPC domains.

```bash
git clone https://github.com/Cranial-XIX/llm-pddl.git third-party/llm-pddl
```

The repository includes domains: `blocksworld`, `floortile`, `tyreworld`, `storage`, `barman`, `grippers`, `termes` (20 problems each).

#### 4.3 CPDDL - Static Analysis (Optional)

CPDDL provides h² reachability analysis for detecting unsolvable problems before planning. Requires Singularity/Apptainer.

```bash
mkdir -p third-party/cpddl

# Option 1: Download pre-built container (~170MB)
# Contact authors for cpddl_latest.sif or build from source

# Option 2: Build from source
git clone https://gitlab.com/danfis/cpddl.git /tmp/cpddl-src
cd /tmp/cpddl-src
singularity build third-party/cpddl/cpddl_latest.sif Singularity
```

**Install Singularity** (Ubuntu):
```bash
sudo apt-get install singularity-container
# Or for newer systems: sudo apt-get install apptainer
```

The pipeline works without CPDDL but won't perform static analysis checks.

#### 4.4 AlfWorld (Optional - for AlfWorld benchmarks)

```bash
pip install alfworld
alfworld-download  # Downloads ~2GB of game data
```

#### 4.5 VirtualHome (Optional - for VirtualHome benchmarks)

```bash
pip install virtualhome

# Download Unity simulator for execution
# See: https://github.com/xavierpuigf/virtualhome/releases
# Extract to: third-party/virtualhome/simulation/
```

### 5. Benchmark Data

Prepare IPC benchmark data from the llm-pddl repository:

```bash
python prepare_llmpp_data.py
```

This creates `data/llm-pddl/{domain}/{problem_id}/` with:
- `nl` - Natural language task + domain description
- `domain.pddl` - Ground-truth PDDL domain
- `problem.pddl` - Ground-truth PDDL problem

## Running Benchmarks

### IPC Domains (Table 1 in paper)

Seven IPC domains: `blocksworld`, `floortile`, `tyreworld`, `storage`, `barman`, `grippers`, `termes`

```bash
# Run LAPIS on a single domain (generates domain + problem from NL)
python run_benchmark.py --domain blocksworld --method lapis --generate_domain --ablation full_adequacy

# Run LAPIS with ground-truth domain provided
python run_benchmark.py --domain blocksworld --method lapis

# Run LLM+P baseline (no refinement)
python run_benchmark.py --domain blocksworld --method llmpp --generate_domain

# Run all 7 IPC domains
python run_benchmark.py --domain all --method lapis --generate_domain --ablation full_adequacy

# Side-by-side comparison (LLM+P vs LAPIS)
python run_benchmark.py --domain blocksworld --method compare --generate_domain
```

**Ablation modes** (`--ablation`):
- `baseline` - Legacy prompts, no schema injection
- `clean_domain` - Clean domain prompt only
- `schema_problem` - Schema-injected problem prompt
- `full` - Clean domain + schema injection (default)
- `full_adequacy` - Full + 3-step CoT adequacy checks (recommended for paper results)

### VirtualHome (Table 2 in paper)

20 household tasks from the EAI benchmark:

```bash
# With ground-truth domain provided (GT variant)
python run_virtualhome_lapis.py --mode gt

# With domain synthesis from NL (Syn variant)
python run_virtualhome_lapis.py --mode syn

# Specific task
python run_virtualhome_lapis.py --task "Watch TV"
```

**VirtualHome Setup** (required for GT execution):
```bash
# Install VirtualHome package
pip install virtualhome

# Download Unity simulator (required for execution)
# See: https://github.com/xavierpuigf/virtualhome
```

### AlfWorld (Table 2 in paper)

20 trials from AlfWorld training split:

```bash
# With ground-truth domain (GT variant)
python run_alfworld_lapis.py --mode gt

# With domain synthesis from NL (Syn variant)
python run_alfworld_lapis.py --mode syn

# Specific task type
python run_alfworld_lapis.py --task_type pick_cool_then_place_in_recep
```

**AlfWorld Setup** (required for GT execution):
```bash
pip install alfworld
alfworld-download  # Downloads game data
```

## Results

Results are written to:
- IPC: `results_llmpp/benchmark_llmpp_{domain}_{method}_{model}/{timestamp}/`
- VirtualHome: `results_virtualhome/{timestamp}/`
- AlfWorld: `results_alfworld/{timestamp}/`

Each problem produces a `manifold.json` with:
- `planning_successful` - Planner found a plan
- `val_valid` - VAL validation passed
- `gt_executable` - Plan runs in ground-truth simulator
- `timing` - LLM and planner timing
- `pddl_refinements` - Number of refinement iterations

## Architecture

```
src/lapis/
  agents/           # LLM wrappers (Claude, GPT, Gemini)
  pipelines/        # Pipeline implementations
    lapis_low_level.py   # Main LAPIS pipeline for IPC
    baseline.py          # LLM+P baseline
  planner/low/      # PDDL generation, refinement, validation
  simulators/       # Ground-truth simulators per domain
  utils/            # Utilities (logging, PDDL parsing)
  validators/       # VAL/CPDDL integration

third-party/
  VAL/              # PDDL validator binary
  llm-pddl/         # IPC domain data + FastDownward
  cpddl/            # Static analysis container

data/
  llm-pddl/         # IPC benchmark data
  virtualhome/      # VirtualHome task data
  alfworld/         # AlfWorld task data
```

## Reproducing Paper Results

To reproduce the exact results from the paper:

```bash
# 1. IPC benchmark (Table 1, Sim-Rob-LAPI(S)^2 column)
for domain in blocksworld floortile tyreworld storage barman grippers termes; do
    python run_benchmark.py --domain $domain --method lapis \
        --generate_domain --ablation full_adequacy --model claude-sonnet-4-6
done

# 2. VirtualHome (Table 2, Syn variant)
python run_virtualhome_lapis.py --mode syn --model claude-sonnet-4-6

# 3. AlfWorld (Table 2, Syn variant)
python run_alfworld_lapis.py --mode syn --model claude-sonnet-4-6
```

## Configuration Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `claude-sonnet-4-6` | LLM model to use |
| `--generate_domain` | False | Generate domain from NL (else use GT) |
| `--ablation` | `full` | Prompt ablation mode |
| `--pddl_gen_iterations` | 3 (LAPIS) / 0 (LLM+P) | Max refinement iterations |
| `--planner_timeout` | 180 | Planner timeout in seconds |
| `--semantic_checks` | False | Enable semantic verification |
| `--refine_domain` | False | Enable domain refinement on failures |

## Planner Backends

| `--planner` | Backend | Notes |
|-------------|---------|-------|
| `pyperplan` (default) | Pyperplan via Unified Planning | Pure Python, no binary needed |
| `up_fd` | FastDownward via Unified Planning | `pip install up-fast-downward` |
| `fd` | FastDownward direct subprocess | Requires `fast-downward` binary |

## Troubleshooting

**Missing API key error:**
```
RuntimeError: Missing ANTHROPIC_API_KEY
```
Set your API key in `.env` or export it: `export ANTHROPIC_API_KEY=sk-ant-...`

**VAL not found:**
Ensure `third-party/VAL/validate` exists and is executable.

**CPDDL static analysis fails:**
Requires Singularity/Apptainer. The container is in `third-party/cpddl/cpddl_latest.sif`.

## License

MIT License
