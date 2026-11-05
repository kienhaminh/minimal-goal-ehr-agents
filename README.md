# minimal-goal-ehr-agents

Code for the paper "From Step-by-Step Instructions to Minimal Clinical Goals: Harness Engineering for Autonomous
Electronic Health Record Agents". The study asks how far an AI agent completes clinical workflows when it gets a good
environment and tools but no detailed instructions. The benchmark is MedAgentBench v2 (300 tasks, 10 task families).
This guide tells a person or an agent how to install the code and run the experiments again.

The modules in `src/` form the Python package `clinical_agent`. The code finds its folders relative to `src/`, so the
name of the clone folder does not matter.

## Design

Four configurations cross the instruction with the environment (`agent.ARMS`). The paper uses the names in the second
column; the code uses the arm names.

| Arm | Paper name | Instruction | Environment |
| --- | --- | --- | --- |
| A | Full + Base | the original task | FHIR tools only |
| B | Full + Institutional | the original task | FHIR tools and institutional tools |
| C | Goal + Base | one goal sentence | FHIR tools only |
| D | Goal + Institutional | one goal sentence | FHIR tools and institutional tools |
| `original` | Original harness | the original task as written | the upstream MedAgentBench v2 harness |
| `original-corrected` | Original harness, clarified task | the corrected task | the upstream MedAgentBench v2 harness |

The goal sentence is "Review <topic> for patient <id> and act according to the hospital's protocols." The institutional
tools are the current time, a terminology, an order catalog and a protocol library. The split is fixed in
`src/tasks.py`: 100 development, 50 validation and 150 test tasks. Each task runs once per arm and run (pass@1).

## Scoring

Every attempt is scored three times (`scores()` in `src/run.py`):

| Outcome | Scorer | Task | Scorer clock | Use in the paper |
| --- | --- | --- | --- | --- |
| `passed` | upstream v2 scorer (`new_refsol.py`) | corrected task | pinned to the task's reference time | every number of the main paper |
| `passed_upstream` | upstream v2 scorer | original task wording | machine clock at scoring time | sensitivity outcome (supplement) |
| `passed_revised` | revised scorer (`src/revised_scorer.py`, six revisions) | corrected task | pinned to the task's reference time | sensitivity outcome (supplement) |

The paper's main results report `passed` only. Every arm, including `original`, is scored with this rule: the
`original` arm gives its agent the task as written, but its attempts are scored on the corrected task with the pinned
clock, so all arms share one scoring rule.

Two mechanical repairs of the shared scoring code apply to every arm and every outcome alike; they are not a new
scorer:

- The scorer patch (`patches/medagentbench-v2.patch`) makes `extract_posts` count a POST only when it is a real
  request that the server accepted.
- `history()` in `src/run.py` passes the scorer only the tool calls that ran without error, so a write that the server
  or the schema refused, which changed nothing, does not count as an action.

## What a reproduction can and cannot give

- The agents are commercial language models. A new run gives new model outputs, so the success rates reproduce within
  sampling variation, not to the attempt.
- The runs used these model identifiers: `gpt-6-luna`, `gpt-6.1-terra`, `claude-haiku-5-5` and `claude-sonnet-5-5`, at
  reasoning effort `medium`. A provider can change or retire a model.
- Every model is called at OpenRouter with one OpenRouter API key, pinned to the model's own provider: OpenAI (its
  standard tier) for the GPT models and Anthropic for the Claude models. Each call records its route in `usage.route`.
- The paper runs reached the models in other ways. The GPT runs called the OpenAI Responses API directly through a
  ChatGPT subscription. The Claude runs used the Claude subscription and moved a call to this OpenRouter route only
  when the subscription hit its rate limit. OpenRouter sends the same messages, tools and settings as Chat Completions
  requests, so the GPT models get their earlier reasoning back through OpenRouter's reasoning details rather than the
  Responses API. A new GPT run can therefore differ slightly from a paper run beyond sampling variation.
- The design lock (`results/design-lock/lock.json`) holds the SHA-256 of the files in `src/`. This release moved the
  files to `src/` and changed the model routes, so its digest differs from the one that the paper runs record.
  The lock keeps that digest in `source_sha256_paper_runs`, and `note_2026-11-05` lists what changed. The prompts,
  tools, institutional entries, task corrections, split, agent loop and scorer are unchanged.
- This repository holds no run outputs. The `results/` folder holds only the lock files that the runners read.

## Requirements

- Python 3.11 or later (the runs used Python 3.12).
- Docker. Each run starts one FHIR container per worker from the upstream MedAgentBench image and removes the
  containers at the end.
- Network access to the model providers.
- Disk space for the run outputs: each attempt keeps its full trajectory and FHIR log.

## Installation

### 1. Make the folder layout

The code finds the benchmark in a `benchmarks/` folder next to this repository:

```text
<workspace>/
  benchmarks/
    MedAgentBench-v2/         MedAgentBench v2: tasks, scorer and upstream tools
  minimal-goal-ehr-agents/    this repository
```

```sh
mkdir workspace
cd workspace
git clone -c core.autocrlf=false https://github.com/kienhaminh/minimal-goal-ehr-agents.git
```

### 2. Get MedAgentBench v2 and apply the scorer patch

The runs used commit `46c93f3` of MedAgentBench v2 with one local patch. The patch makes the scorer count a chart
change only when the agent sent a POST request and the server accepted it, and it lets the upstream calculator call
`datetime.date.today()`.

```sh
git clone -c core.autocrlf=false https://github.com/ericoericochen/medagentbenchv2.git benchmarks/MedAgentBench-v2
git -C benchmarks/MedAgentBench-v2 checkout 46c93f39bd8679af35790eaaa5dd41bc067024a0
git -C benchmarks/MedAgentBench-v2 apply ../../minimal-goal-ehr-agents/patches/medagentbench-v2.patch
```

Check the files. The SHA-256 values must be:

| File under `benchmarks/MedAgentBench-v2/medagentbench_v2/` | SHA-256 |
| --- | --- |
| `src/MedAgentBench/data/medagentbench/new_patient_tasks.json` | `c1645ea5415b808fc3b2f0ba819dca85eff50eddf85a0863c515f7fe876b0d37` |
| `src/medagentbenchevals/new_refsol.py` | `ad299c13bc44e63a3ceac9811f1a36cecf4ec52b9744fbb230e9179f650c5b1d` |
| `src/tool/calculator.py` | `199c24e8fd994168700caec7f8a739b90ed2fdd243ba65fb094e82d2a4cd48d9` |

```sh
sha256sum benchmarks/MedAgentBench-v2/medagentbench_v2/src/MedAgentBench/data/medagentbench/new_patient_tasks.json \
  benchmarks/MedAgentBench-v2/medagentbench_v2/src/medagentbenchevals/new_refsol.py \
  benchmarks/MedAgentBench-v2/medagentbench_v2/src/tool/calculator.py
```

The clone commands set `core.autocrlf=false`, so that git keeps the line endings that the hashes assume.

### 3. Get the FHIR server image

```sh
docker pull jyxsu6/medagentbench@sha256:3fb83d7ed71c5476f9eb6212bd440a909ef7505922bbc757dc488a8fc0701966
```

### 4. Install the package

```sh
cd minimal-goal-ehr-agents
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e .
```

On Linux or macOS, use `.venv/bin/python`. Activate the virtual environment before the commands below, so that
`python` is the one in `.venv`.

### 5. Add the keys

Create `minimal-goal-ehr-agents/.env`:

```sh
OPENROUTER_API_KEY=<your OpenRouter key>
```

The key serves every model. The OpenRouter account must have credit left: a key at its spending limit returns HTTP 403, and the attempt ends with `agent_error`.

### 6. Check the design lock

`run.py` starts a validation or test run only when the files in `src/` match `results/design-lock/lock.json`. Check
the match before a run:

```sh
python -c "import json; from clinical_agent import run; l = json.load(open('results/design-lock/lock.json'))['source_sha256']['combined']; print('source matches the lock:', l == run.source_sha256()['combined'])"
```

The output must be `source matches the lock: True`. The sensitivity runners check their own files against
`results/sensitivity-lock/plan.json` and `results/followup-lock/plan.json` in the same way.

## Run the experiments

`python -m clinical_agent.run` runs every arm and model. Every run calls a paid model and writes
`results/<run id>/`: `config.json` (the command, all parameters and the source digest), `attempts/<task id>.json`,
`summary.json` and the FHIR container receipt. Each run starts its own FHIR containers, and each attempt sees the
original records. The number of workers changes only the speed; the paper runs used 3 to 6 workers. If a run stops
because of a provider or infrastructure error, delete the failed attempt file and start the same command again with
`--resume`.

To try the setup, run one development task under a run identifier of its own, and delete that folder afterwards, so
that no trial attempt mixes with the experiment runs:

```sh
python -m clinical_agent.run --arm D --part dev --tasks task10_17 --run-id check --model claude-haiku-5-5 --workers 1
```

### Primary and repeated test runs (150 test tasks)

The first run of each arm and model is the primary analysis. The three runs of one batch give the pooled outcomes and
pass^3. For GPT-6-Luna the batch is the three runs after the primary run; for the other models the primary run is the
first run of the batch.

```sh
# GPT-6-Luna: primary run and three repeated runs
for arm in A B C D; do
  python -m clinical_agent.run --arm $arm --part test --run-id $arm-test-150 --model gpt-6-luna --workers 6
  for k in 1 2 3; do
    python -m clinical_agent.run --arm $arm --part test --run-id $arm-test-150-rep$k --model gpt-6-luna --workers 6
  done
done

# GPT-6.1-Terra: three runs
for arm in A B C D; do
  for suffix in terra terra-rep2 terra-rep3; do
    python -m clinical_agent.run --arm $arm --part test --run-id $arm-test-150-$suffix --model gpt-6.1-terra --workers 6
  done
done

# Claude Haiku 5.5 and Claude Sonnet 5.5: three runs each
for model in haiku sonnet; do
  for arm in A B C D; do
    for suffix in $model $model-rep2 $model-rep3; do
      python -m clinical_agent.run --arm $arm --part test --run-id $arm-test-150-$suffix --model claude-$model-5-5 --workers 4
    done
  done
done
```

### Original harness (150 test tasks)

The original harness has 8 steps, as upstream. The second run of each model gives it 16 steps, to compare with arms A
to D at 16 steps. Arm `original-corrected` gives it the corrected task of arm A.

```sh
for model in gpt-6-luna:"" gpt-6.1-terra:-terra claude-haiku-5-5:-haiku claude-sonnet-5-5:-sonnet; do
  m=${model%%:*}; s=${model##*:}
  python -m clinical_agent.run --arm original --part test --run-id original-test-150$s --model $m --workers 6
  python -m clinical_agent.run --arm original --part test --run-id original-test-150$s-steps16 --model $m --max-steps 16 --workers 3
  python -m clinical_agent.run --arm original-corrected --part test --run-id original-corrected-test-150$s --model $m --workers 5
done
```

### Validation runs (GPT-6-Luna, 50 validation tasks)

```sh
for arm in A B C D; do
  python -m clinical_agent.run --arm $arm --part validate --run-id $arm-validate-50 --model gpt-6-luna --workers 6
done
```

### Sensitivity analyses of the institutional libraries (GPT-6-Luna, three runs each)

`sensitivity/run_sensitivity.py` replaces the library entries in memory (`sensitivity/libraries.json`):
S1 paraphrased protocols, S2 realistic library, S3 stress library, S4 no protocol tool. `sensitivity/run_followup.py`
replaces the terminology entries or the goal sentence (`sensitivity/followup.json`): S5 terminology without record
locations, S6 reworded goals. The plans are in `results/sensitivity-lock/plan.json` and
`results/followup-lock/plan.json`.

```sh
for k in 1 2 3; do
  for s in S1_paraphrase:s1 S2_realistic:s2 S3_stress:s3 S4_no_protocols:s4; do
    python sensitivity/run_sensitivity.py --condition ${s%%:*} --arm D --part test --run-id D-test-150-${s##*:}-rep$k --model gpt-6-luna --workers 6
  done
  for s in S2_realistic:s2 S3_stress:s3; do
    python sensitivity/run_sensitivity.py --condition ${s%%:*} --arm B --part test --run-id B-test-150-${s##*:}-rep$k --model gpt-6-luna --workers 6
  done
  for arm in B D; do
    python sensitivity/run_followup.py --condition S5_no_record_locations --arm $arm --part test --run-id $arm-test-150-s5-rep$k --model gpt-6-luna --workers 6
  done
  python sensitivity/run_followup.py --condition S6_goal_paraphrase --arm D --part test --run-id D-test-150-s6-rep$k --model gpt-6-luna --workers 6
done

# GPT-6.1-Terra, realistic library, one run per arm
for arm in B D; do
  python sensitivity/run_sensitivity.py --condition S2_realistic --arm $arm --part test --run-id $arm-test-150-s2-terra --model gpt-6.1-terra --workers 6
done
```

### Score saved attempts again

To score saved attempts again without a model call, for example after a change of the scorer:

```sh
python -m clinical_agent.rescore --runs A-test-150 D-test-150 --out rescore-test
```

## Layout

```text
src/                     The package clinical_agent (the locked design)
  settings.py            Paths, agent models and .env loading
  tasks.py               MedAgentBench v2 tasks, task corrections and the split
  tools.py               Base environment: upstream FHIR tools with the mechanical fixes (FIXES)
  institution.py         Institutional environment: base tools and the institutional tools
  agent.py               Arms, instructions, system prompt, review step and the agent loop
  run.py                 Runner: FHIR containers, attempts, scoring, results/<run>/
  rescore.py             Offline rescoring of saved attempts
  revised_scorer.py      Revised scorer for the sensitivity outcome
  fhir_workers.py        Disposable FHIR containers, one per worker
  openrouter_transport.py OpenRouter transport of every model
sensitivity/             Sensitivity runners and their variant entries
patches/                 The local patch of the MedAgentBench v2 scorer and calculator
results/*-lock/          The locks of the design and of the sensitivity plans
```

`python -m clinical_agent.run --help` lists the run options. The sensitivity runners take the same options and add
`--condition`; their docstrings describe them.
