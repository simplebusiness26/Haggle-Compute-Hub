# Kaggle Compute Hub

Central control plane for getting the most value from Kaggle's free compute.

## What it does

- Keeps one queue for GPU, TPU and CPU work.
- Scores jobs by expected value, urgency, priority and compute cost.
- Routes CPU work away from Kaggle so GPU quota is not wasted.
- Supports Kaggle kernel dispatch through an approval-safe command channel.
- Tracks estimated weekly GPU usage and job history.
- Publishes a phone-first dashboard from repository state.
- Keeps job execution allow-listed: no arbitrary shell commands from queue data.

## Control flow

`ChatGPT -> control/command.json -> GitHub Actions -> Compute Hub scheduler -> Kaggle CLI -> Kaggle job -> state/results -> dashboard`

## Job families

The initial catalogue supports:

- Animation / image-to-video
- Visual asset generation and image processing
- Model benchmarking
- Fine-tuning
- Document AI
- Media processing
- Embeddings / RAG preprocessing
- Dataset enrichment
- Game AI / simulation
- Kaggle competition experiments

## Commands

The hub reads `control/command.json`.

### Add a job

```json
{
  "action": "enqueue",
  "request_id": "example-001",
  "job": {
    "title": "Benchmark a T4-friendly image model",
    "family": "model_lab",
    "compute": "gpu",
    "kernel_path": "kernels/smoke-test",
    "estimated_minutes": 20,
    "value": 80,
    "priority": "high"
  }
}
```

### Run the scheduler

```json
{
  "action": "schedule",
  "request_id": "schedule-001"
}
```

### Status only

```json
{
  "action": "status",
  "request_id": "status-001"
}
```

### Cancel a queued job

```json
{
  "action": "cancel",
  "request_id": "cancel-001",
  "job_id": "job-..."
}
```

## One-time credential setup

Real Kaggle dispatch requires a GitHub Actions secret named `KAGGLE_API_TOKEN`. Keep the token only in GitHub Actions secrets; never commit it.

The hub remains safe to run without the secret while the queue is empty or when using status/queue-management commands.

## Dashboard

`dashboard/` is a static phone-first UI. The Pages workflow stages the latest queue, history, resources and catalogue into the published site.

## Safety rails

- The repository is a scheduler, not an always-on GPU server.
- It is designed for ML/data/batch jobs that fit Kaggle's intended use.
- No arbitrary command execution is accepted from job JSON.
- Jobs over the configured weekly budget remain queued.
- Paid compute is never enabled by this project.
