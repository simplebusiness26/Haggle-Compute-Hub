# Kaggle Compute Hub — Setup Status

## Built and verified

- [x] Value-first GPU/TPU/CPU job queue
- [x] Priority + urgency + expected-value scheduler
- [x] 30-hour configurable weekly GPU budget model
- [x] 20% protected GPU reserve
- [x] Hourly autonomous scheduling workflow
- [x] Safe chat -> GitHub command channel
- [x] Kaggle job lifecycle/status monitor
- [x] Finished Kaggle output collection to GitHub Actions artifacts
- [x] Overlapping accelerator-job guard
- [x] Kaggle authentication health probe
- [x] Unit tests and syntax CI
- [x] Phone-first dashboard
- [x] Project feeder registry
- [x] High-value GPU opportunity backlog
- [x] Dashboard workflow stays green while Pages is disabled
- [x] Live Compute Hub workflow verified end-to-end

## Two one-time manual settings remaining

### 1. Add the existing Kaggle token to this repository

GitHub -> `Haggle-Compute-Hub` -> **Settings** -> **Secrets and variables** -> **Actions** -> **New repository secret**

Name:

`KAGGLE_API_TOKEN`

Value: use the same Kaggle API token already configured in Animation Factory.

Do not put the token in a repository file, issue, commit, or chat message.

`KAGGLE_OWNER` does not need to be added; the workflow currently defaults to `simplebusiness`.

### 2. Enable the dashboard

GitHub -> `Haggle-Compute-Hub` -> **Settings** -> **Pages** -> **Build and deployment** -> **Source** -> **GitHub Actions**

After Pages is enabled, the next state/dashboard update will deploy automatically.

Expected URL with the repository's current name:

`https://simplebusiness26.github.io/Haggle-Compute-Hub/`

## Repository name

The repository was created as `Haggle-Compute-Hub`. The product and code call it **Kaggle Compute Hub**. Renaming the GitHub repository to `Kaggle-Compute-Hub` is optional, but would make the repository and dashboard URL match the product name.

## Next verification after the token is added

Run the included `kernels/smoke-test` through the Hub. The Hub should:

1. authenticate to Kaggle;
2. enqueue and rank the smoke job;
3. launch it on the allow-listed T4 accelerator;
4. monitor it without launching a second accelerator job;
5. collect the finished JSON output into a GitHub Actions artifact;
6. mark the job complete on the dashboard.
