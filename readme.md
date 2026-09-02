# RiskOps — Production MLOps & AI Investigation Platform

> An end-to-end fraud detection platform combining **MLOps, LLMOps, Agentic AI, DevSecOps, Kubernetes, and GitOps**.

## 1. Objective

Build a production-style platform that:

1. Ingests transaction data.
2. Validates and versions the data.
3. Trains and evaluates a fraud detection model.
4. Tracks experiments and models.
5. Serves predictions through an API.
6. Uses LangGraph to investigate high-risk transactions.
7. Uses LangChain for tools and RAG.
8. Deploys through Kubernetes and GitOps.
9. Monitors application, infrastructure, and ML/LLM health.
10. Detects drift and triggers retraining.

---

## 2. Architecture

```text
Transaction
    │
    ▼
FastAPI
    │
    ▼
Fraud ML Model
    │
    ├── LOW ───────► Approve
    │
    ├── MEDIUM ────► Review
    │
    └── HIGH
         │
         ▼
     LangGraph
         │
    ┌────┼─────────────┐
    ▼    ▼             ▼
Customer Transaction  History
Profile   Analysis    Analysis
    │       │             │
    └───────┼─────────────┘
            ▼
       Risk Assessment
            │
            ▼
      Investigation Report
            │
            ▼
       Human Decision
            │
            ▼
      Feedback / Retraining
```

### Platform

```text
GitHub
  │
  ▼
CI/CD
  │
  ├── Tests
  ├── Security
  ├── Build
  └── Scan
       │
       ▼
Container Registry
       │
       ▼
GitOps Repository
       │
       ▼
Argo CD
       │
       ▼
Kubernetes
       │
       ├── ML API
       ├── Agent API
       ├── MLflow
       ├── Vector DB
       └── Observability
```

---

# 3. Key Decisions

### ML is deterministic

The fraud model produces the risk score.

**LangChain/LangGraph do not replace the fraud model.**

### Agents investigate

LangGraph orchestrates investigation workflows using controlled tools and structured evidence.

### RAG provides context

LangChain retrieves relevant fraud policies, procedures, and historical cases.

### Everything is reproducible

Data, experiments, models, configurations, and deployments are versioned.

### Deployment is GitOps

Production changes flow through Git → CI → GitOps → Argo CD → Kubernetes.

### Security is part of CI

Security scanning happens before artifacts reach production.

---

# 4. Technology Stack

| Area                | Technology               |
| ------------------- | ------------------------ |
| Language            | Python                   |
| ML                  | Scikit-learn / XGBoost   |
| Data validation     | Pandera                  |
| Experiment tracking | MLflow                   |
| Data versioning     | DVC                      |
| API                 | FastAPI                  |
| Agent framework     | LangGraph                |
| LLM tooling         | LangChain                |
| RAG                 | Qdrant                   |
| Containers          | Docker                   |
| Orchestration       | Kubernetes               |
| Packaging           | Helm                     |
| GitOps              | Argo CD                  |
| IaC                 | Terraform                |
| CI/CD               | GitHub Actions           |
| Security            | Gitleaks, Semgrep, Trivy |
| Metrics             | Prometheus               |
| Dashboards          | Grafana                  |
| Tracing             | OpenTelemetry            |
| Logs                | Loki                     |

Local development will use **Docker + Kind**. Cloud deployment will target **Azure/AKS**.

---

# 5. Build Roadmap

## Phase 1 — Foundation

Create the repository and engineering standards.

```text
Python
├── src
├── tests
├── configs
├── scripts
└── docs
```

Add:

- `pyproject.toml`
- Makefile
- pre-commit
- testing
- linting
- README
- `.gitignore`

**Outcome:** reproducible development environment.

---

## Phase 2 — Data

Build:

```text
Raw Data
   ↓
Validation
   ↓
Feature Engineering
   ↓
Versioned Dataset
```

Add:

- dataset
- validation schema
- DVC
- data tests

**Outcome:** trusted and reproducible training data.

---

## Phase 3 — ML

Build the fraud model.

```text
Dataset
   ↓
Training
   ↓
Evaluation
   ↓
Quality Gate
   ↓
Model
```

Track:

- Precision
- Recall
- F1
- PR-AUC
- ROC-AUC

**Outcome:** reproducible model training.

---

## Phase 4 — MLflow

Add:

```text
Experiments
     ↓
MLflow
     ↓
Model Registry
     ↓
Candidate
     ↓
Production
```

**Outcome:** versioned, traceable models.

---

## Phase 5 — Inference

Build the FastAPI service.

```text
POST /predict
GET  /health
GET  /ready
GET  /model-info
```

**Outcome:** production-style model serving.

---

## Phase 6 — AI Investigation

Add LangChain + LangGraph.

```text
High Risk
   ↓
LangGraph
   ↓
Gather Evidence
   ↓
Tools
   ↓
RAG
   ↓
LLM
   ↓
Investigation Report
```

**Outcome:** explainable AI-assisted investigation.

---

## Phase 7 — DevSecOps

Build CI/CD:

```text
Commit
 ↓
Tests
 ↓
SAST
 ↓
Dependency Scan
 ↓
Secret Scan
 ↓
Docker Build
 ↓
Container Scan
 ↓
Registry
```

**Outcome:** secure automated delivery.

---

## Phase 8 — Kubernetes

Deploy:

- ML API
- Agent API
- MLflow
- Qdrant
- monitoring

Add:

- probes
- resource limits
- HPA
- RBAC
- NetworkPolicies
- secrets

**Outcome:** production-style runtime platform.

---

## Phase 9 — GitOps

```text
Application Repo
      ↓
CI
      ↓
Registry
      ↓
GitOps Repo
      ↓
Argo CD
      ↓
Kubernetes
```

**Outcome:** declarative continuous delivery.

---

## Phase 10 — Observability & Retraining

Monitor:

```text
Infrastructure
Application
ML Model
LLM
Agent
Data Drift
```

Then:

```text
Drift
 ↓
Retraining
 ↓
Evaluation
 ↓
Model Registry
 ↓
Promotion
```

**Outcome:** continuous ML lifecycle.

---

---

# 7. Engineering Principle

> **Build the simplest system that demonstrates production-grade MLOps.**

We will add complexity only when it solves a real engineering problem.
