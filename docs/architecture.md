# RiskOps Architecture

## Purpose

RiskOps is a production-oriented fraud and transaction risk platform combining:

- Machine Learning
- MLOps
- LLMOps
- Agentic AI
- DevSecOps
- Kubernetes
- GitOps
- Observability

The platform is designed to demonstrate how a machine learning system moves from
data ingestion through model training, deployment, monitoring, investigation,
and continuous improvement.

## High-Level Flow

```text
Transaction
    │
    ▼
FastAPI
    │
    ▼
Fraud Detection Model
    │
    ├── LOW ───────► Approve
    │
    ├── MEDIUM ────► Human Review
    │
    └── HIGH
         │
         ▼
   Investigation
         │
         ▼
     LangGraph
         │
         ├── Customer Evidence
         ├── Transaction History
         ├── Velocity Analysis
         ├── Model Explanation
         └── Fraud Knowledge Base
                  │
                  ▼
          Investigation Report
                  │
                  ▼
            Human Decision