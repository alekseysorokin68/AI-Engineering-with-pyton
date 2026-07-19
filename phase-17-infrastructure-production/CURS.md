# Phase 17: Infrastructure & Production

> Инфраструктура и продакшен — от распределённых систем до ML-платформ.

[Вернуться к README](../README.md)

## Оглавление

| # | Урок | Код |
|---|------|-----|
| 201 | [Distributed Systems](#урок-201-distributed-systems) | [Код](201-distributed_systems.py) |
| 202 | [Container Orchestration](#урок-202-container-orchestration) | [Код](202-container_orchestration.py) |
| 203 | [CI/CD for ML](#урок-203-cicd-for-ml) | [Код](203-cicd_ml.py) |
| 204 | [Feature Stores](#урок-204-feature-stores) | [Код](204-feature_stores.py) |
| 205 | [ML Monitoring](#урок-205-ml-monitoring) | [Код](205-ml_monitoring.py) |
| 206 | [A/B Testing for ML](#урок-206-ab-testing-for-ml) | [Код](206-ab_testing_ml.py) |
| 207 | [Model Registry](#урок-207-model-registry--governance) | [Код](207-model_registry.py) |
| 208 | [Data Pipelines](#урок-208-data-pipelines) | [Код](208-data_pipelines.py) |
| 209 | [Edge Deployment](#урок-209-edge-deployment) | [Код](209-edge_deployment.py) |
| 210 | [GPU & Accelerators](#урок-210-gpu--accelerators) | [Код](210-gpu_accelerators.py) |
| 211 | [Cost Optimization](#урок-211-cost-optimization) | [Код](211-cost_optimization.py) |
| 212 | [Disaster Recovery](#урок-212-disaster-recovery) | [Код](212-disaster_recovery.py) |
| 213 | [Compliance & Governance](#урок-213-compliance--governance) | [Код](213-compliance_governance.py) |
| 214 | [Team & Process](#урок-214-team--process) | [Код](214-team_process.py) |
| 215 | [ML Platform](#урок-215-building-ml-platforms) | [Код](215-ml_platform.py) |

---

## Урок 201: Distributed Systems

### CAP Theorem

```
Consistency:     все ноды видят одни данные
Availability:    каждый запрос получает ответ
Partition Tolerance: система работает при разрыве сети

Выбор 2 из 3:
  CP: консистентность + толерантность (ZooKeeper)
  AP: доступность + толерантность (Cassandra)
  CA: консистентность + доступность (нет partition → единичная нода)
```

### Consistency Models

```
Strong:      немедленная консистентность (write → read видит)
Eventual:    со временем все ноды синхронизируются
Causal:      причинно-следственные связи сохраняются
Monotonic:   чтения не откатываются назад
```

### Raft Consensus

```
Leader Election: кандидат → голосование → лидер
Log Replication: лидер записывает → реплицирует → majority → commit
Safety: только полные логи реплицируются
```

---

## Урок 202: Container Orchestration

### Kubernetes Resources

```
StatefulSet:  стабильные имена, персистентные тома
DaemonSet:    под на каждой ноде (мониторинг, логи)
Job:          одноразовые задачи
CronJob:      запуск по расписанию
```

### Service Mesh

```
Sidecar Proxy:  envoy/nginx рядом с каждым подом
mTLS:           шифрование между сервисами
Traffic:        canary, circuit breaking, retry
```

### GitOps

```
Git Repo ( desired state ) → ArgoCD → Kubernetes ( actual state )
Drift detected → automatic sync or alert
```

---

## Урок 203: CI/CD for ML

### ML Pipeline

```
Data Validation → Feature Engineering → Training → Evaluation → Registration → Deployment
     ↓                    ↓                  ↓           ↓
  quality checks     transformations    hyperparams   metrics gate
```

### Model Versioning

```
model-v1.0.0  → staging → canary → production
     ↓
  lineage: data → training run → metrics → deployment
```

---

## Урок 204: Feature Stores

### Architecture

```
Offline Store:   batch features (Parquet, Hive) → training
Online Store:    low-latency features (Redis, DynamoDB) → serving
Registry:        feature metadata, versioning, lineage
```

### Feature Freshness

```
Batch:     обновление раз в день/час
Streaming: обновление в реальном времени
On-demand: вычисление при запросе
```

---

## Урок 205: ML Monitoring

### Data Drift

```
PSI = Σ (A_i - B_i) × ln(A_i / B_i)
PSI < 0.1:   нет дрейфа
PSI 0.1-0.2: умеренный
PSI > 0.2:   значительный

KS test: p-value < 0.05 → дрейф
```

### Model Drift

```
Baseline accuracy: 0.92
Current accuracy:  0.85
Degradation:       7.6%
Trigger retrain if degradation > threshold
```

---

## Урок 206: A/B Testing for ML

### Sample Size

```
n = (Z_α/2 + Z_β)² × 2σ² / δ²
α: significance level (0.05)
β: power (0.8)
δ: minimum detectable effect
```

### Multi-Armed Bandits

```
Epsilon-Greedy:  ε chance random, 1-ε best arm
UCB1:            arm_score = mean + sqrt(2×ln(t)/n)
Thompson:        sample from Beta posterior per arm
```

---

## Урок 207: Model Registry & Governance

### Stage Transitions

```
None → Staging → Production → Archived
         ↓            ↓
      manual QA    canary → full rollout
```

### Approval Workflow

```
Submit → Auto Tests → Human Review → Approve → Deploy
              ↓
         reject + feedback
```

---

## Урок 208: Data Pipelines

### ETL vs ELT

```
ETL:  Extract → Transform → Load (transform before storage)
ELT:  Extract → Load → Transform (transform in warehouse)
```

### Stream Processing

```
Windowing:
  Tumbling: [1-5][6-10][11-15]
  Sliding:  [1-5][3-7][5-9]
  Session:  [1-3][8-12] (gap > threshold)

Exactly-once: idempotent writes + transaction log
```

---

## Урок 209: Edge Deployment

### Model Optimization

```
Pruning:          удаление неважных весов
Quantization:     FP32 → INT8 (4x compression)
Distillation:     large model → small model
```

### ONNX

```
Model → ONNX Export → ONNX Runtime → Cross-platform
Benefits: framework-agnostic, optimized inference
```

---

## Урок 210: GPU & Accelerators

### GPU Architecture

```
SM (Streaming Multiprocessor): 128-2048 CUDA cores
Warp: 32 threads executing in lockstep
Global Memory: 16GB+ HBM (high latency)
Shared Memory: 48-164KB per SM (low latency)
```

### Mixed Precision

```
FP32: 4 bytes, full precision
FP16: 2 bytes, reduced range
BF16: 2 bytes, same range as FP32
Loss Scaling: prevent gradient underflow in FP16
```

---

## Урок 211: Cost Optimization

### Instance Selection

```
Training:   GPU instances (A100, H100)
Inference:  CPU or small GPU (T4, L4)
Batch:      spot instances (70% cheaper)
```

### Auto-Scaling

```
Metric:   CPU > 70% → scale up
Cooldown: 300s (prevent flapping)
Min/Max:  2-10 instances
```

---

## Урок 212: Disaster Recovery

### RTO/RPO

```
RTO (Recovery Time Objective):  máximo время простоя
RPO (Recovery Point Objective): máximo количество потерянных данных
```

### Backup Strategies

```
Full:       полная копия (раз в неделю)
Incremental: изменения с последнего бакапа (ежедневно)
Point-in-time: восстановление на любой момент
```

### Chaos Engineering

```
Hypothesis: система выдержит отказ ноды
Experiment: отключить случайную ноду
Observe:    сервис продолжает работать
Learn:      улучшить отказоустойчивость
```

---

## Урок 213: Compliance & Governance

### GDPR

```
Consent:         явное согласие на обработку
Right to Erasure: право на удаление
Data Minimization: собирать только необходимое
Purpose Limitation: использовать только для заявленной цели
```

### Explainability

```
SHAP:     вклад каждого признака в предсказание
LIME:     локальная интерпретируемая модель
Feature Importance: permutation importance
```

---

## Урок 214: Team & Process

### MLOps Maturity Levels

```
Level 0: Manual (ручной скрипт)
Level 1: Pipeline (автоматический пайплайн)
Level 2: CI/CD (тесты + деплой)
Level 3: Monitoring (мониторинг + алертинг)
Level 4: Full Automation (автономная платформа)
```

### Agile for ML

```
Sprint: 2 недели
Backlog: задачи + эксперименты
Definition of Done: модель протестирована + задеплоена
Retrospective: что улучшить в процессе
```

---

## Урок 215: Building ML Platforms

### Platform Layers

```
Infrastructure:  вычисления, хранилища, сеть
Data Layer:      данные, фичи, качество
Training Layer:  оркестрация, трекинг, версионирование
Serving Layer:   инференс, батчинг, кэширование
Monitoring:      метрики, дрейф, алерты
```

### Adoption Strategy

```
Phase 1: 1-2 пилотных проекта
Phase 2: стандарты и best practices
Phase 3: расширение на команды
Phase 4: полная автоматизация
```
