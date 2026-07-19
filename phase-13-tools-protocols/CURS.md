# Phase 13: Tools & Protocols

> Инструменты и протоколы — от упаковки Python до облаков.

[Вернуться к README](../README.md)

## Оглавление

| # | Урок | Код |
|---|------|-----|
| 141 | [Python Packaging](#урок-141-python-packaging) | [Код](141-python_packaging.py) |
| 142 | [Git Advanced](#урок-142-git-advanced) | [Код](142-git_advanced.py) |
| 143 | [Docker](#урок-143-docker--containerization) | [Код](143-docker_containers.py) |
| 144 | [REST API](#урок-144-rest-api-design) | [Код](144-rest_api.py) |
| 145 | [GraphQL](#урок-145-graphql) | [Код](145-graphql.py) |
| 146 | [gRPC & Protobuf](#урок-146-grpc--protocol-buffers) | [Код](146-grpc_protobuf.py) |
| 147 | [Message Queues](#урок-147-message-queues) | [Код](147-message_queues.py) |
| 148 | [CI/CD Pipelines](#урок-148-cicd-pipelines) | [Код](148-cicd_pipelines.py) |
| 149 | [Infrastructure as Code](#урок-149-infrastructure-as-code) | [Код](149-infrastructure_as_code.py) |
| 150 | [Kubernetes](#урок-150-kubernetes-basics) | [Код](150-kubernetes_basics.py) |
| 151 | [Observability](#урок-151-monitoring--observability) | [Код](151-observability.py) |
| 152 | [Databases](#урок-152-database-fundamentals) | [Код](152-databases.py) |
| 153 | [Caching](#урок-153-caching-strategies) | [Код](153-caching.py) |
| 154 | [Security](#урок-154-security-fundamentals) | [Код](154-security_fundamentals.py) |
| 155 | [Cloud Computing](#урок-155-cloud-computing) | [Код](155-cloud_basics.py) |

---

## Урок 141: Python Packaging

### Структура пакета

```
my_package/
├── pyproject.toml
├── README.md
├── src/
│   └── my_package/
│       ├── __init__.py
│       └── module.py
└── tests/
```

### pyproject.toml

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "my-package"
version = "1.2.3"
dependencies = ["requests>=2.28"]
```

### Версионирование (SemVer)

```
MAJOR.MINOR.PATCH
1.2.3 → 1.2.4 (патч) → 1.3.0 (фича) → 2.0.0 (breaking)
```

---

## Урок 142: Git Advanced

### Ветвление

```
main ────●────●────●
          \        \
           feature   hotfix
           \        /
            ●───●───●
```

### Rebase vs Merge

```
Merge:  создаёт merge commit, сохраняет историю веток
Rebase: переносит коммиты, линейная история
```

### Conventional Commits

```
feat:     добавлена новая фича
fix:      исправлен баг
docs:     обновлена документация
refactor: рефакторинг без изменения поведения
```

---

## Урок 143: Docker & Containerization

### Слои образа

```
Layer 1: FROM python:3.11-slim     (80 MB)
Layer 2: COPY requirements.txt     (1 KB)
Layer 3: RUN pip install           (200 MB)
Layer 4: COPY .                    (50 MB)
```

### Multi-stage Build

```
Stage 1 (builder):  компиляция, зависимости
Stage 2 (runtime):  только бинарники, minimal image
```

### docker-compose

```yaml
services:
  app:
    build: .
    ports: ["8000:8000"]
    depends_on: [db]
  db:
    image: postgres:15
    volumes: ["pgdata:/var/lib/postgresql/data"]
```

---

## Урок 144: REST API Design

### HTTP Methods

```
GET    /users       → список пользователей
POST   /users       → создать пользователя
GET    /users/123   → получить пользователя 123
PUT    /users/123   → обновить пользователя 123
DELETE /users/123   → удалить пользователя 123
```

### Status Codes

```
200 OK            → успех
201 Created       → ресурс создан
400 Bad Request   → ошибка клиента
401 Unauthorized  → нет авторизации
404 Not Found     → ресурс не найден
500 Internal Error → ошибка сервера
```

### JWT Token

```
Header.Payload.Signature
eyJhbGci... . eyJzdWIi... . SflKxwRJ...
```

---

## Урок 145: GraphQL

### Schema

```graphql
type Query {
  user(id: ID!): User
  users(limit: Int): [User]
}

type User {
  id: ID!
  name: String!
  posts: [Post]
}
```

### N+1 Problem

```
Запрос: users { name, posts { title } }
N+1: 1 запрос за users + N запросов за posts каждого
Решение: DataLoader (batch + cache)
```

---

## Урок 146: gRPC & Protocol Buffers

### Protobuf Message

```protobuf
message User {
  int32 id = 1;
  string name = 2;
  string email = 3;
}
```

### RPC Methods

```protobuf
service UserService {
  rpc GetUser(GetUserRequest) returns (User);           // Unary
  rpc ListUsers(ListRequest) returns (stream User);     // Server
  rpc Upload(stream UploadRequest) returns (Response);  // Client
  rpc Chat(stream Msg) returns (stream Msg);            // BiDi
}
```

---

## Урок 147: Message Queues

### Основы

```
Producer → Broker (Queue) → Consumer
                ↓
          ACK (подтверждение)
```

### Pub/Sub

```
Producer → Exchange → Queue A (подписчик 1)
                   → Queue B (подписчик 2)
```

### Kafka

```
Topic → Partition 0: [msg1, msg4, msg7]
       → Partition 1: [msg2, msg5, msg8]
       → Partition 2: [msg3, msg6, msg9]
Consumer Group: каждый consumer читает свои партиции
```

---

## Урок 148: CI/CD Pipelines

### Pipeline Stages

```
Commit → Build → Test → Lint → Security → Deploy to Staging → Deploy to Prod
```

### Тестирование

```
Unit:      быстрые, изолированные (80% покрытия)
Integration: взаимодействие модулей
E2E:       полный сценарий пользователя
```

### Кэширование в CI

```
Cache:  ~/.cache/pip  →  ускорение установки зависимостей
Cache:  node_modules  →  ускорение npm install
```

---

## Урок 149: Infrastructure as Code

### Declarative vs Imperative

```
Declarative (Terraform): "Я хочу VM с 4 CPU и 16GB RAM"
Imperative (Ansible):    "Создай VM, затем установи Nginx"
```

### Terraform Cycle

```
terraform plan   → покажет изменения
terraform apply  → применит изменения
terraform destroy → удалит ресурсы
```

### State Management

```
State file:  хранит текущее состояние инфраструктуры
Locking:     предотвращает параллельные изменения
Backend:     S3 + DynamoDB для командной работы
```

---

## Урок 150: Kubernetes Basics

### Pod

```
apiVersion: v1
kind: Pod
metadata:
  name: my-app
spec:
  containers:
  - name: app
    image: my-app:1.0
    resources:
      requests: { cpu: "250m", memory: "128Mi" }
      limits:   { cpu: "500m", memory: "256Mi" }
```

### Service Types

```
ClusterIP:  внутренний доступ (по умолчанию)
NodePort:   доступ через порт на ноде
LoadBalancer: внешний балансировщик
```

### Rolling Update

```
Replicas: 3
Strategy: RollingUpdate (maxSurge=1, maxUnavailable=0)
Версия 1.0 → 1.1: поочерёдная замена подов
```

---

## Урок 151: Monitoring & Observability

### Три столпа

```
Metrics:    числовые метрики (CPU, latency, errors)
Logs:       текстовые записи событий
Traces:     путь запроса через систему
```

### SLI/SLO/SLA

```
SLI:   Service Level Indicator (доля успешных запросов)
SLO:   Service Level Objective (99.9% за месяц)
SLA:   Service Level Agreement (штрафы за нарушение)
```

### Error Budget

```
SLO = 99.9%
Error Budget = 0.1% = 43.8 минут/месяц
Если превышен → замедляем деплой
```

---

## Урок 152: Database Fundamentals

### SQL JOINs

```sql
SELECT * FROM users u
JOIN orders o ON u.id = o.user_id
WHERE u.active = 1
ORDER BY o.created_at DESC;
```

### Индексы

```
B-tree:   диапазонные запросы (BETWEEN, >, <)
Hash:     точные запросы (=)
Composite: составные (col1, col2)
```

### Connection Pool

```
min_connections: 5
max_connections: 20
timeout: 30s
idle_timeout: 300s
```

---

## Урок 153: Caching Strategies

### Паттерны

```
Cache-Aside:    приложение проверяет кэш → БД
Read-Through:   кэш сам загружает из БД
Write-Through:  запись в кэш + БД одновременно
Write-Behind:   запись в кэш, async в БД
```

### Eviction

```
LRU:  Least Recently Used (удаляем давно использованные)
LFU:  Least Frequently Used (удаляем редко используемые)
TTL:  Time To Live (удаляем по времени)
```

### Consistent Hashing

```
Кольцо: 0 ──── 256
Серверы: A(50), B(150), C(200)
Ключ 75 → сервер A
Ключ 175 → сервер C
При добавлении сервера → перехэшируется ~1/N ключей
```

---

## Урок 154: Security Fundamentals

### Аутентификация

```
Password Hash:   password + salt → bcrypt → hash
JWT:             Header.Payload.Signature
Session:         cookie + server-side storage
```

### Шифрование

```
Симметричное:  AES-256 (один ключ для шифрования/дешифрования)
Асимметричное: RSA (публичный шифрует, приватный дешифрует)
Хеширование:   SHA-256 (необратимый, для целостности)
```

### OWASP Top 10

```
1. Broken Access Control
2. Cryptographic Failures
3. Injection (SQL, XSS)
4. Insecure Design
5. Security Misconfiguration
```

---

## Урок 155: Cloud Computing

### Модели сервисов

```
IaaS:  виртуальные машины (AWS EC2)
PaaS:  платформа для деплоя (Heroku, Railway)
SaaS:  готовое приложение (Gmail, Slack)
FaaS:  функции (AWS Lambda, Cloud Functions)
```

### Serverless

```
Function:  код + триггер
API Gateway: маршрутизация HTTP → function
Event:     S3, SQS, DynamoDB → function
```

### CDN

```
Origin (сервер) → Edge (кэш рядом с пользователем)
Cache-Control: max-age=3600
Invalidation:  purge по тегам/путям
```
