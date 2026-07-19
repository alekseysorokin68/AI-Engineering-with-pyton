"""145 — GraphQL: schemas, resolvers, queries, mutations, subscriptions

Темы:
  1. Schema Definition (types, queries, mutations, enums)
  2. Query Language (fields, arguments, fragments, aliases)
  3. Resolvers & Data Sources (N+1 problem, DataLoader pattern)
  4. Subscriptions & Real-time (WebSocket simulation, event streams)

Самодостаточный файл — не требует numpy, torch, transformers.
"""

import math
import random
import re
import json
import hashlib
import collections
import time
import base64

random.seed(42)

# =============================================================================
# Демо 1: Schema Definition
# =============================================================================

def demo_schema_definition():
    """Демонстрация определения GraphQL-схемы: типы, запросы, мутации, перечисления."""
    print("=" * 70)
    print("Демо 1: Schema Definition (types, queries, mutations, enums)")
    print("=" * 70)

    # --- 1.1 Основные GraphQL-типы ---
    print("\n--- 1.1 Основные скалярные типы GraphQL ---")

    scalars = {
        "Int":     {"description": "Целое число (32 бита)",      "example": 42,           "python": "int"},
        "Float":   {"description": "Число с плавающей точкой",   "example": 3.14,         "python": "float"},
        "String":  {"description": "Строка Unicode",             "example": "hello",      "python": "str"},
        "Boolean": {"description": "Логическое значение",        "example": True,         "python": "bool"},
        "ID":      {"description": "Уникальный идентификатор",   "example": "abc123",     "python": "str"},
    }

    for gtype, info in scalars.items():
        print(f"  {gtype:10s} — {info['description']:35s} | пример: {str(info['example']):10s} | Python: {info['python']}")

    # --- 1.2 Определение типов (SDL) ---
    print("\n--- 1.2 Schema Definition Language (SDL) ---")

    schema_sdl = """
# Тип "Пользователь"
type User {
    id: ID!
    name: String!
    email: String!
    role: Role!
    posts: [Post!]!
    createdAt: DateTime!
}

# Перечисление ролей
enum Role {
    ADMIN
    EDITOR
    VIEWER
}

# Тип "Пост"
type Post {
    id: ID!
    title: String!
    content: String!
    author: User!
    tags: [String!]!
    status: PostStatus!
    likes: Int!
}

enum PostStatus {
    DRAFT
    PUBLISHED
    ARCHIVED
}

# Входные типы для мутаций
input CreatePostInput {
    title: String!
    content: String!
    tags: [String!] = []
}

input UpdatePostInput {
    title: String
    content: String
    tags: [String!]
    status: PostStatus
}

# Корневой тип запросов
type Query {
    user(id: ID!): User
    users(limit: Int, offset: Int): [User!]!
    post(id: ID!): Post
    posts(status: PostStatus, authorId: ID): [Post!]!
}

# Корневой тип мутаций
type Mutation {
    createPost(input: CreatePostInput!): Post!
    updatePost(id: ID!, input: UpdatePostInput!): Post!
    deletePost(id: ID!): Boolean!
}
"""
    # Показать схему
    for line in schema_sdl.strip().split("\n"):
        if line.strip():
            print(f"  {line}")

    # --- 1.3 Программное представление схемы ---
    print("\n--- 1.3 Программное представление GraphQL-схемы ---")

    class GraphQLType:
        """Базовый GraphQL-тип."""
        def __init__(self, name, fields=None, description=""):
            self.name = name
            self.fields = fields or {}
            self.description = description

        def __repr__(self):
            return f"GraphQLType({self.name})"

    class GraphQLField:
        """Поле GraphQL-типа."""
        def __init__(self, name, type_name, nullable=True, is_list=False, is_required=False):
            self.name = name
            self.type_name = type_name
            self.nullable = nullable
            self.is_list = is_list
            self.is_required = is_required

        def to_sdl(self):
            """Генерация SDL-строки для поля."""
            type_str = self.type_name
            if self.is_list:
                type_str = f"[{type_str}!]"
            if not self.nullable:
                type_str += "!"
            return f"{self.name}: {type_str}"

    # Определяем типы
    user_type = GraphQLType("User", description="Пользователь системы")
    user_type.fields = {
        "id":        GraphQLField("id", "ID", nullable=False),
        "name":      GraphQLField("name", "String", nullable=False),
        "email":     GraphQLField("email", "String", nullable=False),
        "role":      GraphQLField("role", "Role", nullable=False),
        "posts":     GraphQLField("posts", "Post", is_list=True),
        "createdAt": GraphQLField("createdAt", "DateTime", nullable=False),
    }

    post_type = GraphQLType("Post", description="Статья/пост")
    post_type.fields = {
        "id":      GraphQLField("id", "ID", nullable=False),
        "title":   GraphQLField("title", "String", nullable=False),
        "content": GraphQLField("content", "String", nullable=False),
        "author":  GraphQLField("author", "User", nullable=False),
        "tags":    GraphQLField("tags", "String", is_list=True),
        "status":  GraphQLField("status", "PostStatus", nullable=False),
        "likes":   GraphQLField("likes", "Int", nullable=False),
    }

    print(f"\n  Тип: {user_type.name} — {user_type.description}")
    print("  type User {")
    for fname, field in user_type.fields.items():
        print(f"    {field.to_sdl()}")
    print("  }")

    print(f"\n  Тип: {post_type.name} — {post_type.description}")
    print("  type Post {")
    for fname, field in post_type.fields.items():
        print(f"    {field.to_sdl()}")
    print("  }")

    # --- 1.4 Enum types ---
    print("\n--- 1.4 Enum types ---")

    enums = {
        "Role":       ["ADMIN", "EDITOR", "VIEWER"],
        "PostStatus": ["DRAFT", "PUBLISHED", "ARCHIVED"],
    }

    for enum_name, values in enums.items():
        print(f"  enum {enum_name} {{")
        for val in values:
            print(f"    {val}")
        print("  }")

    print()


# =============================================================================
# Демо 2: Query Language
# =============================================================================

def demo_query_language():
    """Демонстрация языка запросов GraphQL: поля, аргументы, фрагменты, алиасы."""
    print("=" * 70)
    print("Демо 2: Query Language (fields, arguments, fragments, aliases)")
    print("=" * 70)

    # --- 2.1 Простые запросы ---
    print("\n--- 2.1 Простые запросы (Queries) ---")

    queries = [
        {
            "name": "Получение одного пользователя",
            "query": """{
  user(id: "42") {
    id
    name
    email
    role
  }
}""",
        },
        {
            "name": "Запрос с аргументами",
            "query": """{
  posts(status: PUBLISHED, authorId: "42") {
    id
    title
    likes
    tags
  }
}""",
        },
        {
            "name": "Вложенные связи",
            "query": """{
  post(id: "7") {
    title
    author {
      name
      role
    }
    comments {
      text
      author {
        name
      }
    }
  }
}""",
        },
    ]

    for q in queries:
        print(f"\n  Запрос: {q['name']}")
        for line in q["query"].split("\n"):
            print(f"    {line}")

    # --- 2.2 Фрагменты ---
    print("\n--- 2.2 Фрагменты (Fragments) ---")

    fragment_example = """
# Переиспользуемый фрагмент
fragment UserBasic on User {
    id
    name
    email
}

# Использование фрагмента
{
  currentUser {
    ...UserBasic
    role
  }
  team {
    ...UserBasic
    posts {
      title
    }
  }
}"""
    print("  Fragment-определение и использование:")
    for line in fragment_example.strip().split("\n"):
        print(f"    {line}")

    # Подсчёт переиспользования
    fragment_refs = fragment_example.count("...UserBasic")
    fragment_defs = fragment_example.count("fragment UserBasic")
    print(f"\n  Определений фрагмента: {fragment_defs}")
    print(f"  Использований (...UserBasic): {fragment_refs}")
    print(f"  Экономия строк: ~{(fragment_refs - 1) * 4} строк (без фрагмента)")

    # --- 2.3 Алиасы ---
    print("\n--- 2.3 Алиасы (Aliases) ---")

    alias_example = """
{
  # Два разных запроса к одному полю — нужен алиас
  activePosts: posts(status: PUBLISHED) {
    title
    likes
  }
  draftPosts: posts(status: DRAFT) {
    title
  }
  archivedPosts: posts(status: ARCHIVED) {
    title
  }
}"""
    print("  Алиасы для разных аргументов:")
    for line in alias_example.strip().split("\n"):
        print(f"    {line}")

    # --- 2.4 Мутации ---
    print("\n--- 2.4 Мутации (Mutations) ---")

    mutations = [
        {
            "name": "Создание поста",
            "query": """mutation {
  createPost(input: {
    title: "GraphQL: Полное руководство"
    content: "GraphQL — это язык запросов для API..."
    tags: ["graphql", "api", "tutorial"]
  }) {
    id
    title
    status
    author {
      name
    }
  }
}""",
        },
        {
            "name": "Обновление поста",
            "query": """mutation {
  updatePost(id: "7", input: {
    title: "Обновлённый заголовок"
    status: PUBLISHED
  }) {
    id
    title
    status
  }
}""",
        },
    ]

    for m in mutations:
        print(f"\n  Мутация: {m['name']}")
        for line in m["query"].split("\n"):
            print(f"    {line}")

    # --- 2.5 Интроспекция ---
    print("\n--- 2.5 Интроспекция схемы ---")

    introspection_query = """{
  __schema {
    queryType { name }
    mutationType { name }
    types {
      name
      kind
      fields {
        name
        type {
          name
          kind
        }
      }
    }
  }
}"""
    print("  Запрос интроспекции:")
    for line in introspection_query.split("\n"):
        print(f"    {line}")

    # Имитация ответа интроспекции
    introspection_result = {
        "__schema": {
            "queryType": {"name": "Query"},
            "mutationType": {"name": "Mutation"},
            "types": [
                {
                    "name": "User",
                    "kind": "OBJECT",
                    "fields": [
                        {"name": "id", "type": {"name": None, "kind": "NON_NULL"}},
                        {"name": "name", "type": {"name": None, "kind": "NON_NULL"}},
                        {"name": "email", "type": {"name": None, "kind": "NON_NULL"}},
                        {"name": "role", "type": {"name": "Role", "kind": "ENUM"}},
                        {"name": "posts", "type": {"name": None, "kind": "LIST"}},
                    ],
                }
            ],
        }
    }

    print("\n  Ответ интроспекции (упрощённый):")
    print(json.dumps(introspection_result, indent=4, ensure_ascii=False))

    print()


# =============================================================================
# Демо 3: Resolvers & DataLoader
# =============================================================================

def demo_resolvers_dataloader():
    """Демонстрация резолверов, проблемы N+1 и паттерна DataLoader."""
    print("=" * 70)
    print("Демо 3: Resolvers & DataLoader (N+1 Problem)")
    print("=" * 70)

    # --- 3.1 Имитация базы данных ---
    print("\n--- 3.1 Имитация базы данных ---")

    # Создаём тестовые данные
    users_db = {
        i: {
            "id": i,
            "name": f"Пользователь_{i}",
            "email": f"user{i}@example.com",
            "role": random.choice(["ADMIN", "EDITOR", "VIEWER"]),
        }
        for i in range(1, 11)
    }

    posts_db = {}
    post_id = 1
    for user_id in range(1, 6):  # Первые 5 пользователей имеют посты
        for j in range(random.randint(2, 4)):
            posts_db[post_id] = {
                "id": post_id,
                "title": f"Пост {post_id} от пользователя {user_id}",
                "content": f"Содержимое поста {post_id}...",
                "author_id": user_id,
                "tags": random.sample(["python", "graphql", "api", "ml", "devops"], k=2),
                "likes": random.randint(0, 100),
            }
            post_id += 1

    comments_db = {}
    comment_id = 1
    for pid in range(1, post_id):
        for k in range(random.randint(0, 3)):
            comments_db[comment_id] = {
                "id": comment_id,
                "post_id": pid,
                "author_id": random.randint(1, 10),
                "text": f"Комментарий {comment_id} к посту {pid}",
            }
            comment_id += 1

    print(f"  Пользователей в БД: {len(users_db)}")
    print(f"  Постов в БД: {len(posts_db)}")
    print(f"  Комментариев в БД: {len(comments_db)}")

    # --- 3.2 Резолверы (проблема N+1) ---
    print("\n--- 3.2 Резолверы: Проблема N+1 ---")

    # Счётчик запросов к "БД"
    query_count = [0]

    def resolve_users():
        """Резолвер: получить всех пользователей."""
        query_count[0] += 1
        return list(users_db.values())

    def resolve_user_posts(user_id):
        """Резолвер: получить посты пользователя (N+1 проблема!)."""
        query_count[0] += 1
        return [p for p in posts_db.values() if p["author_id"] == user_id]

    def resolve_post_comments(post_id):
        """Резолвер: получить комментарии к посту."""
        query_count[0] += 1
        return [c for c in comments_db.values() if c["post_id"] == post_id]

    # Имитация запроса: все пользователи → их посты → комментарии
    print("\n  Запрос: все пользователи → посты → комментарии (N+1!)")
    query_count[0] = 0

    all_users = resolve_users()  # 1 запрос
    results_naive = []

    for user in all_users:
        user_posts = resolve_user_posts(user["id"])  # N запросов
        for post in user_posts:
            post_comments = resolve_post_comments(post["id"])  # M запросов
            results_naive.append({
                "user": user["name"],
                "post": post["title"],
                "comments_count": len(post_comments),
            })

    print(f"\n  Результатов: {len(results_naive)}")
    print(f"  Всего запросов к БД: {query_count[0]}")
    print(f"  Формула: 1 (users) + N (user_posts) + M (post_comments) = {query_count[0]}")
    print(f"  Это классическая проблема N+1!")

    # Показать первые 3 результата
    for r in results_naive[:3]:
        print(f"    {r['user']} → {r['post']} ({r['comments_count']} комментариев)")

    # --- 3.3 DataLoader ---
    print("\n--- 3.3 DataLoader: решение проблемы N+1 ---")

    class DataLoader:
        """Простой DataLoader для батчинга запросов."""
        def __init__(self, batch_fn):
            self.batch_fn = batch_fn
            self.cache = {}
            self.queue = []

        def _cache_key(self, key):
            """Преобразование ключа в хешируемый вид (списки → кортежи)."""
            return tuple(key) if isinstance(key, list) else key

        def load(self, key):
            """Загрузить значение по ключу (с кешированием и батчингом)."""
            ck = self._cache_key(key)
            if ck in self.cache:
                return self.cache[ck]

            future = {"key": key, "result": None, "resolved": False}
            self.queue.append(future)
            self.cache[ck] = future
            return future

        def dispatch(self):
            """Выполнить батч и разрешить все futures."""
            if not self.queue:
                return

            keys = [f["key"] for f in self.queue]
            results = self.batch_fn(keys)

            for future, result in zip(self.queue, results):
                future["result"] = result
                future["resolved"] = True

            self.queue.clear()

    # DataLoader для пользователей
    def batch_load_users(ids):
        """Батч-загрузка пользователей."""
        query_count[0] += 1
        return [users_db.get(uid) for uid in ids]

    # DataLoader для постов по author_id
    def batch_load_posts(author_ids):
        """Батч-загрузка постов по author_id."""
        query_count[0] += 1
        result = []
        for aid in author_ids:
            result.append([p for p in posts_db.values() if p["author_id"] == aid])
        return result

    # DataLoader для комментариев по post_id
    def batch_load_comments(post_ids):
        """Батч-загрузка комментариев по post_id (каждый элемент — отдельный post_id)."""
        query_count[0] += 1
        result = []
        for pid in post_ids:
            result.append([c for c in comments_db.values() if c["post_id"] == pid])
        return result

    # Используем DataLoader
    query_count[0] = 0
    user_loader = DataLoader(batch_load_users)
    posts_loader = DataLoader(batch_load_posts)
    comments_loader = DataLoader(batch_load_comments)

    all_users = resolve_users()  # 1 запрос
    results_optimized = []

    for user in all_users:
        user_future = user_loader.load(user["id"])
        user_loader.dispatch()  # Сразу выполняем (уже есть в кеше)

        posts_future = posts_loader.load(user["id"])
        posts_loader.dispatch()

        user_posts = posts_future["result"] or []
        total_comments = 0
        for post in user_posts:
            comment_future = comments_loader.load(post["id"])
            comments_loader.dispatch()
            total_comments += len(comment_future["result"] or [])

        if user_posts:
            results_optimized.append({
                "user": user["name"],
                "posts_count": len(user_posts),
                "total_comments": total_comments,
            })

    print(f"\n  Результатов: {len(results_optimized)}")
    print(f"  Всего запросов к БД: {query_count[0]}")
    print(f"  Экономия: {10 + len(posts_db) + len(comments_db)} → {query_count[0]} запросов")

    for r in results_optimized[:3]:
        print(f"    {r['user']}: {r['posts_count']} постов, {r['total_comments']} комментариев")

    # --- 3.4 Сравнение ---
    print("\n--- 3.4 Сравнение: Naive vs DataLoader ---")

    comparison = [
        ("Запросов к БД",   f"{10 + len(posts_db) + len(comments_db)}", f"{query_count[0]}", "×" + str((10 + len(posts_db) + len(comments_db)) // max(query_count[0], 1))),
        ("Кеширование",     "Нет", "Да", "DataLoader кеширует по ключу"),
        ("Батчинг",         "Нет", "Да", "Группирует запросы в батчи"),
    ]

    print(f"\n  {'Метрика':20s} | {'Naive':10s} | {'DataLoader':12s} | Улучшение")
    print(f"  {'-'*20}-+-{'-'*10}-+-{'-'*12}-+--{'-'*20}")
    for metric, naive, dl, improvement in comparison:
        print(f"  {metric:20s} | {naive:10s} | {dl:12s} | {improvement}")

    print()


# =============================================================================
# Демо 4: Subscriptions & Real-time
# =============================================================================

def demo_subscriptions():
    """Демонстрация подписок GraphQL: WebSocket, event streams."""
    print("=" * 70)
    print("Демо 4: Subscriptions & Real-time (WebSocket simulation)")
    print("=" * 70)

    # --- 4.1 Определение подписок в схеме ---
    print("\n--- 4.1 Определение подписок в GraphQL-схеме ---")

    subscription_sdl = """
# Подписки на реальное время
type Subscription {
    # Подписка на новые посты
    postCreated: Post!

    # Подписка на обновления конкретного поста
    postUpdated(postId: ID!): Post!

    # Подписка на комментарии к посту
    commentAdded(postId: ID!): Comment!

    # Подписка на статус пользователя (онлайн/оффлайн)
    userStatusChanged(userId: ID!): UserStatus!
}

type Comment {
    id: ID!
    text: String!
    author: User!
    postId: ID!
}

type UserStatus {
    userId: ID!
    status: ONLINE | OFFLINE | AWAY
    lastSeen: DateTime!
}
"""
    print("  GraphQL SDL для подписок:")
    for line in subscription_sdl.strip().split("\n"):
        if line.strip():
            print(f"    {line}")

    # --- 4.2 Клиентские запросы подписок ---
    print("\n--- 4.2 Клиентские запросы подписок ---")

    subscription_queries = [
        {
            "name": "Подписка на новые посты",
            "query": """subscription OnNewPost {
  postCreated {
    id
    title
    author {
      name
    }
    createdAt
  }
}""",
        },
        {
            "name": "Подписка на комментарии к посту",
            "query": """subscription OnComment($postId: ID!) {
  commentAdded(postId: $postId) {
    id
    text
    author {
      name
      avatar
    }
  }
}""",
        },
    ]

    for sq in subscription_queries:
        print(f"\n  {sq['name']}:")
        for line in sq["query"].split("\n"):
            print(f"    {line}")

    # --- 4.3 Симуляция WebSocket ---
    print("\n--- 4.3 Симуляция WebSocket-соединения ---")

    class WebSocketSimulator:
        """Симуляция WebSocket для GraphQL Subscription."""

        def __init__(self):
            self.channels = collections.defaultdict(list)  # topic → callbacks
            self.messages = []  # история сообщений
            self.connected = False

        def connect(self, url):
            """Установка WebSocket-соединения."""
            self.connected = True
            self.messages.append({
                "type": "connection",
                "status": "connected",
                "url": url,
                "timestamp": time.time(),
            })
            return True

        def subscribe(self, topic, callback):
            """Подписка на топик."""
            self.channels[topic].append(callback)
            self.messages.append({
                "type": "subscribe",
                "topic": topic,
                "timestamp": time.time(),
            })

        def unsubscribe(self, topic, callback):
            """Отписка от топика."""
            if callback in self.channels[topic]:
                self.channels[topic].remove(callback)

        def publish(self, topic, data):
            """Публикация события в топик."""
            message = {
                "type": "data",
                "topic": topic,
                "payload": data,
                "timestamp": time.time(),
            }
            self.messages.append(message)

            for callback in self.channels.get(topic, []):
                callback(data)

        def disconnect(self):
            """Закрытие WebSocket-соединения."""
            self.connected = False
            self.messages.append({
                "type": "connection",
                "status": "disconnected",
                "timestamp": time.time(),
            })

    # Создаём WebSocket-сервер
    ws = WebSocketSimulator()
    ws.connect("ws://localhost:4000/graphql")

    print(f"  WebSocket подключён: {ws.connected}")

    # Создаём клиентов
    received_messages = {"client1": [], "client2": []}

    def client1_handler(data):
        received_messages["client1"].append(data)

    def client2_handler(data):
        received_messages["client2"].append(data)

    # Подписки
    ws.subscribe("postCreated", client1_handler)
    ws.subscribe("postCreated", client2_handler)
    ws.subscribe("commentAdded:post_7", client1_handler)

    print(f"  Клиенты подписаны на:")
    print(f"    postCreated: 2 клиента")
    print(f"    commentAdded:post_7: 1 клиент")

    # Публикация событий
    print("\n  Публикация событий:")

    event1 = {
        "id": "post_100",
        "title": "Новый пост о GraphQL",
        "author": {"name": "Иван Иванов"},
        "createdAt": "2024-01-15T10:30:00Z",
    }
    ws.publish("postCreated", event1)
    print(f"    → postCreated: '{event1['title']}' (получено клиентами: {len(received_messages['client1']) + len(received_messages['client2'])})")

    event2 = {
        "id": "post_101",
        "title": "Второй пост",
        "author": {"name": "Пётр Петров"},
        "createdAt": "2024-01-15T11:00:00Z",
    }
    ws.publish("postCreated", event2)
    print(f"    → postCreated: '{event2['title']}' (получено клиентами: {len(received_messages['client1']) + len(received_messages['client2'])})")

    event3 = {
        "id": "comment_200",
        "text": "Отличная статья!",
        "author": {"name": "Мария Сидорова"},
        "postId": "post_7",
    }
    ws.publish("commentAdded:post_7", event3)
    print(f"    → commentAdded: '{event3['text'][:30]}...' (только клиент 1)")

    # --- 4.4 Event-driven архитектура ---
    print("\n--- 4.4 Event-driven архитектура ---")

    class EventBus:
        """Простой шина событий для GraphQL Subscriptions."""

        def __init__(self):
            self.subscribers = collections.defaultdict(list)
            self.event_log = []

        def subscribe(self, event_type, handler, filter_fn=None):
            """Подписка на тип события с опциональным фильтром."""
            self.subscribers[event_type].append({
                "handler": handler,
                "filter": filter_fn,
            })

        def emit(self, event_type, payload):
            """Отправка события."""
            self.event_log.append({
                "type": event_type,
                "payload": payload,
                "timestamp": time.time(),
            })

            for sub in self.subscribers[event_type]:
                # Проверяем фильтр
                if sub["filter"] is None or sub["filter"](payload):
                    sub["handler"](payload)

    # Создаём шину событий
    bus = EventBus()

    # Подписчики
    post_notifications = []
    comment_notifications = []

    def on_new_post(payload):
        post_notifications.append(payload)

    def on_comment(payload):
        comment_notifications.append(payload)

    bus.subscribe("post.created", on_new_post)
    bus.subscribe("comment.created", on_comment,
                   filter_fn=lambda p: p.get("post_id") == "post_7")

    # Генерация событий
    print("  Генерация событий:")

    events = [
        ("post.created",   {"id": "p1", "title": "GraphQL入门", "author_id": 1}),
        ("post.created",   {"id": "p2", "title": "高级GraphQL", "author_id": 2}),
        ("comment.created", {"id": "c1", "post_id": "post_7", "text": "Коммент 1"}),
        ("comment.created", {"id": "c2", "post_id": "post_5", "text": "Коммент 2"}),
        ("comment.created", {"id": "c3", "post_id": "post_7", "text": "Коммент 3"}),
    ]

    for event_type, payload in events:
        bus.emit(event_type, payload)
        print(f"    → {event_type}: {json.dumps(payload, ensure_ascii=False)[:60]}")

    print(f"\n  Всего событий в логе: {len(bus.event_log)}")
    print(f"  Уведомлений о постах: {len(post_notifications)}")
    print(f"  Уведомлений о комментариях (фильтр post_7): {len(comment_notifications)}")

    # История WebSocket
    print(f"\n  История WebSocket-сообщений: {len(ws.messages)} сообщений")
    for msg in ws.messages[:5]:
        print(f"    [{msg['type']}] {json.dumps(msg, default=str, ensure_ascii=False)[:60]}")

    ws.disconnect()
    print(f"\n  WebSocket отключён: {ws.connected}")

    print()


# =============================================================================
# Точка входа
# =============================================================================

if __name__ == "__main__":
    demo_schema_definition()
    demo_query_language()
    demo_resolvers_dataloader()
    demo_subscriptions()
