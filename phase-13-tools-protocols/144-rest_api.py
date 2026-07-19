"""144 — REST API Design: HTTP methods, status codes, OpenAPI, authentication

Темы:
  1. HTTP Methods & Status Codes (GET/POST/PUT/DELETE, 2xx/4xx/5xx)
  2. RESTful Resource Design (naming, nesting, pagination)
  3. Authentication (API keys, JWT tokens, OAuth2 flow)
  4. API Documentation (OpenAPI/Swagger spec, schema validation)

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
# Демо 1: HTTP Methods & Status Codes
# =============================================================================

def demo_http_methods():
    """Демонстрация HTTP методов и кодов ответов."""
    print("=" * 70)
    print("Демо 1: HTTP Methods & Status Codes")
    print("=" * 70)

    # --- 1.1 Имитация HTTP-методов ---
    print("\n--- 1.1 Кодирование HTTP-методов ---")

    methods = {
        "GET":    {"description": "Получение ресурса",      "idempotent": True,  "safe": True},
        "POST":   {"description": "Создание ресурса",       "idempotent": False, "safe": False},
        "PUT":    {"description": "Полная замена ресурса",  "idempotent": True,  "safe": False},
        "PATCH":  {"description": "Частичное обновление",   "idempotent": False, "safe": False},
        "DELETE": {"description": "Удаление ресурса",       "idempotent": True,  "safe": False},
    }

    for method, info in methods.items():
        idem_mark = "[идемпотентный]" if info["idempotent"] else "[неидемпотентный]"
        safe_mark = "[безопасный]" if info["safe"] else "[небезопасный]"
        print(f"  {method:8s} — {info['description']:30s} {idem_mark:22s} {safe_mark}")

    # --- 1.2 Коды ответов HTTP ---
    print("\n--- 1.2 Коды ответов HTTP ---")

    status_codes = {
        # 2xx — успех
        200: ("OK",                   "Успешный запрос"),
        201: ("Created",              "Ресурс создан"),
        204: ("No Content",           "Успешно, без тела ответа"),
        # 3xx — перенаправление
        301: ("Moved Permanently",    "Ресурс перемещён навсегда"),
        304: ("Not Modified",         "Кэш актуален"),
        # 4xx — ошибка клиента
        400: ("Bad Request",          "Невалидный запрос"),
        401: ("Unauthorized",         "Необходима аутентификация"),
        403: ("Forbidden",            "Нет прав доступа"),
        404: ("Not Found",            "Ресурс не найден"),
        429: ("Too Many Requests",    "Превышен лимит запросов"),
        # 5xx — ошибка сервера
        500: ("Internal Server Error","Внутренняя ошибка сервера"),
        502: ("Bad Gateway",          "Неверный ответ от upstream"),
        503: ("Service Unavailable",  "Сервис недоступен"),
    }

    for code, (phrase, desc) in status_codes.items():
        category = code // 100
        prefix = {2: "✅ Успех", 3: "↪️  Перенаправление",
                  4: "❌ Ошибка клиента", 5: "💥 Ошибка сервера"}[category]
        print(f"  {code} {phrase:25s} — {desc:35s} [{prefix}]")

    # --- 1.3 Матрица идемпотентности ---
    print("\n--- 1.3 Матрица: метод × код ответа ---")

    allowed = {
        "GET":    [200, 304, 404],
        "POST":   [201, 202, 400, 409],
        "PUT":    [200, 201, 204, 400],
        "DELETE": [200, 204, 404],
    }

    header = f"  {'Метод':8s}" + "".join(f"{c:6d}" for c in [200, 201, 204, 400, 404, 500])
    print(header)
    print("  " + "-" * (8 + 6 * 6))

    for method in ["GET", "POST", "PUT", "DELETE"]:
        row = f"  {method:8s}"
        for code in [200, 201, 204, 400, 404, 500]:
            mark = "  ✓  " if code in allowed[method] else "  ·  "
            row += f"{mark:6s}"
        print(row)

    # --- 1.4 Классификация HTTP-запросов ---
    print("\n--- 1.4 Классификация типичных REST-запросов ---")

    requests = [
        ("GET",    "/api/v1/users",          "Список пользователей",         200),
        ("GET",    "/api/v1/users/42",        "Один пользователь",            200),
        ("POST",   "/api/v1/users",           "Создать пользователя",         201),
        ("PUT",    "/api/v1/users/42",        "Полная замена профиля",        200),
        ("PATCH",  "/api/v1/users/42",        "Обновить email",               200),
        ("DELETE", "/api/v1/users/42",        "Удалить пользователя",         204),
        ("GET",    "/api/v1/users/42/orders", "Заказы пользователя",          200),
        ("POST",   "/api/v1/auth/login",      "Вход в систему",               201),
    ]

    for method, path, purpose, code in requests:
        print(f"  {method:7s} {path:30s} → {code} ({purpose})")

    print()


# =============================================================================
# Демо 2: RESTful Resource Design
# =============================================================================

def demo_resource_design():
    """Демонстрация проектирования REST-ресурсов, именование, вложенность, пагинация."""
    print("=" * 70)
    print("Демо 2: RESTful Resource Design")
    print("=" * 70)

    # --- 2.1 Именование ресурсов ---
    print("\n--- 2.1 Именование REST-ресурсов (CRUD) ---")

    resource = "articles"

    endpoints = [
        ("GET",    f"/{resource}",              f"Список {resource}"),
        ("POST",   f"/{resource}",              f"Создать {resource[:-1]}"),
        ("GET",    f"/{resource}/{{id}}",       f"Одно {resource[:-1]}"),
        ("PUT",    f"/{resource}/{{id}}",       f"Замена {resource[:-1]}"),
        ("PATCH",  f"/{resource}/{{id}}",       f"Обновление {resource[:-1]}"),
        ("DELETE", f"/{resource}/{{id}}",       f"Удаление {resource[:-1]}"),
    ]

    for method, endpoint, desc in endpoints:
        print(f"  {method:7s} {endpoint:35s} — {desc}")

    # --- 2.2 Вложенность ресурсов ---
    print("\n--- 2.2 Вложенность ресурсов ---")

    nested = [
        ("/users/{uid}/posts",                    "Посты пользователя",      "Вложение 1 уровня"),
        ("/users/{uid}/posts/{pid}/comments",     "Комментарии к посту",     "Вложение 2 уровней"),
        ("/users/{uid}/posts/{pid}/comments/{cid}","Один комментарий",      "Вложение 3 уровня"),
        ("/users/{uid}/followers",                "Подписчики",              "Альтернатива вложенности"),
        ("/posts/{pid}?user={uid}",               "Посты (фильтр)",         "Фильтр через query param"),
    ]

    for path, desc, note in nested:
        depth = path.count("{")
        print(f"  {path:50s} — {desc:25s} ({note}, глубина={depth})")

    # --- 2.3 Пагинация ---
    print("\n--- 2.3 Стратегии пагинации ---")

    total_items = 95
    page_size = 20
    total_pages = math.ceil(total_items / page_size)

    print(f"  Всего элементов: {total_items}, размер страницы: {page_size}, страниц: {total_pages}")

    class SimplePaginator:
        """Простая реализация страниц-навигации."""
        def __init__(self, total, per_page):
            self.total = total
            self.per_page = per_page
            self.total_pages = math.ceil(total / per_page)

        def get_page(self, page_num):
            """Получить данные страницы и мета-информацию."""
            page_num = max(1, min(page_num, self.total_pages))
            start = (page_num - 1) * self.per_page
            end = min(start + self.per_page, self.total)
            items = list(range(start + 1, end + 1))

            return {
                "items": items,
                "page": page_num,
                "per_page": self.per_page,
                "total": self.total,
                "total_pages": self.total_pages,
                "has_next": page_num < self.total_pages,
                "has_prev": page_num > 1,
            }

    paginator = SimplePaginator(total_items, page_size)

    # Показать заголовки ответа для каждой страницы
    for page in [1, 2, 5]:
        result = paginator.get_page(page)
        print(f"\n  Страница {result['page']}/{result['total_pages']}:")
        print(f"    X-Total-Count: {result['total']}")
        print(f"    X-Total-Pages: {result['total_pages']}")
        print(f"    Link: <?page={page + 1}>; rel=\"next\"" if result['has_next'] else "    Link: (последняя)")
        print(f"    Элементы: {result['items'][:5]}{'...' if len(result['items']) > 5 else ''}")

    # --- 2.4 Cursor-based пагинация ---
    print("\n--- 2.4 Cursor-based пагинация ---")

    all_data = [
        {"id": i, "name": f"item_{i:03d}", "ts": 1000 + i * 7}
        for i in range(1, 31)
    ]

    def encode_cursor(item_id):
        """Кодирование курсора в base64."""
        raw = f"cursor:{item_id}".encode()
        return base64.b64encode(raw).decode()

    def decode_cursor(cursor_b64):
        """Декодирование курсора из base64."""
        raw = base64.b64decode(cursor_b64).decode()
        return int(raw.split(":")[1])

    def cursor_page(data, cursor=None, limit=10):
        """Получить страницу по курсору."""
        if cursor is not None:
            start_id = decode_cursor(cursor)
            filtered = [d for d in data if d["id"] > start_id]
        else:
            filtered = data

        items = filtered[:limit]
        next_cursor = encode_cursor(items[-1]["id"]) if items and len(filtered) > limit else None
        return items, next_cursor

    # Страница 1
    page1, cursor1 = cursor_page(all_data, None, limit=10)
    print(f"  Страница 1 (без курсора): IDs {[x['id'] for x in page1]}")
    print(f"  Следующий курсор: {cursor1}")

    # Страница 2
    page2, cursor2 = cursor_page(all_data, cursor1, limit=10)
    print(f"  Страница 2 (cursor={cursor1[:20]}...): IDs {[x['id'] for x in page2]}")
    print(f"  Следующий курсор: {cursor2}")

    # Страница 3
    page3, cursor3 = cursor_page(all_data, cursor2, limit=10)
    print(f"  Страница 3: IDs {[x['id'] for x in page3]}")
    print(f"  Следующий курсор: {cursor3} (None = конец)")

    print()


# =============================================================================
# Демо 3: Authentication
# =============================================================================

def demo_authentication():
    """Демонстрация методов аутентификации: API-ключи, JWT, OAuth2."""
    print("=" * 70)
    print("Демо 3: Authentication (API Keys, JWT, OAuth2)")
    print("=" * 70)

    # --- 3.1 API Keys ---
    print("\n--- 3.1 API Keys ---")

    def generate_api_key(prefix="sk"):
        """Генерация API-ключа."""
        random_bytes = bytes(random.getrandbits(8) for _ in range(32))
        key = base64.b32encode(random_bytes).decode().lower().replace("=", "")
        return f"{prefix}_{key}"

    def hash_api_key(key):
        """Хеширование API-ключа для безопасного хранения."""
        return hashlib.sha256(key.encode()).hexdigest()

    keys = {}
    for i in range(3):
        api_key = generate_api_key("sk")
        key_hash = hash_api_key(api_key)
        keys[api_key[:12] + "..."] = {
            "full_key": api_key,
            "hash": key_hash[:16] + "...",
            "prefix": api_key[:2],
        }
        print(f"  Ключ {i+1}: {api_key[:12]}...{api_key[-4:]}")
        print(f"    Префикс: {api_key[:2]}, Хеш (первые 16): {key_hash[:16]}...")
        print(f"    Длина: {len(api_key)} символов")

    # Проверка API-ключа
    test_key = list(keys.values())[0]["full_key"]
    test_hash = hash_api_key(test_key)
    print(f"\n  Проверка ключа {test_key[:12]}...")
    print(f"    Хеш совпадает: {test_hash == keys[test_key[:12] + '...']['hash']}")

    # --- 3.2 JWT (JSON Web Tokens) ---
    print("\n--- 3.2 JWT (JSON Web Tokens) ---")

    def base64url_encode(data):
        """Base64url-кодирование."""
        if isinstance(data, str):
            data = data.encode()
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

    def base64url_decode(s):
        """Base64url-декодирование."""
        s += "=" * (4 - len(s) % 4)
        return base64.urlsafe_b64decode(s)

    def create_jwt(payload, secret="my_secret_key", algo="HS256"):
        """Создание JWT-токена."""
        header = {"alg": algo, "typ": "JWT"}

        header_b64 = base64url_encode(json.dumps(header))
        payload_b64 = base64url_encode(json.dumps(payload))

        # HMAC-SHA256 подпись
        message = f"{header_b64}.{payload_b64}".encode()
        signature = hmac_sha256(secret.encode(), message)

        return f"{header_b64}.{payload_b64}.{signature}"

    def hmac_sha256(key, message):
        """Простая HMAC-SHA256 реализация."""
        # Берём 64 байта из ключа, дополняя нулями
        key_padded = key.ljust(64, b"\x00")
        if len(key_padded) > 64:
            key_padded = hashlib.sha256(key_padded).digest()

        # Inner hash
        inner_key = bytes(k ^ 0x36 for k in key_padded)
        inner = hashlib.sha256(inner_key + message).digest()

        # Outer hash
        outer_key = bytes(k ^ 0x5C for k in key_padded)
        result = hashlib.sha256(outer_key + inner).digest()

        return base64url_encode(result)

    def decode_jwt(token, secret="my_secret_key"):
        """Декодирование и проверка JWT-токена."""
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Невалидный JWT")

        header_b64, payload_b64, signature = parts

        # Проверка подписи
        expected_sig = hmac_sha256(secret.encode(), f"{header_b64}.{payload_b64}".encode())
        if signature != expected_sig:
            raise ValueError("Невалидная подпись")

        header = json.loads(base64url_decode(header_b64))
        payload = json.loads(base64url_decode(payload_b64))
        return header, payload

    # Создание JWT
    jwt_payload = {
        "sub": "user_42",
        "name": "Иван Иванов",
        "role": "admin",
        "iat": int(time.time()),
        "exp": int(time.time()) + 3600,  # 1 час
    }

    token = create_jwt(jwt_payload)
    print(f"  Токен: {token[:50]}...")
    print(f"  Длина токена: {len(token)} символов")

    # Декодирование JWT
    decoded_header, decoded_payload = decode_jwt(token)
    print(f"\n  Decoded header:  {decoded_header}")
    print(f"  Decoded payload:")
    for k, v in decoded_payload.items():
        print(f"    {k}: {v}")

    # Проверка истечения
    now = int(time.time())
    is_valid = decoded_payload["exp"] > now
    remaining = decoded_payload["exp"] - now
    print(f"\n  Токен действителен: {is_valid}")
    print(f"  Осталось секунд: {remaining}")

    # --- 3.3 OAuth2 Flow ---
    print("\n--- 3.3 OAuth2 Authorization Code Flow ---")

    class OAuth2Simulator:
        """Симулятор OAuth2 Authorization Code Flow."""

        def __init__(self, client_id="my_app", client_secret="secret_123"):
            self.client_id = client_id
            self.client_secret = client_secret
            self.auth_codes = {}
            self.tokens = {}

        def get_authorization_url(self, redirect_uri, scope="read write"):
            """Шаг 1: Формирование URL для авторизации."""
            state = hashlib.sha256(str(random.random()).encode()).hexdigest()[:16]
            url = (
                f"https://auth.example.com/authorize"
                f"?response_type=code"
                f"&client_id={self.client_id}"
                f"&redirect_uri={redirect_uri}"
                f"&scope={scope}"
                f"&state={state}"
            )
            return url, state

        def simulate_user_authorize(self, user_id="user_42"):
            """Шаг 2: Пользователь авторизует приложение (симуляция)."""
            code = hashlib.sha256(f"{user_id}_{time.time()}".encode()).hexdigest()[:32]
            self.auth_codes[code] = {
                "user_id": user_id,
                "created_at": time.time(),
                "expires_in": 600,
            }
            return code

        def exchange_code(self, code, redirect_uri):
            """Шаг 3: Обмен кода на токен."""
            if code not in self.auth_codes:
                return None, "Код не найден или уже использован"

            auth_info = self.auth_codes.pop(code)
            access_token = f"at_{hashlib.sha256(f'{code}_access'.encode()).hexdigest()[:32]}"
            refresh_token = f"rt_{hashlib.sha256(f'{code}_refresh'.encode()).hexdigest()[:32]}"

            self.tokens[access_token] = {
                "user_id": auth_info["user_id"],
                "scope": "read write",
                "expires_in": 3600,
                "created_at": time.time(),
            }

            return {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": "Bearer",
                "expires_in": 3600,
                "scope": "read write",
            }, None

        def refresh_access_token(self, refresh_token):
            """Шаг 4: Обновление токена."""
            if not refresh_token.startswith("rt_"):
                return None, "Невалидный refresh токен"

            new_access = f"at_{hashlib.sha256(f'{refresh_token}_new'.encode()).hexdigest()[:32]}"
            new_refresh = f"rt_{hashlib.sha256(f'{refresh_token}_refresh_new'.encode()).hexdigest()[:32]}"

            return {
                "access_token": new_access,
                "refresh_token": new_refresh,
                "token_type": "Bearer",
                "expires_in": 3600,
            }, None

    oauth = OAuth2Simulator()

    # Шаг 1
    redirect_uri = "https://myapp.example.com/callback"
    auth_url, state = oauth.get_authorization_url(redirect_uri)
    print(f"  Шаг 1 — Authorization URL:")
    print(f"    {auth_url[:80]}...")
    print(f"    state={state}")

    # Шаг 2
    auth_code = oauth.simulate_user_authorize()
    print(f"\n  Шаг 2 — Код авторизации получен: {auth_code[:20]}...")

    # Шаг 3
    token_response, error = oauth.exchange_code(auth_code, redirect_uri)
    print(f"\n  Шаг 3 — Обмен кода на токен:")
    print(f"    access_token:  {token_response['access_token'][:30]}...")
    print(f"    refresh_token: {token_response['refresh_token'][:30]}...")
    print(f"    token_type:    {token_response['token_type']}")
    print(f"    expires_in:    {token_response['expires_in']} сек")

    # Шаг 4
    refresh_result, err = oauth.refresh_access_token(token_response["refresh_token"])
    print(f"\n  Шаг 4 — Обновление токена:")
    print(f"    Новый access_token:  {refresh_result['access_token'][:30]}...")
    print(f"    Новый refresh_token: {refresh_result['refresh_token'][:30]}...")

    print()


# =============================================================================
# Демо 4: API Documentation (OpenAPI / Schema Validation)
# =============================================================================

def demo_api_documentation():
    """Демонстрация документирования API: OpenAPI/Swagger, валидация схем."""
    print("=" * 70)
    print("Демо 4: API Documentation (OpenAPI / Schema Validation)")
    print("=" * 70)

    # --- 4.1 OpenAPI Specification ---
    print("\n--- 4.1 OpenAPI 3.0 Specification (минимальный пример) ---")

    openapi_spec = {
        "openapi": "3.0.3",
        "info": {
            "title": "AI Course API",
            "version": "1.0.0",
            "description": "REST API для управления курсами ИИ",
        },
        "paths": {
            "/courses": {
                "get": {
                    "summary": "Получить список курсов",
                    "operationId": "listCourses",
                    "parameters": [
                        {
                            "name": "page",
                            "in": "query",
                            "schema": {"type": "integer", "default": 1},
                        },
                        {
                            "name": "limit",
                            "in": "query",
                            "schema": {"type": "integer", "default": 20, "maximum": 100},
                        },
                    ],
                    "responses": {
                        "200": {
                            "description": "Успешный ответ",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "items": {"type": "array", "items": {"$ref": "#/components/schemas/Course"}},
                                            "total": {"type": "integer"},
                                        },
                                    }
                                }
                            },
                        }
                    },
                },
                "post": {
                    "summary": "Создать курс",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/CourseInput"}
                            }
                        },
                    },
                    "responses": {
                        "201": {"description": "Курс создан"},
                        "400": {"description": "Ошибка валидации"},
                    },
                },
            }
        },
        "components": {
            "schemas": {
                "Course": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "title": {"type": "string"},
                        "level": {"type": "string", "enum": ["beginner", "intermediate", "advanced"]},
                        "duration_hours": {"type": "number"},
                    },
                    "required": ["id", "title", "level"],
                },
                "CourseInput": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "minLength": 3, "maxLength": 100},
                        "level": {"type": "string", "enum": ["beginner", "intermediate", "advanced"]},
                        "duration_hours": {"type": "number", "minimum": 1},
                    },
                    "required": ["title", "level"],
                },
            }
        },
    }

    # Красивый вывод спецификации
    print(json.dumps(openapi_spec, indent=2, ensure_ascii=False)[:1200])
    print("  ... (упрощённый вывод)")

    # --- 4.2 Схемы и типы ---
    print("\n--- 4.2 Типы данных OpenAPI ---")

    types = {
        "string":  {"example": '"hello"',         "formats": ["date", "date-time", "email", "uri", "uuid"]},
        "integer": {"example": "42",               "formats": []},
        "number":  {"example": "3.14",             "formats": ["float", "double"]},
        "boolean": {"example": "true",             "formats": []},
        "array":   {"example": "[1, 2, 3]",        "formats": []},
        "object":  {"example": '{"key": "value"}', "formats": []},
    }

    for type_name, info in types.items():
        formats = ", ".join(info["formats"]) if info["formats"] else "—"
        print(f"  {type_name:10s} | пример: {info['example']:25s} | форматы: {formats}")

    # --- 4.3 Валидация схем ---
    print("\n--- 4.3 Валидация входных данных ---")

    def validate_course_input(data):
        """Валидация данных курса по схеме OpenAPI."""
        errors = []

        # Проверка обязательных полей
        if "title" not in data:
            errors.append("Поле 'title' обязательно")
        elif not isinstance(data["title"], str):
            errors.append("Поле 'title' должно быть строкой")
        elif len(data["title"]) < 3:
            errors.append(f"Поле 'title' минимум 3 символа (получено {len(data['title'])})")

        if "level" not in data:
            errors.append("Поле 'level' обязательно")
        elif data["level"] not in ["beginner", "intermediate", "advanced"]:
            errors.append(f"Поле 'level' должно быть одним из: beginner, intermediate, advanced")

        if "duration_hours" in data:
            if not isinstance(data["duration_hours"], (int, float)):
                errors.append("Поле 'duration_hours' должно быть числом")
            elif data["duration_hours"] < 1:
                errors.append(f"Поле 'duration_hours' минимум 1 (получено {data['duration_hours']})")

        return errors

    test_cases = [
        {"title": "Python для начинающих", "level": "beginner", "duration_hours": 40},
        {"title": "Py", "level": "advanced"},                     # title слишком короткий
        {"title": "ML Crash Course", "level": "expert"},          # невалидный level
        {"title": "Deep Learning", "level": "intermediate", "duration_hours": -5},  # отрицательное время
        {},                                                        # пустой объект
    ]

    for i, case in enumerate(test_cases, 1):
        errors = validate_course_input(case)
        status = "✅ Валидно" if not errors else "❌ Ошибки"
        print(f"\n  Тест {i}: {status}")
        print(f"    Данные: {json.dumps(case, ensure_ascii=False)}")
        if errors:
            for err in errors:
                print(f"    → {err}")

    # --- 4.4 Генерация документации ---
    print("\n--- 4.4 Автоматическая генерация markdown-документации ---")

    def generate_api_docs(spec):
        """Генерация markdown-документации из OpenAPI-спецификации."""
        lines = []
        lines.append(f"# {spec['info']['title']} v{spec['info']['version']}")
        lines.append(f"\n{spec['info']['description']}\n")

        for path, methods in spec.get("paths", {}).items():
            for method, details in methods.items():
                lines.append(f"## {method.upper()} `{path}`")
                lines.append(f"\n**{details.get('summary', 'Нет описания')}**\n")

                # Параметры
                params = details.get("parameters", [])
                if params:
                    lines.append("Параметры:")
                    for p in params:
                        schema = p.get("schema", {})
                        default = f" (по умолчанию: {schema.get('default', '—')})"
                        lines.append(f"- `{p['name']}` ({p['in']}) — тип: {schema.get('type', '?')}{default}")

                # Ответы
                responses = details.get("responses", {})
                if responses:
                    lines.append("\nОтветы:")
                    for code, resp in responses.items():
                        lines.append(f"- **{code}**: {resp['description']}")

                lines.append("")

        # Схемы компонентов
        schemas = spec.get("components", {}).get("schemas", {})
        if schemas:
            lines.append("## Схемы данных\n")
            for name, schema in schemas.items():
                props = schema.get("properties", {})
                required = schema.get("required", [])
                lines.append(f"### {name}")
                for prop, prop_schema in props.items():
                    req_mark = " *(обязательно)*" if prop in required else ""
                    enum_mark = f" — enum: {prop_schema.get('enum', [])}" if "enum" in prop_schema else ""
                    lines.append(f"- `{prop}` ({prop_schema.get('type', '?')}){req_mark}{enum_mark}")
                lines.append("")

        return "\n".join(lines)

    docs = generate_api_docs(openapi_spec)
    print(docs)

    print()


# =============================================================================
# Точка входа
# =============================================================================

if __name__ == "__main__":
    demo_http_methods()
    demo_resource_design()
    demo_authentication()
    demo_api_documentation()
