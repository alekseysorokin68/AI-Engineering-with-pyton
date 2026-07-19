"""146 — gRPC & Protocol Buffers: protobuf, streaming, service definitions

Темы:
  1. Protocol Buffers (message definition, field types, serialization)
  2. Service Definition (RPC methods, request/response types)
  3. Streaming (unary, server-streaming, client-streaming, bidirectional)
  4. gRPC Patterns (interceptors, deadlines, error handling)

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
# Демо 1: Protocol Buffers
# =============================================================================

def demo_protocol_buffers():
    """Демонстрация Protocol Buffers: определение сообщений, типы полей, сериализация."""
    print("=" * 70)
    print("Демо 1: Protocol Buffers (message definition, field types)")
    print("=" * 70)

    # --- 1.1 Определение protobuf-схемы ---
    print("\n--- 1.1 Определение protobuf-схемы (proto3) ---")

    proto_schema = """
// Файл: course.proto
syntax = "proto3";

package ai_course;

// Перечисление уровня курса
enum CourseLevel {
    BEGINNER = 0;
    INTERMEDIATE = 1;
    ADVANCED = 2;
}

// Тип "Курс"
message Course {
    int32 id = 1;
    string title = 2;
    string description = 3;
    CourseLevel level = 4;
    double price = 5;
    int32 duration_hours = 6;
    repeated string tags = 7;           // repeated = массив
    map<string, string> metadata = 8;  // map = словарь
}

// Тип "Студент"
message Student {
    int32 id = 1;
    string name = 2;
    string email = 3;
    repeated int32 enrolled_course_ids = 4;
    map<string, double> grades = 5;     // course_id → оценка
}

// Тип "Запись" (вложение)
message Enrollment {
    int32 student_id = 1;
    int32 course_id = 2;
    int64 enrolled_at = 3;              // timestamp в Unix-секундах
    bool completed = 4;
}
"""
    print("  Proto3-схема:")
    for line in proto_schema.strip().split("\n"):
        if line.strip():
            print(f"    {line}")

    # --- 1.2 Типы полей protobuf ---
    print("\n--- 1.2 Типы полей Protocol Buffers ---")

    field_types = {
        "double":   {"wire_type": "fixed64", "size": "8 байт",   "example": "3.14"},
        "float":    {"wire_type": "fixed32", "size": "4 байта",  "example": "2.71"},
        "int32":    {"wire_type": "varint",  "size": "1-5 байт", "example": "42"},
        "int64":    {"wire_type": "varint",  "size": "1-10 байт","example": "9223372036854775807"},
        "uint32":   {"wire_type": "varint",  "size": "1-5 байт", "example": "0"},
        "bool":     {"wire_type": "varint",  "size": "1 байт",   "example": "true"},
        "string":   {"wire_type": "length",  "size": "переменная","example": "hello"},
        "bytes":    {"wire_type": "length",  "size": "переменная","example": "0xff 0xab"},
    }

    print(f"\n  {'Тип':10s} | {'Wire Type':12s} | {'Размер':14s} | Пример")
    print(f"  {'-'*10}-+-{'-'*12}-+-{'-'*14}-+--{'-'*20}")
    for ptype, info in field_types.items():
        print(f"  {ptype:10s} | {info['wire_type']:12s} | {info['size']:14s} | {info['example']}")

    # --- 1.3 Varint encoding ---
    print("\n--- 1.3 Varint-кодирование (против) ---")

    def encode_varint(value):
        """Кодирование целого числа в varint-формат."""
        result = []
        while value > 127:
            result.append((value & 0x7F) | 0x80)  # 7 бит данных + 1 бит продолжения
            value >>= 7
        result.append(value & 0x7F)
        return bytes(result)

    def decode_varint(data):
        """Декодирование varint."""
        result = 0
        shift = 0
        for byte in data:
            result |= (byte & 0x7F) << shift
            if not (byte & 0x80):
                break
            shift += 7
        return result

    test_values = [1, 127, 128, 300, 16384, 2097151, 268435455]

    print("\n  Значение          → Varint (hex)        → Длина  → Декодировано")
    print(f"  {'-'*17}   {'-'*20}   {'-'*6}   {'-'*10}")
    for val in test_values:
        encoded = encode_varint(val)
        decoded = decode_varint(encoded)
        hex_str = " ".join(f"{b:02x}" for b in encoded)
        assert decoded == val, f"Ошибка декодирования: {decoded} != {val}"
        print(f"  {val:17d}   {hex_str:20s}   {len(encoded):6d}   {decoded:10d}")

    # --- 1.4 Сериализация (JSON-подобная) ---
    print("\n--- 1.4 Сериализация protobuf-сообщений ---")

    class ProtobufMessage:
        """Базовый класс для protobuf-сообщений."""
        def __init__(self, **kwargs):
            for field, value in kwargs.items():
                setattr(self, field, value)

        def serialize(self):
            """Сериализация в protobuf-подобный формат (упрощённый)."""
            result = {}
            for field_name, value in self.__dict__.items():
                if isinstance(value, bool):
                    result[field_name] = {"type": "bool", "value": value}
                elif isinstance(value, int):
                    result[field_name] = {"type": "int32", "value": value}
                elif isinstance(value, float):
                    result[field_name] = {"type": "double", "value": value}
                elif isinstance(value, str):
                    result[field_name] = {"type": "string", "value": value}
                elif isinstance(value, list):
                    result[field_name] = {"type": "repeated", "value": value}
                elif isinstance(value, dict):
                    result[field_name] = {"type": "map", "value": value}
            return result

        def to_json(self):
            """В JSON для демонстрации."""
            return json.dumps(self.serialize(), ensure_ascii=False, indent=2)

    class Course(ProtobufMessage):
        """Тип Course."""
        def __init__(self, id, title, level, price=0.0, tags=None, **kwargs):
            super().__init__(
                id=id, title=title, level=level,
                price=price, tags=tags or [], **kwargs
            )

    class Student(ProtobufMessage):
        """Тип Student."""
        def __init__(self, id, name, email, grades=None, **kwargs):
            super().__init__(
                id=id, name=name, email=email,
                grades=grades or {}, **kwargs
            )

    # Создаём тестовые объекты
    course1 = Course(
        id=1,
        title="Python для AI",
        level="BEGINNER",
        price=99.99,
        tags=["python", "ai", "beginner"],
        metadata={"instructor": "Иванов", "language": "ru"},
    )

    student1 = Student(
        id=42,
        name="Алексей Смирнов",
        email="alex@example.com",
        grades={"1": 4.5, "2": 5.0},
    )

    print("\n  Course (сериализация):")
    print(course1.to_json())

    print("\n  Student (сериализация):")
    print(student1.to_json())

    # Размер данных
    course_json = json.dumps(course1.serialize(), ensure_ascii=False)
    print(f"  Размер JSON-сериализации Course: {len(course_json)} байт")

    print()


# =============================================================================
# Демо 2: Service Definition
# =============================================================================

def demo_service_definition():
    """Демонстрация определения gRPC-сервисов: RPC-методы, типы запросов/ответов."""
    print("=" * 70)
    print("Демо 2: Service Definition (RPC methods, request/response)")
    print("=" * 70)

    # --- 2.1 Определение сервиса ---
    print("\n--- 2.1 Определение gRPC-сервиса ---")

    service_proto = """
service CourseService {
    // Unary RPC: один запрос → один ответ
    rpc GetCourse (GetCourseRequest) returns (CourseResponse);

    // Server-streaming RPC: один запрос → поток ответов
    rpc ListCourses (ListCoursesRequest) returns (stream CourseResponse);

    // Client-streaming RPC: поток запросов → один ответ
    rpc SubmitGrades (stream GradeSubmission) returns (GradeSummary);

    // Bidirectional streaming: поток запросов → поток ответов
    rpc Chat (stream ChatMessage) returns (stream ChatMessage);
}

message GetCourseRequest {
    int32 course_id = 1;
}

message ListCoursesRequest {
    int32 page_size = 1;
    string page_token = 2;
    CourseLevel level_filter = 3;
}

message CourseResponse {
    Course course = 1;
    bool from_cache = 2;
}

message GradeSubmission {
    int32 student_id = 1;
    int32 course_id = 2;
    double score = 3;
}

message GradeSummary {
    int32 total_submitted = 1;
    double average_score = 2;
    repeated string warnings = 3;
}

message ChatMessage {
    string user_id = 1;
    string text = 2;
    int64 timestamp = 3;
}
"""
    print("  Proto-схема сервиса:")
    for line in service_proto.strip().split("\n"):
        if line.strip():
            print(f"    {line}")

    # --- 2.2 Типы RPC-методов ---
    print("\n--- 2.2 Типы RPC-методов ---")

    rpc_types = [
        ("Unary",           "1 запрос → 1 ответ",    "GetCourse",      "Простой вызов функции"),
        ("Server-streaming","1 запрос → N ответов",  "ListCourses",    "Сервер отправляет данные пачками"),
        ("Client-streaming","N запросов → 1 ответ",  "SubmitGrades",   "Клиент отправляет пачками"),
        ("Bidi-streaming",  "N запросов → M ответов","Chat",           "Двунаправленный поток"),
    ]

    for rtype, desc, method, usage in rpc_types:
        print(f"\n  {rtype}:")
        print(f"    Описание: {desc}")
        print(f"    Пример:   {method}")
        print(f"    Использование: {usage}")

    # --- 2.3 Реализация сервиса (Python) ---
    print("\n--- 2.3 Реализация gRPC-сервиса (имитация) ---")

    class CourseServiceServicer:
        """Имитация gRPC-сервиса CourseService."""

        def __init__(self):
            self.courses = {
                1: {"id": 1, "title": "Python для AI", "level": "BEGINNER", "price": 99.99},
                2: {"id": 2, "title": "Deep Learning", "level": "ADVANCED", "price": 199.99},
                3: {"id": 3, "title": "MLOps", "level": "INTERMEDIATE", "price": 149.99},
                4: {"id": 4, "title": "NLP с transformers", "level": "ADVANCED", "price": 249.99},
                5: {"id": 5, "title": "Компьютерное зрение", "level": "INTERMEDIATE", "price": 179.99},
            }
            self.grades = []

        def GetCourse(self, request):
            """Unary RPC: получить курс по ID."""
            course_id = request.get("course_id", 0)
            if course_id in self.courses:
                return {"course": self.courses[course_id], "from_cache": False}
            return {"error": "NOT_FOUND", "message": f"Курс {course_id} не найден"}

        def ListCourses(self, request, stream=False):
            """Server-streaming: список курсов с пагинацией."""
            page_size = request.get("page_size", 2)
            level_filter = request.get("level_filter", None)

            courses = list(self.courses.values())
            if level_filter:
                courses = [c for c in courses if c["level"] == level_filter]

            # Генерируем страницы
            pages = []
            for i in range(0, len(courses), page_size):
                page = courses[i:i + page_size]
                pages.append(page)
            return pages

        def SubmitGrades(self, grade_stream):
            """Client-streaming: приём оценок."""
            submissions = list(grade_stream)
            self.grades.extend(submissions)

            total = len(submissions)
            avg_score = sum(g["score"] for g in submissions) / total if total > 0 else 0
            warnings = []
            if avg_score < 3.0:
                warnings.append("Средний балл ниже 3.0")
            low_grades = [g for g in submissions if g["score"] < 2.0]
            if low_grades:
                warnings.append(f"{len(low_grades)} студентов с баллом ниже 2.0")

            return {
                "total_submitted": total,
                "average_score": round(avg_score, 2),
                "warnings": warnings,
            }

    # Использование
    service = CourseServiceServicer()

    # Unary RPC
    print("\n  Unary RPC — GetCourse:")
    result = service.GetCourse({"course_id": 1})
    print(f"    Запрос: {{course_id: 1}}")
    print(f"    Ответ:  {json.dumps(result, ensure_ascii=False)}")

    result = service.GetCourse({"course_id": 999})
    print(f"    Запрос: {{course_id: 999}}")
    print(f"    Ответ:  {json.dumps(result, ensure_ascii=False)}")

    # Server-streaming
    print("\n  Server-streaming RPC — ListCourses:")
    pages = service.ListCourses({"page_size": 2, "level_filter": "ADVANCED"})
    for i, page in enumerate(pages):
        titles = [c["title"] for c in page]
        print(f"    Страница {i + 1}: {titles}")

    # Client-streaming
    print("\n  Client-streaming RPC — SubmitGrades:")
    grade_stream = [
        {"student_id": 1, "course_id": 1, "score": 4.5},
        {"student_id": 2, "course_id": 1, "score": 3.8},
        {"student_id": 3, "course_id": 1, "score": 5.0},
        {"student_id": 4, "course_id": 1, "score": 1.5},
    ]
    result = service.SubmitGrades(grade_stream)
    print(f"    Отправлено оценок: {len(grade_stream)}")
    print(f"    Результат: {json.dumps(result, ensure_ascii=False)}")

    # --- 2.4 Метрики производительности ---
    print("\n--- 2.4 Сравнение gRPC vs REST ---")

    comparison = [
        ("Протокол",           "HTTP/1.1 + JSON",      "HTTP/2 + Protobuf"),
        ("Кодирование",        "Текст (JSON)",         "Бинарное (protobuf)"),
        ("Размер сообщения",   "~1000 байт",           "~200 байт"),
        ("Скорость (сериализация)", "~50 мкс",          "~5 мкс"),
        ("Стриминг",           "Нет (самодельный)",    "Встроенный"),
        ("Контракт",           "Нет (OpenAPI опционален)", "Строгий (protobuf)"),
        ("Клиенты",            "Любой HTTP-клиент",    "Специальный gRPC-клиент"),
    ]

    print(f"\n  {'Параметр':25s} | {'REST/HTTP':25s} | {'gRPC/protobuf':25s}")
    print(f"  {'-'*25}-+-{'-'*25}-+-{'-'*25}")
    for param, rest, grpc in comparison:
        print(f"  {param:25s} | {rest:25s} | {grpc:25s}")

    print()


# =============================================================================
# Демо 3: Streaming
# =============================================================================

def demo_streaming():
    """Демонстрация стриминга: unary, server-streaming, client-streaming, bidirectional."""
    print("=" * 70)
    print("Демо 3: Streaming (unary, server, client, bidirectional)")
    print("=" * 70)

    # --- 3.1 Unary RPC ---
    print("\n--- 3.1 Unary RPC (простой вызов) ---")

    class UnaryClient:
        """Клиент для Unary RPC."""
        def call(self, method, request):
            """Вызов unary RPC."""
            start = time.time()
            # Имитация сетевого вызова
            time.sleep(random.uniform(0.001, 0.005))
            elapsed = time.time() - start
            return {
                "method": method,
                "request": request,
                "latency_ms": round(elapsed * 1000, 2),
            }

    client = UnaryClient()

    requests = [
        ("GetUser", {"user_id": 42}),
        ("GetCourse", {"course_id": 1}),
        ("GetOrder", {"order_id": 100}),
    ]

    total_latency = 0
    for method, req in requests:
        result = client.call(method, req)
        total_latency += result["latency_ms"]
        print(f"  {method}({req}) → {result['latency_ms']}ms")

    print(f"  Общая задержка: {total_latency:.2f}ms (последовательно)")

    # --- 3.2 Server-streaming ---
    print("\n--- 3.2 Server-streaming RPC ---")

    class ServerStreamingClient:
        """Клиент для Server-streaming RPC."""
        def __init__(self, data_generator):
            self.data_generator = data_generator

        def stream(self, request):
            """Получение потока данных от сервера."""
            chunk_count = 0
            total_items = 0
            for chunk in self.data_generator(request):
                chunk_count += 1
                total_items += len(chunk)
                yield chunk

            return {"chunks": chunk_count, "total_items": total_items}

    def generate_course_pages(request):
        """Генератор страниц курсов."""
        all_courses = [
            f"Курс {i}: {random.choice(['Python', 'AI', 'ML', 'DL', 'NLP'])}"
            for i in range(1, 21)
        ]
        page_size = request.get("page_size", 5)

        for i in range(0, len(all_courses), page_size):
            page = all_courses[i:i + page_size]
            time.sleep(0.001)  # Имитация задержки
            yield page

    streaming_client = ServerStreamingClient(generate_course_pages)
    print("  Сервер отправляет курсы страницами по 5:")

    chunk_num = 0
    for chunk in streaming_client.stream({"page_size": 5}):
        chunk_num += 1
        print(f"    Чанк {chunk_num}: {chunk}")

    # --- 3.3 Client-streaming ---
    print("\n--- 3.3 Client-streaming RPC ---")

    class ClientStreamingClient:
        """Клиент для Client-streaming RPC."""
        def __init__(self):
            self.buffer = []

        def send(self, item):
            """Отправить элемент в буфер."""
            self.buffer.append(item)

        def complete(self):
            """Завершить стрим и получить результат."""
            total = len(self.buffer)
            sum_val = sum(item.get("value", 0) for item in self.buffer)
            avg_val = sum_val / total if total > 0 else 0
            return {
                "total_items": total,
                "sum": sum_val,
                "average": round(avg_val, 2),
            }

    streaming_client = ClientStreamingClient()

    print("  Клиент отправляет данные о производительности:")
    for i in range(10):
        data_point = {
            "timestamp": time.time() + i,
            "value": random.uniform(50, 150),
            "metric": "latency_ms",
        }
        streaming_client.send(data_point)
        print(f"    → Отправлено: value={data_point['value']:.2f}")

    result = streaming_client.complete()
    print(f"\n  Результат агрегации на сервере:")
    print(f"    Всего точек: {result['total_items']}")
    print(f"    Сумма: {result['sum']:.2f}")
    print(f"    Среднее: {result['average']:.2f}")

    # --- 3.4 Bidirectional streaming ---
    print("\n--- 3.4 Bidirectional Streaming (Chat) ---")

    class ChatSession:
        """Сессия чата с двунаправленным стримингом."""
        def __init__(self):
            self.messages = []
            self.handlers = {}

        def register_handler(self, user_id, handler):
            """Регистрация обработчика сообщений."""
            self.handlers[user_id] = handler

        def send(self, user_id, text):
            """Отправка сообщения."""
            msg = {
                "user_id": user_id,
                "text": text,
                "timestamp": time.time(),
                "msg_id": len(self.messages) + 1,
            }
            self.messages.append(msg)

            # Уведомляем всех подписчиков
            for handler in self.handlers.values():
                handler(msg)

            return msg

        def history(self, limit=5):
            """Получение истории сообщений."""
            return self.messages[-limit:]

    chat = ChatSession()

    # Регистрируем обработчики (получатели сообщений)
    received_by = {"alice": [], "bob": [], "charlie": []}

    def alice_handler(msg):
        received_by["alice"].append(msg)

    def bob_handler(msg):
        received_by["bob"].append(msg)

    def charlie_handler(msg):
        received_by["charlie"].append(msg)

    chat.register_handler("alice", alice_handler)
    chat.register_handler("bob", bob_handler)
    chat.register_handler("charlie", charlie_handler)

    # Обмен сообщениями
    print("  Чат между alice, bob и charlie:")

    chat_messages = [
        ("alice",   "Привет! Кто изучает gRPC?"),
        ("bob",     "Я! Только начал изучать Protocol Buffers"),
        ("charlie", "Я уже месяц работаю с gRPC. Могу помочь!"),
        ("alice",   "Отлично! Расскажите о стриминге"),
        ("bob",     "Да, особенно интересует bidirectional streaming"),
        ("charlie", "Это как WebSocket, но с типизацией и protobuf"),
    ]

    for user, text in chat_messages:
        msg = chat.send(user, text)
        print(f"    [{user:10s}] → {text}")

    print(f"\n  Всего сообщений: {len(chat.messages)}")

    # Показать кто что получил
    for user, msgs in received_by.items():
        print(f"  {user} получил {len(msgs)} сообщений")
        for m in msgs[:2]:
            print(f"    [{m['user_id']}] {m['text'][:40]}...")

    print()


# =============================================================================
# Демо 4: gRPC Patterns
# =============================================================================

def demo_grpc_patterns():
    """Демонстрация паттернов gRPC: перехватчики, дедлайны, обработка ошибок."""
    print("=" * 70)
    print("Демо 4: gRPC Patterns (interceptors, deadlines, errors)")
    print("=" * 70)

    # --- 4.1 Interceptors ---
    print("\n--- 4.1 Interceptors (Перехватчики) ---")

    class Interceptor:
        """Базовый перехватчик gRPC."""
        def __init__(self, name):
            self.name = name

        def intercept(self, method, request, metadata):
            """Перехват вызова."""
            pass

    class LoggingInterceptor(Interceptor):
        """Перехватчик логирования."""
        def __init__(self):
            super().__init__("Logging")
            self.logs = []

        def intercept(self, method, request, metadata):
            """Логирование запросов и ответов."""
            start = time.time()
            self.logs.append(f"[{self.name}] → {method}: {json.dumps(request, default=str)[:50]}")

            # Вызов реального метода (имитация)
            time.sleep(random.uniform(0.001, 0.01))
            elapsed = time.time() - start

            self.logs.append(f"[{self.name}] ← {method}: OK ({elapsed*1000:.1f}ms)")
            return elapsed

    class MetricsInterceptor(Interceptor):
        """Перехватчик метрик."""
        def __init__(self):
            super().__init__("Metrics")
            self.metrics = collections.defaultdict(int)
            self.latencies = collections.defaultdict(list)

        def intercept(self, method, request, metadata):
            """Сбор метрик."""
            start = time.time()
            self.metrics[f"{method}.requests"] += 1

            time.sleep(random.uniform(0.001, 0.01))
            elapsed = time.time() - start
            self.latencies[method].append(elapsed * 1000)

            return elapsed

        def summary(self):
            """Сводка метрик."""
            result = {}
            for method, latencies in self.latencies.items():
                result[method] = {
                    "count": len(latencies),
                    "avg_ms": round(sum(latencies) / len(latencies), 2),
                    "min_ms": round(min(latencies), 2),
                    "max_ms": round(max(latencies), 2),
                    "p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0, 2),
                }
            return result

    class AuthInterceptor(Interceptor):
        """Перехватчик аутентификации."""
        def __init__(self, valid_tokens):
            super().__init__("Auth")
            self.valid_tokens = valid_tokens

        def intercept(self, method, request, metadata):
            """Проверка токена."""
            token = metadata.get("authorization", "")
            if token not in self.valid_tokens:
                return {"error": "UNAUTHENTICATED", "message": "Невалидный токен"}

            # Добавляем информацию о пользователе
            metadata["user_id"] = self.valid_tokens[token]
            return None  # OK

    # Создаём перехватчики
    logging = LoggingInterceptor()
    metrics = MetricsInterceptor()
    auth = AuthInterceptor({"token_abc": "user_1", "token_xyz": "user_2"})

    # Имитация вызовов с перехватчиками
    print("  Вызовы с перехватчиками:")

    calls = [
        ("GetUser",       {"user_id": 1},    {"authorization": "token_abc"}),
        ("GetCourse",     {"course_id": 2},  {"authorization": "token_abc"}),
        ("CreateOrder",   {"item": "x"},     {"authorization": "token_xyz"}),
        ("DeleteAccount", {"user_id": 1},    {"authorization": "token_bad"}),
    ]

    for method, request, metadata in calls:
        # Auth interceptor
        auth_result = auth.intercept(method, request, metadata)
        if auth_result and "error" in auth_result:
            print(f"  {method}: {auth_result['error']} — {auth_result['message']}")
            continue

        # Logging interceptor
        logging.intercept(method, request, metadata)

        # Metrics interceptor
        metrics.intercept(method, request, metadata)

        print(f"  {method}: OK (user={metadata.get('user_id', '?')})")

    # Показать логи
    print("\n  Логи перехватчика:")
    for log in logging.logs:
        print(f"    {log}")

    # Показать метрики
    print("\n  Метрики перехватчика:")
    summary = metrics.summary()
    for method, stats in summary.items():
        print(f"    {method}: count={stats['count']}, avg={stats['avg_ms']}ms, p95={stats['p95_ms']}ms")

    # --- 4.2 Deadlines (Таймауты) ---
    print("\n--- 4.2 Deadlines (Таймауты) ---")

    class DeadlineManager:
        """Управление дедлайнами gRPC-вызовов."""
        def __init__(self):
            self.default_timeout = 5.0  # секунд

        def check_deadline(self, deadline_seconds):
            """Проверка, не истёк ли дедлайн."""
            now = time.time()
            remaining = deadline_seconds - now
            if remaining <= 0:
                return False, 0
            return True, remaining

        def create_deadline(self, timeout_seconds):
            """Создание дедлайна."""
            return time.time() + timeout_seconds

    deadline_mgr = DeadlineManager()

    # Имитация вызовов с разными дедлайнами
    print("  Проверка дедлайнов:")

    deadlines = [
        ("GetUser",    5.0,   0.01),  # 方法名, таймаут, задержка
        ("GetCourse",  2.0,   0.05),
        ("HeavyQuery", 0.1,   0.2),   # Превысит дедлайн
    ]

    for method, timeout, delay in deadlines:
        deadline = deadline_mgr.create_deadline(timeout)
        time.sleep(delay)

        is_valid, remaining = deadline_mgr.check_deadline(deadline)
        status = "OK" if is_valid else "DEADLINE_EXCEEDED"
        remaining_ms = round(remaining * 1000, 1) if is_valid else 0
        print(f"    {method:15s}: timeout={timeout:.1f}s, delay={delay:.2f}s → {status} (осталось {remaining_ms}ms)")

    # --- 4.3 Error Handling ---
    print("\n--- 4.3 Обработка ошибок (gRPC Status Codes) ---")

    grpc_errors = {
        "OK":                 (0,  "Успешное выполнение"),
        "CANCELLED":          (1,  "Операция отменена клиентом"),
        "UNKNOWN":            (2,  "Неизвестная ошибка"),
        "INVALID_ARGUMENT":   (3,  "Невалидный аргумент"),
        "NOT_FOUND":          (5,  "Ресурс не найден"),
        "ALREADY_EXISTS":     (6,  "Ресурс уже существует"),
        "PERMISSION_DENIED":  (7,  "Нет прав доступа"),
        "UNAUTHENTICATED":    (16, "Необходима аутентификация"),
        "RESOURCE_EXHAUSTED": (8,  "Ресурсы исчерпаны"),
        "FAILED_PRECONDITION":(9,  "Условие не выполнено"),
        "ABORTED":            (10, "Операция прервана"),
        "UNAVAILABLE":        (14, "Сервис недоступен"),
        "UNIMPLEMENTED":      (12, "Метод не реализован"),
        "INTERNAL":           (13, "Внутренняя ошибка сервера"),
        "DEADLINE_EXCEEDED":  (4,  "Превышен дедлайн"),
    }

    print(f"\n  {'Статус':25s} | {'Код':4s} | Описание")
    print(f"  {'-'*25}-+-{'-'*4}-+--{'-'*40}")
    for status, (code, desc) in grpc_errors.items():
        print(f"  {status:25s} | {code:4d} | {desc}")

    # --- 4.4 Retry Policy ---
    print("\n--- 4.4 Retry Policy (Политика повторных попыток) ---")

    class RetryPolicy:
        """Политика повторных попыток для gRPC-вызовов."""
        def __init__(self, max_retries=3, base_delay=1.0, max_delay=10.0, jitter=True):
            self.max_retries = max_retries
            self.base_delay = base_delay
            self.max_delay = max_delay
            self.jitter = jitter

        def get_delay(self, attempt):
            """Расчёт задержки с экспоненциальным бэкоффем."""
            delay = min(self.base_delay * (2 ** attempt), self.max_delay)
            if self.jitter:
                delay *= random.uniform(0.5, 1.0)
            return delay

        def should_retry(self, status_code, attempt):
            """Определение, стоит ли повторять запрос."""
            if attempt >= self.max_retries:
                return False
            # Повторяем только при определённых ошибках
            retryable = {1, 2, 4, 8, 10, 14}  # CANCELLED, UNKNOWN, DEADLINE_EXCEEDED, ...
            return status_code in retryable

    retry_policy = RetryPolicy(max_retries=4, base_delay=0.1)

    print("  Экспоненциальный бэкфф с джиттером:")
    for attempt in range(retry_policy.max_retries):
        delay = retry_policy.get_delay(attempt)
        should = retry_policy.should_retry(14, attempt)  # UNAVAILABLE
        print(f"    Попытка {attempt + 1}: задержка={delay:.3f}с, повтор={should}")

    # Имитация retry
    print("\n  Имитация повторных попыток:")

    def unreliable_call():
        """Ненадёжный вызов (иногда падает)."""
        return random.choice([True, True, True, False])

    policy = RetryPolicy(max_retries=3, base_delay=0.01, jitter=False)
    attempt = 0

    while attempt <= policy.max_retries:
        attempt += 1
        success = unreliable_call()
        if success:
            print(f"    Попытка {attempt}: ✅ Успех!")
            break
        else:
            delay = policy.get_delay(attempt - 1) if attempt <= policy.max_retries else 0
            print(f"    Попытка {attempt}: ❌ Ошибка, ожидание {delay:.3f}с...")
            if attempt <= policy.max_retries:
                time.sleep(delay)

    if attempt > policy.max_retries:
        print(f"    Все попытки исчерпаны после {attempt} попыток")

    print()


# =============================================================================
# Точка входа
# =============================================================================

if __name__ == "__main__":
    demo_protocol_buffers()
    demo_service_definition()
    demo_streaming()
    demo_grpc_patterns()
