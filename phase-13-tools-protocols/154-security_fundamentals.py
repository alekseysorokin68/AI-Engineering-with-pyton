"""154 — Security Fundamentals: authentication, encryption, OWASP top 10

Темы:
  1. Authentication (passwords, bcrypt, JWT, session tokens)
  2. Encryption (symmetric vs asymmetric, hashing, salting)
  3. OWASP Top 10 (injection, XSS, CSRF, broken auth)
  4. Secure Coding (input validation, parameterized queries, secrets management)

Самодостаточный файл — не требует numpy, torch, transformers.
"""

import math
import random
import re
import json
import hashlib
import collections
import time
import datetime

random.seed(42)


# ==========================================================================
# Демо 1 — Аутентификация: пароли, хэширование, JWT, сессии
# ==========================================================================
def demo_authentication():
    """Демонстрация методов аутентификации пользователей."""

    print("=" * 70)
    print("Демо 1: Аутентификация (Authentication)")
    print("=" * 70)

    # --- Подпример 1: Хэширование паролей ---
    print("\n--- 1.1 Хэширование паролей (SHA-256 + соль) ---")

    def hash_password(password, salt=None):
        """
        Хэширование пароля с солью.
        Формула: hash = SHA256(password + salt)
        Соль предотвращает атаки по таблицам радуг (rainbow tables).
        """
        if salt is None:
            salt = hashlib.sha256(str(random.random()).encode()).hexdigest()[:16]
        # Конкатенация пароля и соли, затем хэширование
        hashed = hashlib.sha256((password + salt).encode()).hexdigest()
        return salt, hashed

    def verify_password(password, salt, stored_hash):
        """Проверка пароля: вычисляем хэш и сравниваем с хранимым."""
        _, computed_hash = hash_password(password, salt)
        return computed_hash == stored_hash

    # Демонстрация: один и тот же пароль → разные хэши из-за разной соли
    password = "MySecureP@ss123"
    salt1, hash1 = hash_password(password)
    salt2, hash2 = hash_password(password)

    print(f"  Пароль: {password}")
    print(f"  Соль 1: {salt1}")
    print(f"  Хэш 1:  {hash1[:40]}...")
    print(f"  Соль 2: {salt2}")
    print(f"  Хэш 2:  {hash2[:40]}...")
    print(f"  Хэши совпадают: {hash1 == hash2}")  # False — разные соли
    print(f"  Проверка верного пароля: {verify_password(password, salt1, hash1)}")
    print(f"  Проверка неверного пароля: {verify_password('WrongPass', salt1, hash1)}")

    # --- Подпример 2: Имитация bcrypt (медленное хэширование) ---
    print("\n--- 1.2 Имитация bcrypt (cost factor) ---")

    def simulate_bcrypt(password, rounds=4):
        """
        Имитация bcrypt: повторное хэширование для замедления brute-force.
        Формула: хэш = SHA256 повторён rounds раз.
        В реальном bcrypt используют Blowfish и 2^rounds итераций.
        """
        result = password.encode()
        for i in range(2 ** rounds):
            result = hashlib.sha256(result).digest()
        return result.hex()

    test_password = "SecurePassword123"
    # Разные cost factors — демонстрация влияния на время
    for cost in [2, 3, 4]:
        start = time.time()
        h = simulate_bcrypt(test_password, rounds=cost)
        elapsed = time.time() - start
        print(f"  Cost={cost} (2^{cost}={2**cost} итераций): {elapsed:.4f}s → {h[:32]}...")

    # --- Подпример 3: JWT (JSON Web Token) ---
    print("\n--- 1.3 JWT — JSON Web Token ---")

    def base64url_encode(data):
        """Base64url кодирование (стандарт для JWT)."""
        if isinstance(data, str):
            data = data.encode()
        import base64
        return base64.urlsafe_b64encode(data).rstrip(b'=').decode()

    def create_jwt(payload, secret_key, expires_in=3600):
        """
        Создание JWT токена.
        Структура: header.payload.signature
        Формула: signature = HMAC-SHA256(base64(header) + '.' + base64(payload), secret)
        """
        # Заголовок
        header = {"alg": "HS256", "typ": "JWT"}
        header_b64 = base64url_encode(json.dumps(header))

        # Полезная нагрузка с временными метками
        payload_full = payload.copy()
        payload_full["iat"] = int(time.time())  # issued at
        payload_full["exp"] = int(time.time()) + expires_in  # expiration
        payload_b64 = base64url_encode(json.dumps(payload_full))

        # Подпись
        message = f"{header_b64}.{payload_b64}"
        signature = hashlib.sha256((message + secret_key).encode()).hexdigest()[:32]
        signature_b64 = base64url_encode(signature)

        return f"{header_b64}.{payload_b64}.{signature_b64}"

    def verify_jwt(token, secret_key):
        """Проверка JWT: пересчитываем подпись и проверяем срок действия."""
        parts = token.split(".")
        if len(parts) != 3:
            return None, "Неверный формат токена"

        header_b64, payload_b64, signature_b64 = parts
        # Пересчитываем подпись
        message = f"{header_b64}.{payload_b64}"
        expected_sig = hashlib.sha256((message + secret_key).encode()).hexdigest()[:32]

        if signature_b64 != base64url_encode(expected_sig):
            return None, "Подпись невалидна"

        # Декодируем payload
        import base64
        padding = 4 - len(payload_b64) % 4
        payload_b64 += "=" * padding
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))

        # Проверяем срок действия
        if payload.get("exp", 0) < time.time():
            return None, "Токен истёк"

        return payload, "OK"

    SECRET = "my_super_secret_key_2024"
    user_payload = {"sub": "user:42", "role": "admin", "name": "Алексей"}

    jwt_token = create_jwt(user_payload, SECRET, expires_in=3600)
    print(f"  JWT Token:")
    parts = jwt_token.split(".")
    print(f"    Header:   {parts[0]}")
    print(f"    Payload:  {parts[1]}")
    print(f"    Signature:{parts[2]}")
    print(f"    Полная строка: {jwt_token[:80]}...")

    # Проверка валидного токена
    payload, status = verify_jwt(jwt_token, SECRET)
    print(f"\n  Проверка: {status}")
    print(f"  Payload: {payload}")

    # Проверка с неверным ключом
    _, status_bad = verify_jwt(jwt_token, "wrong_secret")
    print(f"  Проверка с неверным ключом: {status_bad}")

    # --- Подпример 4: Session Tokens ---
    print("\n--- 1.4 Session Tokens (сессионные токены) ---")

    class SessionManager:
        """Менеджер сессий с хранением в памяти."""
        def __init__(self):
            self._sessions = {}  # token → {user_id, created, expires}

        def create_session(self, user_id, ttl=1800):
            """Создание новой сессии."""
            token = hashlib.sha256(
                f"{user_id}:{time.time()}:{random.random()}".encode()
            ).hexdigest()
            self._sessions[token] = {
                "user_id": user_id,
                "created": datetime.datetime.now().isoformat(),
                "expires": time.time() + ttl,
                "ttl": ttl,
            }
            print(f"  Сессия создана: token={token[:24]}... user={user_id}")
            return token

        def validate_session(self, token):
            """Проверка валидности сессии."""
            if token not in self._sessions:
                return None, "Сессия не найдена"
            session = self._sessions[token]
            if time.time() > session["expires"]:
                del self._sessions[token]
                return None, "Сессия истекла"
            remaining = session["expires"] - time.time()
            return session, f"Валидна (осталось {remaining:.0f}s)"

        def destroy_session(self, token):
            """Уничтожение сессии (logout)."""
            if token in self._sessions:
                del self._sessions[token]
                print(f"  Сессия {token[:24]}... уничтожена")
                return True
            return False

    sm = SessionManager()
    token = sm.create_session("user:42")
    session, status = sm.validate_session(token)
    print(f"  Проверка: {status}")

    # Уничтожаем сессию
    sm.destroy_session(token)
    session, status = sm.validate_session(token)
    print(f"  После logout: {status}")

    print("\n--- Сравнение методов ---")
    methods = {
        "Password + Salt": "Просто, но уязвим к brute-force при слабых паролях",
        "bcrypt": "Медленное хэширование, защита от rainbow tables",
        "JWT": "Stateless, масштабируем, но сложнее отозвать",
        "Session": "Stateful, легко отозвать, но требует хранилища",
    }
    for name, desc in methods.items():
        print(f"  {name:18s} — {desc}")


# ==========================================================================
# Демо 2 — Шифрование: симметричное/ассиметричное, хэширование, соли
# ==========================================================================
def demo_encryption():
    """Демонстрация методов шифрования и хэширования."""

    print("\n" + "=" * 70)
    print("Демо 2: Шифрование (Encryption)")
    print("=" * 70)

    # --- Подпример 1: Симметричное шифрование (XOR) ---
    print("\n--- 2.1 Симметричное шифрование (XOR-шифр) ---")

    def xor_encrypt(text, key):
        """
        Симметричное шифрование XOR.
        Формула: cipher[i] = text[i] ⊕ key[i % len(key)]
        Одно и то же действие для шифрования и дешифрования.
        """
        result = []
        for i, char in enumerate(text):
            # XOR символа текста с символом ключа (с циклическим повтором)
            encrypted_char = ord(char) ^ ord(key[i % len(key)])
            result.append(format(encrypted_char, '02x'))
        return ' '.join(result)

    def xor_decrypt(hex_text, key):
        """Дешифрование XOR (обратная операция)."""
        bytes_list = [int(x, 16) for x in hex_text.split()]
        result = []
        for i, byte in enumerate(bytes_list):
            decrypted_char = chr(byte ^ ord(key[i % len(key)]))
            result.append(decrypted_char)
        return ''.join(result)

    original = "Hello, AI Engineering!"
    key = "SecretKey"

    encrypted = xor_encrypt(original, key)
    decrypted = xor_decrypt(encrypted, key)

    print(f"  Оригинал:  {original}")
    print(f"  Ключ:      {key}")
    print(f"  Зашифровано (hex): {encrypted}")
    print(f"  Расшифровано: {decrypted}")
    print(f"  Совпадает: {original == decrypted}")

    # --- Подпример 2: Асимметричное шифрование (RSA-подобная логика) ---
    print("\n--- 2.2 Асимметричное шифрование (принцип RSA) ---")

    def simple_rsa_encrypt(message_int, e, n):
        """
        Упрощённое RSA-шифрование: cipher = message^e mod n
        В реальном RSA используются большие простые числа.
        """
        # Быстрое возведение в степень по модулю
        cipher = pow(message_int, e, n)
        return cipher

    def simple_rsa_decrypt(cipher_int, d, n):
        """Дешифрование: message = cipher^d mod n."""
        message = pow(cipher_int, d, n)
        return message

    # Упрощённая модель RSA (маленькие числа для демонстрации)
    p, q = 61, 53  # простые числа
    n = p * q       # модуль = 3233
    phi = (p - 1) * (q - 1)  # φ(n) = 3120
    e = 17          # открытая экспонента (gcd(e, φ) = 1)
    d = pow(e, -1, phi)  # закрытая экспонента (e × d ≡ 1 mod φ)

    print(f"  Параметры RSA (упрощённые):")
    print(f"    p={p}, q={q}, n=p×q={n}")
    print(f"    φ(n)={phi}, e={e}, d={d}")
    print(f"    Проверка: e×d mod φ = {(e * d) % phi}")  # Должно быть 1

    # Шифруем сообщение (число < n)
    message = 42
    cipher = simple_rsa_encrypt(message, e, n)
    decrypted = simple_rsa_decrypt(cipher, d, n)

    print(f"\n  Сообщение: {message}")
    print(f"  Зашифровано: {cipher}")
    print(f"  Расшифровано: {decrypted}")
    print(f"  Совпадает: {message == decrypted}")

    # --- Подпример 3: Хэширование и соли ---
    print("\n--- 2.3 Хэширование и соли (_hashes & salts) ---")

    def demonstrate_hashing():
        """Сравнение хэш-функций и влияние соли."""
        test_data = "password123"

        # Разные алгоритмы хэширования
        algorithms = {
            "MD5": hashlib.md5,
            "SHA-1": hashlib.sha1,
            "SHA-256": hashlib.sha256,
            "SHA-512": hashlib.sha512,
        }

        print(f"  Данные: '{test_data}'\n")
        print(f"  {'Алгоритм':<10} {'Длина хэша':<12} {'Хэш (первые 40 символов)'}")
        print(f"  {'-'*70}")

        for name, func in algorithms.items():
            h = func(test_data.encode()).hexdigest()
            print(f"  {name:<10} {len(h):<12} {h[:40]}...")

        # Демонстрация соли
        print(f"\n  Влияние соли:")
        salt = "random_salt_value"
        h1 = hashlib.sha256(test_data.encode()).hexdigest()
        h2 = hashlib.sha256((salt + test_data).encode()).hexdigest()
        print(f"  Без соли:  {h1[:40]}...")
        print(f"  С солью:   {h2[:40]}...")
        print(f"  Разные:    {h1 != h2}")

    demonstrate_hashing()

    # --- Подпример 4: HMAC (Hash-based Message Authentication Code) ---
    print("\n--- 2.4 HMAC — аутентифицированное хэширование ---")

    def compute_hmac(message, key):
        """
        Вычисление HMAC-SHA256.
        Формула: HMAC(K, m) = H((K' ⊕ opad) || H((K' ⊕ ipad) || m))
        Где K' — ключ, дополненный/хэшированный до размера блока
        """
        # Упрощённая версия (для демонстрации концепции)
        inner = hashlib.sha256((key + message).encode()).hexdigest()
        outer = hashlib.sha256((key + inner).encode()).hexdigest()
        return outer

    api_key = "api_secret_key_123"
    message = "GET /api/users"
    signature = compute_hmac(message, api_key)

    print(f"  Сообщение: {message}")
    print(f"  Ключ: {api_key}")
    print(f"  HMAC-SHA256: {signature}")

    # Проверка подписи
    expected = compute_hmac(message, api_key)
    print(f"  Проверка подлинности: {signature == expected}")
    # Подделка — другой ключ
    forged = compute_hmac(message, "wrong_key")
    print(f"  Подделка обнаружена: {signature != forged}")

    print("\n--- Формулы шифрования ---")
    print("  XOR:         C = P ⊕ K")
    print("  RSA:         C = M^e mod n,  M = C^d mod n")
    print("  HMAC:        HMAC(K,M) = H((K'⊕opad) || H((K'⊕ipad) || M))")
    print("  bcrypt:     Uses Blowfish with cost factor 2^rounds")


# ==========================================================================
# Демо 3 — OWASP Top 10: инъекции, XSS, CSRF, слабая аутентификация
# ==========================================================================
def demo_owasp_top10():
    """Демонстрация наиболее опасных уязвимостей по версии OWASP Top 10."""

    print("\n" + "=" * 70)
    print("Демо 3: OWASP Top 10 — наиболее опасные уязвимости")
    print("=" * 70)

    # --- Подпример 1: SQL Injection ---
    print("\n--- 3.1 SQL Injection (внедрение SQL-кода) ---")

    # УЯЗВИМЫЙ код (НИКОГДА так не делайте!)
    def vulnerable_query(username):
        """Опасный способ: конкатенация строк для SQL-запроса."""
        query = f"SELECT * FROM users WHERE name = '{username}'"
        return query

    # Безопасный код: параметризованные запросы
    def safe_query(username):
        """Безопасный способ: параметризованные запросы."""
        # В реальном коде здесь используется prepared statement
        query = "SELECT * FROM users WHERE name = ?"
        params = (username,)
        return query, params

    # Демонстрация атаки
    malicious_input = "admin' OR '1'='1"
    print(f"  Ввод пользователя: {malicious_input}")
    print(f"  Уязвимый запрос: {vulnerable_query(malicious_input)}")
    print(f"  → Атакующий получает ВСЕ записи!")

    safe_q, params = safe_query(malicious_input)
    print(f"  Безопасный запрос: {safe_q}")
    print(f"  Параметры: {params}")
    print(f"  → Запрос безопасен, вредоносный ввод экранирован")

    # --- Подпример 2: Cross-Site Scripting (XSS) ---
    print("\n--- 3.2 XSS — Cross-Site Scripting (межсайтовый скриптинг) ---")

    def escape_html(text):
        """Экранирование HTML-сущностей для предотвращения XSS."""
        replacements = {
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#x27;",
        }
        result = text
        for char, entity in replacements.items():
            result = result.replace(char, entity)
        return result

    # Уязвимый ввод
    xss_payload = '<script>alert("XSS Attack!")</script>'
    print(f"  Уязвимый ввод: {xss_payload}")

    # Без экранирования — выполнится JavaScript!
    print(f"  Без экранирования: {xss_payload}")

    # С экранированием — безопасный текст
    safe_output = escape_html(xss_payload)
    print(f"  С экранированием: {safe_output}")

    # Проверка на наличие скриптов
    has_script = bool(re.search(r'<script[^>]*>', xss_payload, re.IGNORECASE))
    print(f"  Обнаружен скрипт: {has_script}")

    # --- Подпример 3: Cross-Site Request Forgery (CSRF) ---
    print("\n--- 3.3 CSRF — Cross-Site Request Forgery ---")

    def generate_csrf_token(session_id):
        """
        Генерация CSRF-токена для защиты от поддельных запросов.
        Формула: token = HMAC(session_id, secret_key)
        """
        secret = "csrf_secret_key_2024"
        return hashlib.sha256(f"{session_id}:{secret}".encode()).hexdigest()[:32]

    def validate_csrf_token(token, session_id):
        """Проверка CSRF-токена."""
        expected = generate_csrf_token(session_id)
        return token == expected

    # Создание формы с CSRF-токеном
    session_id = "sess_abc123"
    csrf_token = generate_csrf_token(session_id)
    print(f"  Session ID: {session_id}")
    print(f"  CSRF Token: {csrf_token}")

    # Валидная форма
    is_valid = validate_csrf_token(csrf_token, session_id)
    print(f"  Валидный токен: {is_valid}")

    # Поддельная форма (без токена или с неверным токеном)
    fake_token = generate_csrf_token("other_session")
    is_fake_valid = validate_csrf_token(fake_token, session_id)
    print(f"  Поддельный токен: {is_fake_valid}")

    # --- Подпример 4: Broken Authentication ---
    print("\n--- 3.4 Broken Authentication (слабая аутентификация) ---")

    class SecureAuth:
        """Демонстрация защищённой аутентификации."""
        def __init__(self):
            self._users = {}
            self._login_attempts = collections.defaultdict(int)
            self._max_attempts = 5
            self._lockout_time = 300  # 5 минут

        def register(self, username, password):
            """Регистрация с валидацией пароля."""
            # Проверка сложности пароля
            if len(password) < 8:
                return False, "Пароль слишком короткий (минимум 8 символов)"
            if not re.search(r'[A-Z]', password):
                return False, "Пароль должен содержать заглавные буквы"
            if not re.search(r'[a-z]', password):
                return False, "Пароль должен содержать строчные буквы"
            if not re.search(r'\d', password):
                return False, "Пароль должен содержать цифры"

            salt, hashed = hash_password_simple(password)
            self._users[username] = {"salt": salt, "hash": hashed}
            return True, "Регистрация успешна"

        def login(self, username, password):
            """Вход с защитой от brute-force."""
            # Проверка на блокировку
            if self._login_attempts[username] >= self._max_attempts:
                return False, "Аккаунт заблокирован (слишком много попыток)"

            if username not in self._users:
                # Не раскрываем, существует ли пользователь
                time.sleep(0.1)  # Задержка для предотвращения timing attack
                return False, "Неверное имя пользователя или пароль"

            user = self._users[username]
            salt, hashed = hash_password_simple(password, user["salt"])

            if hashed == user["hash"]:
                self._login_attempts[username] = 0
                return True, "Вход выполнен успешно"
            else:
                self._login_attempts[username] += 1
                remaining = self._max_attempts - self._login_attempts[username]
                return False, f"Неверный пароль. Осталось попыток: {remaining}"

    def hash_password_simple(password, salt=None):
        """Упрощённое хэширование пароля."""
        if salt is None:
            salt = hashlib.sha256(str(random.random()).encode()).hexdigest()[:16]
        hashed = hashlib.sha256((salt + password).encode()).hexdigest()
        return salt, hashed

    auth = SecureAuth()

    # Регистрация
    success, msg = auth.register("user@example.com", "StrongP@ss1")
    print(f"  Регистрация: {msg}")

    # Успешный вход
    success, msg = auth.login("user@example.com", "StrongP@ss1")
    print(f"  Вход (верный пароль): {msg}")

    # Неверные попытки
    for i in range(4):
        success, msg = auth.login("user@example.com", "wrong_password")
        print(f"  Попытка {i+1}: {msg}")

    # Аккаунт заблокирован
    success, msg = auth.login("user@example.com", "StrongP@ss1")
    print(f"  Попытка после блокировки: {msg}")

    print("\n--- OWASP Top 10 краткий обзор ---")
    owasp = [
        ("A01:Broken Access Control", "Нарушение контроля доступа"),
        ("A02:Cryptographic Failures", "Ошибки шифрования"),
        ("A03:Injection", "Внедрение (SQL, NoSQL, OS, LDAP)"),
        ("A04:Insecure Design", "Небезопасный дизайн"),
        ("A05:Security Misconfiguration", "Неправильная настройка безопасности"),
        ("A06:Vulnerable Components", "Уязвимые компоненты"),
        ("A07:Auth Failures", "Ошибки аутентификации"),
        ("A08:Data Integrity Failures", "Нарушение целостности данных"),
        ("A09:Logging Failures", "Ошибки логирования"),
        ("A10:SSRF", "Серверное запросное подделывание (SSRF)"),
    ]
    for code, desc in owasp:
        print(f"  {code:<28s} — {desc}")


# ==========================================================================
# Демо 4 — Безопасное программирование: валидация, параметризованные запросы
# ==========================================================================
def demo_secure_coding():
    """Демонстрация принципов безопасного кодирования."""

    print("\n" + "=" * 70)
    print("Демо 4: Безопасное кодирование (Secure Coding)")
    print("=" * 70)

    # --- Подпример 1: Input Validation ---
    print("\n--- 4.1 Валидация входных данных ---")

    def validate_email(email):
        """
        Валидация email с регулярным выражением.
        Формат: local@domain.tld
        """
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))

    def validate_integer(value, min_val=None, max_val=None):
        """Валидация целого числа с проверкой диапазона."""
        try:
            num = int(value)
            if min_val is not None and num < min_val:
                return None, f"Число {num} меньше минимума {min_val}"
            if max_val is not None and num > max_val:
                return None, f"Число {num} больше максимума {max_val}"
            return num, "OK"
        except ValueError:
            return None, f"'{value}' не является целым числом"

    def validate_string(value, min_len=1, max_len=100, pattern=None):
        """Валидация строки: длина и формат."""
        if not isinstance(value, str):
            return False, "Ожидалась строка"
        if len(value) < min_len:
            return False, f"Строка слишком короткая ({len(value)} < {min_len})"
        if len(value) > max_len:
            return False, f"Строка слишком длинная ({len(value)} > {max_len})"
        if pattern and not re.match(pattern, value):
            return False, f"Строка не соответствует шаблону"
        return True, "OK"

    # Тестирование валидации
    test_emails = [
        "user@example.com",
        "invalid-email",
        "name@domain.co.uk",
        "",
        "user+tag@domain.com",
    ]
    print("  Валидация email:")
    for email in test_emails:
        valid = validate_email(email)
        status = "✓" if valid else "✗"
        print(f"    {status} '{email}'")

    # Валидация чисел
    print("\n  Валидация чисел (1-100):")
    test_numbers = ["42", "abc", "150", "0", "7"]
    for num in test_numbers:
        val, msg = validate_integer(num, 1, 100)
        print(f"    '{num}' → {val} ({msg})")

    # --- Подпример 2: Parameterized Queries ---
    print("\n--- 4.2 Параметризованные запросы ---")

    class SafeDatabase:
        """Имитация БД с параметризованными запросами."""
        def __init__(self):
            self._data = [
                {"id": 1, "name": "Алиса", "email": "alice@example.com", "role": "admin"},
                {"id": 2, "name": "Борис", "email": "boris@example.com", "role": "user"},
                {"id": 3, "name": "Вера", "email": "vera@example.com", "role": "user"},
            ]

        def query(self, sql_template, params):
            """
            Параметризованный запрос: SQL и данные разделены.
            Параметры экранируются автоматически.
            """
            print(f"    SQL шаблон: {sql_template}")
            print(f"    Параметры:  {params}")

            # Простая имитация: парсим WHERE условие
            results = self._data
            if "WHERE name = ?" in sql_template and len(params) > 0:
                results = [r for r in results if r["name"] == params[0]]
            elif "WHERE role = ?" in sql_template and len(params) > 0:
                results = [r for r in results if r["role"] == params[0]]
            elif "WHERE id > ? AND role = ?" in sql_template:
                results = [r for r in results if r["id"] > params[0] and r["role"] == params[1]]

            print(f"    Результат:  {len(results)} записей")
            return results

    db = SafeDatabase()

    # Безопасные запросы
    print("\n  Запрос пользователей по имени:")
    db.query("SELECT * FROM users WHERE name = ?", ("Алиса",))

    print("\n  Запрос администраторов:")
    db.query("SELECT * FROM users WHERE role = ?", ("admin",))

    print("\n  Составной запрос:")
    db.query("SELECT * FROM users WHERE id > ? AND role = ?", (1, "user"))

    # --- Подпример 3: Secrets Management ---
    print("\n--- 4.3 Управление секретами (Secrets Management) ---")

    class SecretsManager:
        """Менеджер секретов с шифрованием и контролем доступа."""
        def __init__(self, master_key):
            self._master_key = master_key
            self._secrets = {}
            self._access_log = []

        def _encrypt(self, data, key):
            """Шифрование данных мастер-ключом."""
            # Простой XOR для демонстрации (в реальности — AES-256-GCM)
            encrypted = []
            for i, byte in enumerate(data.encode()):
                encrypted.append(chr(byte ^ ord(key[i % len(key)])))
            return ''.join(encrypted)

        def _decrypt(self, data, key):
            """Дешифрование данных."""
            decrypted = []
            for i, char in enumerate(data):
                decrypted.append(chr(ord(char) ^ ord(key[i % len(key)])))
            return ''.join(decrypted)

        def store_secret(self, name, value, environment="prod"):
            """Сохранение секрета с метаданными."""
            encrypted = self._encrypt(value, self._master_key)
            self._secrets[name] = {
                "encrypted": encrypted,
                "environment": environment,
                "created": datetime.datetime.now().isoformat(),
                "access_count": 0,
            }
            self._access_log.append(f"STORE: {name}")
            print(f"  Секрет '{name}' сохранён (environment={environment})")

        def get_secret(self, name, requester="system"):
            """Получение секрета с логированием доступа."""
            if name not in self._secrets:
                return None
            secret = self._secrets[name]
            secret["access_count"] += 1
            decrypted = self._decrypt(secret["encrypted"], self._master_key)
            self._access_log.append(f"ACCESS: {name} by {requester}")
            print(f"  Секрет '{name}' получен (запросил: {requester})")
            return decrypted

        def rotate_secret(self, name, new_value):
            """Ротация секрета (замена значения)."""
            if name in self._secrets:
                old_encrypted = self._secrets[name]["encrypted"]
                self._secrets[name]["encrypted"] = self._encrypt(new_value, self._master_key)
                self._secrets[name]["created"] = datetime.datetime.now().isoformat()
                self._access_log.append(f"ROTATE: {name}")
                print(f"  Секрет '{name}' обновлён (ротация)")
                return True
            return False

        def audit_log(self):
            """Аудит-журнал доступа к секретам."""
            print(f"\n  Аудит-журнал ({len(self._access_log)} записей):")
            for entry in self._access_log[-5:]:
                print(f"    {entry}")

    # Демонстрация
    sm = SecretsManager("master_key_for_encryption")

    # Сохранение секретов
    sm.store_secret("db_password", "SuperSecret123!")
    sm.store_secret("api_key", "sk-1234567890abcdef")
    sm.store_secret("jwt_secret", "jwt_signing_key_xyz")

    # Получение секретов
    db_pass = sm.get_secret("db_password", requester="app_server")
    api_key = sm.get_secret("api_key", requester="worker_process")

    # Ротация секрета
    sm.rotate_secret("api_key", "sk-new_key_abcdef123456")

    # Аудит
    sm.audit_log()

    # --- Подпример 4: Security Headers ---
    print("\n--- 4.4 Security Headers (HTTP-заголовки безопасности) ---")

    security_headers = {
        "Content-Security-Policy": "default-src 'self'; script-src 'self'",
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    }

    print("  Рекомендуемые HTTP-заголовки безопасности:")
    for header, value in security_headers.items():
        print(f"    {header}:")
        print(f"      {value}")

    print("\n--- Принципы безопасного кодирования ---")
    principles = [
        "Не доверяй пользовательскому вводу — всегда валидируй",
        "Используй параметризованные запросы — никогда не конкатен SQL",
        "Экранируй вывод — предотвращай XSS",
        "Используй CSRF-токены — защищай от поддельных форм",
        "Хэшируй пароли с солью — никогда не храни открытым текстом",
        "Логируй аудит-события — отслеживай подозрительную активность",
        "Шифруй данные в покое и при передаче",
        "Регулярно обновляй зависимости — исправляй уязвимости",
    ]
    for p in principles:
        print(f"  • {p}")


# ==========================================================================
# Точка входа
# ==========================================================================
if __name__ == "__main__":
    print("УРОК 154: SECURITY FUNDAMENTALS")
    print("Темы: Authentication, Encryption, OWASP Top 10, Secure Coding")
    print("=" * 70)

    demo_authentication()
    demo_encryption()
    demo_owasp_top10()
    demo_secure_coding()

    print("\n" + "=" * 70)
    print("Урок завершён. Все 4 демо выполнены успешно.")
    print("=" * 70)
