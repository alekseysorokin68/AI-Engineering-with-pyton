"""169 — Browser Agents: автоматизация браузера, навигация, заполнение форм

Темы:
  1. Web Interaction — HTTP-запросы, парсинг HTML, отправка форм
  2. Navigation — переходы по ссылкам, управление URL, обработка редиректов
  3. Content Extraction — обход DOM, извлечение текста, структурированные данные
  4. Browser Automation Patterns — управление сессиями, куки, ограничение частоты

Самодостаточный файл — не требует numpy, torch, transformers.
"""

import math
import random
import re
import json
import hashlib
import collections
import time

random.seed(42)


# ===========================================================================
# Демо 1: Web Interaction — HTTP-запросы, парсинг HTML, отправка форм
# ===========================================================================
def demo_web_interaction():
    print("=" * 70)
    print("ДЕМО 1: Web Interaction — HTTP-запросы и парсинг")
    print("=" * 70)

    # --- 1.1 Моделирование HTTP-запросов ---
    print("\n--- 1.1 Моделирование HTTP-запросов ---")

    class HTTPRequest:
        """Модель HTTP-запроса."""

        def __init__(self, method, url, headers=None, body=None):
            self.method = method
            self.url = url
            self.headers = headers or {}
            self.body = body
            self.timestamp = time.time()

        def to_dict(self):
            """Преобразует запрос в словарь."""
            return {
                "method": self.method,
                "url": self.url,
                "headers": self.headers,
                "body": self.body,
                "timestamp": self.timestamp,
            }

        def __repr__(self):
            return f"HTTPRequest({self.method} {self.url})"

    class HTTPResponse:
        """Модель HTTP-ответа."""

        def __init__(self, status_code, body, headers=None):
            self.status_code = status_code
            self.body = body
            self.headers = headers or {}
            self.ok = 200 <= status_code < 300

        def json(self):
            """Парсит JSON из тела ответа."""
            return json.loads(self.body)

    # Создаём запросы разных типов
    requests = [
        HTTPRequest("GET", "https://api.example.com/users"),
        HTTPRequest("POST", "https://api.example.com/login",
                     headers={"Content-Type": "application/json"},
                     body='{"username": "admin", "password": "***"}'),
        HTTPRequest("GET", "https://api.example.com/products?page=2"),
        HTTPRequest("DELETE", "https://api.example.com/users/42"),
    ]

    print("Созданные HTTP-запросы:")
    for req in requests:
        print(f"  {req}")

    # Группировка по методу
    method_counts = collections.Counter(r.method for r in requests)
    print(f"По методам: {dict(method_counts)}")

    # --- 1.2 Парсинг HTML (регулярные выражения) ---
    print("\n--- 1.2 Парсинг HTML-документов ---")

    class SimpleHTMLParser:
        """Простой HTML-парсер на регулярных выражениях."""

        def __init__(self, html):
            self.html = html

        def find_tags(self, tag_name):
            """Находит все вхождения тега."""
            pattern = rf"<{tag_name}[^>]*>(.*?)</{tag_name}>"
            return re.findall(pattern, self.html, re.DOTALL)

        def find_links(self):
            """Извлекает все ссылки."""
            pattern = r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>'
            return re.findall(pattern, self.html, re.DOTALL)

        def find_images(self):
            """Извлекает все изображения."""
            pattern = r'<img\s+[^>]*src=["\']([^"\']+)["\'][^>]*/?>'
            return re.findall(pattern, self.html)

        def find_inputs(self):
            """Извлекает все поля ввода формы."""
            pattern = r'<input\s+[^>]*name=["\']([^"\']+)["\'][^>]*/?>'
            return re.findall(pattern, self.html)

        def get_text(self):
            """Удаляет все теги и возвращает текст."""
            text = re.sub(r"<[^>]+>", " ", self.html)
            text = re.sub(r"\s+", " ", text)
            return text.strip()

    # Пример HTML-документа
    html_doc = """
    <html>
    <head><title>Тестовая страница</title></head>
    <body>
        <h1>Добро пожаловать</h1>
        <p>Это тестовый документ для парсинга.</p>
        <a href="/about">О нас</a>
        <a href="/contacts">Контакты</a>
        <a href="https://example.com">Внешняя ссылка</a>
        <img src="/logo.png" alt="Логотип"/>
        <img src="/banner.jpg" alt="Баннер"/>
        <form>
            <input name="username" type="text"/>
            <input name="password" type="password"/>
            <input name="email" type="email"/>
        </form>
        <ul>
            <li>Пункт 1</li>
            <li>Пункт 2</li>
            <li>Пункт 3</li>
        </ul>
    </body>
    </html>
    """

    parser = SimpleHTMLParser(html_doc)
    print(f"Заголовок страницы: {parser.find_tags('title')}")
    print(f"Найдено ссылок: {len(parser.find_links())}")
    for href, text in parser.find_links():
        print(f"  {href} -> {text.strip()}")
    print(f"Найдено изображений: {len(parser.find_images())}")
    print(f"Поля формы: {parser.find_inputs()}")
    print(f"Текст страницы: {parser.get_text()[:80]}...")

    # --- 1.3 Отправка форм ---
    print("\n--- 1.3 Моделирование отправки форм ---")

    class FormHandler:
        """Обработчик HTML-форм."""

        def __init__(self):
            self.forms = []

        def parse_form(self, html, form_index=0):
            """Парсит форму из HTML."""
            # Находим все формы
            form_pattern = r"<form[^>]*>(.*?)</form>"
            forms = re.findall(form_pattern, html, re.DOTALL)

            if form_index >= len(forms):
                return None

            form_html = forms[form_index]
            # Извлекаем action и method
            action_match = re.search(r'action=["\']([^"\']*)["\']', html)
            method_match = re.search(r'method=["\']([^"\']*)["\']', html)

            action = action_match.group(1) if action_match else "/"
            method = method_match.group(1).upper() if method_match else "GET"

            # Извлекаем поля
            inputs = parser.find_inputs() if hasattr(parser, 'find_inputs') else []
            fields = []
            input_pattern = r'<input\s+([^>]*)/?>'
            for attrs in re.findall(input_pattern, form_html):
                name_match = re.search(r'name=["\']([^"\']+)["\']', attrs)
                type_match = re.search(r'type=["\']([^"\']+)["\']', attrs)
                if name_match:
                    fields.append({
                        "name": name_match.group(1),
                        "type": type_match.group(1) if type_match else "text",
                    })

            return {"action": action, "method": method, "fields": fields}

        def submit_form(self, action, method, data):
            """Моделирует отправку формы."""
            if method == "GET":
                query = "&".join(f"{k}={v}" for k, v in data.items())
                url = f"{action}?{query}"
            else:
                url = action
            return {
                "url": url,
                "method": method,
                "data": data,
                "status": "sent",
            }

    handler = FormHandler()
    form_data = handler.parse_form(html_doc)
    if form_data:
        print(f"Action: {form_data['action']}")
        print(f"Method: {form_data['method']}")
        print(f"Поля: {[f['name'] for f in form_data['fields']]}")

        # Заполняем и отправляем форму
        submit_data = {"username": "user123", "password": "secret", "email": "test@example.com"}
        result = handler.submit_form(form_data["action"], form_data["method"], submit_data)
        print(f"Результат отправки: {result}")

    # --- 1.4 Симуляция ответов ---
    print("\n--- 1.4 Симуляция HTTP-ответов ---")

    class MockHTTPServer:
        """Мок HTTP-сервера для тестирования."""

        def __init__(self):
            self.routes = {}
            self.request_log = []

        def add_route(self, method, path, handler):
            """Добавляет маршрут."""
            self.routes[(method, path)] = handler

        def handle(self, method, path, body=None):
            """Обрабатывает запрос."""
            self.request_log.append({"method": method, "path": path, "body": body})
            key = (method, path)
            if key in self.routes:
                return self.routes[key](body)
            return HTTPResponse(404, '{"error": "Not Found"}')

    server = MockHTTPServer()
    # Регистрируем обработчики
    server.add_route("GET", "/users", lambda b: HTTPResponse(200, '{"users": ["Alice", "Bob"]}'))
    server.add_route("POST", "/users", lambda b: HTTPResponse(201, '{"status": "created"}'))

    # Обрабатываем запросы
    resp1 = server.handle("GET", "/users")
    print(f"GET /users -> {resp1.status_code}: {resp1.body}")

    resp2 = server.handle("POST", "/users", '{"name": "Charlie"}')
    print(f"POST /users -> {resp2.status_code}: {resp2.body}")

    resp3 = server.handle("GET", "/nonexistent")
    print(f"GET /nonexistent -> {resp3.status_code}: {resp3.body}")

    print(f"Всего запросов в логе: {len(server.request_log)}")


# ===========================================================================
# Демо 2: Navigation — переходы, URL, редиректы
# ===========================================================================
def demo_navigation():
    print("\n" + "=" * 70)
    print("ДЕМО 2: Navigation — навигация по сайтам")
    print("=" * 70)

    # --- 2.1 Управление URL ---
    print("\n--- 2.1 Управление URL ---")

    class URLManager:
        """Менеджер URL-адресов для навигации."""

        def __init__(self, base_url):
            self.base_url = base_url.rstrip("/")
            self.history = []
            self.current = base_url

        def navigate(self, path):
            """Переходит по относительному пути."""
            # Сохраняем текущий URL в историю
            self.history.append(self.current)
            # Формируем новый URL
            if path.startswith("http"):
                self.current = path
            elif path.startswith("/"):
                self.current = self.base_url + path
            else:
                # Относительный путь
                base = self.current.rsplit("/", 1)[0]
                self.current = f"{base}/{path}"
            return self.current

        def go_back(self):
            """Возвращает на предыдущую страницу."""
            if self.history:
                self.current = self.history.pop()
            return self.current

        def get_relative(self, url):
            """Вычисляет относительный путь между URL."""
            if url.startswith(self.base_url):
                return url[len(self.base_url):]
            return url

    manager = URLManager("https://example.com")
    print(f"Начальный URL: {manager.current}")

    # Навигация
    paths = ["/products", "/products/42", "reviews", "/cart"]
    for path in paths:
        result = manager.navigate(path)
        print(f"  Перешли по '{path}' -> {result}")

    # Возврат назад
    for _ in range(3):
        back = manager.go_back()
        print(f"  Назад -> {back}")

    # --- 2.2 Обработка редиректов ---
    print("\n--- 2.2 Обработка редиректов ---")

    class RedirectHandler:
        """Обработчик HTTP-редиректов."""

        def __init__(self, max_redirects=5):
            self.max_redirects = max_redirects
            self.redirect_chain = []

        def process_redirects(self, url, redirect_map):
            """Обрабатывает цепочку редиректов."""
            current = url
            self.redirect_chain = [current]

            for _ in range(self.max_redirects):
                if current in redirect_map:
                    next_url = redirect_map[current]
                    self.redirect_chain.append(next_url)
                    current = next_url
                else:
                    break

            return {
                "final_url": current,
                "chain_length": len(self.redirect_chain) - 1,
                "chain": self.redirect_chain,
            }

    redirect_map = {
        "http://example.com": "https://example.com",
        "https://example.com": "https://www.example.com",
        "https://www.example.com/login": "https://www.example.com/auth/login",
    }

    handler = RedirectHandler()
    result = handler.process_redirects("http://example.com", redirect_map)
    print(f"Цепочка редиректов ({result['chain_length']} переходов):")
    for i, url in enumerate(result["chain"]):
        if i < len(result["chain"]) - 1:
            print(f"  {url} ->")
        else:
            print(f"  {url} (финальный)")

    # --- 2.3 Граф навигации ---
    print("\n--- 2.3 Граф навигации (сайт-карта) ---")

    class SiteGraph:
        """Граф навигации сайта."""

        def __init__(self):
            self.edges = collections.defaultdict(list)

        def add_link(self, from_url, to_url):
            """Добавляет ссылку между страницами."""
            self.edges[from_url].append(to_url)

        def get_links(self, url):
            """Возвращает все ссылки со страницы."""
            return self.edges.get(url, [])

        def bfs(self, start, max_depth=3):
            """Обход в ширину (имитация краулера)."""
            visited = set()
            queue = [(start, 0)]
            result = []

            while queue:
                url, depth = queue.pop(0)
                if url in visited or depth > max_depth:
                    continue
                visited.add(url)
                result.append({"url": url, "depth": depth})
                for link in self.get_links(url):
                    if link not in visited:
                        queue.append((link, depth + 1))

            return result

    graph = SiteGraph()
    # Строим граф сайта
    links = [
        ("/", "/about"), ("/", "/products"), ("/", "/contacts"),
        ("/products", "/products/1"), ("/products", "/products/2"),
        ("/products/1", "/products/1/reviews"),
        ("/about", "/about/team"),
    ]
    for from_url, to_url in links:
        graph.add_link(from_url, to_url)

    # BFS обход
    crawl_result = graph.bfs("/", max_depth=2)
    print(f"Обход сайта от '/' (макс. глубина 2):")
    for item in crawl_result:
        indent = "  " * (item["depth"] + 1)
        print(f"{indent}{item['url']} (глубина {item['depth']})")

    # --- 2.4 Rate limiting ---
    print("\n--- 2.4 Ограничение частоты запросов ---")

    class RateLimiter:
        """Ограничитель частоты запросов."""

        def __init__(self, max_requests, window_seconds):
            self.max_requests = max_requests
            self.window = window_seconds
            self.timestamps = []

        def allow_request(self):
            """Проверяет, можно ли сделать запрос."""
            now = time.time()
            # Удаляем старые записи
            self.timestamps = [t for t in self.timestamps if now - t < self.window]
            if len(self.timestamps) < self.max_requests:
                self.timestamps.append(now)
                return True
            return False

        def wait_time(self):
            """Возвращает время ожидания до следующего запроса."""
            if not self.timestamps:
                return 0
            oldest = self.timestamps[0]
            return max(0, self.window - (time.time() - oldest))

    limiter = RateLimiter(max_requests=3, window_seconds=1.0)

    # Пытаемся сделать 5 быстрых запросов
    print("Тест rate limiter (макс. 3 запроса в секунду):")
    for i in range(5):
        allowed = limiter.allow_request()
        status = "РАЗРЕШЕНО" if allowed else "ОГРАНИЧЕНО"
        print(f"  Запрос {i + 1}: {status}")
    print(f"  Записей в буфере: {len(limiter.timestamps)}")


# ===========================================================================
# Демо 3: Content Extraction — обход DOM, текст, структурированные данные
# ===========================================================================
def demo_content_extraction():
    print("\n" + "=" * 70)
    print("ДЕМО 3: Content Extraction — извлечение данных из страниц")
    print("=" * 70)

    # --- 3.1 Обход DOM ---
    print("\n--- 3.1 Обход DOM-дерева ---")

    class DOMNode:
        """Узел DOM-дерева."""

        def __init__(self, tag, attributes=None, text=None):
            self.tag = tag
            self.attributes = attributes or {}
            self.text = text
            self.children = []

        def add_child(self, child):
            """Добавляет дочерний узел."""
            self.children.append(child)
            return child

        def find_by_tag(self, tag_name):
            """Находит все узлы по тегу."""
            results = []
            if self.tag == tag_name:
                results.append(self)
            for child in self.children:
                results.extend(child.find_by_tag(tag_name))
            return results

        def find_by_class(self, class_name):
            """Находит все узлы по классу."""
            results = []
            if class_name in self.attributes.get("class", "").split():
                results.append(self)
            for child in self.children:
                results.extend(child.find_by_class(class_name))
            return results

        def get_text_recursive(self):
            """Рекурсивно собирает весь текст."""
            parts = []
            if self.text:
                parts.append(self.text)
            for child in self.children:
                parts.append(child.get_text_recursive())
            return " ".join(parts)

    # Строим DOM-дерево
    root = DOMNode("html")
    body = root.add_child(DOMNode("body"))
    h1 = body.add_child(DOMNode("h1", {"class": "title"}, "Заголовок статьи"))
    article = body.add_child(DOMNode("div", {"class": "article", "id": "main"}))
    p1 = article.add_child(DOMNode("p", {"class": "text"}, "Первый абзац текста."))
    p2 = article.add_child(DOMNode("p", {"class": "text"}, "Второй абзац текста."))
    sidebar = body.add_child(DOMNode("aside", {"class": "sidebar"}, "Боковая панель"))

    # Поиск по тегам
    paragraphs = root.find_by_tag("p")
    print(f"Найдено тегов <p>: {len(paragraphs)}")
    for p in paragraphs:
        print(f"  Текст: '{p.text}'")

    # Поиск по классу
    articles = root.find_by_class("article")
    print(f"Найдено элементов с классом 'article': {len(articles)}")
    for a in articles:
        print(f"  ID: {a.attributes.get('id', 'нет')}")

    # --- 3.2 Извлечение текста ---
    print("\n--- 3.2 Извлечение и очистка текста ---")

    class TextExtractor:
        """Извлекает и очищает текст из HTML."""

        @staticmethod
        def strip_tags(html):
            """Удаляет все HTML-теги."""
            return re.sub(r"<[^>]+>", "", html)

        @staticmethod
        def extract_emails(text):
            """Извлекает email-адреса."""
            pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
            return re.findall(pattern, text)

        @staticmethod
        def extract_phones(text):
            """Извлекает телефоны."""
            pattern = r"\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}"
            return re.findall(pattern, text)

        @staticmethod
        def extract_urls(text):
            """Извлекает URL."""
            pattern = r"https?://[^\s<>\"']+"
            return re.findall(pattern, text)

        @staticmethod
        def summarize(text, max_words=20):
            """Создаёт краткую выжимку текста."""
            words = text.split()
            if len(words) <= max_words:
                return text
            return " ".join(words[:max_words]) + "..."

    extractor = TextExtractor()
    sample_html = """
    <div class="content">
        <h2>Контакты</h2>
        <p>Свяжитесь с нами: info@example.com или support@test.org</p>
        <p>Телефон: +7 (495) 123-45-67</p>
        <p>Сайт: https://example.com и https://test.org</p>
        <p>Длинный текст о компании, которая занимается разработкой 
        искусственного интеллекта и машинного обучения для решения 
        сложных бизнес-задач в различных отраслях экономики.</p>
    </div>
    """

    clean_text = extractor.strip_tags(sample_html)
    print(f"Очищенный текст: {clean_text[:100]}...")
    print(f"Email-адреса: {extractor.extract_emails(clean_text)}")
    print(f"Телефоны: {extractor.extract_phones(clean_text)}")
    print(f"URL: {extractor.extract_urls(clean_text)}")
    print(f"Саммари: {extractor.summarize(clean_text, max_words=10)}")

    # --- 3.3 Структурированные данные ---
    print("\n--- 3.3 Извлечение структурированных данных ---")

    class StructuredExtractor:
        """Извлекает структурированные данные (таблицы, списки)."""

        @staticmethod
        def extract_table(html):
            """Извлекает таблицу из HTML."""
            rows = []
            row_pattern = r"<tr[^>]*>(.*?)</tr>"
            cell_pattern = r"<t[dh][^>]*>(.*?)</t[dh]>"

            for row_match in re.finditer(row_pattern, html, re.DOTALL):
                row_html = row_match.group(1)
                cells = [re.sub(r"<[^>]+>", "", c).strip()
                         for c in re.findall(cell_pattern, row_html, re.DOTALL)]
                if cells:
                    rows.append(cells)
            return rows

        @staticmethod
        def extract_list(html):
            """Извлекает ненумерованный список."""
            item_pattern = r"<li[^>]*>(.*?)</li>"
            return [re.sub(r"<[^>]+>", "", item).strip()
                    for item in re.findall(item_pattern, html, re.DOTALL)]

        @staticmethod
        def extract_metadata(html):
            """Извлекает мета-данные страницы."""
            meta = {}
            title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.DOTALL)
            if title_match:
                meta["title"] = title_match.group(1).strip()

            desc_match = re.search(r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']*)["\']', html, re.IGNORECASE)
            if desc_match:
                meta["description"] = desc_match.group(1)

            # Извлекаем все мета-теги
            meta_pattern = r'<meta[^>]*name=["\']([^"\']+)["\'][^>]*content=["\']([^"\']*)["\']'
            for name, content in re.findall(meta_pattern, html, re.IGNORECASE):
                meta[name] = content
            return meta

    struct_extractor = StructuredExtractor()

    # Пример таблицы
    table_html = """
    <table>
        <tr><th>Имя</th><th>Возраст</th><th>Город</th></tr>
        <tr><td>Алиса</td><td>25</td><td>Москва</td></tr>
        <tr><td>Борис</td><td>30</td><td>СПб</td></tr>
        <tr><td>Вера</td><td>22</td><td>Казань</td></tr>
    </table>
    """
    table_data = struct_extractor.extract_table(table_html)
    print("Извлечённая таблица:")
    for row in table_data:
        print(f"  {' | '.join(row)}")

    # Пример списка
    list_html = "<ul><li>Python</li><li>JavaScript</li><li>Rust</li><li>Go</li></ul>"
    items = struct_extractor.extract_list(list_html)
    print(f"Извлечённый список: {items}")

    # Мета-данные
    meta_html = """
    <html><head>
        <title>Мой сайт</title>
        <meta name="description" content="Описание сайта"/>
        <meta name="keywords" content="AI, машинное обучение"/>
    </head></html>
    """
    metadata = struct_extractor.extract_metadata(meta_html)
    print(f"Мета-данные: {metadata}")

    # --- 3.4 Валидация извлечённых данных ---
    print("\n--- 3.4 Валидация извлечённых данных ---")

    class DataValidator:
        """Валидатор извлечённых данных."""

        @staticmethod
        def validate_email(email):
            """Проверяет формат email."""
            pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
            return bool(re.match(pattern, email))

        @staticmethod
        def validate_url(url):
            """Проверяет формат URL."""
            pattern = r"^https?://[^\s<>\"']+$"
            return bool(re.match(pattern, url))

        @staticmethod
        def validate_phone(phone):
            """Проверяет формат телефона."""
            digits = re.sub(r"\D", "", phone)
            return 10 <= len(digits) <= 15

    validator = DataValidator()

    # Тест валидации
    test_emails = ["valid@example.com", "invalid@", "user@domain.co", "@no-local.com"]
    print("Валидация email:")
    for email in test_emails:
        status = "OK" if validator.validate_email(email) else "ОШИБКА"
        print(f"  {email}: {status}")

    test_urls = ["https://example.com", "ftp://files.com", "not a url", "http://test.org/path"]
    print("Валидация URL:")
    for url in test_urls:
        status = "OK" if validator.validate_url(url) else "ОШИБКА"
        print(f"  {url}: {status}")


# ===========================================================================
# Демо 4: Browser Automation Patterns — сессии, куки, rate limiting
# ===========================================================================
def demo_browser_automation():
    print("\n" + "=" * 70)
    print("ДЕМО 4: Browser Automation Patterns — паттерны автоматизации")
    print("=" * 70)

    # --- 4.1 Управление сессиями ---
    print("\n--- 4.1 Управление сессиями ---")

    class SessionManager:
        """Менеджер веб-сессий."""

        def __init__(self):
            self.sessions = {}
            self.active = None

        def create_session(self, session_id):
            """Создаёт новую сессию."""
            self.sessions[session_id] = {
                "id": session_id,
                "cookies": {},
                "headers": {},
                "created_at": time.time(),
                "requests_count": 0,
            }
            return self.sessions[session_id]

        def activate(self, session_id):
            """Активирует сессию."""
            if session_id in self.sessions:
                self.active = session_id
                return True
            return False

        def add_cookie(self, name, value, path="/"):
            """Добавляет cookie в активную сессию."""
            if self.active and self.active in self.sessions:
                self.sessions[self.active]["cookies"][name] = {
                    "value": value,
                    "path": path,
                }

        def get_cookies(self):
            """Возвращает cookie активной сессии."""
            if self.active and self.active in self.sessions:
                return self.sessions[self.active]["cookies"]
            return {}

        def make_request(self):
            """Имитирует запрос с cookie сессии."""
            if self.active:
                self.sessions[self.active]["requests_count"] += 1
                return {
                    "session": self.active,
                    "cookies": list(self.get_cookies().keys()),
                    "request_num": self.sessions[self.active]["requests_count"],
                }
            return None

        def destroy_session(self, session_id):
            """Уничтожает сессию."""
            if session_id in self.sessions:
                del self.sessions[session_id]
                if self.active == session_id:
                    self.active = None

    sm = SessionManager()
    sm.create_session("sess_001")
    sm.create_session("sess_002")
    sm.activate("sess_001")

    # Добавляем cookie
    sm.add_cookie("session_id", "abc123")
    sm.add_cookie("user_pref", "dark_mode")

    print(f"Активная сессия: {sm.active}")
    print(f"Cookie: {sm.get_cookies()}")

    # Делаем запросы
    for i in range(3):
        resp = sm.make_request()
        print(f"Запрос {resp['request_num']}: cookies={resp['cookies']}")

    # Переключаем сессию
    sm.activate("sess_002")
    sm.add_cookie("session_id", "xyz789")
    print(f"\nПосле переключения: {sm.active}")
    print(f"Cookie: {sm.get_cookies()}")

    # --- 4.2 Cookie-менеджер ---
    print("\n--- 4.2 Cookie-менеджер ---")

    class CookieJar:
        """Хранилище cookies с поддержкой TTL."""

        def __init__(self):
            self.cookies = {}

        def set(self, name, value, max_age=None):
            """Устанавливает cookie с опциональным TTL."""
            self.cookies[name] = {
                "value": value,
                "created": time.time(),
                "max_age": max_age,
            }

        def get(self, name):
            """Получает cookie, проверяя срок жизни."""
            if name not in self.cookies:
                return None
            cookie = self.cookies[name]
            if cookie["max_age"] is not None:
                age = time.time() - cookie["created"]
                if age > cookie["max_age"]:
                    del self.cookies[name]
                    return None
            return cookie["value"]

        def delete(self, name):
            """Удаляет cookie."""
            if name in self.cookies:
                del self.cookies[name]

        def all_names(self):
            """Возвращает имена всех активных cookie."""
            result = []
            for name in list(self.cookies.keys()):
                if self.get(name) is not None:
                    result.append(name)
            return result

        def clear_expired(self):
            """Очищает протухшие cookie."""
            expired = []
            for name in self.cookies:
                if self.get(name) is None:
                    expired.append(name)
            for name in expired:
                self.cookies.pop(name, None)
            return len(expired)

    jar = CookieJar()
    jar.set("session", "abc123", max_age=3600)
    jar.set("pref", "dark", max_age=None)
    jar.set("temp", "data", max_age=0)  # Протухшая cookie

    print(f"Все cookie: {jar.all_names()}")
    print(f"session = {jar.get('session')}")
    print(f"temp = {jar.get('temp')}")  # Должна быть None
    cleaned = jar.clear_expired()
    print(f"Очищено протухших: {cleaned}")
    print(f"Оставшиеся cookie: {jar.all_names()}")

    # --- 4.3 Стратегии краулинга ---
    print("\n--- 4.3 Стратегии краулинга ---")

    class CrawlStrategy:
        """Разные стратегии обхода сайтов."""

        @staticmethod
        def breadth_first(graph, start):
            """Обход в ширину (BFS)."""
            visited = []
            queue = [start]
            seen = {start}
            while queue:
                node = queue.pop(0)
                visited.append(node)
                for neighbor in graph.get(node, []):
                    if neighbor not in seen:
                        seen.add(neighbor)
                        queue.append(neighbor)
            return visited

        @staticmethod
        def depth_first(graph, start):
            """Обход в глубину (DFS)."""
            visited = []
            stack = [start]
            seen = {start}
            while stack:
                node = stack.pop()
                visited.append(node)
                for neighbor in graph.get(node, []):
                    if neighbor not in seen:
                        seen.add(neighbor)
                        stack.append(neighbor)
            return visited

        @staticmethod
        def random_walk(graph, start, steps=5):
            """Случайный обход графа."""
            visited = [start]
            current = start
            for _ in range(steps):
                neighbors = graph.get(current, [])
                if neighbors:
                    current = random.choice(neighbors)
                    visited.append(current)
            return visited

    site_graph = {
        "/": ["/about", "/products", "/blog"],
        "/about": ["/about/team", "/about/history"],
        "/products": ["/products/1", "/products/2"],
        "/blog": ["/blog/post1", "/blog/post2"],
        "/products/1": ["/products/1/reviews"],
        "/blog/post1": ["/blog/post2"],
    }

    print("BFS обход от '/':")
    bfs_result = CrawlStrategy.breadth_first(site_graph, "/")
    print(f"  Порядок: {' -> '.join(bfs_result)}")

    print("DFS обход от '/':")
    dfs_result = CrawlStrategy.depth_first(site_graph, "/")
    print(f"  Порядок: {' -> '.join(dfs_result)}")

    print("Случайный обход от '/' (5 шагов):")
    random.seed(42)
    rw_result = CrawlStrategy.random_walk(site_graph, "/", steps=5)
    print(f"  Путь: {' -> '.join(rw_result)}")

    # --- 4.4 Сбор данных ---
    print("\n--- 4.4 Паттерн сбора данных ---")

    class DataCollector:
        """Сборщик данных с нескольких страниц."""

        def __init__(self):
            self.collected = []
            self.errors = []

        def collect_page(self, url, parser_func):
            """Собирает данные со страницы."""
            try:
                # Имитируем загрузку страницы
                data = parser_func(url)
                self.collected.append({"url": url, "data": data, "status": "ok"})
                return data
            except Exception as e:
                self.errors.append({"url": url, "error": str(e)})
                return None

        def summary(self):
            """Возвращает сводку сбора."""
            return {
                "total_pages": len(self.collected) + len(self.errors),
                "successful": len(self.collected),
                "failed": len(self.errors),
                "success_rate": (len(self.collected) / max(1, len(self.collected) + len(self.errors))) * 100,
            }

    # Пример парсера
    def mock_parser(url):
        """Мок-парсер, извлекающий данные."""
        pages_data = {
            "/page1": {"title": "Страница 1", "items": 5},
            "/page2": {"title": "Страница 2", "items": 8},
            "/page3": {"title": "Страница 3", "items": 3},
        }
        return pages_data.get(url, {"title": "Неизвестно", "items": 0})

    collector = DataCollector()
    pages = ["/page1", "/page2", "/page3", "/page4"]

    for page in pages:
        collector.collect_page(page, mock_parser)

    summary = collector.summary()
    print(f"Сбор данных завершён:")
    print(f"  Всего страниц: {summary['total_pages']}")
    print(f"  Успешно: {summary['successful']}")
    print(f"  Ошибок: {summary['failed']}")
    print(f"  Процент успеха: {summary['success_rate']:.1f}%")

    print("\nСобранные данные:")
    for item in collector.collected:
        print(f"  {item['url']}: {item['data']}")


# ===========================================================================
# Запуск всех демонстраций
# ===========================================================================
if __name__ == "__main__":
    print("╔" + "═" * 68 + "╗")
    print("║  169 — Browser Agents: автоматизация браузера, навигация          ║")
    print("╚" + "═" * 68 + "╝")
    print()

    demo_web_interaction()
    demo_navigation()
    demo_content_extraction()
    demo_browser_automation()

    print("\n" + "=" * 70)
    print("Все демонстрации завершены")
    print("=" * 70)
