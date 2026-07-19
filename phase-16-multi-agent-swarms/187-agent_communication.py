"""
187 — Agent Communication: речевые акты, форматы сообщений, протоколы

Темы:
  1. Speech Acts — inform, request, propose, agree, refuse, confirm
  2. Message Formats — JSON-сообщения, онтологии, языки контента
  3. Conversation Protocols — request-inform, contract-net, auction
  4. Language Grounding — отображение символов на значения, общий словарь

Самодостаточный файл — не требует numpy, torch, transformers.
"""

import math
import random
import re
import json
import hashlib
import collections
import time
import itertools

random.seed(42)


# ─────────────────────────── Демо 1: Речевые акты (Speech Acts) ───────────────────────────

def demo_speech_acts():
    """Демонстрация типов речевых актов: inform, request, propose, agree, refuse, confirm."""
    print("=" * 70)
    print("ДЕМО 1: Speech Acts — речевые акты агентов")
    print("=" * 70)

    # 1.1 Классификация речевых актов по Searle
    print("Классификация речевых актов (Searle):")
    print("-" * 50)
    speech_acts = {
        "representatives": {
            "describe": "описание состояния мира",
            "inform": "сообщение факта",
            "assert": "утверждение истинности",
        },
        "directives": {
            "request": "просьба о действии",
            "command": "приказ",
            "ask": "вопрос",
        },
        "commissives": {
            "promise": "обещание",
            "commit": "обязательство",
            "guarantee": "гарантия",
        },
        "expressives": {
            "thank": "благодарность",
            "apologize": "извинение",
            "congratulate": "поздравление",
        },
        "declaratives": {
            "declare": "объявление",
            "appoint": "назначение",
            "resign": "отставка",
        },
    }
    for category, acts in speech_acts.items():
        print(f"\n  {category.upper()}:")
        for act, desc in acts.items():
            print(f"    {act:18s} — {desc}")

    # 1.2 Реализация речевых актов как сообщений
    print("\nРеализация речевых актов:")
    print("-" * 50)

    class SpeechAct:
        def __init__(self, act_type, sender, receiver, content):
            self.act_type = act_type
            self.sender = sender
            self.receiver = receiver
            self.content = content
            self.timestamp = time.time()

        def to_dict(self):
            return {
                "type": self.act_type,
                "sender": self.sender,
                "receiver": self.receiver,
                "content": self.content,
                "timestamp": self.timestamp,
            }

        def describe(self):
            descriptions = {
                "inform": f"{self.sender} сообщает {self.receiver}: {self.content}",
                "request": f"{self.sender} просит {self.receiver}: {self.content}",
                "propose": f"{self.sender} предлагает {self.receiver}: {self.content}",
                "agree": f"{self.sender} соглашается с {self.receiver}: {self.content}",
                "refuse": f"{self.sender} отказывает {self.receiver}: {self.content}",
                "confirm": f"{self.sender} подтверждает для {self.receiver}: {self.content}",
            }
            return descriptions.get(self.act_type, f"{self.act_type}: {self.content}")

    # Примеры речевых актов
    acts = [
        SpeechAct("inform", "Агент_A", "Агент_B", {"температура": 25}),
        SpeechAct("request", "Агент_A", "Агент_C", "выполнить_задачу_1"),
        SpeechAct("propose", "Агент_B", "Агент_A", {"цена": 100, "условие": "быстрая_доставка"}),
        SpeechAct("agree", "Агент_A", "Агент_B", "условия_контракта"),
        SpeechAct("refuse", "Агент_C", "Агент_A", "задача_невыполнима"),
        SpeechAct("confirm", "Агент_B", "Агент_A", "оплата_получена"),
    ]
    for act in acts:
        print(f"  {act.describe()}")

    # 1.3 Протокол завершения (completion protocols)
    print("\nПротоколы завершения:")
    print("-" * 50)

    # Success path: request → agree → inform → confirm
    print("  Успешный путь (success path):")
    success_path = ["request", "agree", "inform", "confirm"]
    for i, step in enumerate(success_path, 1):
        print(f"    Шаг {i}: {step}")

    # Failure path: request → refuse
    print("\n  Путь отказа (failure path):")
    failure_path = ["request", "refuse"]
    for i, step in enumerate(failure_path, 1):
        print(f"    Шаг {i}: {step}")

    # Timeout path: request → (нет ответа)
    print("\n  Путь таймаута (timeout path):")
    print("    Шаг 1: request → (нет ответа → таймаут)")

    # 1.4 Локуционные, иллокуционные, перлокуционные эффекты
    print("\nТри уровня речевого акта:")
    print("-" * 50)
    levels = {
        "локуционный": "произнесение предложения (форма)",
        "иллокуционный": "намерение говорящего (намерение)",
        "перлокуционный": "влияние на слушателя (эффект)",
    }
    example = '"Температура 25 градусов"'
    print(f"  Пример: {example}")
    for level, desc in levels.items():
        print(f"    {level:20s} — {desc}")
    print(f"    {'':20s}   локуционный: 'Температура 25 градусов'")
    print(f"    {'':20s}   иллокуционный: inform (сообщение факта)")
    print(f"    {'':20s}   перлокуционный: слушатель узнаёт температуру")

    print()


# ─────────────────────────── Демо 2: Форматы сообщений ───────────────────────────

def demo_message_formats():
    """Демонстрация JSON-сообщений, онтологий, языков контента."""
    print("=" * 70)
    print("ДЕМО 2: Message Formats — JSON-сообщения, онтологии, языки контента")
    print("=" * 70)

    # 2.1 JSON-формат сообщений
    print("JSON-формат сообщений:")
    print("-" * 50)

    # Простое сообщение
    simple_msg = {
        "from": "agent-1",
        "to": "agent-2",
        "type": "inform",
        "payload": {
            "action": "data_update",
            "values": {"sensor_id": 42, "reading": 36.5}
        },
        "metadata": {
            "timestamp": time.time(),
            "priority": "high",
            "retry_count": 0,
        }
    }
    print(json.dumps(simple_msg, indent=2, ensure_ascii=False))

    # 2.2 Онтологии — формальное описание домена
    print("\nОнтологии (Ontology) — формальное описание:")
    print("-" * 50)

    ontology = {
        "name": "SmartHome",
        "version": "1.0",
        "concepts": {
            "Device": {
                "properties": ["id", "name", "type", "state"],
                "relations": ["controls", "monitors", "connected_to"],
            },
            "Room": {
                "properties": ["id", "name", "area"],
                "relations": ["contains", "adjacent_to"],
            },
            "Action": {
                "properties": ["id", "name", "parameters"],
                "relations": ["performed_by", "acted_on"],
            },
        },
        "instances": {
            "Device_1": {"type": "thermostat", "room": "living_room"},
            "Device_2": {"type": "light", "room": "bedroom"},
            "Room_1": {"name": "living_room", "area": 25.5},
        }
    }
    for concept, details in ontology["concepts"].items():
        props = ", ".join(details["properties"])
        rels = ", ".join(details["relations"])
        print(f"  Концепт '{concept}':")
        print(f"    Свойства: {props}")
        print(f"    Связи: {rels}")

    # 2.3 Языки контента (Content Languages)
    print("\nЯзыки контента:")
    print("-" * 50)

    # SL — Simple Language
    sl_content = "(price widget 100.0)"
    print(f"  SL (Simple Language):  {sl_content}")

    # RDF-like
    rdf_content = {"subject": "widget", "predicate": "hasPrice", "object": "100.0"}
    print(f"  RDF-like:              {json.dumps(rdf_content)}")

    # KIF (Knowledge Interchange Format)
    kif_content = "(=> (instance ?x Widget) (hasPrice ?x 100.0))"
    print(f"  KIF:                   {kif_content}")

    # JSON-KIF гибрид
    json_kif = {"implies": [{"instance": ["?x", "Widget"]}, {"hasPrice": ["?x", "100.0"]}]}
    print(f"  JSON-KIF:              {json.dumps(json_kif)}")

    # 2.4 Сериализация и десериализация
    print("\nСериализация/десериализация:")
    print("-" * 50)

    class MessageSerializer:
        """Сериализатор сообщений в разных форматах."""

        @staticmethod
        def to_json(msg):
            return json.dumps(msg, ensure_ascii=False)

        @staticmethod
        def from_json(data):
            return json.loads(data)

        @staticmethod
        def to_sl(msg):
            """Преобразование в SL-формат."""
            parts = []
            for key, val in msg.items():
                if isinstance(val, dict):
                    val_str = " ".join(f"{k}:{v}" for k, v in val.items())
                    parts.append(f"({key} {val_str})")
                else:
                    parts.append(f"({key} {val})")
            return "(" + " ".join(parts) + ")"

    original = {"performative": "inform", "sender": "A", "content": {"temp": 25}}
    serializer = MessageSerializer()
    json_repr = serializer.to_json(original)
    sl_repr = serializer.to_sl(original)
    restored = serializer.from_json(json_repr)

    print(f"  Оригинал:     {original}")
    print(f"  JSON:          {json_repr[:80]}...")
    print(f"  SL:            {sl_repr[:80]}...")
    print(f"  Десериализация: {restored}")
    print(f"  Совпадает:     {original == restored}")

    print()


# ─────────────────────────── Демо 3: Протоколы разговоров ───────────────────────────

def demo_conversation_protocols():
    """Демонстрация протоколов request-inform, contract-net, auction."""
    print("=" * 70)
    print("ДЕМО 3: Conversation Protocols — request-inform, contract-net, auction")
    print("=" * 70)

    # 3.1 Протокол request-inform
    print("Протокол request-inform:")
    print("-" * 50)

    class RequestInformProtocol:
        def __init__(self, initiator, responder):
            self.initiator = initiator
            self.responder = responder
            self.log = []

        def execute(self, request_content, response_content):
            self.log.append((self.initiator, "request", request_content))
            self.log.append((self.responder, "agree", f"принято: {request_content}"))
            self.log.append((self.responder, "inform", response_content))
            self.log.append((self.initiator, "confirm", "получено"))
            return self.log

    protocol = RequestInformProtocol("Клиент", "Сервер")
    log = protocol.execute("get_weather", {"temp": 22, "condition": "sunny"})
    for sender, performative, content in log:
        print(f"  {sender:10s} → {performative:10s}: {content}")

    # 3.2 Протокол contract-net
    print("\nПротокол Contract-Net:")
    print("-" * 50)

    def contract_net_protocol(manager, contractors, task):
        """Протокол contract-net для распределения задач."""
        results = []
        # 1. Call for Proposal (CFP)
        results.append(f"1. {manager} → broadcast: CFP для '{task}'")

        # 2. Получение предложений
        proposals = {}
        for c in contractors:
            cost = random.randint(10, 100)
            time_est = random.randint(1, 10)
            proposals[c] = {"cost": cost, "time": time_est}
            results.append(f"2. {c} → {manager}: proposal(cost={cost}, time={time_est})")

        # 3. Оценка и выбор лучшего
        best = min(proposals.items(), key=lambda x: x[1]["cost"] + x[1]["time"])
        results.append(f"3. {manager} оценивает提案: лучший = {best[0]} (cost={best[1]['cost']}, time={best[1]['time']})")

        # 4. Отправка контракта
        results.append(f"4. {manager} → {best[0]}: award (контракт)")

        # 5. Отказ остальным
        for c in contractors:
            if c != best[0]:
                results.append(f"5. {manager} → {c}: reject (отказ)")

        return best[0], results

    winner, steps = contract_net_protocol("Менеджер", ["Подрядчик_1", "Подрядчик_2", "Подрядчик_3"], "строительство_моста")
    for step in steps:
        print(f"  {step}")
    print(f"  Итог: победитель = {winner}")

    # 3.3 Протокол аукциона
    print("\nПротокол аукциона (ascending/bidding):")
    print("-" * 50)

    def auction_protocol(item, participants, rounds=3):
        """Простой аукцион с повышением цены."""
        current_price = 10
        print(f"  Товар: {item}, начальная цена: {current_price}")
        active_bidders = list(participants)

        for rnd in range(1, rounds + 1):
            print(f"\n  Раунд {rnd}:")
            bids = {}
            for bidder in active_bidders:
                bid = current_price + random.randint(5, 20)
                bids[bidder] = bid
                print(f"    {bidder} делает ставку: {bid}")

            max_bid = max(bids.values())
            winners = [b for b, bid in bids.items() if bid == max_bid]
            current_price = max_bid
            print(f"  Текущая цена: {current_price}")

            # Участники с низкими ставками выбывают
            active_bidders = [b for b, bid in bids.items() if bid >= current_price - 5]
            if len(active_bidders) <= 1:
                break

        return current_price, active_bidders

    final_price, remaining = auction_protocol("Робот-манипулятор", ["Компания_A", "Компания_B", "Компания_C"])
    print(f"\n  Итог: цена = {final_price}, осталось участников: {remaining}")

    # 3.4 Протокол negotiation (торг)
    print("\nПротокол negotiation (торг):")
    print("-" * 50)

    def negotiation_protocol(buyer, seller, item, buyer_max, seller_min, rounds=5):
        """Протокол торговых переговоров."""
        buyer_offer = buyer_max * 0.8
        seller_offer = seller_min * 1.2
        log = []

        for rnd in range(1, rounds + 1):
            if buyer_offer >= seller_offer:
                deal_price = (buyer_offer + seller_offer) / 2
                log.append(f"  Раунд {rnd}: СДЕЛКА! Цена: {deal_price:.2f}")
                return deal_price, log

            # Покупатель повышает, продавец снижает
            buyer_step = (seller_offer - buyer_offer) * 0.3
            seller_step = (seller_offer - buyer_offer) * 0.4
            buyer_offer += buyer_step
            seller_offer -= seller_step

            log.append(f"  Раунд {rnd}: покупатель → {buyer_offer:.2f}, продавец → {seller_offer:.2f}")

        deal_price = (buyer_offer + seller_offer) / 2
        log.append(f"  Итог: компромиссная цена: {deal_price:.2f}")
        return deal_price, log

    deal, log = negotiation_protocol("Покупатель", "Продавец", "Датчик_LiDAR", buyer_max=500, seller_min=300)
    for entry in log:
        print(entry)

    print()


# ─────────────────────────── Демо 4: Заземление языка ───────────────────────────

def demo_language_grounding():
    """Демонстрация отображения символов на значения, общий словарь."""
    print("=" * 70)
    print("ДЕМО 4: Language Grounding — отображение символов на значения")
    print("=" * 70)

    # 4.1 Общий словарь агентов
    print("Общий словарь (Shared Vocabulary):")
    print("-" * 50)

    vocabulary = {
        "temperature": {"unit": "celsius", "range": (-50, 50), "type": "numeric"},
        "humidity": {"unit": "percent", "range": (0, 100), "type": "numeric"},
        "light_level": {"unit": "lux", "range": (0, 100000), "type": "numeric"},
        "motion": {"values": ["none", "detected", "fast"], "type": "enum"},
        "alarm": {"values": ["off", "warning", "critical"], "type": "enum"},
    }

    for symbol, meaning in vocabulary.items():
        print(f"  {symbol:15s} → {meaning}")

    # 4.2 Отображение символов на физические значения (grounding)
    print("\nGrounding — отображение на физические значения:")
    print("-" * 50)

    def ground_value(vocab, symbol, raw_value):
        """Преобразование сырых данных в семантически значимое значение."""
        if symbol not in vocab:
            return f"ОШИБКА: символ '{symbol}' не найден"
        entry = vocab[symbol]
        if entry["type"] == "numeric":
            low, high = entry["range"]
            if low <= raw_value <= high:
                normalized = (raw_value - low) / (high - low)
                if normalized < 0.2:
                    level = "очень_низкое"
                elif normalized < 0.4:
                    level = "низкое"
                elif normalized < 0.6:
                    level = "нормальное"
                elif normalized < 0.8:
                    level = "высокое"
                else:
                    level = "очень_высокое"
                return f"{raw_value} {entry['unit']} ({level}, норм.={normalized:.2f})"
            else:
                return f"ОШИБКА: {raw_value} вне диапазона {entry['range']}"
        elif entry["type"] == "enum":
            if raw_value in entry["values"]:
                return f"{raw_value} (валидное значение)"
            else:
                return f"ОШИБКА: '{raw_value}' не в {entry['values']}"
        return str(raw_value)

    # Тестирование grounding
    test_readings = [
        ("temperature", 22.5),
        ("temperature", -60),  # вне диапазона
        ("humidity", 45.0),
        ("motion", "detected"),
        ("alarm", "warning"),
        ("unknown_sensor", 100),
    ]
    for symbol, value in test_readings:
        grounded = ground_value(vocabulary, symbol, value)
        print(f"  {symbol:15s} = {str(value):8s} → {grounded}")

    # 4.3 Семантическое сходство между агентами
    print("\nСемантическое сходство словарей агентов:")
    print("-" * 50)

    vocab_agent_A = {"temp", "humidity", "pressure", "wind", "rain"}
    vocab_agent_B = {"temperature", "humidity", "barometric_pressure", "wind_speed", "precipitation"}
    vocab_agent_C = {"temp", "humidity", "pressure"}

    def jaccard_similarity(set_a, set_b):
        """Коэффициент Жаккара — |A∩B| / |A∪B|."""
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0

    pairs = [("A-B", vocab_agent_A, vocab_agent_B), ("A-C", vocab_agent_A, vocab_agent_C), ("B-C", vocab_agent_B, vocab_agent_C)]
    for name, va, vb in pairs:
        sim = jaccard_similarity(va, vb)
        print(f"  {name}: Jaccard = {sim:.3f}")
        print(f"    A: {va}")
        print(f"    B: {vb}")
        common = va & vb
        print(f"    Общие: {common}")

    # 4.4 Трансляция сообщений между агентами с разными словарями
    print("\nТрансляция сообщений между агентами:")
    print("-" * 50)

    class MessageTranslator:
        """Транслятор сообщений между словарями разных агентов."""
        def __init__(self, mapping):
            self.mapping = mapping  # {source_term: target_term}

        def translate(self, message):
            translated = {}
            for key, value in message.items():
                new_key = self.mapping.get(key, key)
                translated[new_key] = value
            return translated

    # Маппинг от словаря A к словарю B
    mapping_a_to_b = {
        "temp": "temperature",
        "wind": "wind_speed",
        "rain": "precipitation",
        "pressure": "barometric_pressure",
    }
    translator = MessageTranslator(mapping_a_to_b)

    original_msg = {"temp": 22.5, "humidity": 65, "wind": 12.3, "rain": 0.0}
    translated_msg = translator.translate(original_msg)

    print(f"  Оригинал (словарь A):    {original_msg}")
    print(f"  Трансляция (словарь B):  {translated_msg}")

    # Обратная трансляция
    mapping_b_to_a = {v: k for k, v in mapping_a_to_b.items()}
    translator_back = MessageTranslator(mapping_b_to_a)
    back_translated = translator_back.translate(translated_msg)
    print(f"  Обратная трансляция:     {back_translated}")
    print(f"  Циклическая трансляция:  {original_msg == back_translated}")

    print()


if __name__ == "__main__":
    demo_speech_acts()
    demo_message_formats()
    demo_conversation_protocols()
    demo_language_grounding()
