import math
import random
from collections import defaultdict

random.seed(42)


# ============================================================
#  Формула Байеса
# ============================================================

def bayes(prior, likelihood, false_positive_rate):
    evidence = likelihood * prior + false_positive_rate * (1 - prior)
    posterior = likelihood * prior / evidence
    return posterior


# ============================================================
#  Наивный Байес — классификатор
# ============================================================

class NaiveBayes:
    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing
        self.class_counts = defaultdict(int)
        self.word_counts = defaultdict(lambda: defaultdict(int))
        self.class_word_totals = defaultdict(int)
        self.vocab = set()

    def train(self, documents, labels):
        for doc, label in zip(documents, labels):
            self.class_counts[label] += 1
            words = doc.lower().split()
            for word in words:
                self.word_counts[label][word] += 1
                self.class_word_totals[label] += 1
                self.vocab.add(word)

    def predict(self, document):
        words = document.lower().split()
        total_docs = sum(self.class_counts.values())
        vocab_size = len(self.vocab)
        best_class = None
        best_score = float("-inf")
        for cls in self.class_counts:
            score = math.log(self.class_counts[cls] / total_docs)
            for word in words:
                count = self.word_counts[cls].get(word, 0)
                total = self.class_word_totals[cls]
                score += math.log((count + self.smoothing) / (total + self.smoothing * vocab_size))
            if score > best_score:
                best_score = score
                best_class = cls
        return best_class

    def get_word_prob(self, cls, word):
        vocab_size = len(self.vocab)
        total = self.class_word_totals[cls]
        count = self.word_counts[cls].get(word, 0)
        return (count + self.smoothing) / (total + self.smoothing * vocab_size)


# ============================================================
#  Beta-Binomial (сопряжённое распределение)
# ============================================================

def beta_mean(a, b):
    return a / (a + b)

def beta_sample(a, b, n=1):
    samples = []
    for _ in range(n):
        # Наивный сэмплер через Gamma
        x = sum(-math.log(random.random()) for _ in range(a))
        y = sum(-math.log(random.random()) for _ in range(b))
        samples.append(x / (x + y))
    return samples if n > 1 else samples[0]


# ============================================================
#  Демо 1: Формула Байеса — медицинский тест
# ============================================================

print("=" * 55)
print("ДЕМО 1: Медицинский тест")
print("=" * 55)

p_sick = 0.0001
p_pos_given_sick = 0.99
p_pos_given_healthy = 0.01

posterior = bayes(p_sick, p_pos_given_sick, p_pos_given_healthy)
print(f"Болезнь: 1 из 10 000 → P(sick) = {p_sick}")
print(f"Тест точен на 99% → P(+|sick) = {p_pos_given_sick}")
print(f"Ложный позитив: 1% → P(+|healthy) = {p_pos_given_healthy}")
print(f"\nP(sick|+) = {posterior:.4f} = {posterior*100:.2f}%")
print(f"→ Менее 1%! Даже точный тест mostly ложные срабатывания.")

# Два теста подряд
posterior2 = bayes(posterior, p_pos_given_sick, p_pos_given_healthy)
print(f"\nПосле второго положительного теста:")
print(f"P(sick|+++) = {posterior2:.4f} = {posterior2*100:.2f}%")


# ============================================================
#  Демо 2: Спам-фильтр
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 2: Спам-фильтр (одно слово)")
print("=" * 55)

p_spam = 0.3
p_lottery_given_spam = 0.05
p_lottery_given_ham = 0.001

p_lottery = p_lottery_given_spam * p_spam + p_lottery_given_ham * (1 - p_spam)
p_spam_given_lottery = p_lottery_given_spam * p_spam / p_lottery

print(f"P(spam) = {p_spam}")
print(f"P('lottery'|spam) = {p_lottery_given_spam}")
print(f"P('lottery'|ham) = {p_lottery_given_ham}")
print(f"\nP(spam|'lottery') = {p_spam_given_lottery:.4f} = {p_spam_given_lottery*100:.1f}%")
print(f"→ Одно слово сдвигает вероятность с 30% до 95.5%!")


# ============================================================
#  Демо 3: Наивный Байес — обучение на спаме
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 3: Наивный Байес — классификатор спама")
print("=" * 55)

train_docs = [
    "win free money now",
    "free lottery ticket winner",
    "claim your prize today free",
    "urgent offer free cash",
    "congratulations you won free",
    "meeting tomorrow at noon",
    "project update attached",
    "can we schedule a call",
    "quarterly report review",
    "lunch on thursday sounds good",
    "team standup notes attached",
    "please review the pull request",
]

train_labels = [
    "spam", "spam", "spam", "spam", "spam",
    "ham", "ham", "ham", "ham", "ham", "ham", "ham",
]

classifier = NaiveBayes()
classifier.train(train_docs, train_labels)

test_messages = [
    "free money waiting for you",
    "meeting rescheduled to friday",
    "you won a free prize",
    "please review the attached report",
]

print("\nПредсказания:")
for msg in test_messages:
    pred = classifier.predict(msg)
    print(f"  '{msg}' -> {pred}")


# ============================================================
#  Демо 4: Топ слова для каждого класса
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 4: Самые вероятные слова")
print("=" * 55)

def show_top_words(classifier, cls, n=5):
    vocab_size = len(classifier.vocab)
    total = classifier.class_word_totals[cls]
    probs = {}
    for word in classifier.vocab:
        count = classifier.word_counts[cls].get(word, 0)
        probs[word] = (count + classifier.smoothing) / (total + classifier.smoothing * vocab_size)
    sorted_words = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    for word, prob in sorted_words[:n]:
        print(f"    {word}: {prob:.4f}")

print("Топ спам-слов:")
show_top_words(classifier, "spam")
print("\nТоп нормальных слов:")
show_top_words(classifier, "ham")


# ============================================================
#  Демо 5: Влияние сглаживания (smoothing)
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 5: Влияние smoothing на вероятности")
print("=" * 55)

for smooth in [0.01, 0.1, 1.0, 10.0]:
    clf = NaiveBayes(smoothing=smooth)
    clf.train(train_docs, train_labels)
    p_free_spam = clf.get_word_prob("spam", "free")
    p_meeting_spam = clf.get_word_prob("spam", "meeting")
    print(f"  smoothing={smooth:5.2f}  P('free'|spam)={p_free_spam:.4f}  P('meeting'|spam)={p_meeting_spam:.4f}")


# ============================================================
#  Демо 6: Beta-Binomial обновление
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 6: Байесово обновление (Beta-Binomial)")
print("=" * 55)

a, b = 1, 1  # Beta(1,1) = априори без знаний
print(f"Начало: Beta({a},{b}), среднее = {beta_mean(a,b):.3f}")

# День 1: 7 орлов, 3 решки
a, b = a + 7, b + 3
print(f"День 1 (7H, 3T): Beta({a},{b}), среднее = {beta_mean(a,b):.3f}")

# День 2: ещё 5 орлов, 5 решек
a, b = a + 5, b + 5
print(f"День 2 (5H, 5T): Beta({a},{b}), среднее = {beta_mean(a,b):.3f}")

# День 3: ещё 2 орла, 8 решек
a, b = a + 2, b + 8
print(f"День 3 (2H, 8T): Beta({a},{b}), среднее = {beta_mean(a,b):.3f}")


# ============================================================
#  Демо 7: A/B тестирование (Monte Carlo)
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 7: A/B тестирование")
print("=" * 55)

# Вариант A: 50 кликов / 1000 просмотров
# Вариант B: 65 кликов / 1000 просмотров
a_a, b_a = 1 + 50, 1 + 950   # Beta(51, 951)
a_b, b_b = 1 + 65, 1 + 935   # Beta(66, 936)

print(f"Вариант A: {50}/1000 кликов → Beta({a_a},{b_a}), среднее = {beta_mean(a_a,b_a):.4f}")
print(f"Вариант B: {65}/1000 кликов → Beta({a_b},{b_b}), среднее = {beta_mean(a_b,b_b):.4f}")

# Monte Carlo
n_samples = 100000
samples_a = beta_sample(a_a, b_a, n_samples)
samples_b = beta_sample(a_b, b_b, n_samples)
p_b_better = sum(1 for a, b in zip(samples_a, samples_b) if b > a) / n_samples

print(f"\nMonte Carlo ({n_samples} сэмплов):")
print(f"  P(B > A) = {p_b_better:.4f} = {p_b_better*100:.1f}%")

if p_b_better > 0.95:
    print(f"  → Запускаем вариант B! (> 95% уверенности)")
elif p_b_better < 0.05:
    print(f"  → Запускаем вариант A!")
else:
    print(f"  → Ещё рано решать, собирайте больше данных.")


# ============================================================
#  Демо 8: Множественные тесты
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 8: Два теста подряд")
print("=" * 55)

p_sick = 0.0001
p_pos = 0.99
p_false = 0.01

# Первый тест
p1 = bayes(p_sick, p_pos, p_false)
print(f"Тест 1: P(sick|+) = {p1:.6f} = {p1*100:.4f}%")

# Второй тест (постериор первого = априори второго)
p2 = bayes(p1, p_pos, p_false)
print(f"Тест 2: P(sick|++) = {p2:.6f} = {p2*100:.2f}%")

# Третий тест
p3 = bayes(p2, p_pos, p_false)
print(f"Тест 3: P(sick|+++) = {p3:.6f} = {p3*100:.1f}%")
