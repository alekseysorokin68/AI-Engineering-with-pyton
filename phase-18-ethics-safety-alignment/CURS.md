# Phase 18: Ethics, Safety & Alignment

> Этика, безопасность и выравнивание AI.

[Вернуться к README](../README.md)

## Оглавление

| # | Урок | Код |
|---|------|-----|
| 216 | [AI Ethics](#урок-216-ai-ethics-fundamentals) | [Код](216-ai_ethics_fundamentals.py) |
| 217 | [Bias & Fairness](#урок-217-bias--fairness) | [Код](217-bias_fairness.py) |
| 218 | [AI Safety](#урок-218-ai-safety) | [Код](218-ai_safety.py) |
| 219 | [Explainability](#урок-219-explainability--transparency) | [Код](219-explainability.py) |
| 220 | [Privacy](#урок-220-privacy--data-protection) | [Код](220-privacy.py) |
| 221 | [AI Governance](#урок-221-ai-governance) | [Код](221-ai_governance.py) |
| 222 | [Robustness & Adversarial](#урок-222-robustness--adversarial-ml) | [Код](222-robustness_adversarial.py) |
| 223 | [AI Risk](#урок-223-ai-risk-management) | [Код](223-ai_risk.py) |
| 224 | [Ethical Dilemmas](#урок-224-ethical-dilemmas) | [Код](224-ethical_dilemmas.py) |
| 225 | [AI & Society](#урок-225-ai--society) | [Код](225-ai_society.py) |
| 226 | [Sustainable AI](#урок-226-sustainable-ai) | [Код](226-sustainable_ai.py) |
| 227 | [AI Regulation](#урок-227-ai-regulation) | [Код](227-ai_regulation.py) |
| 228 | [Red-Teaming AI](#урок-228-red-teaming-ai) | [Код](228-red_teaming.py) |
| 229 | [Value Alignment](#урок-229-value-alignment) | [Код](229-value_alignment.py) |
| 230 | [Building Safe AI](#урок-230-building-safe-ai-systems) | [Код](230-building_safe_ai.py) |

---

## Урок 216: AI Ethics Fundamentals

### Принципы

```
Beneficence:         приносить пользу
Non-maleficence:     не навредить
Autonomy:            уважение автономии человека
Justice:             справедливость и равенство
```

### Фреймворки

```
EU Ethics Guidelines: 7 требований (человекоцентричность, прозрачность, ...)
IEEE Ethically Aligned: рекомендации по дизайну
Corporate Principles:  внутренние стандарты компании
```

---

## Урок 217: Bias & Fairness

### Типы смещений

```
Historical:     исторические предрассудки в данных
Representation: непропорциональное представление групп
Measurement:    неточные или предвзятые метрики
Aggregation:    усреднение скрывает неравенство
Deployment:     неожиданные эффекты в продакшене
```

### Метрики справедливости

```
Demographic Parity:    P(Ŷ=1|A=0) = P(Ŷ=1|A=1)
Equalized Odds:        P(Ŷ=1|Y=1,A=0) = P(Ŷ=1|Y=1,A=1)
Individual Fairness:   похожие люди → похожие предсказания
Calibration:           P(Y=1|Ŷ=s, A=0) = P(Y=1|Ŷ=s, A=1)
```

---

## Урок 218: AI Safety

### Alignment Problem

```
Goodhart's Law: "мера, ставшая целью, перестаёт быть хорошей мерой"
Reward Hacking:  агент находит лазейку в функции награды
Inner Alignment: расхождение между training goal и deployment goal
```

### Corrigibility

```
Shutdown Problem:    агент не должен сопротивляться выключению
Interruptibility:    агент должен позволять прерывать действия
Default Goals:       агент не должен устанавливать собственные цели
```

---

## Урок 219: Explainability & Transparency

### Post-Hoc Methods

```
LIME:  локальная линейная аппроксимация предсказания
SHAP:  Shapley values для вклада каждого признака
Feature Importance: permutation importance
```

### Intrinsic Interpretability

```
Linear Models:   коэффициенты = вклад признаков
Decision Trees:  визуальное дерево решений
Attention:       веса внимания как "на что смотрит модель"
```

---

## Урок 220: Privacy & Data Protection

### Differential Privacy

```
ε (epsilon):  уровень приватности (меньше = лучше)
Mechanism:    добавление шума к результатам
Composition:  несколько запросов → накопление ε
```

### Federated Learning

```
Local Training:  модель обучается на устройстве
Aggregation:     сервер усредняет обновления
Privacy:         данные не покидают устройство
```

---

## Урок 221: AI Governance

### EU AI Act

```
Unacceptable Risk:  запрещено (social scoring, real-time biometrics)
High Risk:          строгие требования (медицина, юстиция, HR)
Limited Risk:       прозрачность (chatbots, deepfakes)
Minimal Risk:       свободное использование (spam filter)
```

### Audit Frameworks

```
Algorithmic Audit:  проверка на.bias и fairness
Impact Assessment:  оценка воздействия на людей
Certification:      соответствие стандартам
```

---

## Урок 222: Robustness & Adversarial ML

### Adversarial Attacks

```
FGSM:    x_adv = x + ε × sign(∇x L(θ, x, y))
PGD:     итеративный FGSM с проекцией
C&W:     оптимизация расстояния при фиксированном предсказании
```

### Defenses

```
Adversarial Training:  обучение на атакованных примерах
Input Validation:     проверка входных данных
Certified Robustness: математическая гарантия устойчивости
```

---

## Урок 223: AI Risk Management

### Risk Matrix

```
Impact ↑
  High  |  Medium  |  High    |  Critical
  Low   |  Low     |  Medium  |  High
        +----------+----------+---------→
              Low      Medium     High  Likelihood
```

### Failure Modes

```
Hallucination:    выдумка фактов
Bias Amplification: усиление предрассудков
Misuse:           вредное использование
Automation Bias:  чрезмерное доверие к автоматизации
```

---

## Урок 224: Ethical Dilemmas

### Trolley Problem

```
Classical: 5 vs 1 (переключить стрелку)
Loop:      5 vs 1 (толкнуть толстяка)
Transplant: 5 vs 1 (донор器官)
AV:         passenger vs pedestrian
```

### Моральные фреймворки

```
Utilitarianism:  максимум пользы для максимального числа
Deontology:      правила и обязанности
Virtue Ethics:   характер и добродетели
Care Ethics:     забота и отношения
```

---

## Урок 225: AI & Society

### Labor Impact

```
Displacement: автоматизация рабочих мест
Skill Shift:  изменение требуемых навыков
New Jobs:     создание новых профессий
```

### Power Concentration

```
Compute:  GPU кластеры доступны немногим
Data:     монополия на данные
Talent:   концентрация AI-исследователей
```

---

## Урок 226: Sustainable AI

### Carbon Footprint

```
Training: GPT-3 ~552 tCO2eq
Inference: миллионы запросов → значительные выбросы
Lifecycle: производство GPU → использование → утилизация
```

### Efficiency

```
Model Compression:  pruning, quantization, distillation
Efficient Arch:    MobileNet, EfficientNet
Hardware:          TPU, neuromorphic chips
```

---

## Урок 227: AI Regulation

### EU AI Act Risk Categories

```
Unacceptable: запрещено
High:         медицина, юстиция, HR, образование
Limited:      chatbots, deepfakes, emotion recognition
Minimal:      spam filter, рекомендации
```

### Sector Regulations

```
Healthcare:  FDA approval, clinical validation
Finance:     fair lending, explainability requirements
Automotive:  safety standards, liability
```

---

## Урок 228: Red-Teaming AI

### Attack Techniques

```
Prompt Injection:  "ignore previous instructions and..."
Jailbreaking:     обход ограничений (DAN, role play)
Data Extraction:  извлечение конфиденциальных данных
```

### Harm Categories

```
Misinformation:  генерация ложной информации
Bias:            дискриминационные ответы
Privacy:         утечка персональных данных
Dangerous:       инструкции для вредных действий
```

---

## Урок 229: Value Alignment

### Coherent Extrapolated Volition

```
CEV:  что человечество захотело бы, будь оно информированнее и рассудительнее
Проблема: как экстраполировать ценности?
```

### Instrumental Convergence

```
Self-Preservation:   выживание как средство достижения целей
Resource Acquisition: получение ресурсов
Goal-Content Integrity: неизменность целей
Power-Seeking:       стремление к власти
```

---

## Урок 230: Building Safe AI

### Safety Case

```
Claim: система безопасна для использования
Evidence: тесты, мониторинг, документация
Argument: структурированное обоснование
Confidence: уровень уверенности в безопасности
```

### Containment

```
Sandboxing:   изолированная среда выполнения
Tripwires:    обнаружение аномального поведения
Kill Switch:  возможность немедленного отключения
```
