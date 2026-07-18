# AI Engineering from Scratch

Практический курс по математике для AI/ML — от линейной алгебры до ансамблей методов. Каждый урок: теория + рабочий код на Python без фреймворков.

> **[Конспект теории (CURS.md)](CURS.md)** — все формулы, таблицы и связи между уроками.

## О чём этот курс

Полный курс AI/ML от основ математики до компьютерного зрения. **60 уроков, 4 фазы, ~350 файлов кода.**

Чтобы понимать, как работают нейросети, PyTorch и LLM, нужно знать математику **под капотом**. Этот курс проходит Phase 1-4 курса [AI Engineering from Scratch](https://aiengineeringfromscratch.com/):

- **Phase 1: Math Foundations** — линейная алгебра, calculus, вероятности, оптимизация, Фурье
- **Phase 2: ML Fundamentals** — линейные модели, деревья, SVM, ансамбли, оценка моделей
- **Phase 3: Deep Learning Core** — нейросети с нуля: перцептрон, backprop, оптимизаторы, регуляризация
- **Phase 4: Vision** — CNN, свёртки, transfer learning, детекция и сегментация

Все алгоритмы реализованы **с нуля**: autograd движок, оптимизаторы, PCA, SVD, CNN — без импорта `torch` или `sklearn`. Только Python и понимание.

## Структура курса

### Phase 1: Math Foundations (уроки 01-20)

| # | Урок | Что изучаем |
|---|------|-------------|
| 01 | Линейная алгебра — интуиция | Векторы, dot product, проекция, ранг, Грамм-Шмидт |
| 02 | Векторы, матрицы и операции | Матричное умножение, broadcasting, слой нейросети |
| 03 | [Матричные преобразования](phase-01-math-foundations/03-matrix_transformations.py) | Поворот, масштаб, собственные числа, определитель |
| 04 | [Calculus для ML](phase-01-math-foundations/04-calculus_for_ml.py) | Производные, градиенты, гессиан, ряд Тейлора |
| 05 | [Chain Rule & Autograd](phase-01-math-foundations/05-chain_rule_autodiff.py) | Цепное правило, micrograd, gradient checking |
| 06 | [Вероятности и распределения](phase-01-math-foundations/06-probability_distributions.py) | Bernoulli, Normal, softmax, cross-entropy |
| 07 | [Bayes' Theorem](phase-01-math-foundations/07-bayes_theorem.py) | Наивный Байес, A/B тестирование, Beta-Binomial |
| 08 | [Оптимизация](phase-01-math-foundations/08-optimization.py) | GD, SGD, Momentum, Adam, расписания lr |
| 09 | [Теория информации](phase-01-math-foundations/09-information_theory.py) | Энтропия, KL-расхождение, взаимная информация |
| 10 | [Размерность](phase-01-math-foundations/10-dimensionality_reduction.py) | PCA, t-SNE, UMAP, Kernel PCA |
| 11 | [SVD](phase-01-math-foundations/11-svd.py) | Сжатие изображений, шумоподавление, рекомендации |
| 12 | [Tensor Operations](phase-01-math-foundations/12-tensor_operations.py) | Shape, broadcasting, einsum, multi-head attention |
| 13 | [Numerical Stability](phase-01-math-foundations/13-numerical_stability.py) | Overflow, stable softmax, gradient checking, mixed precision |
| 14 | [Нормы и расстояния](phase-01-math-foundations/14-norms_distances.py) | L1, L2, cosine, Mahalanобис, Jaccard, расстояние редактирования |
| 15 | [Статистика для ML](phase-01-math-foundations/15-statistics_for_ml.py) | Корреляция, t-test, bootstrap, Cohen's d, A/B тестирование |
| 16 | [Методы семплирования](phase-01-math-foundations/16-sampling_methods.py) | Inverse CDF, rejection, temperature, top-k, top-p, MCMC |
| 17 | [Линейные системы](phase-01-math-foundations/17-linear_systems.py) | Gaussian elimination, LU, Cholesky, least squares, ridge regression |
| 18 | [Выпуклая оптимизация](phase-01-math-foundations/18-convex_optimization.py) | Выпуклость, Ньютон, множители Лагранжа, KKT, L1/L2 регуляризация |
| 19 | [Комплексные числа](phase-01-math-foundations/19-complex_numbers.py) | Комплексная арифметика, Euler, DFT, RoPE в Transformer |
| 20 | [Преобразование Фурье](phase-01-math-foundations/20-fourier_transform.py) | DFT, FFT, спектральный анализ, свёртка, спектрограммы |

### Phase 2: ML Fundamentals (уроки 21-38)

| # | Урок | Что изучаем |
|---|------|-------------|
| 21 | [Что такое ML](phase-02-ml-fundamentals/21-ml_intro.py) | Типы ML, classification/regression, overfitting, bias-variance |
| 22 | [Linear Regression](phase-02-ml-fundamentals/22-linear_regression.py) | GD vs Normal Equation, polynomial, Ridge регуляризация |
| 23 | [Logistic Regression](phase-02-ml-fundamentals/23-logistic_regression.py) | Sigmoid, binary cross-entropy, softmax, метрики |
| 24 | [Decision Trees & Random Forests](phase-02-ml-fundamentals/24-decision_trees.py) | Gini, entropy, information gain, bootstrap, feature importance |
| 25 | [Support Vector Machines](phase-02-ml-fundamentals/25-support_vector_machines.py) | Hinge loss, margin, kernel trick, RBF |
| 26 | [KNN & Distance Metrics](phase-02-ml-fundamentals/26-knn.py) | K-Nearest Neighbors, метрики расстояния |
| 27 | [Unsupervised Learning](phase-02-ml-fundamentals/27-unsupervised_learning.py) | K-Means, DBSCAN, кластеризация |
| 28 | [Feature Engineering](phase-02-ml-fundamentals/28-feature_engineering.py) | Нормализация, one-hot, полиномиальные признаки |
| 29 | [Model Evaluation](phase-02-ml-fundamentals/29-model_evaluation.py) | Метрики, cross-validation, confusion matrix |
| 30 | [Bias-Variance](phase-02-ml-fundamentals/30-bias-variance.py) | Bias-variance tradeoff, learning curves |
| 31 | [Ensemble Methods](phase-02-ml-fundamentals/31-ensemble_methods.py) | Bagging, AdaBoost, Voting |
| 32 | [Hyperparameter Tuning](phase-02-ml-fundamentals/32-hyperparameter_tuning.py) | Grid Search, Random Search |
| 33 | [ML Pipelines](phase-02-ml-fundamentals/33-ml_pipelines.py) | Пайплайны, experiment tracking |
| 34 | [Naive Bayes](phase-02-ml-fundamentals/34-naive_bayes.py) | Gaussian/Multinomial Naive Bayes |
| 35 | [Time Series](phase-02-ml-fundamentals/35-time_series.py) | Скользящее среднее, экспоненциальное сглаживание |
| 36 | [Anomaly Detection](phase-02-ml-fundamentals/36-anomaly_detection.py) | Z-score, IQR, Isolation Forest |
| 37 | [Imbalanced Data](phase-02-ml-fundamentals/37-imbalanced_data.py) | Oversampling, undersampling, class weights |
| 38 | [Feature Selection](phase-02-ml-fundamentals/38-feature_selection.py) | Filter, wrapper, embedded methods |

### Phase 3: Deep Learning Core (уроки 39-51)

| # | Урок | Что изучаем |
|---|------|-------------|
| 39 | [The Perceptron](phase-03-deep-learning-core/39-perceptron.py) | Перцепptron, граница решения, XOR |
| 40 | [Multi-Layer Networks](phase-03-deep-learning-core/40-multi_layer_networks.py) | Neuron, Layer, MLP, активации |
| 41 | [Backpropagation](phase-03-deep-learning-core/41-backpropagation.py) | Value class, autodiff, gradient checking |
| 42 | [Activation Functions](phase-03-deep-learning-core/42-activation_functions.py) | Sigmoid, ReLU, GELU, затухание градиентов |
| 43 | [Loss Functions](phase-03-deep-learning-core/43-loss_functions.py) | MSE, Cross-Entropy, Huber, Contrastive |
| 44 | [Optimizers](phase-03-deep-learning-core/44-optimizers.py) | SGD, Momentum, RMSProp, Adam, AdamW |
| 45 | [Weight Initialization](phase-03-deep-learning-core/45-weight_initialization.py) | Random, Xavier, He |
| 46 | [Batch Normalization](phase-03-deep-learning-core/46-batch_normalization.py) | Нормализация, running stats |
| 47 | [Dropout & Regularization](phase-03-deep-learning-core/47-dropout_regularization.py) | Dropout, L1/L2, early stopping |
| 48 | [Learning Rate Schedules](phase-03-deep-learning-core/48-learning_rate_schedules.py) | Step, cosine, warmup |
| 49 | [Neural Network Framework](phase-03-deep-learning-core/49-neural_network_framework.py) | Полный фреймворк с нуля |
| 50 | [Training Loop](phase-03-deep-learning-core/50-training_loop.py) | Forward, backward, update, eval |
| 51 | [Debugging Neural Networks](phase-03-deep-learning-core/51-debugging_neural_networks.py) | Gradient checking, NaN, loss curves |

### Phase 4: Vision (уроки 52-60)

| # | Урок | Что изучаем |
|---|------|-------------|
| 52 | [Image Representations](phase-04-vision/52-image_representations.py) | Пиксели, фильтры, гистограммы |
| 53 | [Convolution](phase-04-vision/53-convolution.py) | 2D свёртка, ядра, padding, stride |
| 54 | [Pooling](phase-04-vision/54-pooling.py) | Max, average, global average pooling |
| 55 | [Building a CNN](phase-04-vision/55-building_cnn.py) | ConvLayer, PoolingLayer, CNN |
| 56 | [Classic Architectures](phase-04-vision/56-classic_architectures.py) | LeNet-5, ResNet, residual connections |
| 57 | [Transfer Learning](phase-04-vision/57-transfer_learning.py) | Feature extraction, fine-tuning |
| 58 | [Data Augmentation](phase-04-vision/58-data_augmentation.py) | Flip, rotation, noise, crop |
| 59 | [Object Detection](phase-04-vision/59-object_detection.py) | IoU, NMS, anchor boxes, mAP |
| 60 | [Image Segmentation](phase-04-vision/60-image_segmentation.py) | Thresholding, K-Means, connected components |

### Phase 5: NLP (уроки 61-68)

| # | Урок | Что изучаем |
|---|------|-------------|
| 61 | [Text Representations](phase-05-nlp/61-text_representations.py) | BoW, TF-IDF, N-grams |
| 62 | [Word Embeddings](phase-05-nlp/62-word_embeddings.py) | Word2Vec, косинусное сходство |
| 63 | [RNN](phase-05-nlp/63-rnn.py) | Рекуррентные сети, BPTT |
| 64 | [LSTM & GRU](phase-05-nlp/64-lstm_gru.py) | Вентили, затухание градиентов |
| 65 | [Seq2Seq](phase-05-nlp/65-seq2seq.py) | Encoder-Decoder, teacher forcing |
| 66 | [Attention](phase-05-nlp/66-attention.py) | Scaled dot-product, multi-head |
| 67 | [Transformer](phase-05-nlp/67-transformer.py) | Positional encoding, encoder-decoder |
| 68 | [Pre-trained Models](phase-05-nlp/68_pretrained_models.py) | BERT, GPT, fine-tuning |

### Phase 6: Speech & Audio (уроки 69-75)

| # | Урок | Что изучаем |
|---|------|-------------|
| 69 | [Audio Signal Processing](phase-06-speech-audio/69-audio_signal_processing.py) | FFT, фильтры, окнонание |
| 70 | [Spectrograms](phase-06-speech-audio/70-spectrograms.py) | STFT, мел-шкала, MFCC |
| 71 | [Speech Recognition](phase-06-speech-audio/71-speech_recognition.py) | DTW, HMM, распознавание команд |
| 72 | [Speaker Verification](phase-06-speech-audio/72-speaker_verification.py) | MFCC, cosine similarity, порог |
| 73 | [Text-to-Speech](phase-06-speech-audio/73-text_to_speech.py) | Формантный синтез |
| 74 | [Music Generation](phase-06-speech-audio/74-music_generation.py) | Марковские цепи, пентатоника |
| 75 | [Audio Augmentation](phase-06-speech-audio/75-audio_augmentation.py) | Шум, time stretching, pitch shift |

### Phase 7: Reinforcement Learning (уроки 76-82)

| # | Урок | Что изучаем |
|---|------|-------------|
| 76 | [Introduction to RL](phase-07-reinforcement-learning/76-introduction_rl.py) | Среда, агент, вознаграждение |
| 77 | [Multi-Armed Bandits](phase-07-reinforcement-learning/77-multi_armed_bandits.py) | Epsilon-greedy, UCB, Thompson |
| 78 | [Q-Learning](phase-07-reinforcement-learning/78-q_learning.py) | Q-таблица,贝尔曼 |
| 79 | [DQN](phase-07-reinforcement-learning/79-dqn.py) | Experience replay, target network |
| 80 | [Policy Gradient](phase-07-reinforcement-learning/80-policy_gradient.py) | REINFORCE, baseline |
| 81 | [Actor-Critic](phase-07-reinforcement-learning/81-actor_critic.py) | A2C, advantage |
| 82 | [RLHF](phase-07-reinforcement-learning/82-rlhf.py) | Reward model, PPO, KL penalty |

## Быстрый старт

```bash
git clone https://github.com/alekseysorokin68/AI-Engineering-with-pyton.git
cd AI-Engineering-with-pyton
python phase-01-math-foundations/08-optimization.py
```

Или через VS Code: открой папку → `phase-01-math-foundations/` → файл → **F5**.

## Конспект

Файл `CURS.md` — оглавление со ссылками на конспекты каждой фазы. В каждой папке фазы свой `CURS.md` с полным конспектом.

## Лицензия

MIT
