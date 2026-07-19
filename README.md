# AI Engineering from Scratch

Практический курс по математике для AI/ML — от линейной алгебры до ансамблей методов. Каждый урок: теория + рабочий код на Python без фреймворков.

> **[Конспект теории (CURS.md)](CURS.md)** — все формулы, таблицы и связи между уроками.

## О чём этот курс

Полный курс AI/ML от основ математики до генеративного AI. **215 уроков, 17 фаз, ~1050 файлов кода.**

Чтобы понимать, как работают нейросети, PyTorch и LLM, нужно знать математику **под капотом**. Этот курс проходит Phase 1-10 курса [AI Engineering from Scratch](https://aiengineeringfromscratch.com/):

- **Phase 1: Math Foundations** — линейная алгебра, calculus, вероятности, оптимизация, Фурье
- **Phase 2: ML Fundamentals** — линейные модели, деревья, SVM, ансамбли, оценка моделей
- **Phase 3: Deep Learning Core** — нейросети с нуля: перцептрон, backprop, оптимизаторы, регуляризация
- **Phase 4: Vision** — CNN, свёртки, transfer learning, детекция и сегментация
- **Phase 5: NLP** — текстовые представления, word embeddings, RNN, LSTM, Transformer
- **Phase 6: Speech & Audio** — спектрограммы, распознавание речи, TTS, генерация музыки
- **Phase 7: Reinforcement Learning** — Q-learning, DQN, Policy Gradient, Actor-Critic, RLHF
- **Phase 8: Generative AI** — Autoencoders, VAE, GAN, Diffusion Models, Stable Diffusion
- **Phase 9: RL Advanced** — Model-Based, Monte Carlo, TD Learning, Multi-Agent RL
- **Phase 10: LLMs from Scratch** — Токенизация, Transformer, GPT, Fine-tuning, RAG, LoRA
- **Phase 11: LLM Engineering** — API Design, Serving, Security, Agents, Monitoring, Production
- **Phase 12: Multimodal AI** — Vision Transformers, CLIP, Diffusion, Video, Audio, 3D, Multimodal Agents
- **Phase 13: Tools & Protocols** — Python Packaging, Git, Docker, REST/GraphQL/gRPC, CI/CD, K8s, Security
- **Phase 14: Agent Engineering** — Architecture, Tools, Planning, Memory, Multi-Agent, Security, Deployment
- **Phase 15: Autonomous Systems** — Self-Learning, World Models, Swarm Intelligence, Continual Learning
- **Phase 16: Multi-Agent & Swarms** — Communication, Game Theory, MARL, LLM Agents, Production
- **Phase 17: Infrastructure & Production** — Distributed Systems, K8s, CI/CD, Monitoring, MLOps

Все алгоритмы реализованы **с нуля**: autograd движок, оптимизаторы, PCA, SVD, CNN, Transformer, RL-агенты — без импорта `torch` или `sklearn`. Только Python и понимание.

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

### Phase 8: Generative AI (уроки 83-90)

| # | Урок | Что изучаем |
|---|------|-------------|
| 83 | [Autoencoders](phase-08-generative-ai/83-autoencoders.py) | Encoder, Decoder, latent space |
| 84 | [VAE](phase-08-generative-ai/84-vae.py) | Reparameterization, ELBO |
| 85 | [GAN](phase-08-generative-ai/85-gan.py) | Generator, Discriminator, adversarial training |
| 86 | [GAN Stability](phase-08-generative-ai/86-gan_stability.py) | Label smoothing, WGAN, gradient penalty |
| 87 | [Diffusion Models](phase-08-generative-ai/87-diffusion_models.py) | Forward/reverse process, DDPM |
| 88 | [Denoising Diffusion](phase-08-generative-ai/88-denoising_diffusion.py) | UNet, DDIM, CFG |
| 89 | [Stable Diffusion](phase-08-generative-ai/89-stable_diffusion.py) | VAE + UNet + Text Encoder |
| 90 | [Text-to-Image](phase-08-generative-ai/90_text_to_image.py) | CLIP, prompt engineering |

### Phase 9: RL Advanced (уроки 91-95)

| # | Урок | Что изучаем |
|---|------|-------------|
| 91 | [Model-Based RL](phase-09-reinforcement-advanced/91-model_based_rl.py) | Динамика среды, Dyna-Q |
| 92 | [Monte Carlo](phase-09-reinforcement-advanced/92-monte_carlo.py) | MC evaluation, MC control |
| 93 | [TD Learning](phase-09-reinforcement-advanced/93-temporal_difference.py) | TD(0), TD(λ), SARSA |
| 94 | [SARSA vs Q-Learning](phase-09-reinforcement-advanced/94-sarsa_vs_qlearning.py) | On-policy vs off-policy |
| 95 | [Multi-Agent RL](phase-09-reinforcement-advanced/95_multi_agent_rl.py) | Cooperative, competitive, communication |

### Phase 10: LLMs from Scratch (уроки 96-110)

| # | Урок | Что изучаем |
|---|------|-------------|
| 96 | [Tokenization](phase-10-llms-from-scratch/96-tokenization.py) | BPE, WordPiece |
| 97 | [Embeddings](phase-10-llms-from-scratch/97-embeddings.py) | Token embeddings, positional encoding |
| 98 | [Self-Attention](phase-10-llms-from-scratch/98-self_attention.py) | Q, K, V, scaled dot-product |
| 99 | [Multi-Head Attention](phase-10-llms-from-scratch/99_multi_head_attention.py) | Heads, projection |
| 100 | [Transformer Block](phase-10-llms-from-scratch/100-transformer_block.py) | LayerNorm, residual, FFN |
| 101 | [GPT Architecture](phase-10-llms-from-scratch/101-gpt_architecture.py) | Decoder-only, causal mask |
| 102 | [Pre-training](phase-10-llms-from-scratch/102-pretraining.py) | Next token, masked LM |
| 103 | [Fine-tuning](phase-10-llms-from-scratch/103-fine_tuning.py) | SFT, instruction tuning |
| 104 | [Prompting](phase-10-llms-from-scratch/104-prompting.py) | Zero-shot, few-shot, CoT |
| 105 | [Chain-of-Thought](phase-10-llms-from-scratch/105-chain_of_thought.py) | CoT, self-consistency, ToT |
| 106 | [RAG](phase-10-llms-from-scratch/106-rag.py) | Retrieval, generation |
| 107 | [LoRA](phase-10-llms-from-scratch/107-lora.py) | Low-rank adaptation |
| 108 | [Quantization](phase-10-llms-from-scratch/108-quantization.py) | INT8, INT4 |
| 109 | [KV Cache](phase-10-llms-from-scratch/109-kv_cache.py) | Efficient inference |
| 110 | [LLM Evaluation](phase-10-llms-from-scratch/110-llm_evaluation.py) | Perplexity, BLEU, benchmarks |

### Phase 11: LLM Engineering (уроки 111-125)

| # | Урок | Что изучаем |
|---|------|-------------|
| 111 | [LLM API Design](phase-11-llm-engineering/111-llm_api_design.py) | System prompts, message formats, temperature, top-p |
| 112 | [Token Management](phase-11-llm-engineering/112-token_management.py) | Context windows, truncation, sliding window |
| 113 | [Prompt Engineering Advanced](phase-11-llm-engineering/113-prompt_engineering_advanced.py) | Structured output, JSON mode, prompt chaining |
| 114 | [LLM Serving](phase-11-llm-engineering/114-llm_serving.py) | Batching, streaming, load balancing, rate limiting |
| 115 | [Inference Optimization](phase-11-llm-engineering/115-inference_optimization.py) | Speculative decoding, continuous batching, paged attention |
| 116 | [LLM Security](phase-11-llm-engineering/116-llm_security.py) | Prompt injection, jailbreaking, guardrails |
| 117 | [Agents & Tool Use](phase-11-llm-engineering/117-agents_tool_use.py) | Function calling, ReAct, tool orchestration |
| 118 | [Multi-Modal LLMs](phase-11-llm-engineering/118-multimodal_llms.py) | Vision-language, audio-language, fusion |
| 119 | [LLM Memory](phase-11-llm-engineering/119-llm_memory.py) | Conversation history, long-term memory, retrieval |
| 120 | [Fine-tuning at Scale](phase-11-llm-engineering/120-finetuning_at_scale.py) | Data prep, distributed training, hyperparameters |
| 121 | [LLM Testing](phase-11-llm-engineering/121-llm_testing.py) | Red-teaming, adversarial testing, A/B testing |
| 122 | [LLM Deployment](phase-11-llm-engineering/122-llm_deployment.py) | Containerization, scaling, health checks |
| 123 | [LLM Monitoring](phase-11-llm-engineering/123-llm_monitoring.py) | Logging, metrics, drift detection, alerting |
| 124 | [Cost Optimization](phase-11-llm-engineering/124-cost_optimization.py) | Token costs, caching, model routing, distillation |
| 125 | [LLM Production](phase-11-llm-engineering/125-llm_production.py) | A/B testing, incident response, quality monitoring |

### Phase 12: Multimodal AI (уроки 126-140)

| # | Урок | Что изучаем |
|---|------|-------------|
| 126 | [Vision Transformers](phase-12-multimodal-ai/126-vision_transformers.py) | ViT, patch embeddings, DeiT, Swin |
| 127 | [CLIP & Alignment](phase-12-multimodal-ai/127-clip_alignment.py) | Contrastive learning, zero-shot classification |
| 128 | [Image Generation](phase-12-multimodal-ai/128-image_generation.py) | Autoregressive, VQ-VAE, discrete tokens |
| 129 | [Diffusion Advanced](phase-12-multimodal-ai/129-diffusion_advanced.py) | DDIM, classifier-free guidance, noise schedules |
| 130 | [Video Understanding](phase-12-multimodal-ai/130-video_understanding.py) | Temporal modeling, action recognition, video QA |
| 131 | [Audio & Speech](phase-12-multimodal-ai/131-audio_speech_models.py) | Whisper, audio tokens, speech-to-text |
| 132 | [Document Understanding](phase-12-multimodal-ai/132-document_understanding.py) | Layout analysis, OCR, table extraction |
| 133 | [3D Vision](phase-12-multimodal-ai/133-3d_vision.py) | Point clouds, depth estimation, PointNet |
| 134 | [Multimodal RAG](phase-12-multimodal-ai/134-multimodal_rag.py) | Cross-modal retrieval, hybrid indexes |
| 135 | [Multimodal Agents](phase-12-multimodal-ai/135-multimodal_agents.py) | Vision tool use, GUI automation, embodied AI |
| 136 | [Cross-Modal Transfer](phase-12-multimodal-ai/136-cross_modal_transfer.py) | Adapters, few-shot, knowledge distillation |
| 137 | [Multimodal Benchmarks](phase-12-multimodal-ai/137-multimodal_benchmarks.py) | VQA, captioning metrics, retrieval metrics |
| 138 | [Efficient Multimodal](phase-12-multimodal-ai/138-efficient_multimodal.py) | Token pruning, streaming, hardware optimization |
| 139 | [Multimodal Safety](phase-12-multimodal-ai/139-multimodal_safety.py) | Hallucination, bias, content filtering |
| 140 | [Building Multimodal Apps](phase-12-multimodal-ai/140-building_multimodal_apps.py) | Pipelines, API design, production deployment |

### Phase 13: Tools & Protocols (уроки 141-155)

| # | Урок | Что изучаем |
|---|------|-------------|
| 141 | [Python Packaging](phase-13-tools-protocols/141-python_packaging.py) | pip, pyproject.toml, wheel, venv |
| 142 | [Git Advanced](phase-13-tools-protocols/142-git_advanced.py) | Branching, merging, rebasing, hooks |
| 143 | [Docker](phase-13-tools-protocols/143-docker_containers.py) | Images, layers, multi-stage builds |
| 144 | [REST API](phase-13-tools-protocols/144-rest_api.py) | HTTP methods, JWT, OpenAPI |
| 145 | [GraphQL](phase-13-tools-protocols/145-graphql.py) | Schemas, resolvers, subscriptions |
| 146 | [gRPC & Protobuf](phase-13-tools-protocols/146-grpc_protobuf.py) | Protocol Buffers, streaming, interceptors |
| 147 | [Message Queues](phase-13-tools-protocols/147-message_queues.py) | RabbitMQ, Kafka, async patterns |
| 148 | [CI/CD Pipelines](phase-13-tools-protocols/148-cicd_pipelines.py) | GitHub Actions, testing, deployment |
| 149 | [Infrastructure as Code](phase-13-tools-protocols/149-infrastructure_as_code.py) | Terraform, declarative config, state |
| 150 | [Kubernetes](phase-13-tools-protocols/150-kubernetes_basics.py) | Pods, services, deployments, scaling |
| 151 | [Observability](phase-13-tools-protocols/151-observability.py) | Prometheus, Grafana, distributed tracing |
| 152 | [Databases](phase-13-tools-protocols/152-databases.py) | SQL, indexing, connection pooling, ORMs |
| 153 | [Caching](phase-13-tools-protocols/153-caching.py) | Redis, LRU/LFU, invalidation, consistent hashing |
| 154 | [Security](phase-13-tools-protocols/154-security_fundamentals.py) | Auth, encryption, OWASP, secure coding |
| 155 | [Cloud Computing](phase-13-tools-protocols/155-cloud_basics.py) | IaaS/PaaS/SaaS, serverless, CDN |

### Phase 14: Agent Engineering (уроки 156-170)

| # | Урок | Что изучаем |
|---|------|-------------|
| 156 | [Agent Architecture](phase-14-agent-engineering/156-agent_architecture.py) | Agent loop, state machine, lifecycle |
| 157 | [Tool Integration](phase-14-agent-engineering/157-tool_integration.py) | Function calling, tool registry, validation |
| 158 | [Planning & Reasoning](phase-14-agent-engineering/158-planning_reasoning.py) | ReAct, plan-and-execute, reflection |
| 159 | [Memory Systems](phase-14-agent-engineering/159-memory_systems.py) | Short/long-term, episodic, semantic memory |
| 160 | [Multi-Agent Systems](phase-14-agent-engineering/160-multi_agent_systems.py) | Communication, cooperation, conflict resolution |
| 161 | [Agent Evaluation](phase-14-agent-engineering/161-agent_evaluation.py) | Metrics, benchmarks, failure modes |
| 162 | [Prompt Engineering for Agents](phase-14-agent-engineering/162-prompt_engineering_agents.py) | System prompts, few-shot, dynamic prompting |
| 163 | [Agent Frameworks](phase-14-agent-engineering/163-agent_frameworks.py) | LangChain, AutoGPT, CrewAI concepts |
| 164 | [Autonomous Agents](phase-14-agent-engineering/164-autonomous_agents.py) | Goal-driven, self-reflection, error recovery |
| 165 | [Agent Security](phase-14-agent-engineering/165-agent_security.py) | Sandboxing, permissions, input validation |
| 166 | [Agent Deployment](phase-14-agent-engineering/166-agent_deployment.py) | Scaling, monitoring, cost management |
| 167 | [RAG for Agents](phase-14-agent-engineering/167-rag_for_agents.py) | Retrieval-augmented generation, knowledge bases |
| 168 | [Code Agents](phase-14-agent-engineering/168-code_agents.py) | Code generation, execution, debugging |
| 169 | [Browser Agents](phase-14-agent-engineering/169-browser_agents.py) | Web automation, navigation, form filling |
| 170 | [Agent Use Cases](phase-14-agent-engineering/170-agent_use_cases.py) | Support, research, data analysis, creative |

### Phase 15: Autonomous Systems (уроки 171-185)

| # | Урок | Что изучаем |
|---|------|-------------|
| 171 | [Autonomous Architecture](phase-15-autonomous-systems/171-autonomous_architecture.py) | Autonomy levels, decision architecture, state |
| 172 | [Self-Learning](phase-15-autonomous-systems/172-self_learning.py) | Online learning, experience replay, reward shaping |
| 173 | [Adaptive Behavior](phase-15-autonomous-systems/173-adaptive_behavior.py) | Behavior trees, utility systems, GOAP |
| 174 | [World Models](phase-15-autonomous-systems/174-world_models.py) | Predictive models, MCTS, mental simulation |
| 175 | [Self-Improvement](phase-15-autonomous-systems/175-self_improvement.py) | Meta-learning, prompt evolution, self-optimization |
| 176 | [Multi-Objectives](phase-15-autonomous-systems/176-multi_objectives.py) | Pareto optimality, constraint satisfaction, NSGA |
| 177 | [Resource-Aware](phase-15-autonomous-systems/177-resource_aware.py) | Compute budgets, priority scheduling, lazy eval |
| 178 | [Long-Horizon](phase-15-autonomous-systems/178-long_horizon.py) | Task chaining, checkpointing, failure recovery |
| 179 | [Human-AI Teaming](phase-15-autonomous-systems/179-human_ai_teaming.py) | Handoff protocols, trust calibration, escalation |
| 180 | [Autonomous Decisions](phase-15-autonomous-systems/180-autonomous_decisions.py) | Expected utility, Bayesian decisions, sequential |
| 181 | [Self-Monitoring](phase-15-autonomous-systems/181-self_monitoring.py) | Anomaly detection, calibration, diagnostics |
| 182 | [Swarm Intelligence](phase-15-autonomous-systems/182-swarm_intelligence.py) | ACO, PSO, flocking, emergent behavior |
| 183 | [Continual Learning](phase-15-autonomous-systems/183-continual_learning.py) | Forgetting, EWC, progressive networks, distillation |
| 184 | [Autonomous Debugging](phase-15-autonomous-systems/184-autonomous_debugging.py) | Root cause, fix generation, delta debugging |
| 185 | [Production Autonomy](phase-15-autonomous-systems/185-production_autonomy.py) | SLA, graceful degradation, circuit breaker |

### Phase 16: Multi-Agent & Swarms (уроки 186-200)

| # | Урок | Что изучаем |
|---|------|-------------|
| 186 | [Multi-Agent Fundamentals](phase-16-multi-agent-swarms/186-multi_agent_fundamentals.py) | Agent societies, FIPA ACL, coordination |
| 187 | [Agent Communication](phase-16-multi-agent-swarms/187-agent_communication.py) | Speech acts, protocols, grounding |
| 188 | [Cooperative Strategies](phase-16-multi-agent-swarms/188-cooperative_strategies.py) | Joint planning, Shapley value, coalitions |
| 189 | [Competitive Agents](phase-16-multi-agent-swarms/189-competitive_agents.py) | Game theory, Nash equilibrium, auctions |
| 190 | [Agent Organizations](phase-16-multi-agent-swarms/190-agent_organizations.py) | Hierarchies, roles, teams, learning |
| 191 | [Swarm Algorithms](phase-16-multi-agent-swarms/191-swarm_algorithms.py) | ACO, PSO, ABC, firefly |
| 192 | [Emergent Behavior](phase-16-multi-agent-swarms/192-emergent_behavior.py) | Self-organization, cellular automata, stigmergy |
| 193 | [Agent Negotiation](phase-16-multi-agent-swarms/193-agent_negotiation.py) | Bargaining, contract net, multi-issue |
| 194 | [MARL](phase-16-multi-agent-swarms/194-marl.py) | QMIX, self-play, mixed motivation |
| 195 | [LLM Multi-Agent](phase-16-multi-agent-swarms/195_llm_multi_agent.py) | Agent roles, debate, workflow orchestration |
| 196 | [Simulation & Testing](phase-16-multi-agent-swarms/196-simulation_testing.py) | ABM, scenario design, validation |
| 197 | [Communication Networks](phase-16-multi-agent-swarms/197-communication_networks.py) | Topologies, gossip, cascades, resilience |
| 198 | [Task Distribution](phase-16-multi-agent-swarms/198-task_distribution.py) | Allocation, load balancing, work stealing |
| 199 | [Scaling Agents](phase-16-multi-agent-swarms/199-scaling_agents.py) | Hierarchical control, abstraction, optimization |
| 200 | [Production Multi-Agent](phase-16-multi-agent-swarms/200-production_multi_agent.py) | Deployment, monitoring, debugging |

### Phase 17: Infrastructure & Production (уроки 201-215)

| # | Урок | Что изучаем |
|---|------|-------------|
| 201 | [Distributed Systems](phase-17-infrastructure-production/201-distributed_systems.py) | CAP theorem, consistency, Raft consensus |
| 202 | [Container Orchestration](phase-17-infrastructure-production/202-container_orchestration.py) | K8s advanced, operators, service mesh, GitOps |
| 203 | [CI/CD for ML](phase-17-infrastructure-production/203-cicd_ml.py) | ML pipelines, model versioning, experiment tracking |
| 204 | [Feature Stores](phase-17-infrastructure-production/204-feature_stores.py) | Feature engineering, serving, freshness, monitoring |
| 205 | [ML Monitoring](phase-17-infrastructure-production/205-ml_monitoring.py) | Data drift, model drift, performance, alerting |
| 206 | [A/B Testing for ML](phase-17-infrastructure-production/206-ab_testing_ml.py) | Statistical testing, bandits, online evaluation |
| 207 | [Model Registry](phase-17-infrastructure-production/207-model_registry.py) | Versioning, lineage, governance, approval |
| 208 | [Data Pipelines](phase-17-infrastructure-production/208-data_pipelines.py) | ETL, stream processing, data quality, orchestration |
| 209 | [Edge Deployment](phase-17-infrastructure-production/209-edge_deployment.py) | Pruning, quantization, ONNX, mobile inference |
| 210 | [GPU & Accelerators](phase-17-infrastructure-production/210-gpu_accelerators.py) | CUDA basics, mixed precision, tensor cores |
| 211 | [Cost Optimization](phase-17-infrastructure-production/211-cost_optimization.py) | Spot instances, auto-scaling, right-sizing |
| 212 | [Disaster Recovery](phase-17-infrastructure-production/212-disaster_recovery.py) | Backup, failover, chaos engineering, RTO/RPO |
| 213 | [Compliance & Governance](phase-17-infrastructure-production/213-compliance_governance.py) | GDPR, audit trails, explainability, model cards |
| 214 | [Team & Process](phase-17-infrastructure-production/214-team_process.py) | Agile for ML, MLOps maturity, team roles |
| 215 | [ML Platform](phase-17-infrastructure-production/215-ml_platform.py) | Architecture, tooling, adoption, evolution |

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
