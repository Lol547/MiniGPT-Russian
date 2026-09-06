# MiniGPT-Russian - Генерация русского текста с нуля

> **Реализация GPT-подобной модели с нуля на PyTorch для генерации русского текста. Архитектура: Decoder-only Transformer (51 млн параметров) с позиционным кодированием и weight tying. Обучена на корпусе ru-text-corpus (потоковая загрузка). Поддерживает жадную, random, top-k, top-p и комбинированную стратегии сэмплирования.**

---

## Навигация
- [Особенности](#особенности)
- [Архитектура модели](#архитектура-модели)
- [Результаты](#результаты)
- [Примеры генерации](#примеры-генерации)
- [Установка](#установка)
- [Использование](#использование)
  - [Генерация текста](#генерация-текста)
  - [Обучение](#обучение)
- [Структура проекта](#структура-проекта)
- [Веса модели](#веса-модели)
---

## Особенности

- **GPT-подобный Transformer с нуля** - реализация Decoder-only архитектуры на PyTorch.
- **Потоковое обучение** - датасет загружается по частям, что позволяет работать с большими корпусами без переполнения памяти.
- **Пять стратегий генерации**:
  - **Greedy** - детерминированный выбор наиболее вероятного токена.
  - **Random Sampling** - сэмплирование с температурой.
  - **Top-K Sampling** - выбор из K наиболее вероятных токенов.
  - **Top-P (Nucleus) Sampling** - выбор из токенов с кумулятивной вероятностью p.
  - **Top-K + Top-P** - комбинированная стратегия.
- **Регуляризация** - Dropout, Weight Decay, Label Smoothing, Gradient Clipping.
- **Weight Tying** - веса эмбеддингов и классификатора связаны для экономии параметров.
- **Смешанная точность** - ускорение обучения на GPU.
- **Контекст** - 256 токенов


---

## Архитектура модели

Модель представляет собой **Decoder-only Transformer** (как в GPT):

| Компонент | Описание |
|-----------|----------|
| **Positional Embedding** | Позиционное кодирование с синусоидальными функциями (sequence_length=256, hidden_dim=512) |
| **Decoder Layer** | 8 слоёв, каждый с Masked Self-Attention (8 heads) и Feed-Forward (intermediate_dim=2048) |
| **Функция потерь** | CrossEntropyLoss с Label Smoothing (0.03) и ignore_index=PAD_ID |
| **Оптимизатор** | AdamW (betas=(0.9, 0.95), weight_decay=0.1) |
| **Планировщик** | NoamScheduler (warmup_steps=10000) |
| **Регуляризация** | Dropout 0.2, Weight Tying (shared embedding weights) |

**Количество параметров:** ~51.1 млн

---

## Результаты

Модель обучалась 150 000 шагов (15 эпох × 10 000 шагов) на потоковом датасете **PotatoHD/ru-text-corpus**. Лучший чекпоинт на шаге **120 000**:

| Метрика | Значение |
|---------|----------|
| **Train Loss** | 5.2709 |
| **Validation Loss** | **5.0679** |
| **Validation Token Accuracy** | **28.20%** |
| **Validation Perplexity** | **158.84** |

---

## Примеры генерации

**Промпт:** `"Рецепт очень простой"`

| Стратегия | Результат |
|-----------|-----------|
| **Greedy** | Рецепт очень простой, но не очень удобный. - Не нужно использовать его для того, чтобы он был в качестве основы... |
| **Random (T=0.8)** | Рецепт очень простой, но оченьчный, поэтому думаю, что это так многое спасибо. Я свойства воды были просто... |
| **Top-k (k=5, T=0.7)** | Рецепт очень простой и удобный для меня. В конце концов, я не могу сказать, что это очень важно... |
| **Top-p (p=0.9, T=0.7)** | Рецепт очень простой и быстро решить проблему, в которой мы не будем использовать эти дополнительные формы... |
| **Top-k + Top-p** | Рецепт очень простой и простой. Для тех, кто хочет стать одним из самых популярных представителей бизнеса... |

> Полные примеры генерации для разных промптов можно найти в ноутбуке `notebooks/mini_gpt.ipynb`.

---

## Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/Lol547/MiniGPT-Russian.git
cd MiniGPT-Russian
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

**Требования:**
- Python 3.8+
- PyTorch (с поддержкой CUDA, если есть GPU)
- transformers, datasets, torchvision, numpy, matplotlib, tqdm

---

## Использование

### Генерация текста

```python
from src.model import MiniGPT
from src.generate import generate_text, random_sample, top_k_sample
import torch
from transformers import AutoTokenizer

# 1. Загрузка токенизатора
tokenizer = AutoTokenizer.from_pretrained("ai-forever/rugpt3small_based_on_gpt2")
tokenizer.add_special_tokens({"pad_token": "[PAD]", "eos_token": "[EOS]", "bos_token": "[BOS]"})

# 2. Загрузка модели
checkpoint = torch.load("checkpoints/mini_gpt_final.pt", map_location="cuda")
model = MiniGPT(**checkpoint["config"]).to("cuda")
model.load_state_dict(checkpoint["model_state"])
model.eval()

# 3. Генерация текста
prompt = "Рецепт очень простой"
generated = generate_text(
    prompt, model, tokenizer,
    max_new_tokens=64,
    sample_fn=lambda logits: top_k_sample(logits, k=5, temperature=0.7),
    device="cuda"
)
print(generated)
```

### Командная строка

```bash
python src/generate.py --prompt "Рецепт очень простой" --method top_k --k 5 --temperature 0.7 --length 64
```

### Обучение

```bash
python src/train.py --epochs 15 --batch_size 16 --hidden_dim 512 --num_layers 8 --num_heads 8
```

Все гиперпараметры можно настроить через аргументы командной строки или изменить в конфигурационном файле.

---

## Структура проекта

```
mini-gpt-russian/
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── mini_gpt.ipynb                # Исходный ноутбук с экспериментами
│
├── src/
│   ├── __init__.py
│   ├── model.py                      # MiniGPT, PositionalEmbedding, TransformerDecoderLayer
│   ├── dataset.py                    # StreamingTextDataset, CachedValDataset
│   ├── generate.py                   # Функции генерации (greedy, random, top-k, top-p)
│   ├── train.py                      # Скрипт обучения
│   └── utils.py                      # NoamScheduler, EarlyStopping
│
├── configs/
│   └── default.yaml                  # Конфигурация гиперпараметров
│
├── checkpoints/                      # Папка для весов (в .gitignore)
│   ├── best_model.pt
│   └── tokenizer/
│
└── data/                             # Кэш датасета (в .gitignore)
```

---

## Веса модели

Для работы необходимы следующие файлы:
1. **`mini_gpt_final.pt`** - полный чекпоинт модели (веса + конфиг).
2. **Токенизатор** - папка с файлами токенизатора (можно сохранить через `tokenizer.save_pretrained()`).

**Ссылки для скачивания:**
[Веса и токенизатор](https://drive.google.com/drive/folders/1dTpc97mPnjc0aV177SXn5ujAwMuMSaHm?usp=sharing)

---
## Контакты

По вопросам или найденным ошибкам пишите: sokolovkirill489@gmail.com TG: @qqkiru
