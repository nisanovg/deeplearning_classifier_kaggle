Fashion‑MNIST Classifier

О проекте

Это репозиторий решения соревнования по классификации изображений одежды Fashion‑MNIST в рамках курса «Глубокое обучение в науках о данных». Цель — построить модель, правильно определяющую класс предмета одежды на изображении 28 × 28 px.

Цель соревнования

Метрика: Accuracy

Порог зачёта: ≥ 0.85 (для зачёта на курсе достаточно 0.80)

Мой результат: 0.90 на Public Leaderboard

Каждый участник может отправить не более 20 решений в сутки. Платформа принимает сабмиты только в формате, идентичном sample_submission.csv.

Датасет

Fashion‑MNIST — современный аналог MNIST, содержащий 70 000 grayscale‑изображений (28 × 28) десяти классов одежды.

Классы (label):
0 — T‑shirt/top • 1 — Trouser • 2 — Pullover • 3 — Dress • 4 — Coat •5 — Sandal • 6 — Shirt • 7 — Sneaker • 8 — Bag • 9 — Ankle boot

Структура данных

fmnist_train.csv   # обучающая выборка (label + id + 784 pixel_xx)
fmnist_test.csv    # тестовая выборка (id + 784 pixel_xx)
sample_submission.csv
notebooks/
  └── baseline_cnn.ipynb   # код решения (достигает accuracy 0.90)
src/
  ├── dataset.py           # Dataset & transforms
  ├── model.py             # Архитектура CNN
  ├── train.py             # Тренировочный скрипт
  └── predict.py           # Генерация submission.csv
requirements.txt

Быстрый старт

git clone https://github.com/<username>/fashion-mnist-classifier.git
cd fashion-mnist-classifier
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/baseline_cnn.ipynb

Первичный анализ данных

В notebooks/eda.ipynb:

проверка df.isnull().sum() — отсутствуют NaN в train и test;

визуализация распределения классов;

просмотр примеров изображений.

Архитектура модели

Conv2d (32 → 64 → 128) → BatchNorm → ReLU → MaxPool

Dropout 0.25

Fully Connected (128 → 10)

Softmax

Оптимизатор — Adam (lr = 1e‑3 с CosineAnnealing), функция потерь — CrossEntropyLoss.Аугментации: RandomHorizontalFlip, RandomCrop(28, padding=2), нормализация.

Обучение

python -m src.train \
    --epochs 20 \
    --batch-size 256 \
    --lr 1e-3 \
    --seed 42

Модель и логи сохраняются в checkpoints/. Для просмотра: tensorboard --logdir runs.

Инференс и сабмит

python -m src.predict \
    --checkpoint checkpoints/best.pth \
    --output submission.csv

Загрузите submission.csv на платформу. Ожидаемая accuracy ≈ 0.90.

Лицензия

MIT

