Fashion‑MNIST Classifier

# Построение классификатора Fashion-MNIST

## О проекте
Это соревнование в рамках курса **«Глубокое обучение в науках о данных»**. Ваша цель — построить модель, способную классифицировать изображения предметов одежды из датасета Fashion-MNIST.

## Данные
- **fmnist_train.csv** — обучающая выборка (786 столбцов: `label`, 784 пикселя, `id`).
- **fmnist_test.csv** — тестовая выборка (то же, кроме столбца `label`).
- **sample_submission.csv** — пример формата отправки.

**Fashion-MNIST** — аналог классического MNIST, но с 10 классами одежды:

- 0 — T-shirt/top (футболка)
- 1 — Trouser (брюки)
- 2 — Pullover (свитер)
- 3 — Dress (платье)
- 4 — Coat (пальто)
- 5 — Sandal (сандалия)
- 6 — Shirt (рубашка)
- 7 — Sneaker (кроссовок)
- 8 — Bag (сумка)
- 9 — Ankle boot (сапог)

Каждое изображение хранится в виде 784 пикселей (28×28), значения от 0 до 255 в чёрно-белой шкале.

## Задача
1. Провести первичный анализ (EDA), проверить данные на наличие NaN.  
2. Построить классификатор и получить предсказания для тестовой выборки.  
3. Отправить решение в формате `sample_submission.csv`.

## Критерии успеха
- Метрика: **accuracy**.  
- Public Leaderboard ≥ 0.85 для зачёта; минимум 0.80.  
- Не более 20 отправок в день.

## Выбранный способ реализации
- **Среда и библиотеки**  
  Python 3.8+, TensorFlow 2.x (Keras), NumPy, Pandas, Matplotlib, scikit-learn.  
- **Предобработка**  
  1. Загрузка CSV, удаление строк с NaN.  
  2. Нормализация пикселей: `X = X / 255.0`.  
  3. reshape `(-1, 28, 28, 1)`.  
  4. One-hot кодирование меток через `to_categorical`.  
  5. Разбиение на обучение и валидацию (80/20) с `train_test_split(stratify=y)`.  
- **Архитектура модели**  
  1. Conv2D(32, 3×3, ReLU, padding=‘same’) → MaxPooling2D(2×2) → Dropout(0.25)  
  2. Conv2D(64, 3×3, ReLU, padding=‘same’) → MaxPooling2D(2×2) → Dropout(0.25)  
  3. Conv2D(128, 3×3, ReLU, padding=‘same’) → MaxPooling2D(2×2) → Dropout(0.30)  
  4. Flatten → Dense(128, ReLU) → Dropout(0.50) → Dense(10, softmax)  
- **Обучение**  
  - Optimizer: `SGD(lr=0.001, momentum=0.9, clipnorm=1.0)`  
  - Loss: `categorical_crossentropy`, метрика `accuracy`  
  - Batch size = 64, epochs ≤ 300  
  - Callback: `EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)`  
- **Результат**  
  Модель достигла **0.90** accuracy на Public Leaderboard.


## Лицензия
MIT
