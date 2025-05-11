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

## Критерии успеха
- Метрика: **accuracy**.  
- Решение зачтётся, если **Public Leaderboard ≥ 0.85**.  
- Минимальное требование для зачёта — **accuracy ≥ 0.80**.  
- В день допускается не более **20 отправок**.

## Выбранный способ реализации
- **Среда и библиотеки**  
  - Python 3.8+  
  - TensorFlow 2.x (Keras API), NumPy, Pandas, Matplotlib  
- **Предобработка данных**  
  1. Загрузка CSV, извлечение меток и пикселей.  
  2. Нормализация пикселей: `X / 255.0`.  
  3. Преобразование в формат `(-1, 28, 28, 1)`.  
  4. One-hot кодирование меток через `to_categorical`.  
  5. Разбиение на обучение и валидацию: `train_test_split(test_size=0.2, random_state=42, stratify=y)`.  
- **Архитектура модели**  
  Построена как `Sequential`:
  1. `Conv2D(32, 3×3, activation='relu', padding='same', input_shape=(28,28,1))`  
     + `MaxPooling2D(2×2)` + `Dropout(0.25)`  
  2. `Conv2D(64, 3×3, activation='relu', padding='same')`  
     + `MaxPooling2D(2×2)` + `Dropout(0.25)`  
  3. `Conv2D(128, 3×3, activation='relu', padding='same')`  
     + `MaxPooling2D(2×2)` + `Dropout(0.30)`  
  4. `Flatten()`  
  5. `Dense(128, activation='relu')` + `Dropout(0.50)`  
  6. `Dense(10, activation='softmax')`
     
- **Результат**  
  - Достигнута точность **0.90** на Public Leaderboard.

## Результаты
- Моя модель достигла **accuracy = 0.90** на Public Leaderboard.


## Лицензия
MIT
