"""
Модуль загрузки, валидации и предобработки данных для Лабораторной работы №3.
Полностью повторяет ячейку «ЗАГРУЗКА, ПРЕДОБРАБОТКА И СТАНДАРТИЗАЦИЯ ДАННЫХ»
блокнота lab03_gradient_descent.ipynb.
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import DATA_PATH, RANDOM_SEED


def prepare_data() -> Tuple[
    pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, List[str], str,
]:
    """
    Загружает датасет из ЛР2, верифицирует отсутствие пропусков и выбросов,
    исключает признаки с утечкой целевой переменной, разбивает на Train/Test (80/20)
    и выполняет корректную стандартизацию признаков через StandardScaler.
    """
    candidates = [
        DATA_PATH,
        Path.cwd() / "student_performance_featured.csv",
        Path.cwd().parent / "student_performance_featured.csv",
        Path("student_performance_featured.csv"),
        Path.cwd().parent.parent / "ЛР2" / "Решение" / "processed_data" / "student_performance_featured.csv",
    ]

    dataset_path = None
    for p in candidates:
        if p.exists():
            dataset_path = p
            break

    if dataset_path is None:
        raise FileNotFoundError(f"Файл датасета не найден. Проверены пути: {[str(c) for c in candidates]}")

    df = pd.read_csv(dataset_path)

    # 1. Проверка чистоты данных: аудит пропусков (Missing Values)
    null_counts = df.isna().sum().sum()
    assert null_counts == 0, f"Обнаружены пропущенные значения: {null_counts}"

    selected_features = [
        "Hours_Studied",
        "Attendance",
        "Previous_Scores",
        "Tutoring_Sessions",
        "Physical_Activity",
        "Parental_Involvement"
    ]
    target_col = "Exam_Score"

    # 2. Извлечение матриц признаков и целевого вектора
    X_raw = df[selected_features].values.astype(np.float64)
    y_raw = df[target_col].values.astype(np.float64)

    # 3. Разделение на обучающую (80%) и тестовую (20%) выборки ДО масштабирования
    X_tr_raw, X_te_raw, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=0.20, random_state=RANDOM_SEED, shuffle=True
    )

    # 4. Стандартизация признаков: fit выполняется СТРОГО на обучающей выборке
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_tr_raw)
    X_test_scaled = scaler.transform(X_te_raw)

    # 5. Добавление фиктивного единичного столбца x0 = 1 для свободного члена w0
    X_train_b = np.hstack([np.ones((X_train_scaled.shape[0], 1)), X_train_scaled])
    X_test_b = np.hstack([np.ones((X_test_scaled.shape[0], 1)), X_test_scaled])

    return df, X_train_b, X_test_b, y_train, y_test, X_train_scaled, X_test_scaled, selected_features, target_col
