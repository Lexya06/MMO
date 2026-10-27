"""
Модуль загрузки, валидации и предобработки данных для Лабораторной работы №3.
Осуществляет извлечение данных из ЛР2, разбиение выборки, стандартизацию
(без Data Leakage) и добавление фиктивного единичного столбца для свободного члена (bias).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import (
    LR2_DATA_PATH,
    LR2_CLEANED_DATA_PATH,
    SELECTED_FEATURES,
    TARGET_COLUMN,
    LEAKAGE_COLUMNS,
    RANDOM_SEED,
)


@dataclass
class DatasetContainer:
    """Контейнер для хранения подготовленных матриц и метаданных датасета."""

    X_train_raw: np.ndarray
    X_test_raw: np.ndarray
    X_train_scaled: np.ndarray
    X_test_scaled: np.ndarray
    X_train_b: np.ndarray  # С единичным столбцом (x0 = 1) для w0
    X_test_b: np.ndarray  # С единичным столбцом (x0 = 1) для w0
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    scaler: StandardScaler
    df_raw: pd.DataFrame


def load_and_prepare_dataset(
    feature_mode: str = "selected",
    test_size: float = 0.2,
    random_state: int = RANDOM_SEED,
) -> DatasetContainer:
    """
    Загружает датасет из ЛР2, фильтрует признаки и подготавливает матрицы.

    Parameters:
    -----------
    feature_mode : str
        'selected' для 6 ключевых интерпретируемых признаков,
        'all' для всех признаков кроме утечек и таргета.
    test_size : float
        Доля тестовой выборки (по умолчанию 0.2).
    random_state : int
        Случайное зерно для воспроизводимости разбиения.

    Returns:
    --------
    DatasetContainer
        Структурированный объект со всеми выборками и скейлером.
    """
    if LR2_DATA_PATH.exists():
        df = pd.read_csv(LR2_DATA_PATH)
    elif LR2_CLEANED_DATA_PATH.exists():
        df = pd.read_csv(LR2_CLEANED_DATA_PATH)
    else:
        raise FileNotFoundError(
            f"Не найден датасет из ЛР2 по путям:\n{LR2_DATA_PATH}\n{LR2_CLEANED_DATA_PATH}"
        )

    # Исключение признаков с утечкой целевой переменной
    for col in LEAKAGE_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Выбор признаков
    if feature_mode == "selected":
        feature_names = [f for f in SELECTED_FEATURES if f in df.columns]
    elif feature_mode == "all":
        feature_names = [c for c in df.columns if c != TARGET_COLUMN]
    else:
        raise ValueError(f"Неизвестный режим признаков: {feature_mode}")

    if len(feature_names) < 2:
        raise ValueError(
            f"По заданию признаков должно быть не менее 2. Найдено: {len(feature_names)}"
        )

    X = df[feature_names].values.astype(np.float64)
    y = df[TARGET_COLUMN].values.astype(np.float64)

    # Разделение на обучающую и тестовую выборки ДО масштабирования
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Обучение скейлера строго на Train, предотвращая Data Leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # Добавление единичного столбца для смещения (bias) w0: X_b = [1, X]
    ones_train = np.ones((X_train_scaled.shape[0], 1), dtype=np.float64)
    ones_test = np.ones((X_test_scaled.shape[0], 1), dtype=np.float64)

    X_train_b = np.hstack([ones_train, X_train_scaled])
    X_test_b = np.hstack([ones_test, X_test_scaled])

    return DatasetContainer(
        X_train_raw=X_train_raw,
        X_test_raw=X_test_raw,
        X_train_scaled=X_train_scaled,
        X_test_scaled=X_test_scaled,
        X_train_b=X_train_b,
        X_test_b=X_test_b,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        scaler=scaler,
        df_raw=df,
    )


if __name__ == "__main__":
    data = load_and_prepare_dataset(feature_mode="selected")
    print(f"Загружено признаков: {len(data.feature_names)}")
    print(f"Признаки: {data.feature_names}")
    print(f"Размерность X_train_b: {data.X_train_b.shape}")
    print(f"Размерность X_test_b:  {data.X_test_b.shape}")
    print(f"Среднее y_train: {data.y_train.mean():.2f}, std: {data.y_train.std():.2f}")
