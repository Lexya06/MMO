# -*- coding: utf-8 -*-
"""
Модуль генерации новых признаков (Feature Engineering)
и кодирования категориальных переменных.
"""

from typing import Dict, List, Tuple
import pandas as pd


def generate_new_features(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Генерирует 5 новых предметных признаков для повышения
    предсказательной способности моделей.

    :param df: Очищенный DataFrame.
    :return: Кортеж (DataFrame с новыми признаками, список имен признаков).
    """
    df_feat = df.copy()

    # 1. Study_Sleep_Ratio: баланс учебы и сна
    df_feat['Study_Sleep_Ratio'] = (
        df_feat['Hours_Studied'] / df_feat['Sleep_Hours']
    ).round(3)

    # 2. Total_Study_Effort: совокупные академические усилия
    df_feat['Total_Study_Effort'] = (
        df_feat['Hours_Studied']
        + 2.0 * df_feat['Tutoring_Sessions']
        + 0.1 * df_feat['Attendance']
    ).round(2)

    # 3. Score_Improvement: академический прогресс относительно прошлого уровня
    df_feat['Score_Improvement'] = (
        df_feat['Exam_Score'] - df_feat['Previous_Scores']
    ).round(2)

    # 4. Physical_Study_Balance: баланс спорта и учебной нагрузки
    df_feat['Physical_Study_Balance'] = (
        df_feat['Physical_Activity'] / (df_feat['Hours_Studied'] + 1)
    ).round(3)

    # 5. High_Attendance_Flag: бинарный флаг дисциплины (посещаемость >= 85%)
    df_feat['High_Attendance_Flag'] = (
        df_feat['Attendance'] >= 85
    ).astype(int)

    new_feature_names = [
        'Study_Sleep_Ratio',
        'Total_Study_Effort',
        'Score_Improvement',
        'Physical_Study_Balance',
        'High_Attendance_Flag'
    ]
    return df_feat, new_feature_names


def encode_binary_features(
    df: pd.DataFrame,
    mappings: Dict[str, Dict[str, int]]
) -> pd.DataFrame:
    """
    Выполняет бинаризацию дихотомических признаков (отображение в 0 и 1).

    :param df: DataFrame с категориальными признаками.
    :param mappings: Словарь {имя_колонки: {значение: 0_или_1}}.
    :return: Преобразованный DataFrame.
    """
    df_encoded = df.copy()
    for col, mapping in mappings.items():
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].map(mapping)
    return df_encoded


def encode_ordinal_features(
    df: pd.DataFrame,
    mappings: Dict[str, Dict[str, int]]
) -> pd.DataFrame:
    """
    Выполняет порядковое кодирование (Ordinal Encoding) ранговых категорий.

    :param df: DataFrame с категориальными признаками.
    :param mappings: Словарь {имя_колонки: {категория: ранг}}.
    :return: Преобразованный DataFrame.
    """
    df_encoded = df.copy()
    for col, mapping in mappings.items():
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].map(mapping)
    return df_encoded


def encode_nominal_features(
    df: pd.DataFrame,
    columns: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Выполняет One-Hot Encoding (OHE) для номинальных категориальных признаков.

    :param df: DataFrame.
    :param columns: Список номинальных признаков для прямого кодирования.
    :return: Кортеж (DataFrame с OHE-колонками, список добавленных имен).
    """
    df_encoded = pd.get_dummies(
        df, columns=columns, prefix=columns, dtype=int
    )
    new_ohe_cols = [
        c for c in df_encoded.columns
        if any(c.startswith(f"{orig}_") for orig in columns)
    ]
    return df_encoded, new_ohe_cols
