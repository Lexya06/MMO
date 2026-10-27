# -*- coding: utf-8 -*-
"""
Модуль стандартизации и нормализации данных.
Реализует алгоритмы MinMaxScaler и StandardScaler,
а также сравнительный статистический анализ.
"""

from typing import Dict, List
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def apply_minmax_scaling(
    df: pd.DataFrame, columns: List[str]
) -> pd.DataFrame:
    """
    Выполняет нормализацию числовых признаков в диапазон [0, 1].

    :param df: Исходный DataFrame.
    :param columns: Список колонок для масштабирования.
    :return: DataFrame с нормализованными признаками.
    """
    df_scaled = df.copy()
    scaler = MinMaxScaler()
    df_scaled[columns] = scaler.fit_transform(df[columns])
    return df_scaled


def apply_standard_scaling(
    df: pd.DataFrame, columns: List[str]
) -> pd.DataFrame:
    """
    Выполняет стандартизацию признаков к mean=0 и std=1 (Z-score).

    :param df: Исходный DataFrame.
    :param columns: Список колонок для масштабирования.
    :return: DataFrame со стандартизированными признаками.
    """
    df_scaled = df.copy()
    scaler = StandardScaler()
    df_scaled[columns] = scaler.fit_transform(df[columns])
    return df_scaled


def compare_scaling_stats(
    df_orig: pd.DataFrame,
    df_minmax: pd.DataFrame,
    df_std: pd.DataFrame,
    columns: List[str]
) -> Dict[str, pd.DataFrame]:
    """
    Формирует сравнительные таблицы статистических параметров
    до и после различных методов масштабирования.

    :param df_orig: Исходный DataFrame.
    :param df_minmax: DataFrame после MinMaxScaler.
    :param df_std: DataFrame после StandardScaler.
    :param columns: Список анализируемых колонок.
    :return: Словарь {признак: сводная_таблица}.
    """
    results: Dict[str, pd.DataFrame] = {}
    metric_names = [
        'Среднее (Mean)', 'Станд. откл. (Std)',
        'Минимум (Min)', 'Максимум (Max)'
    ]

    for col in columns:
        col_comparison = pd.DataFrame({
            'Оригинал': [
                df_orig[col].mean(), df_orig[col].std(),
                df_orig[col].min(), df_orig[col].max()
            ],
            'MinMaxScaler': [
                df_minmax[col].mean(), df_minmax[col].std(),
                df_minmax[col].min(), df_minmax[col].max()
            ],
            'StandardScaler': [
                df_std[col].mean(), df_std[col].std(),
                df_std[col].min(), df_std[col].max()
            ]
        }, index=metric_names)
        results[col] = col_comparison.round(4)

    return results
