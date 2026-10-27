# -*- coding: utf-8 -*-
"""
Модуль загрузки и первичного осмотра данных.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Загружает CSV датасет в pandas DataFrame.

    :param filepath: Абсолютный или относительный путь к CSV-файлу.
    :return: DataFrame с загруженными данными.
    :raises FileNotFoundError: Если файл не найден.
    """
    return pd.read_csv(filepath)


def get_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Разделяет признаки датасета на числовые и категориальные.

    :param df: Исходный датасет.
    :return: Кортеж (числовые_колонки, категориальные_колонки).
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    return numeric_cols, categorical_cols


def get_numeric_summary(
    df: pd.DataFrame, numeric_cols: List[str]
) -> pd.DataFrame:
    """
    Формирует таблицу описательных статистик для числовых колонок.

    :param df: Исходный датасет.
    :param numeric_cols: Список числовых признаков.
    :return: Транспонированный DataFrame со статистиками.
    """
    cols_to_show = [
        'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'
    ]
    return df[numeric_cols].describe().round(2).T[cols_to_show]
