# -*- coding: utf-8 -*-
"""
Модуль предварительной очистки данных:
- Обнаружение и импутация пропущенных значений модой
- Расчет статистических границ по формуле межквартильного размаха (IQR)
- Устранение логических аномалий и винсоризация (clipping) выбросов
"""

from typing import Dict, List, Tuple
import pandas as pd


def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Вычисляет количество и процентное соотношение пропусков в столбцах.

    :param df: Анализируемый DataFrame.
    :return: DataFrame с колонками 'Пропусков (шт.)' и 'Процент (%)'.
    """
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df) * 100).round(2)
    summary = pd.DataFrame({
        'Пропусков (шт.)': missing_counts,
        'Процент (%)': missing_pct
    })
    return summary[summary['Пропусков (шт.)'] > 0].sort_values(
        by='Пропусков (шт.)', ascending=False
    )


def impute_missing_with_mode(
    df: pd.DataFrame, columns: List[str]
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Заполняет пропущенные значения категориальных признаков их модой.

    :param df: Исходный датасет.
    :param columns: Список колонок для заполнения.
    :return: Кортеж (очищенный DataFrame, словарь подставленных мод).
    """
    df_clean = df.copy()
    imputed_modes: Dict[str, str] = {}
    for col in columns:
        if df_clean[col].isnull().any():
            mode_val = str(df_clean[col].mode()[0])
            imputed_modes[col] = mode_val
            df_clean[col] = df_clean[col].fillna(mode_val)
    return df_clean, imputed_modes


def calculate_iqr_bounds(
    df: pd.DataFrame, numeric_cols: List[str]
) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    """
    Рассчитывает статистические границы выбросов по правилу 1.5 * IQR.

    :param df: Анализируемый DataFrame.
    :param numeric_cols: Список числовых признаков.
    :return: Таблица статистик выбросов и словарь {признак: (нижняя, верхняя)}.
    """
    outlier_stats = []
    bounds_dict: Dict[str, Tuple[float, float]] = {}

    for col in numeric_cols:
        q1 = float(df[col].quantile(0.25))
        q3 = float(df[col].quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        bounds_dict[col] = (lower, upper)

        outliers_count = int(
            ((df[col] < lower) | (df[col] > upper)).sum()
        )
        pct = (outliers_count / len(df) * 100)

        outlier_stats.append({
            'Признак': col,
            'Q1': q1,
            'Q3': q3,
            'IQR': iqr,
            'Нижняя граница': lower,
            'Верхняя граница': upper,
            'Мин. значение': float(df[col].min()),
            'Макс. значение': float(df[col].max()),
            'Кол-во выбросов': outliers_count,
            'Процент (%)': round(pct, 2)
        })

    summary_df = pd.DataFrame(outlier_stats).set_index('Признак')
    return summary_df, bounds_dict


def handle_outliers(
    df: pd.DataFrame,
    bounds_dict: Dict[str, Tuple[float, float]],
    target_cols: List[str]
) -> pd.DataFrame:
    """
    Устраняет логические аномалии и применяет винсоризацию (clipping)
    к числовым признакам с выбросами.

    :param df: Исходный датасет.
    :param bounds_dict: Словарь с границами IQR.
    :param target_cols: Признаки, подлежащие винсоризации.
    :return: DataFrame с устраненными выбросами.
    """
    df_clean = df.copy()

    # 1. Корректировка логической аномалии: Exam_Score > 100
    if 'Exam_Score' in df_clean.columns:
        df_clean.loc[df_clean['Exam_Score'] > 100, 'Exam_Score'] = 100

    # 2. Винсоризация (clipping)
    for col in target_cols:
        low, high = bounds_dict[col]
        # Нижняя граница не может быть отрицательной для счетчиков занятий
        low = max(low, 0.0)
        # Верхняя граница экзамена не выше 100
        if col == 'Exam_Score':
            high = min(high, 100.0)

        df_clean[col] = df_clean[col].clip(lower=low, upper=high)

    return df_clean
