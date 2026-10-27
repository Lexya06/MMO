# -*- coding: utf-8 -*-
"""
Модуль конфигурации лабораторной работы № 2.
Определяет файловые пути, глобальные константы и словари кодирования категорий.
"""

import os
from typing import Dict

# Базовые пути проекта
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH: str = os.path.abspath(
    os.path.join(
        BASE_DIR, '..', 'Задание', 'StudentPerformanceFactors',
        'StudentPerformanceFactors.csv'
    )
)
IMAGES_DIR: str = os.path.join(BASE_DIR, 'images')
PROCESSED_DATA_DIR: str = os.path.join(BASE_DIR, 'processed_data')

REPORT_DIR: str = os.path.abspath(os.path.join(BASE_DIR, '..', 'Отчет'))
REPORT_IMAGES_DIR: str = os.path.join(REPORT_DIR, 'images')
REPORT_OVERLEAF_IMAGES_DIR: str = os.path.join(
    REPORT_DIR, 'Отчет_Overleaf', 'images'
)

# Словари для бинаризации категориальных признаков (0 / 1)
BINARY_MAPPINGS: Dict[str, Dict[str, int]] = {
    'Extracurricular_Activities': {'No': 0, 'Yes': 1},
    'Internet_Access': {'No': 0, 'Yes': 1},
    'Learning_Disabilities': {'No': 0, 'Yes': 1},
    'School_Type': {'Public': 0, 'Private': 1},
    'Gender': {'Male': 0, 'Female': 1}
}

# Словари для порядкового кодирования (Ordinal Encoding)
ORDINAL_MAPPINGS: Dict[str, Dict[str, int]] = {
    'Parental_Involvement': {'Low': 0, 'Medium': 1, 'High': 2},
    'Access_to_Resources': {'Low': 0, 'Medium': 1, 'High': 2},
    'Motivation_Level': {'Low': 0, 'Medium': 1, 'High': 2},
    'Family_Income': {'Low': 0, 'Medium': 1, 'High': 2},
    'Teacher_Quality': {'Low': 0, 'Medium': 1, 'High': 2},
    'Distance_from_Home': {'Near': 0, 'Moderate': 1, 'Far': 2},
    'Parental_Education_Level': {
        'High School': 0,
        'College': 1,
        'Postgraduate': 2
    }
}

# Признаки для One-Hot кодирования
OHE_COLUMNS: list[str] = ['Peer_Influence']

# Непрерывные числовые признаки для стандартизации и нормализации
CONTINUOUS_COLUMNS_TO_SCALE: list[str] = [
    'Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores',
    'Tutoring_Sessions', 'Physical_Activity', 'Exam_Score',
    'Study_Sleep_Ratio', 'Total_Study_Effort', 'Score_Improvement',
    'Physical_Study_Balance'
]

# Ключевые демонстрационные признаки для детального анализа распределений
DEMO_COLUMNS: list[str] = [
    'Hours_Studied', 'Attendance', 'Exam_Score', 'Total_Study_Effort'
]
