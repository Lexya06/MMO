"""
Централизованный конфигурационный модуль для Лабораторной работы №3.
«Градиентный спуск, стохастический градиентный спуск и регуляризация».
"""

import os
from pathlib import Path
import numpy as np

# Базовые директории
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
LR2_DATA_PATH = (
    PROJECT_ROOT.parent / "ЛР2" / "Решение" / "processed_data" / "student_performance_featured.csv"
)
LR2_CLEANED_DATA_PATH = (
    PROJECT_ROOT.parent / "ЛР2" / "Решение" / "processed_data" / "student_performance_cleaned.csv"
)

# Каталоги сохранения результатов и графиков
IMAGES_DIR = BASE_DIR / "images"
REPORT_IMAGES_DIR = PROJECT_ROOT / "Отчет" / "images"
RESULTS_DIR = BASE_DIR / "results"

for d in [IMAGES_DIR, REPORT_IMAGES_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Ключевые признаки (интерпретируемый набор: 6 признаков + свободный член w0 = 7 весов >= 3)
SELECTED_FEATURES = [
    "Hours_Studied",
    "Attendance",
    "Previous_Scores",
    "Tutoring_Sessions",
    "Physical_Activity",
    "Parental_Involvement",
]

# Целевая переменная
TARGET_COLUMN = "Exam_Score"

# Запрещенные признаки (утечка данных)
LEAKAGE_COLUMNS = ["Score_Improvement"]

# Гиперпараметры градиентного спуска (BGD)
GD_PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 3000,
    "tol": 1e-6,
}

# Экспериментальные скорости обучения
LEARNING_RATES_EXPERIMENT = [0.0005, 0.005, 0.02, 0.05, 0.1, 0.25]

# Гиперпараметры стохастического градиентного спуска (SGD)
SGD_PARAMS = {
    "learning_rate": 0.01,
    "decay_rate": 0.0001,
    "max_iter": 10000,
    "tol": 1e-6,
}

# 20 коэффициентов регуляризации в широком логарифмическом диапазоне от 0.001 до 1000
REGULARIZATION_LAMBDAS = np.logspace(-3, 3, 20)

# Фиксация случайного зерна для 100% воспроизводимости
RANDOM_SEED = 42

# Настройки графиков
PLOT_CONFIG = {
    "dpi": 300,
    "style": "whitegrid",
    "palette": "tab10",
    "font_scale": 1.1,
}
