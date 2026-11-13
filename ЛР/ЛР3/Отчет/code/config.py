"""
Централизованный конфигурационный модуль Лабораторной работы №3.
«Градиентный спуск, стохастический градиентный спуск и регуляризация».

Содержимое полностью повторяет первую (служебную) ячейку блокнота
lab03_gradient_descent.ipynb: импорты, настройку вычислительной среды,
каталоги для артефактов и функцию сохранения графиков.
"""

import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

# Высокопроизводительные библиотеки векторных вычислений и таблиц
import numpy as np
import pandas as pd

# Академическая визуализация публикационного качества (300 DPI)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy import stats

# Эталонные алгоритмы машинного обучения Scikit-Learn для валидации
from sklearn.linear_model import Lasso, LinearRegression, Ridge, SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Интерактивный вывод в Jupyter Notebook
from IPython.display import HTML, display

# Headless-режим: консольный запуск без интерактивных окон
plt.switch_backend("Agg")

# 1. Фиксация зерна генератора псевдослучайных чисел для 100% воспроизводимости
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# 2. Подавление несущественных предупреждений
warnings.filterwarnings("ignore")

# 3. Настройка эстетики графиков Seaborn и Matplotlib
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#2c3e50"
plt.rcParams["axes.linewidth"] = 1.0

# 4. Настройка форматирования чисел в Pandas
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 1000)
pd.set_option("display.precision", 4)

# 5. Каталоги для сохранения локальных артефактов и отчета.
#    Код лежит в ЛР3/Отчет/code, поэтому корень проекта — двумя уровнями выше.
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[1]
SOLUTION_DIR = PROJECT_ROOT / "Решение"
DATA_PATH = SOLUTION_DIR / "student_performance_featured.csv"
IMAGES_DIR = SOLUTION_DIR / "images"
RESULTS_DIR = SOLUTION_DIR / "results"
REPORT_IMAGES_DIR = PROJECT_ROOT / "Отчет" / "images"

for _directory in (IMAGES_DIR, RESULTS_DIR, REPORT_IMAGES_DIR):
    _directory.mkdir(parents=True, exist_ok=True)


def save_plot(fig, filename):
    """Сохраняет фигуру в Решение/images и в Отчет/images (как в блокноте)."""
    fig.tight_layout()
    target_path = IMAGES_DIR / filename
    fig.savefig(target_path, dpi=300, bbox_inches="tight")
    if REPORT_IMAGES_DIR.parent.exists():
        fig.savefig(REPORT_IMAGES_DIR / filename, dpi=300, bbox_inches="tight")
    print(f"  [+] График сохранен: {filename}")
