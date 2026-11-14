# ==============================================================================
# ИМПОРТ БИБЛИОТЕК И НАСТРОЙКА ВЫЧИСЛИТЕЛЬНОЙ СРЕДЫ
# ==============================================================================

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

# 5. Каталоги для сохранения локальных артефактов и отчета
BASE_DIR = Path.cwd()
IMAGES_DIR = BASE_DIR / "images"
RESULTS_DIR = BASE_DIR / "results"
if BASE_DIR.name == "Решение":
    REPORT_IMAGES_DIR = BASE_DIR.parent / "Отчет" / "images"
else:
    REPORT_IMAGES_DIR = BASE_DIR / "Отчет" / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
if REPORT_IMAGES_DIR.parent.exists():
    REPORT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def save_plot(fig, filename):
    fig.tight_layout()
    target_path = IMAGES_DIR / filename
    fig.savefig(target_path, dpi=300, bbox_inches="tight")
    if REPORT_IMAGES_DIR.parent.exists():
        fig.savefig(REPORT_IMAGES_DIR / filename, dpi=300, bbox_inches="tight")
    print(f"  [+] График сохранен: {filename}")


print("✅ Вычислительная среда успешно инициализирована!")
print(
    f"📌 Версии: Python {sys.version.split()[0]} | NumPy {np.__version__} | "
    f"Pandas {pd.__version__} | Matplotlib {plt.matplotlib.__version__}"
)
print(f"📌 Папка сохранения графиков: {IMAGES_DIR}")