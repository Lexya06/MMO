"""
Главный исполняемый модуль и точка входа Лабораторной работы №3.
«Градиентный спуск, стохастический градиентный спуск и регуляризация».
БГУИР, ФКСиС, Кафедра ПОИТ, 2026.
"""

import sys
import time
from pathlib import Path

# Гарантия корректного вывода кириллицы в консоли Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from config import IMAGES_DIR, RESULTS_DIR
from data_loader import load_and_prepare_dataset
from experiments import (
    run_learning_rate_experiment,
    run_gd_vs_sgd_experiment,
    run_regularization_path_experiment,
    run_benchmark_experiment,
)


def print_header() -> None:
    print("=" * 80)
    print("  БЕЛОРУССКИЙ ГОСУДАРСТВЕННЫЙ УНИВЕРСИТЕТ ИНФОРМАТИКИ И РАДИОЭЛЕКТРОНИКИ")
    print("  Факультет компьютерных систем и сетей | Кафедра ПОИТ")
    print("  Дисциплина: Методы машинного обучения (ММО)")
    print("  ЛАБОРАТОРНАЯ РАБОТА № 3: Градиентный спуск и регуляризация")
    print("  Вариант: 2 (Четный — L1 регуляризация Lasso + сравнительный анализ L2 Ridge)")
    print("  Выполнила: студентка гр. 351002 Алексейчик Яна Николаевна")
    print("=" * 80)


def main() -> None:
    print_header()
    start_time = time.perf_counter()

    # 1. Загрузка и подготовка данных из ЛР2
    print("\n[ШАГ 1] Загрузка подготовленного набора данных из ЛР2...")
    data = load_and_prepare_dataset(feature_mode="selected", test_size=0.2)
    print(f"  [+] Загружено наблюдений: {len(data.df_raw)} строк")
    print(f"  [+] Размер обучающей выборки (Train): {data.X_train_b.shape[0]} объектов")
    print(f"  [+] Размер тестовой выборки (Test):   {data.X_test_b.shape[0]} объектов")
    print(f"  [+] Отобрано признаков: {len(data.feature_names)} (число весов с bias w0 = {data.X_train_b.shape[1]})")
    print(f"  [+] Признаки: {', '.join(data.feature_names)}")
    print(f"  [+] Целевая переменная: Exam_Score")

    # 2. Эксперимент 1: Скорость обучения
    exp1_res = run_learning_rate_experiment(data)

    # 3. Эксперимент 2: GD vs SGD
    exp2_res = run_gd_vs_sgd_experiment(data)

    # 4. Эксперимент 3: Регуляризация (L1 для варианта 2 и L2)
    exp3_res = run_regularization_path_experiment(data)

    # 5. Эксперимент 4: Сравнение со Scikit-Learn
    exp4_res = run_benchmark_experiment(data)

    total_time = time.perf_counter() - start_time
    print("\n" + "=" * 80)
    print(f"  [✓] Все расчеты и эксперименты успешно завершены за {total_time:.2f} сек.")
    print(f"  [✓] Графические артефакты сохранены в папку: {IMAGES_DIR}")
    print(f"  [✓] Сводные таблицы и CSV сохранены в папку: {RESULTS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
