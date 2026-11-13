"""
Главный исполняемый модуль Лабораторной работы №3.
Последовательно выполняет ячейки блокнота lab03_gradient_descent.ipynb:
подготовка данных, проверка класса, четыре вычислительных эксперимента.
"""

import pandas as pd

from config import HTML, IMAGES_DIR, RESULTS_DIR, display, plt, save_plot, sns
from data_loader import prepare_data
from experiments import (
    run_benchmark_experiment,
    run_gd_vs_sgd_experiment,
    run_learning_rate_experiment,
    run_regularization_experiment,
)
from linear_regression import CustomLinearRegression


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
    print("✅ Вычислительная среда успешно инициализирована!")
    print(f"📌 Папка сохранения графиков: {IMAGES_DIR}")

    # 1. Подготовка данных (ячейка 2 блокнота)
    df_raw, X_train_b, X_test_b, y_train, y_test, X_tr_s, X_te_s, feature_names, target_name = prepare_data()

    print(f"📊 Общий объем выборки: {len(df_raw)} строк")
    print(f"✅ Пропущенных значений во всем датасете: {df_raw.isna().sum().sum()} (датасет полностью чист)")
    print(f"📈 Обучающая выборка (Train 80%): X_train_b = {X_train_b.shape}, y_train = {y_train.shape}")
    print(f"📉 Тестовая выборка   (Test 20%):  X_test_b  = {X_test_b.shape}, y_test  = {y_test.shape}")
    print(f"📌 Количество признаков m = {len(feature_names)}, векторов весов (с учетом w0) = {X_train_b.shape[1]}")

    # Визуализация корреляционной матрицы отобранных признаков с целевой переменной
    corr_matrix = df_raw[feature_names + [target_name]].corr()
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="coolwarm", cbar=True, square=True, linewidths=0.5)
    plt.title("Корреляционная матрица Пирсона отобранных факторов успеваемости", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    save_plot(plt.gcf(), "00_correlation_matrix.png")
    plt.show()

    # Табличный предпросмотр первых строк (ячейка 2 блокнота)
    display(HTML("<b>Таблица 1 — Первые 5 наблюдений стандартизованной обучающей выборки:</b>"))
    df_preview = pd.DataFrame(X_train_b[:5], columns=["x0 (bias)"] + feature_names)
    df_preview["Target: Exam_Score"] = y_train[:5]
    display(df_preview.style.format("{:.4f}"))

    # 2. Проверка класса (ячейка 3 блокнота)
    ols_test = CustomLinearRegression(method="normal_eq").fit(X_train_b, y_train)
    print("✅ Класс CustomLinearRegression успешно реализован!")
    print(
        f"📊 Аналитическое МНК-решение: Test MSE = {ols_test.mse(X_test_b, y_test):.4f}, "
        f"R² = {ols_test.score(X_test_b, y_test):.4f}"
    )
    print(f"📌 Свободный член w0 (intercept) = {ols_test.intercept_:.4f}")

    # 3. Эксперименты 1–4
    run_learning_rate_experiment(X_train_b, X_test_b, y_train, y_test)
    exp2 = run_gd_vs_sgd_experiment(X_train_b, X_test_b, y_train, y_test)
    run_regularization_experiment(X_train_b, y_train, feature_names)
    run_benchmark_experiment(
        X_train_b,
        X_test_b,
        X_tr_s,
        X_te_s,
        y_train,
        y_test,
        feature_names,
        exp2["bgd_model"],
        exp2["sgd_model"],
    )

    print("=" * 80)
    print("  [✓] Все расчеты и эксперименты успешно завершены.")
    print(f"  [✓] Графические артефакты сохранены в папку: {IMAGES_DIR}")
    print(f"  [✓] Сводные таблицы и CSV сохранены в папку: {RESULTS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
