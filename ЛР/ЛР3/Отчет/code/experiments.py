"""
Модуль проведения всех вычислительных экспериментов Лабораторной работы №3.
Тела функций полностью повторяют ячейки блокнота lab03_gradient_descent.ipynb
(эксперименты 1–4), включая построение и сохранение графиков.
"""

import time
from typing import Any, Dict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.linear_model import Lasso, LinearRegression, Ridge, SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from IPython.display import HTML, display

from config import RANDOM_SEED, RESULTS_DIR, save_plot
from linear_regression import CustomLinearRegression


def run_learning_rate_experiment(X_train_b, X_test_b, y_train, y_test):
    """Эксперимент 1: влияние скорости обучения eta на сходимость BGD."""
    # ЭКСПЕРИМЕНТ 1: ВЛИЯНИЕ СКОРОСТИ ОБУЧЕНИЯ (LEARNING RATE) НА СХОДИМОСТЬ BGD
    # ==============================================================================

    learning_rates = [0.0005, 0.005, 0.02, 0.05, 0.1, 0.25]
    lr_results = []
    lr_loss_histories = {}

    for lr in learning_rates:
        model = CustomLinearRegression(
            method="gd",
            learning_rate=lr,
            max_iter=3000,
            tol=1e-6,
        )
        t0 = time.perf_counter()
        model.fit(X_train_b, y_train)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        test_mse = model.mse(X_test_b, y_test)
        test_r2 = model.score(X_test_b, y_test)
        train_mse = model.loss_history_[-1]

        lr_results.append({
            "Learning Rate (η)": lr,
            "Итераций": model.n_iter_,
            "Сошелся (tol=1e-6)": "Да" if model.converged_ else "Нет (max_iter)",
            "Train MSE": train_mse,
            "Test MSE": test_mse,
            "Test R²": test_r2,
            "Время (мс)": elapsed_ms,
        })
        lr_loss_histories[lr] = model.loss_history_

    df_lr = pd.DataFrame(lr_results)
    df_lr.to_csv(RESULTS_DIR / "01_learning_rate_results.csv", index=False)

    display(HTML("<b>Таблица 2 — Сравнительные характеристики сходимости BGD при различных Learning Rate:</b>"))
    display(df_lr.style.format({
        "Learning Rate (η)": "{:.4f}",
        "Train MSE": "{:.4f}",
        "Test MSE": "{:.4f}",
        "Test R²": "{:.4f}",
        "Время (мс)": "{:.2f}",
    }).highlight_min(subset=["Итераций", "Время (мс)"], color="#d4edda"))

    # ------------------------------------------------------------------------------
    # ВИЗУАЛИЗАЦИЯ: Динамика сходимости MSE в линейной и логарифмической шкалах
    # ------------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    palette = sns.color_palette("tab10", len(learning_rates))

    # Панель 1: Линейная шкала на первых 150 шагах
    for i, lr in enumerate(learning_rates):
        history = lr_loss_histories[lr][:150]
        ax1.plot(range(len(history)), history, label=f"η = {lr}", color=palette[i], linewidth=2.0)

    ax1.set_title("Сходимость функции потерь MSE (Линейная шкала, шаги 0..150)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Номер итерации $k$", fontsize=11)
    ax1.set_ylabel("Среднеквадратичная ошибка (MSE)", fontsize=11)
    ax1.set_ylim(bottom=0.0, top=100.0)
    ax1.legend(loc="upper right", frameon=True, framealpha=0.9)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Панель 2: Логарифмическая шкала с ясными скалярными отметками (без научной нотации)
    for i, lr in enumerate(learning_rates):
        history = lr_loss_histories[lr][:400]
        ax2.semilogy(range(len(history)), history, label=f"η = {lr}", color=palette[i], linewidth=2.0)

    ax2.set_title("Скорость сходимости MSE (Логарифмическая шкала semilogy)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Номер итерации $k$", fontsize=11)
    ax2.set_ylabel("Среднеквадратичная ошибка (MSE)", fontsize=11)
    y_ticks_lr = [2, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000]
    ax2.set_yticks(y_ticks_lr)
    ax2.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
    ax2.set_ylim(bottom=1.8, top=5500)
    ax2.legend(loc="upper right", frameon=True, framealpha=0.9)
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()
    save_plot(plt.gcf(), "01_learning_rate_experiment.png")
    plt.show()

    return {"df_lr": df_lr, "loss_histories": lr_loss_histories}


def run_gd_vs_sgd_experiment(X_train_b, X_test_b, y_train, y_test):
    """Эксперимент 2: прямое сравнение сходимости BGD и SGD на едином графике."""
    # ЭКСПЕРИМЕНТ 2: СРАВНЕНИЕ СХОДИМОСТИ BGD И SGD НА ЕДИНОМ ГРАФИКЕ
    # ==============================================================================

    # 1. Обучение пакетного градиентного спуска (BGD)
    bgd_model = CustomLinearRegression(
        method="gd",
        learning_rate=0.05,
        max_iter=1000,
        tol=1e-6,
    )
    t0 = time.perf_counter()
    bgd_model.fit(X_train_b, y_train)
    t_bgd = (time.perf_counter() - t0) * 1000.0

    # 2. Обучение стохастического градиентного спуска (SGD)
    sgd_model = CustomLinearRegression(
        method="sgd",
        learning_rate=0.01,
        decay_rate=0.0001,
        max_iter=10000,
        tol=1e-6,
        random_state=RANDOM_SEED,
    )
    t0 = time.perf_counter()
    sgd_model.fit(X_train_b, y_train)
    t_sgd = (time.perf_counter() - t0) * 1000.0

    # Сводная таблица сравнительных метрик
    df_gd_vs_sgd = pd.DataFrame([
        {
            "Метод оптимизации": "BGD (Batch Gradient Descent)",
            "Размер пакета": f"Все l = {X_train_b.shape[0]}",
            "Шагов до останова": bgd_model.n_iter_,
            "Train MSE": bgd_model.loss_history_[-1],
            "Test MSE": bgd_model.mse(X_test_b, y_test),
            "Test R²": bgd_model.score(X_test_b, y_test),
            "Время обучения (мс)": t_bgd,
        },
        {
            "Метод оптимизации": "SGD (Stochastic Gradient Descent)",
            "Размер пакета": "1 случайный объект",
            "Шагов до останова": sgd_model.n_iter_,
            "Train MSE": sgd_model.loss_history_[-1],
            "Test MSE": sgd_model.mse(X_test_b, y_test),
            "Test R²": sgd_model.score(X_test_b, y_test),
            "Время обучения (мс)": t_sgd,
        }
    ])
    df_gd_vs_sgd.to_csv(RESULTS_DIR / "02_gd_vs_sgd_results.csv", index=False)

    display(HTML("<b>Таблица 3 — Сравнительные характеристики BGD и SGD:</b>"))
    display(df_gd_vs_sgd.style.format({
        "Train MSE": "{:.4f}",
        "Test MSE": "{:.4f}",
        "Test R²": "{:.4f}",
        "Время обучения (мс)": "{:.2f}",
    }))

    # ------------------------------------------------------------------------------
    # ВИЗУАЛИЗАЦИЯ: Нанесение кривых сходимости BGD и SGD на единое графическое поле
    # ------------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Построение глобальной кривой (шаги 0..400)
    steps_bgd = len(bgd_model.loss_history_)
    steps_to_plot = 400

    # Выравнивание массивов для корректного отображения
    bgd_losses_ext = list(bgd_model.loss_history_)
    if len(bgd_losses_ext) < steps_to_plot:
        bgd_losses_ext.extend([bgd_losses_ext[-1]] * (steps_to_plot - len(bgd_losses_ext)))

    ax1.semilogy(
        range(steps_to_plot),
        bgd_losses_ext[:steps_to_plot],
        label="BGD (векторный градиент по всем 5285 прецедентам)",
        color="#1f77b4",
        linewidth=2.5,
    )
    ax1.semilogy(
        range(steps_to_plot),
        sgd_model.loss_history_[:steps_to_plot],
        label="SGD (градиент по 1 случайному студенту)",
        color="#ff7f0e",
        linewidth=1.5,
        alpha=0.85,
    )

    ax1.set_title("Сравнение сходимости BGD и SGD (Полный спуск, шаги 0..400)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Номер итерации (шага) $k$", fontsize=11)
    ax1.set_ylabel("Среднеквадратичная ошибка (MSE)", fontsize=11)
    y_ticks_sgd = [2, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000]
    ax1.set_yticks(y_ticks_sgd)
    ax1.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
    ax1.set_ylim(bottom=1.8, top=5500)
    ax1.legend(loc="upper right", frameon=True, framealpha=0.95)
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)

    # Увеличенный масштаб вблизи оптимума (шаги 150..400, линейная шкала)
    ax2.plot(
        range(150, steps_to_plot),
        bgd_losses_ext[150:steps_to_plot],
        label=f"BGD: устойчивое дно (MSE = {bgd_model.loss_history_[-1]:.4f})",
        color="#1f77b4",
        linewidth=2.5,
    )
    ax2.plot(
        range(150, steps_to_plot),
        sgd_model.loss_history_[150:steps_to_plot],
        label="SGD: стохастический шум и колебания весов",
        color="#ff7f0e",
        linewidth=1.2,
        alpha=0.9,
    )

    ax2.set_title(
        "Увеличенный вид окрестности минимума (Линейная шкала, шаги 150..400)",
        fontsize=13, fontweight="bold",
    )
    ax2.set_xlabel("Номер итерации (шага) $k$", fontsize=11)
    ax2.set_ylabel("Среднеквадратичная ошибка (MSE)", fontsize=11)
    ax2.set_ylim(bottom=1.8, top=6.0)
    ax2.legend(loc="upper right", frameon=True, framealpha=0.95)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    save_plot(plt.gcf(), "02_gd_vs_sgd_convergence.png")
    plt.show()

    return {"bgd_model": bgd_model, "sgd_model": sgd_model, "df": df_gd_vs_sgd}


def run_regularization_experiment(X_train_b, y_train, feature_names):
    """Эксперимент 3: регуляризационные пути L1 (Lasso, вариант 2) и L2 (Ridge) для 20 значений lambda."""
    # ЭКСПЕРИМЕНТ 3: РЕГУЛЯРИЗАЦИОННЫЕ ПУТИ ДЛЯ 20 ЗНАЧЕНИЙ LAMBDA
    # ==============================================================================

    # Генерация 20 значений коэффициента регуляризации в логарифмической шкале [0.001, 1000]
    lambdas_20 = np.logspace(-3, 3, 20)

    l1_weights_history = []
    l2_weights_history = []
    l1_sparsity_counts = []

    for lmbd in lambdas_20:
        # 1. Обучение модели L1 (Lasso — Вариант 2)
        l1_model = CustomLinearRegression(
            method="l1",
            learning_rate=0.02,
            lmbd=lmbd,
            max_iter=3000,
            tol=1e-6,
        )
        l1_model.fit(X_train_b, y_train)
        l1_weights_history.append(l1_model.coef_)

        # Подсчет строго нулевых коэффициентов (|w_j| < 1e-4)
        zero_cnt = int(np.sum(np.isclose(l1_model.coef_, 0.0, atol=1e-4)))
        l1_sparsity_counts.append(zero_cnt)

        # 2. Обучение модели L2 (Ridge)
        l2_model = CustomLinearRegression(
            method="l2",
            learning_rate=0.02,
            lmbd=lmbd,
            max_iter=3000,
            tol=1e-6,
        )
        l2_model.fit(X_train_b, y_train)
        l2_weights_history.append(l2_model.coef_)

    l1_weights_arr = np.array(l1_weights_history)
    l2_weights_arr = np.array(l2_weights_history)

    # Формирование и сохранение сводной таблицы L1 (Lasso)
    df_l1_table = pd.DataFrame(l1_weights_arr, columns=feature_names)
    df_l1_table.insert(0, "lambda", lambdas_20)
    df_l1_table.insert(1, "log10(lambda)", np.round(np.log10(lambdas_20), 2))
    df_l1_table["Занулено признаков"] = l1_sparsity_counts
    df_l1_table.to_csv(RESULTS_DIR / "03_l1_weights_by_lambda.csv", index=False)

    # Формирование и сохранение сводной таблицы L2 (Ridge)
    df_l2_table = pd.DataFrame(l2_weights_arr, columns=feature_names)
    df_l2_table.insert(0, "lambda", lambdas_20)
    df_l2_table.insert(1, "log10(lambda)", np.round(np.log10(lambdas_20), 2))
    df_l2_table.to_csv(RESULTS_DIR / "04_l2_weights_by_lambda.csv", index=False)

    display(HTML("<b>Таблица 4 — Динамика весов L1-регуляризации (Lasso, Вариант 2) и отбор признаков:</b>"))
    display(df_l1_table.style.format({
        "lambda": "{:.4f}",
        "log10(lambda)": "{:.2f}",
        "Hours_Studied": "{:.4f}",
        "Attendance": "{:.4f}",
        "Previous_Scores": "{:.4f}",
        "Tutoring_Sessions": "{:.4f}",
        "Physical_Activity": "{:.4f}",
        "Parental_Involvement": "{:.4f}",
    }))

    # ------------------------------------------------------------------------------
    # График 03: Регуляризационный путь L1 (Lasso, Вариант 2) - отбор признаков
    # ------------------------------------------------------------------------------
    fig_l1, ax_l1 = plt.subplots(figsize=(11, 6))
    colors_l1 = sns.color_palette("tab10", len(feature_names))
    markers_list = ["o", "s", "^", "D", "v", "<"]
    log_lambdas_arr = np.log10(lambdas_20)

    for j, feat in enumerate(feature_names):
        ax_l1.plot(
            log_lambdas_arr, l1_weights_arr[:, j],
            label=feat, color=colors_l1[j], linewidth=2.2,
            marker=markers_list[j % len(markers_list)], markersize=6
        )
    ax_l1.axhline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    ax_l1.set_title(
        "Регуляризационный путь L1 (Lasso) для 20 значений $\\lambda$: зануление признаков (Sparsity)",
        fontsize=12,
        fontweight="bold",
    )
    ax_l1.set_xlabel("$\\log_{10}(\\lambda)$ (от $10^{-3}$ до $10^3$)", fontsize=11)
    ax_l1.set_ylabel("Значение весового коэффициента $w_j$", fontsize=11)
    ax_l1.legend(loc="upper right", frameon=True, framealpha=0.9, title="Признаки модели")
    ax_l1.grid(True, linestyle="--", alpha=0.6)
    save_plot(fig_l1, "03_l1_regularization_path.png")
    plt.show()

    # ------------------------------------------------------------------------------
    # График 04: Регуляризационный путь L2 (Ridge) - непрерывное сжатие весов
    # ------------------------------------------------------------------------------
    fig_l2, ax_l2 = plt.subplots(figsize=(11, 6))
    for j, feat in enumerate(feature_names):
        ax_l2.plot(
            log_lambdas_arr, l2_weights_arr[:, j],
            label=feat, color=colors_l1[j], linewidth=2.2,
            marker=markers_list[j % len(markers_list)], markersize=6
        )
    ax_l2.axhline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    ax_l2.set_title(
        "Регуляризационный путь L2 (Ridge) для 20 значений $\\lambda$: асимптотическое сжатие (Shrinkage)",
        fontsize=12,
        fontweight="bold",
    )
    ax_l2.set_xlabel("$\\log_{10}(\\lambda)$ (от $10^{-3}$ до $10^3$)", fontsize=11)
    ax_l2.set_ylabel("Значение весового коэффициента $w_j$", fontsize=11)
    ax_l2.legend(loc="upper right", frameon=True, framealpha=0.9, title="Признаки модели")
    ax_l2.grid(True, linestyle="--", alpha=0.6)
    save_plot(fig_l2, "04_l2_regularization_path.png")
    plt.show()

    # ------------------------------------------------------------------------------
    # График 05: Покомпонентное сопоставление L1 и L2 для каждого отдельного фактора
    # ------------------------------------------------------------------------------
    fig_comp, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for i, feat in enumerate(feature_names):
        ax = axes[i]
        ax.plot(
            lambdas_20, l1_weights_arr[:, i],
            marker="o", color="#d62728",
            label="L1 (Lasso: строгое зануление)", linewidth=2.0,
        )
        ax.plot(
            lambdas_20, l2_weights_arr[:, i],
            marker="s", color="#1f77b4",
            label="L2 (Ridge: плавное сжатие)", linewidth=2.0,
        )
        ax.axhline(0, color="black", linestyle="--", alpha=0.5)
        ax.set_xscale("log")
        ax.set_title(f"Фактор: {feat}", fontsize=11, fontweight="bold")
        ax.set_xlabel("$\\lambda$ (log-шкала)", fontsize=10)
        ax.set_ylabel("Вес $w_j$", fontsize=10)
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.suptitle(
        "Покомпонентное сравнение сжатия весов: L1 (Lasso) vs L2 (Ridge)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    save_plot(fig_comp, "05_regularization_comparison.png")
    plt.show()

    return {
            "l1_weights": l1_weights_arr,
            "l2_weights": l2_weights_arr,
            "df_l1": df_l1_table,
            "df_l2": df_l2_table,
        }


def run_benchmark_experiment(
    X_train_b,
    X_test_b,
    X_tr_s,
    X_te_s,
    y_train,
    y_test,
    feature_names,
    bgd_model,
    sgd_model,
):
    """Эксперимент 4: эталонное тестирование против моделей Scikit-Learn."""
    # ЭКСПЕРИМЕНТ 4: ЭТАЛОННОЕ ТЕСТИРОВАНИЕ ПРОТИВ SCIKIT-LEARN
    # ==============================================================================

    # 1. Библиотечная модель OLS (LinearRegression)
    sk_ols = LinearRegression(fit_intercept=True)
    sk_ols.fit(X_tr_s, y_train)

    # 2. Библиотечная модель SGDRegressor
    sk_sgd = SGDRegressor(
        loss="squared_error",
        penalty=None,
        learning_rate="invscaling",
        eta0=0.01,
        max_iter=10000,
        tol=1e-6,
        random_state=RANDOM_SEED,
    )
    sk_sgd.fit(X_tr_s, y_train)

    # 3. Кастомная и библиотечная модели Lasso (L1)
    custom_l1 = CustomLinearRegression(method="l1", learning_rate=0.02, lmbd=0.5, max_iter=3000, tol=1e-6)
    custom_l1.fit(X_train_b, y_train)

    sk_lasso = Lasso(alpha=0.25, fit_intercept=True, max_iter=3000, tol=1e-6, random_state=RANDOM_SEED)
    sk_lasso.fit(X_tr_s, y_train)

    # 4. Кастомная и библиотечная модели Ridge (L2)
    custom_l2 = CustomLinearRegression(method="l2", learning_rate=0.02, lmbd=1.0, max_iter=3000, tol=1e-6)
    custom_l2.fit(X_train_b, y_train)

    sk_ridge = Ridge(alpha=X_tr_s.shape[0] * 1.0, fit_intercept=True, max_iter=3000, tol=1e-6, random_state=RANDOM_SEED)
    sk_ridge.fit(X_tr_s, y_train)

    # Оценка моделей и расчет метрик
    def calc_metrics(name: str, model, is_custom: bool = True) -> Dict[str, Any]:
        if is_custom:
            intercept = float(model.intercept_)
            coef = np.ravel(model.coef_)
            y_pred = np.ravel(model.predict(X_test_b))
        else:
            intercept = float(np.ravel(model.intercept_)[0])
            coef = np.ravel(model.coef_)
            y_pred = np.ravel(model.predict(X_te_s))

        w_norm = float(np.linalg.norm(coef))
        test_mse = float(mean_squared_error(y_test, y_pred))
        test_mae = float(mean_absolute_error(y_test, y_pred))
        test_r2 = float(r2_score(y_test, y_pred))

        return {
            "Модель": name,
            "Intercept (w0)": intercept,
            "Норма весов ||w||": w_norm,
            "Test MSE": test_mse,
            "Test MAE": test_mae,
            "Test R²": test_r2,
            "_coef": coef,
            "_pred": y_pred,
        }

    benchmark_records = [
        calc_metrics("Custom BGD", bgd_model, is_custom=True),
        calc_metrics("Sklearn OLS", sk_ols, is_custom=False),
        calc_metrics("Custom SGD", sgd_model, is_custom=True),
        calc_metrics("Sklearn SGD", sk_sgd, is_custom=False),
        calc_metrics("Custom L1 (λ=0.5)", custom_l1, is_custom=True),
        calc_metrics("Sklearn Lasso (α=0.25)", sk_lasso, is_custom=False),
        calc_metrics("Custom L2 (λ=1.0)", custom_l2, is_custom=True),
        calc_metrics("Sklearn Ridge (α=5285.0)", sk_ridge, is_custom=False),
    ]

    df_benchmark = pd.DataFrame(benchmark_records).drop(columns=["_coef", "_pred"])
    df_benchmark.to_csv(RESULTS_DIR / "05_benchmark_comparison.csv", index=False)

    display(HTML("<b>Таблица 5 — Сверка кастомных алгоритмов с библиотечными эталонами Scikit-Learn:</b>"))
    display(df_benchmark.style.format({
        "Intercept (w0)": "{:.4f}",
        "Норма весов ||w||": "{:.4f}",
        "Test MSE": "{:.4f}",
        "Test MAE": "{:.4f}",
        "Test R²": "{:.4f}",
    }))

    # ------------------------------------------------------------------------------
    # 4-ПАНЕЛЬНЫЙ ДИАГНОСТИЧЕСКИЙ ДАШБОРД ВАЛИДАЦИИ
    # ------------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Панель 1: Диаграмма рассеяния «Прогноз vs Факт» (BGD)
    y_pred_bgd = bgd_model.predict(X_test_b)
    axes[0, 0].scatter(y_test, y_pred_bgd, alpha=0.35, color="#1f77b4", edgecolor="none", s=25)
    ideal_line = np.linspace(min(y_test), max(y_test), 100)
    axes[0, 0].plot(
        ideal_line, ideal_line, color="#d62728", linestyle="--", linewidth=2.0,
        label="Идеальный прогноз $y = \\hat{y}$",
    )
    axes[0, 0].set_title(
        f"Диаграмма «Прогноз vs Факт» (Custom BGD, $R^2 = {bgd_model.score(X_test_b, y_test):.4f}$)",
        fontsize=12, fontweight="bold",
    )
    axes[0, 0].set_xlabel("Истинная оценка студента $y$", fontsize=11)
    axes[0, 0].set_ylabel("Предсказанная оценка $\\hat{y}$", fontsize=11)
    axes[0, 0].legend(loc="upper left", framealpha=0.9)
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)

    # Панель 2: Распределение остатков ошибки (Residuals Plot)
    residuals_bgd = y_test - y_pred_bgd
    sns.histplot(residuals_bgd, kde=True, ax=axes[0, 1], color="#2ca02c", stat="density", bins=30)
    mu_res, std_res = float(np.mean(residuals_bgd)), float(np.std(residuals_bgd))
    x_gauss = np.linspace(residuals_bgd.min(), residuals_bgd.max(), 100)
    axes[0, 1].plot(
        x_gauss, stats.norm.pdf(x_gauss, mu_res, std_res), "r--",
        linewidth=2.0, label="Теоретическое норм. распределение",
    )
    axes[0, 1].set_title(
        f"Плотность распределения остатков ($\\mu = {mu_res:.3f}, \\sigma = {std_res:.3f}$)",
        fontsize=12, fontweight="bold",
    )
    axes[0, 1].set_xlabel("Величина ошибки прогноза $\\varepsilon = y - \\hat{y}$", fontsize=11)
    axes[0, 1].set_ylabel("Плотность вероятности", fontsize=11)
    axes[0, 1].legend(loc="upper right", framealpha=0.9)
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)

    # Панель 3: Покомпонентное сравнение весов (Custom BGD vs Sklearn OLS)
    bar_width = 0.35
    indices = np.arange(len(feature_names))
    axes[1, 0].bar(indices - bar_width/2, bgd_model.coef_, bar_width, label="Custom BGD", color="#1f77b4")
    axes[1, 0].bar(indices + bar_width/2, sk_ols.coef_, bar_width, label="Sklearn OLS", color="#aec7e8")
    axes[1, 0].set_xticks(indices)
    axes[1, 0].set_xticklabels(feature_names, rotation=25, ha="right", fontsize=10)
    axes[1, 0].set_title("Покомпонентное совпадение весов (BGD vs Scikit-Learn)", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel("Величина коэффициента $w_j$", fontsize=11)
    axes[1, 0].legend(loc="upper right", framealpha=0.9)
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)

    # Панель 4: Сравнение ошибок Test MSE всех моделей
    models_names = [m["Модель"] for m in benchmark_records]
    mse_vals = [m["Test MSE"] for m in benchmark_records]
    colors_bar = sns.color_palette("tab10", len(models_names))
    axes[1, 1].barh(models_names, mse_vals, color=colors_bar)
    axes[1, 1].set_title("Сравнение тестовой ошибки Test MSE моделей", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Среднеквадратичная ошибка (MSE)", fontsize=11)
    for i, val in enumerate(mse_vals):
        axes[1, 1].text(val + 0.05, i, f"{val:.4f}", va="center", fontsize=10, fontweight="bold")
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    plt.suptitle(
        "Комплексный диагностический дашборд валидации регрессионных моделей",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    save_plot(plt.gcf(), "06_model_validation_benchmark.png")
    plt.show()

    return {"df_benchmark": df_benchmark}
