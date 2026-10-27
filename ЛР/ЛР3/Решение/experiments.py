"""
Модуль проведения всех вычислительных экспериментов Лабораторной работы №3:
1. Исследование влияния скорости обучения (Learning Rate eta) на сходимость GD;
2. Сравнительный анализ динамики градиентного (GD) и стохастического (SGD) спуска;
3. Построение регуляризационных путей для 20 значений lambda (L1 для четного вар. 2 и L2);
4. Эталонное бенчмарк-тестирование против моделей Scikit-Learn.
"""

import time
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge, SGDRegressor

from config import (
    GD_PARAMS,
    SGD_PARAMS,
    LEARNING_RATES_EXPERIMENT,
    REGULARIZATION_LAMBDAS,
    RESULTS_DIR,
)
from data_loader import DatasetContainer
from linear_regression import CustomLinearRegression
from visualizer import (
    plot_learning_rate_experiment,
    plot_gd_vs_sgd_convergence,
    plot_regularization_path,
    plot_regularization_comparison,
    plot_validation_benchmark,
)


def run_learning_rate_experiment(data: DatasetContainer) -> Dict[float, Dict[str, Any]]:
    """
    Эксперимент 1: Исследование скорости обучения eta на градиентный спуск.
    """
    print("\n" + "=" * 70)
    print(">>> ЭКСПЕРИМЕНТ 1: Влияние скорости обучения (Learning Rate) на сходимость GD")
    print("=" * 70)

    lr_losses: Dict[float, List[float]] = {}
    summary_rows = []

    for lr in LEARNING_RATES_EXPERIMENT:
        model = CustomLinearRegression(
            method="gd",
            learning_rate=lr,
            max_iter=GD_PARAMS["max_iter"],
            tol=GD_PARAMS["tol"],
        )
        t0 = time.perf_counter()
        model.fit(data.X_train_b, data.y_train)
        elapsed = time.perf_counter() - t0

        test_mse = model.mse(data.X_test_b, data.y_test)
        test_r2 = model.score(data.X_test_b, data.y_test)

        lr_losses[lr] = model.loss_history_
        summary_rows.append(
            {
                "Learning Rate (eta)": lr,
                "Итераций": model.n_iter_,
                "Сошелся (tol=1e-6)": "Да" if model.converged_ else "Нет (max_iter)",
                "Train MSE": round(model.loss_history_[-1], 4),
                "Test MSE": round(test_mse, 4),
                "Test R2": round(test_r2, 4),
                "Время (мс)": round(elapsed * 1000, 2),
            }
        )

    df_lr = pd.DataFrame(summary_rows)
    print(df_lr.to_string(index=False))
    df_lr.to_csv(RESULTS_DIR / "01_learning_rate_results.csv", index=False)

    # Генерация графиков
    plot_learning_rate_experiment(lr_losses, max_steps_to_plot=200)
    return {"summary": df_lr, "loss_histories": lr_losses}


def run_gd_vs_sgd_experiment(data: DatasetContainer) -> Dict[str, Any]:
    """
    Эксперимент 2: Сравнение скорости сходимости GD и SGD.
    """
    print("\n" + "=" * 70)
    print(">>> ЭКСПЕРИМЕНТ 2: Сравнение сходимости Градиентного (GD) и Стохастического (SGD) спуска")
    print("=" * 70)

    # 1. Пакетный градиентный спуск (BGD)
    gd_model = CustomLinearRegression(
        method="gd",
        learning_rate=GD_PARAMS["learning_rate"],
        max_iter=1000,
        tol=GD_PARAMS["tol"],
    )
    t0 = time.perf_counter()
    gd_model.fit(data.X_train_b, data.y_train)
    t_gd = time.perf_counter() - t0

    # 2. Стохастический градиентный спуск (SGD)
    sgd_model = CustomLinearRegression(
        method="sgd",
        learning_rate=SGD_PARAMS["learning_rate"],
        decay_rate=SGD_PARAMS["decay_rate"],
        max_iter=10000,
        tol=SGD_PARAMS["tol"],
    )
    t0 = time.perf_counter()
    sgd_model.fit(data.X_train_b, data.y_train)
    t_sgd = time.perf_counter() - t0

    # Сравнение метрик
    gd_mse = gd_model.mse(data.X_test_b, data.y_test)
    sgd_mse = sgd_model.mse(data.X_test_b, data.y_test)
    gd_r2 = gd_model.score(data.X_test_b, data.y_test)
    sgd_r2 = sgd_model.score(data.X_test_b, data.y_test)

    df_compare = pd.DataFrame(
        [
            {
                "Метод": "BGD (Batch Gradient Descent)",
                "Размер пакета": f"Все N={len(data.y_train)}",
                "Шагов до сост.": gd_model.n_iter_,
                "Train MSE": round(gd_model.loss_history_[-1], 4),
                "Test MSE": round(gd_mse, 4),
                "Test R2": round(gd_r2, 4),
                "Время (мс)": round(t_gd * 1000, 2),
            },
            {
                "Метод": "SGD (Stochastic Gradient)",
                "Размер пакета": "1 объект (случайный)",
                "Шагов до сост.": sgd_model.n_iter_,
                "Train MSE": round(sgd_model.loss_history_[-1], 4),
                "Test MSE": round(sgd_mse, 4),
                "Test R2": round(sgd_r2, 4),
                "Время (мс)": round(t_sgd * 1000, 2),
            },
        ]
    )
    print(df_compare.to_string(index=False))
    df_compare.to_csv(RESULTS_DIR / "02_gd_vs_sgd_results.csv", index=False)

    # Построение графика сходимости на одном координатном поле
    plot_gd_vs_sgd_convergence(
        gd_losses=gd_model.loss_history_,
        sgd_losses=sgd_model.loss_history_,
        n_samples=len(data.y_train),
        max_iter_plot=400,
    )

    return {
        "gd_model": gd_model,
        "sgd_model": sgd_model,
        "summary": df_compare,
    }


def run_regularization_path_experiment(data: DatasetContainer) -> Dict[str, Any]:
    """
    Эксперимент 3: Построение регуляризационных путей для 20 значений lambda.
    Вариант 2 — L1 регуляризация (Lasso). Дополнительно реализуется L2 (Ridge).
    """
    print("\n" + "=" * 70)
    print(">>> ЭКСПЕРИМЕНТ 3: Регуляризация (L1 - Вариант 2 и L2) для 20 значений lambda")
    print(f"Диапазон lambda: от {REGULARIZATION_LAMBDAS[0]:.4f} до {REGULARIZATION_LAMBDAS[-1]:.1f}")
    print("=" * 70)

    l1_weights_list = []
    l2_weights_list = []
    sparsity_list = []

    for lmbd in REGULARIZATION_LAMBDAS:
        # 1. Модель с L1 регуляризацией (Lasso)
        l1_mod = CustomLinearRegression(
            method="l1",
            learning_rate=0.02,
            lmbd=lmbd,
            max_iter=3000,
            tol=1e-6,
            l1_solver="soft_threshold",
        )
        l1_mod.fit(data.X_train_b, data.y_train)
        l1_weights_list.append(l1_mod.coef_)

        # Подсчет разреженности: сколько весов строго равны нулю
        zero_weights_cnt = int(np.sum(np.isclose(l1_mod.coef_, 0.0, atol=1e-5)))
        sparsity_list.append(zero_weights_cnt)

        # 2. Модель с L2 регуляризацией (Ridge)
        l2_mod = CustomLinearRegression(
            method="l2",
            learning_rate=0.02,
            lmbd=lmbd,
            max_iter=3000,
            tol=1e-6,
        )
        l2_mod.fit(data.X_train_b, data.y_train)
        l2_weights_list.append(l2_mod.coef_)

    l1_matrix = np.array(l1_weights_list)  # (20, n_features)
    l2_matrix = np.array(l2_weights_list)  # (20, n_features)

    # Формирование таблицы изменения весов для L1 (по заданию варианта 2)
    df_l1_table = pd.DataFrame(
        l1_matrix,
        columns=data.feature_names,
    )
    df_l1_table.insert(0, "lambda", REGULARIZATION_LAMBDAS)
    df_l1_table.insert(1, "log10(lambda)", np.round(np.log10(REGULARIZATION_LAMBDAS), 2))
    df_l1_table["Занулено признаков"] = sparsity_list
    df_l1_table.to_csv(RESULTS_DIR / "03_l1_weights_by_lambda.csv", index=False)

    df_l2_table = pd.DataFrame(
        l2_matrix,
        columns=data.feature_names,
    )
    df_l2_table.insert(0, "lambda", REGULARIZATION_LAMBDAS)
    df_l2_table.insert(1, "log10(lambda)", np.round(np.log10(REGULARIZATION_LAMBDAS), 2))
    df_l2_table.to_csv(RESULTS_DIR / "04_l2_weights_by_lambda.csv", index=False)

    print("\nТаблица весов L1 (Lasso) для первых 5 и последних 5 значений lambda:")
    print(pd.concat([df_l1_table.head(5), df_l1_table.tail(5)]).to_string(index=False))

    # Построение графиков путей
    plot_regularization_path(
        lambdas=REGULARIZATION_LAMBDAS,
        weights_matrix=l1_matrix,
        feature_names=data.feature_names,
        reg_type="L1 (Lasso) — Четный вариант 2",
        filename="03_l1_regularization_path.png",
    )

    plot_regularization_path(
        lambdas=REGULARIZATION_LAMBDAS,
        weights_matrix=l2_matrix,
        feature_names=data.feature_names,
        reg_type="L2 (Ridge)",
        filename="04_l2_regularization_path.png",
    )

    plot_regularization_comparison(
        lambdas=REGULARIZATION_LAMBDAS,
        l1_weights=l1_matrix,
        l2_weights=l2_matrix,
        feature_names=data.feature_names,
        filename="05_regularization_comparison.png",
    )

    return {
        "l1_matrix": l1_matrix,
        "l2_matrix": l2_matrix,
        "df_l1": df_l1_table,
        "df_l2": df_l2_table,
    }


def run_benchmark_experiment(data: DatasetContainer) -> Dict[str, Any]:
    """
    Эксперимент 4: Сравнительное тестирование реализованных моделей
    против эталонных библиотечных аналогов scikit-learn.
    """
    print("\n" + "=" * 70)
    print(">>> ЭКСПЕРИМЕНТ 4: Эталонное тестирование против Scikit-Learn")
    print("=" * 70)

    # 1. Custom BGD
    custom_gd = CustomLinearRegression(method="gd", learning_rate=0.05, max_iter=3000, tol=1e-6)
    custom_gd.fit(data.X_train_b, data.y_train)

    # 2. Sklearn LinearRegression (МНК - OLS)
    sk_lr = LinearRegression()
    sk_lr.fit(data.X_train_scaled, data.y_train)

    # 3. Custom SGD
    custom_sgd = CustomLinearRegression(
        method="sgd", learning_rate=0.01, decay_rate=0.0001, max_iter=10000, tol=1e-6
    )
    custom_sgd.fit(data.X_train_b, data.y_train)

    # 4. Sklearn SGDRegressor
    sk_sgd = SGDRegressor(loss="squared_error", penalty=None, max_iter=10000, tol=1e-6, random_state=42)
    sk_sgd.fit(data.X_train_scaled, data.y_train)

    # 5. Custom L1 (Lasso, lambda=0.5)
    lmbd_test = 0.5
    custom_l1 = CustomLinearRegression(
        method="l1", learning_rate=0.02, lmbd=lmbd_test, max_iter=3000, tol=1e-6
    )
    custom_l1.fit(data.X_train_b, data.y_train)

    # 6. Sklearn Lasso (alpha = lambda / 2 из-за множителя 1/(2n))
    sk_lasso = Lasso(alpha=lmbd_test / 2.0, max_iter=10000, random_state=42)
    sk_lasso.fit(data.X_train_scaled, data.y_train)

    # 7. Custom L2 (Ridge, lambda=1.0)
    lmbd_l2 = 1.0
    custom_l2 = CustomLinearRegression(
        method="l2", learning_rate=0.02, lmbd=lmbd_l2, max_iter=3000, tol=1e-6
    )
    custom_l2.fit(data.X_train_b, data.y_train)

    # 8. Sklearn Ridge (alpha = N * lambda)
    sk_ridge = Ridge(alpha=len(data.y_train) * lmbd_l2, max_iter=10000, random_state=42)
    sk_ridge.fit(data.X_train_scaled, data.y_train)

    # Сбор метрик в итоговую таблицу
    models_dict = {
        "Custom BGD": (custom_gd.predict(data.X_test_b), custom_gd.intercept_, custom_gd.coef_),
        "Sklearn OLS": (sk_lr.predict(data.X_test_scaled), sk_lr.intercept_, sk_lr.coef_),
        "Custom SGD": (custom_sgd.predict(data.X_test_b), custom_sgd.intercept_, custom_sgd.coef_),
        "Sklearn SGD": (sk_sgd.predict(data.X_test_scaled), float(sk_sgd.intercept_[0]), sk_sgd.coef_),
        f"Custom L1 (λ={lmbd_test})": (custom_l1.predict(data.X_test_b), custom_l1.intercept_, custom_l1.coef_),
        f"Sklearn Lasso (α={lmbd_test/2})": (sk_lasso.predict(data.X_test_scaled), float(sk_lasso.intercept_), sk_lasso.coef_),
        f"Custom L2 (λ={lmbd_l2})": (custom_l2.predict(data.X_test_b), custom_l2.intercept_, custom_l2.coef_),
        f"Sklearn Ridge (α={len(data.y_train)*lmbd_l2})": (sk_ridge.predict(data.X_test_scaled), float(sk_ridge.intercept_), sk_ridge.coef_),
    }

    benchmark_rows = []
    y_test = data.y_test

    for name, (y_pred, intercept, coef) in models_dict.items():
        mse = float(np.mean((y_test - y_pred) ** 2))
        mae = float(np.mean(np.abs(y_test - y_pred)))
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = float(1.0 - ss_res / ss_tot)
        norm_w = float(np.linalg.norm(coef))

        benchmark_rows.append(
            {
                "Модель": name,
                "Intercept (w0)": round(intercept, 4),
                "Норма весов ||w||": round(norm_w, 4),
                "Test MSE": round(mse, 4),
                "Test MAE": round(mae, 4),
                "Test R2": round(r2, 4),
            }
        )

    df_bench = pd.DataFrame(benchmark_rows)
    print(df_bench.to_string(index=False))
    df_bench.to_csv(RESULTS_DIR / "05_benchmark_comparison.csv", index=False)

    # Построение графика прогнозов и остатков
    plot_validation_benchmark(
        y_test=data.y_test,
        y_pred_gd=custom_gd.predict(data.X_test_b),
        y_pred_sklearn=sk_lr.predict(data.X_test_scaled),
        filename="06_model_validation_benchmark.png",
    )

    return {
        "benchmark_df": df_bench,
        "custom_gd": custom_gd,
        "sklearn_lr": sk_lr,
    }
