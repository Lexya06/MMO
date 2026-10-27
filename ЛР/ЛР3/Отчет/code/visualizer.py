"""
Модуль визуализации публикационного качества (300 DPI) для Лабораторной работы №3.
Создает все графические иллюстрации, требуемые кафедрой ПОИТ:
- Эксперимент с коэффициентом скорости обучения (Learning Rate);
- Совмещенный график сходимости GD vs SGD;
- Регуляризационные пути L1 (Lasso) и L2 (Ridge) для 20 значений lambda;
- Сравнительный анализ разреживания весов L1 vs L2;
- Валидационные графики (прогноз vs факт, анализ остатков).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from config import IMAGES_DIR, REPORT_IMAGES_DIR, PLOT_CONFIG

# Настройка единого академического стиля визуализации
sns.set_theme(style=PLOT_CONFIG["style"], font_scale=PLOT_CONFIG["font_scale"])
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#2c3e50"
plt.rcParams["axes.linewidth"] = 1.0


def _save_figure(fig: plt.Figure, filename: str) -> None:
    """Сохраняет фигуру в высоком разрешении в папки Решение/images и Отчет/images."""
    fig.tight_layout()
    for output_dir in [IMAGES_DIR, REPORT_IMAGES_DIR]:
        output_path = output_dir / filename
        fig.savefig(output_path, dpi=PLOT_CONFIG["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  [+] Сохранен график: {filename}")


def plot_learning_rate_experiment(
    results: Dict[float, List[float]],
    max_steps_to_plot: int = 150,
    filename: str = "01_learning_rate_experiment.png",
) -> None:
    """
    Строит графики динамики MSE при различных скоростях обучения (eta).
    Две панели: линейная шкала и логарифмическая шкала semilogy.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    colors = sns.color_palette(PLOT_CONFIG["palette"], len(results))

    for (lr, loss_hist), color in zip(results.items(), colors):
        steps = np.arange(len(loss_hist[:max_steps_to_plot]))
        losses = np.array(loss_hist[:max_steps_to_plot])

        label = f"$\\eta = {lr}$"
        ax1.plot(steps, losses, label=label, color=color, linewidth=2.0)
        ax2.semilogy(steps, np.maximum(losses, 1e-12), label=label, color=color, linewidth=2.0)

    # Настройки линейного графика
    ax1.set_title("Сходимость MSE при различных $\\eta$ (Линейная шкала)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Номер итерации $k$", fontsize=11)
    ax1.set_ylabel("Среднеквадратичная ошибка $Q(w^k)$ (MSE)", fontsize=11)
    ax1.set_ylim(bottom=0, top=min(150, max(ax1.get_ylim()[1], 50)))
    ax1.legend(loc="upper right", frameon=True, framealpha=0.9)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Настройки логарифмического графика
    ax2.set_title("Скорость сходимости $\\log(MSE)$ (Semilogy шкала)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Номер итерации $k$", fontsize=11)
    ax2.set_ylabel("$\\log_{10}(Q(w^k))$", fontsize=11)
    ax2.legend(loc="upper right", frameon=True, framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.6)

    _save_figure(fig, filename)


def plot_gd_vs_sgd_convergence(
    gd_losses: List[float],
    sgd_losses: List[float],
    n_samples: int,
    max_iter_plot: int = 400,
    filename: str = "02_gd_vs_sgd_convergence.png",
) -> None:
    """
    Строит сравнительный график среднеквадратичной ошибки для GD и SGD
    на одном координатном поле в соответствии с заданием кафедры.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    k_gd = np.arange(len(gd_losses[:max_iter_plot]))
    loss_gd = np.array(gd_losses[:max_iter_plot])

    k_sgd = np.arange(len(sgd_losses[:max_iter_plot]))
    loss_sgd = np.array(sgd_losses[:max_iter_plot])

    # Панель 1: Сравнение по числу итераций (градиентных шагов)
    ax1.plot(k_gd, loss_gd, label="BGD (Пакетный градиентный спуск, $N=5285$)", color="#1f77b4", linewidth=2.5)
    ax1.plot(
        k_sgd,
        loss_sgd,
        label="SGD (Стохастический спуск, 1 объект)",
        color="#ff7f0e",
        linewidth=1.8,
        alpha=0.85,
    )

    ax1.set_title("Сравнение сходимости: GD vs SGD (по числу итераций)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Итерация $k$ (число градиентных шагов)", fontsize=11)
    ax1.set_ylabel("Среднеквадратичная ошибка (MSE)", fontsize=11)
    ax1.set_yscale("log")
    ax1.legend(loc="upper right", frameon=True, framealpha=0.9)
    ax1.grid(True, which="both", linestyle="--", alpha=0.6)

    # Панель 2: Сравнение по эффективным эпохам (проходам по данным)
    # Для GD 1 шаг = 1 эпоха (N вычислений градиента). Для SGD N шагов = 1 эпоха.
    epochs_gd = np.arange(len(gd_losses))
    epochs_sgd = np.arange(len(sgd_losses)) / float(n_samples)

    ax2.plot(epochs_gd[:50], gd_losses[:50], label="BGD (1 шаг = 1 эпоха)", color="#1f77b4", linewidth=2.5)
    ax2.plot(
        epochs_sgd,
        sgd_losses,
        label="SGD ($N$ шагов = 1 эпоха)",
        color="#2ca02c",
        linewidth=1.8,
        alpha=0.85,
    )

    ax2.set_title("Сходимость в зависимости от числа эпох (проходов по датасету)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Эффективные эпохи (число проходов $N$ объектов)", fontsize=11)
    ax2.set_ylabel("Среднеквадратичная ошибка (MSE)", fontsize=11)
    ax2.set_yscale("log")
    ax2.set_xlim(-0.05, 5.0)
    ax2.legend(loc="upper right", frameon=True, framealpha=0.9)
    ax2.grid(True, which="both", linestyle="--", alpha=0.6)

    _save_figure(fig, filename)


def plot_regularization_path(
    lambdas: np.ndarray,
    weights_matrix: np.ndarray,
    feature_names: List[str],
    reg_type: str = "L1 (Lasso)",
    filename: str = "03_l1_regularization_path.png",
) -> None:
    """
    Строит регуляризационный путь: динамику весов признаков w_j
    в зависимости от коэффициента регуляризации lambda (в log-шкале).
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = sns.color_palette("tab10", len(feature_names))
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]

    log_lambdas = np.log10(lambdas)

    for j, (fname, color) in enumerate(zip(feature_names, colors)):
        marker = markers[j % len(markers)]
        weights_j = weights_matrix[:, j]
        ax.plot(
            log_lambdas,
            weights_j,
            label=f"{fname}",
            color=color,
            linewidth=2.2,
            marker=marker,
            markersize=6,
            markevery=1,
        )

    # Опорная горизонтальная линия нуля
    ax.axhline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.7)

    title_suffix = "Зануление признаков (Sparsity)" if "L1" in reg_type else "Асимптотическое сжатие (Shrinkage)"
    ax.set_title(
        f"Регуляризационный путь {reg_type} для 20 значений $\\lambda$: {title_suffix}",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("$\\log_{10}(\\lambda)$ (коэффициент регуляризации от $10^{-3}$ до $10^3$)", fontsize=11)
    ax.set_ylabel("Значение весового коэффициента $w_j$", fontsize=11)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, title="Признаки модели")
    ax.grid(True, linestyle="--", alpha=0.6)

    _save_figure(fig, filename)


def plot_regularization_comparison(
    lambdas: np.ndarray,
    l1_weights: np.ndarray,
    l2_weights: np.ndarray,
    feature_names: List[str],
    filename: str = "05_regularization_comparison.png",
) -> None:
    """
    Строит двухпанельное сравнение L1 (Lasso) vs L2 (Ridge):
    демонстрирует ключевую разницу между разреживанием весов и сжатием.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    log_lambdas = np.log10(lambdas)
    colors = sns.color_palette("tab10", len(feature_names))

    for j, (fname, color) in enumerate(zip(feature_names, colors)):
        ax1.plot(log_lambdas, l1_weights[:, j], label=fname, color=color, linewidth=2.0)
        ax2.plot(log_lambdas, l2_weights[:, j], label=fname, color=color, linewidth=2.0)

    ax1.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax2.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)

    ax1.set_title("L1 Регуляризация (Lasso): Отбор признаков (Sparsity)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("$\\log_{10}(\\lambda)$", fontsize=11)
    ax1.set_ylabel("Веса $w_j$", fontsize=11)
    ax1.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.set_title("L2 Регуляризация (Ridge): Плавное сжатие (Shrinkage)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("$\\log_{10}(\\lambda)$", fontsize=11)
    ax2.set_ylabel("Веса $w_j$", fontsize=11)
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.6)

    _save_figure(fig, filename)


def plot_validation_benchmark(
    y_test: np.ndarray,
    y_pred_gd: np.ndarray,
    y_pred_sklearn: np.ndarray,
    filename: str = "06_model_validation_benchmark.png",
) -> None:
    """
    Строит график соответствия предсказаний Custom GD фактическим значениям
    и распределение остатков в сравнении со Scikit-Learn.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Панель 1: Диаграмма рассеяния "Прогноз vs Факт"
    ax1.scatter(y_test, y_pred_gd, alpha=0.35, color="#1f77b4", edgecolor="none", s=25, label="Custom BGD")
    min_val = min(y_test.min(), y_pred_gd.min()) - 2
    max_val = max(y_test.max(), y_pred_gd.max()) + 2
    ax1.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=2.0, label="Идеальный прогноз ($y = \\hat{y}$)")

    ax1.set_title("Диаграмма соответствия: Прогноз vs Истинные значения", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Истинный балл за экзамен ($y_{true}$)", fontsize=11)
    ax1.set_ylabel("Предсказанный балл ($\\hat{y}$)", fontsize=11)
    ax1.set_xlim(min_val, max_val)
    ax1.set_ylim(min_val, max_val)
    ax1.legend(loc="upper left", frameon=True, framealpha=0.9)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Панель 2: График плотности распределения остатков (Errors)
    residuals_custom = y_test - y_pred_gd
    residuals_sklearn = y_test - y_pred_sklearn

    sns.kdeplot(residuals_custom, ax=ax2, color="#1f77b4", fill=True, alpha=0.3, linewidth=2.0, label="Остатки Custom GD")
    sns.kdeplot(residuals_sklearn, ax=ax2, color="#ff7f0e", linestyle="--", linewidth=2.0, label="Остатки Scikit-Learn")

    ax2.axvline(0, color="red", linestyle=":", linewidth=1.5)
    ax2.set_title("Плотность распределения остатков (KDE Residuals)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Величина ошибки $(y_{true} - \\hat{y})$", fontsize=11)
    ax2.set_ylabel("Плотность вероятности", fontsize=11)
    ax2.legend(loc="upper right", frameon=True, framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.6)

    _save_figure(fig, filename)
