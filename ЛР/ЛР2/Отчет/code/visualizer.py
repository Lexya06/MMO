# -*- coding: utf-8 -*-
"""
Модуль расширенной графической визуализации данных.
Создает информативные, наглядные и эстетичные диаграммы с аннотациями,
статистическими сводками и поддержкой кириллицы.
"""

from typing import Dict, List, Optional, Tuple
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")

# Глобальная настройка типографики и стилей
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font="DejaVu Sans")


def plot_missing_heatmap(df: pd.DataFrame, title: str, save_path: str) -> None:
    """
    Строит наглядную тепловую карту пропущенных значений.
    """
    plt.figure(figsize=(14, 6))
    missing_sum = int(df.isnull().sum().sum())

    cmap = sns.color_palette(["#1f77b4", "#d62728"], as_cmap=True)
    ax = sns.heatmap(
        df.isnull(),
        cbar=True,
        yticklabels=False,
        cmap=cmap,
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Наличие пропуска (0 — нет, 1 — да)", "ticks": [0, 1]},
    )

    # Настройка подписей шкалы (colorbar)
    cbar = ax.collections[0].colorbar
    if cbar:
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["0 (Нет)", "1 (Пропуск)"])

    plt.title(
        f"{title}\n(Всего записей: {len(df):,}, суммарно пропусков: {missing_sum})",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    plt.xlabel("Признаки датасета", fontsize=12, fontweight="bold", labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=10)

    # Если пропусков 0, размещаем информационный бейдж по центру
    if missing_sum == 0:
        ax.text(
            0.5,
            0.5,
            "✓ ПРОПУСКОВ НЕТ (100% заполненность выборки)",
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            color="white",
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.8",
                facecolor="#2ca02c",
                edgecolor="none",
                alpha=0.9,
            ),
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_missing_bar(
    missing_table: pd.DataFrame,
    total_rows: int,
    save_path: str,
    imputed_modes: Optional[Dict[str, str]] = None,
) -> None:
    """
    Строит эстетичную горизонтальную диаграмму пропусков с точными числами,
    долями и планируемыми модальными значениями импутации.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    palette = sns.color_palette("rocket", n_colors=len(missing_table))
    y_pos = np.arange(len(missing_table))
    counts = missing_table["Пропусков (шт.)"].values

    bars = ax.barh(
        y_pos, counts, color=palette, height=0.55, edgecolor="black", linewidth=0.8
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(missing_table.index, fontsize=11, fontweight="bold")
    ax.invert_yaxis()  # Самый частый пропуск вверху

    ax.set_xlabel("Количество пропущенных записей", fontsize=11, fontweight="bold")
    ax.set_title(
        "Количество и доля пропущенных значений по признакам\n"
        f"(Всего объектов выборки: {total_rows})",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )

    # Добавляем текстовые аннотации к каждому столбцу
    for i, bar in enumerate(bars):
        w = bar.get_width()
        pct = (w / total_rows) * 100
        col_name = missing_table.index[i]
        mode_str = (
            f" | Мода: '{imputed_modes[col_name]}'"
            if (imputed_modes and col_name in imputed_modes)
            else ""
        )
        label_text = f" {int(w)} шт. ({pct:.2f}%){mode_str}"
        ax.text(
            w + 1.5,
            bar.get_y() + bar.get_height() / 2.0,
            label_text,
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
            color="#2c3e50",
        )

    ax.set_xlim(0, max(counts) * 1.35)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_distributions(
    df: pd.DataFrame,
    columns: List[str],
    title: str,
    save_path: str,
    color: str = "royalblue",
) -> None:
    """
    Строит сетку гистограмм с KDE, линиями среднего и медианы,
    а также карточками ключевых статистик для каждого признака.
    """
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(16, 12))
    axes_flat = axes.flatten()

    for i, col in enumerate(columns):
        ax = axes_flat[i]
        data = df[col]
        mean_val = float(data.mean())
        median_val = float(data.median())
        std_val = float(data.std())
        skew_val = float(data.skew())

        # Гистограмма с KDE
        sns.histplot(
            data, kde=True, ax=ax, color=color, bins=20, edgecolor="white", alpha=0.65
        )

        # Вертикальные линии для Mean и Median
        ax.axvline(
            mean_val, color="#d62728", linestyle="--", linewidth=1.8, label=f"Mean: {
                mean_val:.1f}"
        )
        ax.axvline(
            median_val, color="#2ca02c", linestyle=":", linewidth=2.0, label=f"Median: {
                median_val:.1f}"
        )

        ax.set_title(f"Признак: {col}", fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel("Частота", fontsize=10)
        ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

        # Плашка со статистиками
        stats_text = f"Std: {std_val:.2f}\nSkew: {skew_val:+.2f}"
        ax.text(
            0.04,
            0.93,
            stats_text,
            transform=ax.transAxes,
            fontsize=8.5,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="white",
                edgecolor="gray",
                alpha=0.85,
            ),
        )

    # Удаление неиспользуемых осей
    for j in range(len(columns), len(axes_flat)):
        fig.delaxes(axes_flat[j])

    plt.suptitle(title, fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_boxplots(
    df: pd.DataFrame,
    columns: List[str],
    title: str,
    save_path: str,
    color: str = "salmon",
    bounds_dict: Optional[Dict[str, Tuple[float, float]]] = None,
) -> None:
    """
    Строит сетку BoxPlot с нанесением теоретических границ 1.5*IQR,
    маркерами среднего значения и подсчетом количества выбросов.
    """
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(18, 9))
    axes_flat = axes.flatten()

    for i, col in enumerate(columns):
        ax = axes_flat[i]
        data = df[col]

        sns.boxplot(
            y=data,
            ax=ax,
            color=color,
            showmeans=True,
            meanprops=dict(
                marker="D",
                markeredgecolor="black",
                markerfacecolor="yellow",
                markersize=6,
            ),
            flierprops=dict(
                marker="o",
                markerfacecolor="#e74c3c",
                markeredgecolor="none",
                markersize=5,
                alpha=0.8,
            ),
        )

        ax.set_title(f"BoxPlot: {col}", fontsize=12, fontweight="bold", pad=8)
        ax.set_ylabel("Значение", fontsize=10)

        # Если переданы границы выбросов, наносим их пунктиром
        if bounds_dict and col in bounds_dict:
            low, high = bounds_dict[col]
            outliers_cnt = int(((data < low) | (data > high)).sum())

            # Линии границ
            ax.axhline(low, color="#c0392b", linestyle="--", linewidth=1.2, alpha=0.7)
            ax.axhline(high, color="#c0392b", linestyle="--", linewidth=1.2, alpha=0.7)

            status_text = (
                f"Выбросов: {outliers_cnt}" if outliers_cnt > 0 else "Выбросов нет"
            )
            badge_color = "#f9d5d5" if outliers_cnt > 0 else "#d5f9d5"
            text_color = "#900c3f" if outliers_cnt > 0 else "#1e8449"

            ax.text(
                0.05,
                0.94,
                status_text,
                transform=ax.transAxes,
                fontsize=9,
                fontweight="bold",
                verticalalignment="top",
                color=text_color,
                bbox=dict(
                    boxstyle="round,pad=0.35",
                    facecolor=badge_color,
                    edgecolor=text_color,
                    alpha=0.9,
                ),
            )

            # Стрелка к явной ошибке 101 в Exam_Score
            if col == "Exam_Score" and (data > 100).any():
                ax.annotate(
                    "Аномалия: 101 (>100)",
                    xy=(0, 101),
                    xytext=(0.25, 95),
                    arrowprops=dict(
                        facecolor="red", shrink=0.08, width=1.5, headwidth=6
                    ),
                    fontsize=9,
                    fontweight="bold",
                    color="red",
                )
        else:
            # Для постобработки
            ax.text(
                0.05,
                0.94,
                "✓ Выбросы устранены",
                transform=ax.transAxes,
                fontsize=9,
                fontweight="bold",
                verticalalignment="top",
                color="#1e8449",
                bbox=dict(
                    boxstyle="round,pad=0.35",
                    facecolor="#d5f9d5",
                    edgecolor="#1e8449",
                    alpha=0.9,
                ),
            )

    for j in range(len(columns), len(axes_flat)):
        fig.delaxes(axes_flat[j])

    plt.suptitle(title, fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_correlation_matrix(
    df: pd.DataFrame, columns: List[str], save_path: str
) -> None:
    """
    Строит информативную треугольную тепловую карту корреляции Пирсона,
    отсортированную по силе связи с целевой переменной Exam_Score.
    """
    # Сортируем столбцы по убыванию корреляции с целевой переменной Exam_Score
    corr_full = df[columns].corr()
    if "Exam_Score" in corr_full.columns:
        sorted_cols = (
            corr_full["Exam_Score"].sort_values(ascending=False).index.tolist()
        )
        corr = df[sorted_cols].corr()
    else:
        corr = corr_full

    plt.figure(figsize=(10, 8))
    # Маска для верхнего треугольника (убираем дублирование)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.8,
        cbar_kws={"shrink": 0.8, "label": "Коэффициент корреляции Пирсона (r)"},
        annot_kws={"size": 10, "weight": "bold"},
    )

    plt.title(
        "Матрица корреляций числовых признаков (Пирсон)\n"
        "(Отсортировано по силе связи с Exam_Score)",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_feature_engineering(
    df: pd.DataFrame, feature_names: List[str], save_path: str
) -> None:
    """
    Визуализирует распределения и аналитическую ценность новых
    сгенерированных признаков: сравнение с экзаменационными баллами
    и физический смысл.
    """
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(17, 10))
    axes_flat = axes.flatten()

    descriptions = {
        "Study_Sleep_Ratio": (
            "Отношение часов учебы к часам сна\n(баланс нагрузки и отдыха)"
        ),
        "Total_Study_Effort": (
            "Суммарный индекс учебных усилий\n"
            "(учеба + 2*репетитор + 0.1*посещаемость)"
        ),
        "Score_Improvement": (
            "Прирост баллов относительно прошлых тестов\n"
            "(Exam_Score - Previous_Scores)"
        ),
        "Physical_Study_Balance": (
            "Баланс физической и умственной активности\n"
            "(Physical / (Study + 1))"
        ),
        "High_Attendance_Flag": (
            "Сравнение среднего балла экзамена:\nПосещаемость ≥85% vs <85%"
        ),
    }

    for i, col in enumerate(feature_names):
        ax = axes_flat[i]
        desc = descriptions.get(col, "")

        if col == "High_Attendance_Flag":
            # Сравниваем средний балл экзамена для двух групп посещаемости!
            group_scores = df.groupby(col)["Exam_Score"].agg(["mean", "count"])
            labels = ["Низкая (<85%)", "Высокая (≥85%)"]
            colors = ["#e67e22", "#27ae60"]

            bars = ax.bar(
                labels, group_scores["mean"], color=colors, width=0.5, edgecolor="black"
            )
            ax.set_ylabel(
                "Средний балл экзамена (Exam_Score)", fontsize=10, fontweight="bold"
            )
            ax.set_ylim(50, 78)

            for bar, (_, row) in zip(bars, group_scores.iterrows()):
                h = bar.get_height()
                cnt = int(row["count"])
                ax.annotate(
                    f"{h:.2f} б.\n({cnt} студ.)",
                    (bar.get_x() + bar.get_width() / 2.0, h),
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                    xytext=(0, 3),
                    textcoords="offset points",
                )

            ax.set_title(f"{col}\n{desc}", fontsize=11, fontweight="bold", pad=8)

        elif col == "Score_Improvement":
            # Дивергентная гистограмма: зеленый для прогресса (>0), красный для
            # регресса (<0)
            data = df[col]
            sns.histplot(
                data, kde=True, ax=ax, color="#2980b9", bins=25, edgecolor="white"
            )
            ax.axvline(
                0, color="red", linestyle="--", linewidth=2.0, label="Нулевой прирост"
            )
            ax.set_title(f"{col}\n{desc}", fontsize=11, fontweight="bold", pad=8)
            ax.set_xlabel(col, fontsize=10)
            ax.set_ylabel("Количество студентов", fontsize=10)
            ax.legend(loc="upper right", fontsize=9)

        else:
            data = df[col]
            sns.histplot(
                data, kde=True, ax=ax, color="#8e44ad", bins=25, edgecolor="white"
            )
            m = float(data.mean())
            ax.axvline(
                m, color="#f39c12", linestyle="--", linewidth=2.0, label=f"Среднее: {
                    m:.2f}"
            )
            ax.set_title(f"{col}\n{desc}", fontsize=11, fontweight="bold", pad=8)
            ax.set_xlabel(col, fontsize=10)
            ax.set_ylabel("Плотность / Частота", fontsize=10)
            ax.legend(loc="upper right", fontsize=9)

    fig.delaxes(axes_flat[5])
    plt.suptitle(
        "Аналитический профиль сгенерированных признаков (Feature Engineering)",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_scaling_comparison(
    df_orig: pd.DataFrame,
    df_minmax: pd.DataFrame,
    df_std: pd.DataFrame,
    demo_cols: List[str],
    save_path: str,
) -> None:
    """
    Строит наглядную матрицу 3х4 для сравнения исходных и масштабированных данных
    с нанесением параметров формы, границ и зон стандартного отклонения.
    """
    fig, axes = plt.subplots(nrows=3, ncols=len(demo_cols), figsize=(18, 12))

    for i, col in enumerate(demo_cols):
        # 1. Исходные
        orig_data = df_orig[col]
        sns.kdeplot(
            orig_data,
            ax=axes[0, i],
            color="#2980b9",
            fill=True,
            alpha=0.35,
            linewidth=2.2,
        )
        axes[0, i].set_title(f"Исходный: {col}", fontsize=11, fontweight="bold")
        axes[0, i].set_xlabel("Исходное значение", fontsize=9.5)
        axes[0, i].set_ylabel("Плотность", fontsize=9.5)
        stats_orig = f"μ = {
            orig_data.mean():.2f}\nσ = {
            orig_data.std():.2f}\n[{
            orig_data.min():.1f}, {
                orig_data.max():.1f}]"
        axes[0, i].text(
            0.05,
            0.92,
            stats_orig,
            transform=axes[0, i].transAxes,
            fontsize=8.5,
            va="top",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="#2980b9",
                alpha=0.85,
            ),
        )

        # 2. MinMaxScaler
        mm_data = df_minmax[col]
        sns.kdeplot(
            mm_data,
            ax=axes[1, i],
            color="#e67e22",
            fill=True,
            alpha=0.35,
            linewidth=2.2,
        )
        axes[1, i].set_title(f"MinMax: {col} [0, 1]", fontsize=11, fontweight="bold")
        axes[1, i].set_xlabel("Нормализованное значение", fontsize=9.5)
        axes[1, i].set_ylabel("Плотность", fontsize=9.5)
        stats_mm = f"μ = {mm_data.mean():.2f}\nσ = {mm_data.std():.2f}\n[0.00, 1.00]"
        axes[1, i].text(
            0.05,
            0.92,
            stats_mm,
            transform=axes[1, i].transAxes,
            fontsize=8.5,
            va="top",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="#e67e22",
                alpha=0.85,
            ),
        )

        # 3. StandardScaler
        std_data = df_std[col]
        sns.kdeplot(
            std_data,
            ax=axes[2, i],
            color="#27ae60",
            fill=True,
            alpha=0.35,
            linewidth=2.2,
        )
        axes[2, i].set_title(
            f"Standard: {col} (Z-score)", fontsize=11, fontweight="bold"
        )
        axes[2, i].set_xlabel("Стандартизированное значение (Z)", fontsize=9.5)
        axes[2, i].set_ylabel("Плотность", fontsize=9.5)
        axes[2, i].axvline(0, color="gray", linestyle=":", linewidth=1.2)
        axes[2, i].axvline(
            -1, color="#27ae60", linestyle="--", linewidth=0.9, alpha=0.6
        )
        axes[2, i].axvline(
            +1, color="#27ae60", linestyle="--", linewidth=0.9, alpha=0.6
        )
        stats_std = f"μ = {
            std_data.mean():.2f} ≈ 0\nσ = {
            std_data.std():.2f} = 1\n[{
            std_data.min():.2f}, {
                std_data.max():.2f}]"
        axes[2, i].text(
            0.05,
            0.92,
            stats_std,
            transform=axes[2, i].transAxes,
            fontsize=8.5,
            va="top",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="#27ae60",
                alpha=0.85,
            ),
        )

    plt.suptitle(
        "Сравнительный анализ трансформаций признакового пространства\n"
        "(Сохранение геометрии распределений при различных масштабах)",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
