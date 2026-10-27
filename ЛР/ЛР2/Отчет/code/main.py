# -*- coding: utf-8 -*-
"""
Главный исполняемый модуль лабораторной работы № 2.
Координирует выполнение всех этапов разведочного и предиктивного анализа:
1. Загрузка и первичный осмотр данных
2. Анализ и импутация пропусков, винсоризация выбросов
3. Инженерия признаков и кодирование категориальных переменных
4. Нормализация и стандартизация признаков
5. Построение графиков и экспорт обработанных данных
"""

import os
import shutil
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from config import (  # noqa: E402
    BINARY_MAPPINGS,
    CONTINUOUS_COLUMNS_TO_SCALE,
    DATASET_PATH,
    DEMO_COLUMNS,
    IMAGES_DIR,
    OHE_COLUMNS,
    ORDINAL_MAPPINGS,
    PROCESSED_DATA_DIR,
    REPORT_IMAGES_DIR,
    REPORT_OVERLEAF_IMAGES_DIR
)
from data_loader import (  # noqa: E402
    get_column_types,
    get_numeric_summary,
    load_dataset
)
from feature_engineering import (  # noqa: E402
    encode_binary_features,
    encode_nominal_features,
    encode_ordinal_features,
    generate_new_features
)
from preprocessor import (  # noqa: E402
    calculate_iqr_bounds,
    get_missing_summary,
    handle_outliers,
    impute_missing_with_mode
)
from scaler import (  # noqa: E402
    apply_minmax_scaling,
    apply_standard_scaling,
    compare_scaling_stats
)
from visualizer import (  # noqa: E402
    plot_boxplots,
    plot_correlation_matrix,
    plot_distributions,
    plot_feature_engineering,
    plot_missing_bar,
    plot_missing_heatmap,
    plot_scaling_comparison
)


def print_banner(title: str) -> None:
    """Выводит форматированный заголовок раздела."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def copy_images_to_reports() -> None:
    """Копирует сгенерированные графики в каталоги отчетов."""
    os.makedirs(REPORT_IMAGES_DIR, exist_ok=True)
    os.makedirs(REPORT_OVERLEAF_IMAGES_DIR, exist_ok=True)

    for img_name in os.listdir(IMAGES_DIR):
        if img_name.endswith('.png'):
            src = os.path.join(IMAGES_DIR, img_name)
            shutil.copy2(src, os.path.join(REPORT_IMAGES_DIR, img_name))
            shutil.copy2(
                src, os.path.join(REPORT_OVERLEAF_IMAGES_DIR, img_name)
            )


def main() -> None:
    """Точка входа пайплайна предварительной обработки данных."""
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # --------------------------------------------------------------------------
    # ЭТАП 1: ИМПОРТ И ПЕРВИЧНЫЙ АНАЛИЗ ДАТАСЕТА
    # --------------------------------------------------------------------------
    print_banner("ЭТАП 1: ИМПОРТ И ПЕРВИЧНЫЙ АНАЛИЗ ДАТАСЕТА")
    print(f"Путь к файлу: {DATASET_PATH}")
    df_raw = load_dataset(DATASET_PATH)
    total_rows, total_cols = df_raw.shape
    print(f"Размерность выборки: {total_rows} строк, {total_cols} столбцов.\n")

    numeric_cols, categorical_cols = get_column_types(df_raw)
    print(f"Числовые признаки ({len(numeric_cols)}): {numeric_cols}")
    print(f"Категориальные признаки ({len(categorical_cols)}): {categorical_cols}\n")

    print("Описательные статистики числовых признаков:")
    print(get_numeric_summary(df_raw, numeric_cols))

    # --------------------------------------------------------------------------
    # ЭТАП 2.1: АНАЛИЗ И ЗАПОЛНЕНИЕ ПРОПУСКОВ
    # --------------------------------------------------------------------------
    print_banner("ЭТАП 2.1: АНАЛИЗ И ОБРАБОТКА ПРОПУСКОВ (MISSING VALUES)")
    missing_summary = get_missing_summary(df_raw)
    print("Признаки с пропущенными значениями:")
    print(missing_summary)

    # График 1: Тепловая карта до очистки
    plot_missing_heatmap(
        df_raw,
        "Тепловая карта пропущенных значений (ДО заполнения)",
        os.path.join(IMAGES_DIR, '01_missing_values_before_heatmap.png')
    )

    # Заполнение пропусков модой
    df_imputed, imputed_modes = impute_missing_with_mode(
        df_raw, missing_summary.index.tolist()
    )
    print("\nЗаполнение пропусков наиболее частым значением (модой):")
    for col, mode_val in imputed_modes.items():
        print(f"  - Признак '{col}': заполнено модой '{mode_val}'")

    print(
        f"Осталось пропусков в датасете: {df_imputed.isnull().sum().sum()}"
    )

    # График 2: Столбчатая диаграмма количества пропусков
    plot_missing_bar(
        missing_summary,
        total_rows,
        os.path.join(IMAGES_DIR, '02_missing_values_bar.png'),
        imputed_modes=imputed_modes
    )

    # График 3: Тепловая карта после очистки
    plot_missing_heatmap(
        df_imputed,
        "Тепловая карта пропущенных значений (ПОСЛЕ заполнения)",
        os.path.join(IMAGES_DIR, '03_missing_values_after_heatmap.png')
    )

    # --------------------------------------------------------------------------
    # ЭТАП 2.2: ВЫЯВЛЕНИЕ И ОБРАБОТКА ВЫБРОСОВ
    # --------------------------------------------------------------------------
    print_banner("ЭТАП 2.2: ВЫЯВЛЕНИЕ И УСТРАНЕНИЕ ВЫБРОСОВ (OUTLIERS)")
    iqr_table, bounds_dict = calculate_iqr_bounds(df_imputed, numeric_cols)
    print("Статистика выбросов по методу межквартильного размаха (1.5 * IQR):")
    cols_to_print = [
        'Нижняя граница', 'Верхняя граница', 'Мин. значение',
        'Макс. значение', 'Кол-во выбросов', 'Процент (%)'
    ]
    print(iqr_table[cols_to_print])

    # График 4: Распределения до устранения выбросов
    plot_distributions(
        df_imputed, numeric_cols,
        "Гистограммы и KDE числовых признаков (ДО устранения выбросов)",
        os.path.join(IMAGES_DIR, '04_distributions_before.png'),
        color='royalblue'
    )

    # График 5: BoxPlot до устранения выбросов
    plot_boxplots(
        df_imputed, numeric_cols,
        "Диаграммы размаха BoxPlot (ДО устранения выбросов)",
        os.path.join(IMAGES_DIR, '05_boxplots_before.png'),
        color='salmon',
        bounds_dict=bounds_dict
    )

    # Обработка выбросов и логических аномалий
    features_with_outliers = ['Hours_Studied', 'Exam_Score', 'Tutoring_Sessions']
    print(
        "\nПрименение винсоризации (clipping) к границам "
        "[Q1 - 1.5*IQR, Q3 + 1.5*IQR]:"
    )
    for feat in features_with_outliers:
        low, high = bounds_dict[feat]
        low = max(low, 0.0)
        if feat == 'Exam_Score':
            high = min(high, 100.0)
        print(f"  - Признак '{feat}': ограничение в диапазон [{low}, {high}]")

    df_cleaned = handle_outliers(
        df_imputed, bounds_dict, features_with_outliers
    )

    # График 6: BoxPlot после устранения выбросов
    plot_boxplots(
        df_cleaned, numeric_cols,
        "Диаграммы размаха BoxPlot (ПОСЛЕ устранения выбросов)",
        os.path.join(IMAGES_DIR, '06_boxplots_after.png'),
        color='mediumseagreen'
    )

    # График 7: Распределения после устранения выбросов
    plot_distributions(
        df_cleaned, numeric_cols,
        "Гистограммы и KDE числовых признаков (ПОСЛЕ устранения выбросов)",
        os.path.join(IMAGES_DIR, '07_distributions_after.png'),
        color='teal'
    )

    # График 8: Тепловая карта корреляции
    plot_correlation_matrix(
        df_cleaned, numeric_cols,
        os.path.join(IMAGES_DIR, '08_correlation_matrix.png')
    )

    cleaned_csv_path = os.path.join(
        PROCESSED_DATA_DIR, 'student_performance_cleaned.csv'
    )
    df_cleaned.to_csv(cleaned_csv_path, index=False)
    print(f"\nОчищенный датасет сохранен в: {cleaned_csv_path}")

    # --------------------------------------------------------------------------
    # ЭТАП 3.1: ГЕНЕРАЦИЯ НОВЫХ ПРИЗНАКОВ (FEATURE ENGINEERING)
    # --------------------------------------------------------------------------
    print_banner("ЭТАП 3.1: ГЕНЕРАЦИЯ НОВЫХ ПРИЗНАКОВ (FEATURE ENGINEERING)")
    df_featured, new_feats = generate_new_features(df_cleaned)
    print("Сгенерировано 5 новых признаков:")
    for nf in new_feats:
        print(
            f"  - {nf}: min={df_featured[nf].min()}, "
            f"max={df_featured[nf].max()}, mean={df_featured[nf].mean():.2f}"
        )

    # График 9: Распределения новых признаков
    plot_feature_engineering(
        df_featured, new_feats,
        os.path.join(IMAGES_DIR, '09_feature_engineering.png')
    )

    # --------------------------------------------------------------------------
    # ЭТАП 3.2: КОДИРОВАНИЕ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ
    # --------------------------------------------------------------------------
    print_banner("ЭТАП 3.2: КОДИРОВАНИЕ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ")
    print("1. Бинаризация бинарных категорий (0 / 1):")
    for col, m in BINARY_MAPPINGS.items():
        print(f"  - {col}: {m}")
    df_encoded = encode_binary_features(df_featured, BINARY_MAPPINGS)

    print("\n2. Порядковое кодирование ранговых категорий (Ordinal Encoding):")
    for col, m in ORDINAL_MAPPINGS.items():
        print(f"  - {col}: {m}")
    df_encoded = encode_ordinal_features(df_encoded, ORDINAL_MAPPINGS)

    print("\n3. One-Hot Encoding (OHE) номинальных переменных:")
    df_encoded, ohe_added = encode_nominal_features(df_encoded, OHE_COLUMNS)
    print(f"  Добавленные OHE-столбцы: {ohe_added}")

    print(
        f"\nИтоговая размерность признакового пространства: "
        f"{df_encoded.shape[0]} строк, {df_encoded.shape[1]} столбцов."
    )
    featured_csv_path = os.path.join(
        PROCESSED_DATA_DIR, 'student_performance_featured.csv'
    )
    df_encoded.to_csv(featured_csv_path, index=False)
    print(f"Датасет с признаками сохранен в: {featured_csv_path}")

    # --------------------------------------------------------------------------
    # ЭТАП 4: СТАНДАРТИЗАЦИЯ И НОРМАЛИЗАЦИЯ
    # --------------------------------------------------------------------------
    print_banner("ЭТАП 4: СТАНДАРТИЗАЦИЯ И НОРМАЛИЗАЦИЯ (SCALING)")
    print(
        f"Масштабируемые непрерывные признаки ({len(CONTINUOUS_COLUMNS_TO_SCALE)}):"
    )
    print(CONTINUOUS_COLUMNS_TO_SCALE)

    # MinMaxScaler
    df_minmax = apply_minmax_scaling(
        df_encoded, CONTINUOUS_COLUMNS_TO_SCALE
    )
    minmax_csv_path = os.path.join(
        PROCESSED_DATA_DIR, 'student_performance_scaled_minmax.csv'
    )
    df_minmax.to_csv(minmax_csv_path, index=False)

    # StandardScaler
    df_standard = apply_standard_scaling(
        df_encoded, CONTINUOUS_COLUMNS_TO_SCALE
    )
    std_csv_path = os.path.join(
        PROCESSED_DATA_DIR, 'student_performance_scaled_standard.csv'
    )
    df_standard.to_csv(std_csv_path, index=False)

    # Сравнение статистик
    scaling_tables = compare_scaling_stats(
        df_encoded, df_minmax, df_standard, DEMO_COLUMNS
    )
    print(
        "\nСравнение статистических показателей "
        "(Исходные vs MinMax vs StandardScaler):"
    )
    for col_name, stats_df in scaling_tables.items():
        print(f"\n--- Признак: {col_name} ---")
        print(stats_df)

    # График 10: Сравнение плотностей масштабирования
    plot_scaling_comparison(
        df_encoded, df_minmax, df_standard, DEMO_COLUMNS,
        os.path.join(IMAGES_DIR, '10_scaling_comparison.png')
    )

    # Копирование в отчетные каталоги
    copy_images_to_reports()
    print(f"\nВсе графики синхронизированы с отчетом: {REPORT_IMAGES_DIR}")

    print_banner("ВСЕ ЭТАПЫ ЛАБОРАТОРНОЙ РАБОТЫ УСПЕШНО ВЫПОЛНЕНЫ!")


if __name__ == '__main__':
    main()
