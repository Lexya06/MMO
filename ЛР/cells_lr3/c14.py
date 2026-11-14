# ==============================================================================
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

plt.suptitle("Покомпонентное сравнение сжатия весов: L1 (Lasso) vs L2 (Ridge)", fontsize=14, fontweight="bold", y=1.02)
save_plot(fig_comp, "05_regularization_comparison.png")
plt.show()