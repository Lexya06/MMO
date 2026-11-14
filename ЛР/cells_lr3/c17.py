# ==============================================================================
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