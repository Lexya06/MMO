# ==============================================================================
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

ax2.set_title("Увеличенный вид окрестности минимума (Линейная шкала, шаги 150..400)", fontsize=13, fontweight="bold")
ax2.set_xlabel("Номер итерации (шага) $k$", fontsize=11)
ax2.set_ylabel("Среднеквадратичная ошибка (MSE)", fontsize=11)
ax2.set_ylim(bottom=1.8, top=6.0)
ax2.legend(loc="upper right", frameon=True, framealpha=0.95)
ax2.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
save_plot(plt.gcf(), "02_gd_vs_sgd_convergence.png")
plt.show()