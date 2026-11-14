# ==============================================================================
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