# ==============================================================================
# КЛАСС КАСТОМНОЙ ЛИНЕЙНОЙ РЕГРЕССИИ НА ЧИСТОМ NUMPY
# ==============================================================================

class CustomLinearRegression:
    """
    Класс множественной линейной регрессии на чистом NumPy,
    реализующий алгоритмы обучения в соответствии со стандартами Scikit-Learn:
    - gd: Пакетный градиентный спуск (BGD)
    - sgd: Стохастический градиентный спуск (SGD) с расписанием Роббинса-Монро
    - l1: Градиентный спуск с L1-регуляризацией (Lasso) через оператор ISTA
    - l2: Градиентный спуск с L2-регуляризацией (Ridge) через Weight Decay
    - normal_eq: Аналитическое МНК-решение нормального уравнения Гаусса
    """

    def __init__(
        self,
        method: Literal["gd", "sgd", "l1", "l2", "normal_eq"] = "gd",
        learning_rate: float = 0.05,
        max_iter: int = 3000,
        tol: float = 1e-6,
        lmbd: float = 0.0,
        decay_rate: float = 0.0,
        record_loss_every: int = 1,
        random_state: Optional[int] = RANDOM_SEED,
    ):
        """
        Инициализация гиперпараметров регрессионной модели:
        :param method: Метод обучения ('gd', 'sgd', 'l1', 'l2', 'normal_eq')
        :param learning_rate: Темп (шаг) градиентного спуска (eta > 0)
        :param max_iter: Максимальное допустимое число итераций
        :param tol: Критерий ранней остановки по норме смещения весов ||Delta w|| < tol
        :param lmbd: Коэффициент силы регуляризации (штраф за сложность модели)
        :param decay_rate: Коэффициент затухания шага обучения по Роббинсу-Монро (для SGD)
        :param record_loss_every: Периодичность сохранения истории MSE и параметров
        :param random_state: Фиксатор генератора случайных чисел для воспроизводимости
        """
        self.method = method
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.lmbd = lmbd
        self.decay_rate = decay_rate
        self.record_loss_every = record_loss_every
        self.random_state = random_state

        # Вычисляемые атрибуты процесса обучения
        self.weights_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        self.coef_: Optional[np.ndarray] = None
        self.loss_history_: List[float] = []
        self.weights_history_: List[np.ndarray] = []
        self.n_iter_: int = 0
        self.converged_: bool = False

    @staticmethod
    def _add_bias(X: np.ndarray) -> np.ndarray:
        """Добавляет фиктивный единичный столбец x0 = 1, если он не добавлен."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if np.allclose(X[:, 0], 1.0):
            return X
        return np.hstack([np.ones((X.shape[0], 1), dtype=np.float64), X])

    @staticmethod
    def _compute_mse(X_b: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        """Векторное вычисление среднеквадратичной ошибки: MSE = (1/l) * ||X_b * w - y||^2."""
        residuals = X_b.dot(w) - y
        return float(np.mean(residuals ** 2))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CustomLinearRegression":
        """
        Обучение параметров линейной модели в соответствии с выбранным методом.
        :param X: Матрица признаков (l, m) или расширенная матрица (l, m+1) с единичной колонкой
        :param y: Вектор целевых значений (l,)
        """
        X_b = self._add_bias(X)
        n_samples, n_features = X_b.shape
        y = y.astype(np.float64)

        # Очистка истории предыдущих обучений
        self.loss_history_ = []
        self.weights_history_ = []
        self.converged_ = False

        # 1. Аналитическое МНК-решение: w* = (X_b^T X_b)^(-1) X_b^T y
        if self.method == "normal_eq":
            XtX = X_b.T.dot(X_b)
            # Добавка 1e-9*I для численной устойчивости обращения матрицы
            XtX_reg = XtX + 1e-9 * np.eye(n_features)
            Xty = X_b.T.dot(y)
            w = np.linalg.solve(XtX_reg, Xty)

            self.weights_ = w
            self.intercept_ = float(w[0])
            self.coef_ = w[1:].copy()
            self.n_iter_ = 1
            self.converged_ = True
            self.loss_history_.append(self._compute_mse(X_b, y, w))
            self.weights_history_.append(w.copy())
            return self

        # 2. Инициализация весов нулями
        w = np.zeros(n_features, dtype=np.float64)
        rng = np.random.RandomState(self.random_state)

        # Сохранение начального состояния
        self.loss_history_.append(self._compute_mse(X_b, y, w))
        self.weights_history_.append(w.copy())

        # Итерационный цикл оптимизации
        for k in range(1, self.max_iter + 1):
            w_prev = w.copy()

            if self.method == "gd":
                # Пакетный градиентный спуск: точный градиент по всей обучающей выборке
                residuals = X_b.dot(w_prev) - y
                grad = (2.0 / n_samples) * X_b.T.dot(residuals)
                w = w_prev - self.learning_rate * grad

            elif self.method == "sgd":
                # Стохастический градиентный спуск: градиент по 1 случайному объекту
                rand_idx = rng.randint(0, n_samples)
                x_i = X_b[rand_idx]
                y_i = y[rand_idx]

                # Стохастический градиент
                grad_i = 2.0 * (np.dot(x_i, w_prev) - y_i) * x_i

                # Расписание затухания шага Роббинса-Монро
                eta_k = self.learning_rate / (1.0 + self.decay_rate * k)
                w = w_prev - eta_k * grad_i

            elif self.method == "l1":
                # L1 Lasso: проксимальный градиентный спуск Soft-Thresholding (ISTA)
                residuals = X_b.dot(w_prev) - y
                grad_mse = (2.0 / n_samples) * X_b.T.dot(residuals)

                # Обычный градиентный шаг
                w_tilde = w_prev - self.learning_rate * grad_mse

                # w0 (bias) не регуляризуется
                w[0] = w_tilde[0]

                # Оператор мягкого порога Soft-Thresholding для весов признаков w1..m
                threshold = self.learning_rate * self.lmbd
                w[1:] = np.sign(w_tilde[1:]) * np.maximum(0.0, np.abs(w_tilde[1:]) - threshold)

            elif self.method == "l2":
                # L2 Ridge: безусловно устойчивый оператор сжатия Weight Decay
                residuals = X_b.dot(w_prev) - y
                grad_mse = (2.0 / n_samples) * X_b.T.dot(residuals)

                # w0 (bias) не регуляризуется
                w[0] = w_prev[0] - self.learning_rate * grad_mse[0]

                # Неявный шаг сжатия весов
                w[1:] = (w_prev[1:] - self.learning_rate * grad_mse[1:]) / (
                    1.0 + 2.0 * self.learning_rate * self.lmbd
                )
            else:
                raise ValueError(f"Неизвестный метод оптимизации: {self.method}")

            # Запись истории изменения функции потерь
            if k % self.record_loss_every == 0 or k == self.max_iter:
                self.loss_history_.append(self._compute_mse(X_b, y, w))
                self.weights_history_.append(w.copy())

            # Проверка критерия останова: ||w^k - w^{k-1}|| < tol
            step_norm = float(np.linalg.norm(w - w_prev))
            if step_norm < self.tol:
                self.converged_ = True
                self.n_iter_ = k
                break
        else:
            self.n_iter_ = self.max_iter

        self.weights_ = w
        self.intercept_ = float(w[0])
        self.coef_ = w[1:].copy()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Прогнозирование целевой переменной: y_pred = X_b * w."""
        if self.weights_ is None:
            raise RuntimeError("Модель не обучена. Вызовите .fit().")
        X_b = self._add_bias(X)
        return X_b.dot(self.weights_)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Коэффициент детерминации R^2."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return float(1.0 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0

    def mse(self, X: np.ndarray, y: np.ndarray) -> float:
        """Среднеквадратичная ошибка (MSE)."""
        return float(np.mean((y - self.predict(X)) ** 2))

    def mae(self, X: np.ndarray, y: np.ndarray) -> float:
        """Средняя абсолютная ошибка (MAE)."""
        return float(np.mean(np.abs(y - self.predict(X))))


# Верификация работоспособности аналитического МНК решения
ols_test = CustomLinearRegression(method="normal_eq").fit(X_train_b, y_train)
print("✅ Класс CustomLinearRegression успешно реализован!")
print(
    f"📊 Аналитическое МНК-решение: Test MSE = {ols_test.mse(X_test_b, y_test):.4f}, "
    f"R² = {ols_test.score(X_test_b, y_test):.4f}"
)
print(f"📌 Свободный член w0 (intercept) = {ols_test.intercept_:.4f}")