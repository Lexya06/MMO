"""
Модуль собственной реализации линейной регрессии с оптимизацией методами:
- Batch Gradient Descent (BGD / GD)
- Stochastic Gradient Descent (SGD)
- L1-регуляризация (Lasso) через субградиентный спуск и Soft-Thresholding (ISTA)
- L2-регуляризация (Ridge) через градиентный спуск со сжатием весов (Weight Decay)
"""

from typing import Dict, List, Literal, Optional, Tuple, Union
import numpy as np


class CustomLinearRegression:
    """
    Пользовательская модель линейной регрессии, обучаемая методами градиентного спуска.

    Параметры:
    ----------
    method : {'gd', 'sgd', 'l1', 'l2'}, default='gd'
        Метод оптимизации функционала качества:
        - 'gd': классический (пакетный) градиентный спуск;
        - 'sgd': стохастический градиентный спуск с выбором одного объекта;
        - 'l1': градиентный спуск с L1-регуляризацией (Lasso);
        - 'l2': градиентный спуск с L2-регуляризацией (Ridge).
    learning_rate : float, default=0.05
        Базовый шаг градиентного спуска (eta).
    max_iter : int, default=3000
        Максимальное число итераций оптимизации.
    tol : float, default=1e-6
        Критерий остановки по норме разности весов: ||w^k - w^{k-1}|| < tol.
    lmbd : float, default=0.0
        Коэффициент регуляризации (lambda).
    decay_rate : float, default=0.0
        Параметр затухания скорости обучения для SGD: eta_k = eta_0 / (1 + decay * k).
    l1_solver : {'soft_threshold', 'subgradient'}, default='soft_threshold'
        Способ решения L1: Soft-Thresholding (ISTA, точное зануление) или субградиент.
    record_loss_every : int, default=1
        Частота записи значения MSE в историю.
    random_state : Optional[int], default=42
        Случайное зерно для воспроизводимости стохастического выбора объектов.
    """

    def __init__(
        self,
        method: Literal["gd", "sgd", "l1", "l2"] = "gd",
        learning_rate: float = 0.05,
        max_iter: int = 3000,
        tol: float = 1e-6,
        lmbd: float = 0.0,
        decay_rate: float = 0.0,
        l1_solver: Literal["soft_threshold", "subgradient"] = "soft_threshold",
        record_loss_every: int = 1,
        random_state: Optional[int] = 42,
    ):
        self.method = method
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.lmbd = lmbd
        self.decay_rate = decay_rate
        self.l1_solver = l1_solver
        self.record_loss_every = record_loss_every
        self.random_state = random_state

        # Атрибуты, формируемые после обучения
        self.weights_: Optional[np.ndarray] = None  # Полный вектор [w0, w1, ..., wm]
        self.intercept_: Optional[float] = None     # w0
        self.coef_: Optional[np.ndarray] = None      # [w1, ..., wm]
        self.loss_history_: List[float] = []        # История MSE
        self.weights_history_: List[np.ndarray] = []# История весов
        self.n_iter_: int = 0                       # Реальное число шагов
        self.converged_: bool = False               # Флаг сходимости по tol

    @staticmethod
    def _add_bias(X: np.ndarray) -> np.ndarray:
        """Добавляет фиктивный единичный столбец x0 = 1, если он отсутствует."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if np.allclose(X[:, 0], 1.0):
            return X
        ones = np.ones((X.shape[0], 1), dtype=np.float64)
        return np.hstack([ones, X])

    @staticmethod
    def _compute_mse(X_b: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        """Вычисляет среднеквадратичную ошибку (MSE) для текущих весов."""
        errors = X_b.dot(w) - y
        return float(np.mean(errors ** 2))

    @staticmethod
    def _soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
        """
        Проксимальный оператор Soft-Thresholding для L1 регуляризации:
        S_thresh(v) = sign(v) * max(0, |v| - thresh)
        """
        return np.sign(values) * np.maximum(0.0, np.abs(values) - threshold)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CustomLinearRegression":
        """
        Обучает линейную регрессию на выборке (X, y).

        Parameters:
        -----------
        X : np.ndarray размерности (n_samples, n_features) или (n_samples, n_features + 1)
            Матрица признаков (с единичным столбцом или без него).
        y : np.ndarray размерности (n_samples,)
            Вектор истинных значений целевой переменной.

        Returns:
        --------
        self : CustomLinearRegression
        """
        X_b = self._add_bias(X)
        n_samples, n_features_with_bias = X_b.shape
        y = np.asarray(y, dtype=np.float64)

        rng = np.random.RandomState(self.random_state)

        # 1. Инициализация начальных весов w^0 и ε
        w = np.zeros(n_features_with_bias, dtype=np.float64)
        self.loss_history_ = []
        self.weights_history_ = []
        self.converged_ = False

        # Запись начальной ошибки до первого шага
        initial_mse = self._compute_mse(X_b, y, w)
        self.loss_history_.append(initial_mse)
        self.weights_history_.append(w.copy())

        # 2. Организовать цикл по количеству итераций
        for k in range(1, self.max_iter + 1):
            w_prev = w.copy()

            if self.method == "gd":
                # 3. Расчет новых весов: w^k = w^{k-1} - eta * grad_Q
                # grad_Q = (2/n) * X^T * (X * w - y)
                errors = X_b.dot(w_prev) - y
                grad = (2.0 / n_samples) * X_b.T.dot(errors)
                w = w_prev - self.learning_rate * grad

            elif self.method == "sgd":
                # Стохастический градиентный спуск: выбор одного случайного объекта
                idx = rng.randint(0, n_samples)
                x_i = X_b[idx]
                y_i = y[idx]
                pred_i = np.dot(x_i, w_prev)
                error_i = pred_i - y_i
                grad_i = 2.0 * error_i * x_i

                # Расписание скорости обучения Robbins-Monro
                if self.decay_rate > 0:
                    eta_k = self.learning_rate / (1.0 + self.decay_rate * k)
                else:
                    eta_k = self.learning_rate

                w = w_prev - eta_k * grad_i

            elif self.method == "l1":
                # L1-регуляризация (Lasso): смещение w0 НЕ регуляризуется!
                errors = X_b.dot(w_prev) - y
                grad_mse = (2.0 / n_samples) * X_b.T.dot(errors)

                if self.l1_solver == "soft_threshold":
                    # Проксимальный градиентный шаг (ISTA)
                    w_temp = w_prev - self.learning_rate * grad_mse
                    w = np.zeros_like(w_prev)
                    w[0] = w_temp[0]  # w0 без штрафа
                    w[1:] = self._soft_threshold(w_temp[1:], self.learning_rate * self.lmbd)
                else:
                    # Субградиентный шаг
                    subgrad = np.zeros_like(w_prev)
                    subgrad[0] = grad_mse[0]
                    subgrad[1:] = grad_mse[1:] + self.lmbd * np.sign(w_prev[1:])
                    w = w_prev - self.learning_rate * subgrad

            elif self.method == "l2":
                # L2-регуляризация (Ridge): смещение w0 НЕ регуляризуется!
                errors = X_b.dot(w_prev) - y
                grad_mse = (2.0 / n_samples) * X_b.T.dot(errors)

                w = np.zeros_like(w_prev)
                # w0 обновляется стандартным шагом GD
                w[0] = w_prev[0] - self.learning_rate * grad_mse[0]
                # Безусловно устойчивый оператор сжатия для признаков w_1..m:
                # w^{k} = (w^{k-1} - eta * grad_mse) / (1 + 2 * eta * lambda)
                w[1:] = (w_prev[1:] - self.learning_rate * grad_mse[1:]) / (
                    1.0 + 2.0 * self.learning_rate * self.lmbd
                )
            else:
                raise ValueError(f"Неизвестный метод оптимизации: {self.method}")

            # Запись истории ошибки
            if k % self.record_loss_every == 0 or k == self.max_iter:
                current_mse = self._compute_mse(X_b, y, w)
                self.loss_history_.append(current_mse)
                self.weights_history_.append(w.copy())

            # 4. Завершить цикл, если ||w^k - w^{k-1}|| < tol (или для SGD проверка сходимости)
            diff_norm = float(np.linalg.norm(w - w_prev))
            if diff_norm < self.tol:
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
        """
        Вычисляет предсказания целевой переменной: y_hat = X_b * w.
        """
        if self.weights_ is None:
            raise RuntimeError("Модель еще не обучена! Вызовите .fit() перед вызовом .predict().")
        X_b = self._add_bias(X)
        return X_b.dot(self.weights_)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Вычисляет коэффициент детерминации R^2:
        R^2 = 1 - SS_res / SS_tot
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot == 0:
            return 0.0
        return float(1.0 - (ss_res / ss_tot))

    def mse(self, X: np.ndarray, y: np.ndarray) -> float:
        """Вычисляет среднеквадратичную ошибку (MSE) на переданной выборке."""
        y_pred = self.predict(X)
        return float(np.mean((y - y_pred) ** 2))

    def mae(self, X: np.ndarray, y: np.ndarray) -> float:
        """Вычисляет среднюю абсолютную ошибку (MAE) на переданной выборке."""
        y_pred = self.predict(X)
        return float(np.mean(np.abs(y - y_pred)))
