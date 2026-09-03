"""https://introml.mit.edu/notes/regression.html"""
import numpy as np
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class LinearRegression:
    def __init__(self, X, y,lam):
        self.X = X  # design matrix
        self.y = y  # target class
        self.n = y.shape[0]  # number of datapoints
        self.d = X.shape[1]  # number of feature dimensions
        self.lam = lam

    def compute_loss(self, w, b):
        y_pred = self.X @ w + b
        mse = np.mean(np.square(y_pred - self.y))
        reg = self.lam * np.sum(np.square(w))
        loss = mse + reg
        return loss

    def compute_gradient(self, w, b):
        y_pred = self.X @ w + b
        d_w = (2.0 / self.n) * self.X.T @ (y_pred - self.y)+ 2 * self.lam * w  # derivatives of weights
        d_b = (2.0 / self.n) * np.sum(y_pred - self.y)  # derivative of the bias
        return d_w, d_b


class GradientDescent:
    def __init__(self, learning_rate=0.03, epochs=2000):
        self.lr = learning_rate
        self.epochs = epochs

    def optimize(self, loss_func):
        w = np.zeros(loss_func.d)  # weights of linear regression model
        b = 0.0  # bias of linear regression model

        for epoch in range(self.epochs):
            d_w, d_b = loss_func.compute_gradient(w, b)
            w = w - self.lr * d_w
            b = b - self.lr * d_b

            if epoch % 200 == 0:
                loss = loss_func.compute_loss(w, b)
                print(f"Epoch {epoch:5d} | Loss = {loss:.4f} ")

        return w, b


def objective(param, X, y, lam):
    dim = X.shape[1]
    w = param[:dim]
    b = param[dim]
    y_pred = X @ w + b
    mse = np.mean(np.square(y_pred - y))
    reg = lam * np.sum(np.square(w))
    return mse + reg


if __name__ == "__main__":
    real_estate_valuation = fetch_ucirepo(id=477)

    X_raw = real_estate_valuation.data.features.values
    y_raw = real_estate_valuation.data.targets.values.ravel()

    print("Dataset name:", real_estate_valuation.metadata.name)
    print("Number of samples:", X_raw.shape[0])
    print("Number of features:", X_raw.shape[1])

    X = (X_raw - np.mean(X_raw, axis=0)) / np.std(X_raw, axis=0)

    lam=0.01
    loss = LinearRegression(X, y_raw,lam=lam)
    optimizer = GradientDescent(learning_rate=0.03, epochs=50000)
    w_opt, b_opt = optimizer.optimize(loss)

    print("=== Training Finished ===")
    print("Optimal weight w:", w_opt)
    print("Optimal bias b:", b_opt)
    final_loss = loss.compute_loss(w_opt, b_opt)
    print("Final Loss:", final_loss)

    init_guess = np.zeros(X.shape[1] + 1)
    res_scipy = minimize(objective, init_guess, args=(X, y_raw, lam))
    w_sp = res_scipy.x[:X.shape[1]]
    b_sp = res_scipy.x[X.shape[1]]

    print("=== Scipy.optimize.minimize ===")
    print("w_scipy:", w_sp)
    print("b_scipy:", b_sp)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i in range(loss.d):
        xi = X[:, i]
        x_line = np.linspace(xi.min(), xi.max(), 100)
        y_gd_line = w_opt[i] * x_line + b_opt
        y_sp_line = w_sp[i] * x_line + b_sp

        axes[i].scatter(xi, y_raw, alpha=0.5, color="gray")
        axes[i].plot(x_line, y_sp_line, 'r-', lw=3, alpha=0.7, label="Scipy minimize")
        axes[i].plot(x_line, y_gd_line, 'b-', lw=2, alpha=0.7, label="Gradient‑Descent")
        axes[i].set_xlabel(f"Feature {i}")
        axes[i].set_ylabel("House Price")
        axes[i].set_title(f"Feature {i} Partial‑Effect Plot")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()



