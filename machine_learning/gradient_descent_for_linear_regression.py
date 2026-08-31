"""https://introml.mit.edu/notes/regression.html"""
import numpy as np
from ucimlrepo import fetch_ucirepo


class LinearRegression:
    def __init__(self, X, y):
        self.X = X  # derivatives of weights
        self.y = y  # derivatives of weights
        self.n = y.shape[0]  # number of datapoints

    def compute_loss(self, w, b):
        y_pred = self.X @ w + b
        mse = np.sum((y_pred - self.y) ** 2) / self.n
        return mse

    def compute_gradient(self, w, b):
        y_pred = self.X @ w + b
        d_w = (2 / self.n) * self.X.T @ (y_pred - self.y)  # derivatives of weights
        d_b = (2 / self.n) * np.sum(y_pred - self.y)  # derivatives of the bias
        return d_w, d_b


class GradientDescent:
    def __init__(self, learning_rate=0.03, epochs=2000):
        self.lr = learning_rate
        self.epochs = epochs

    def optimize(self, loss_func):
        feature_num = loss_func.X.shape[1]
        w = np.zeros(feature_num)
        b = 0.0

        for epoch in range(self.epochs):
            d_w, d_b = loss_func.compute_gradient(w, b)
            w = w - self.lr * d_w
            b = b - self.lr * d_b

            if epoch % 200 == 0:
                loss = loss_func.compute_loss(w, b)
                print(f"Epoch {epoch:5d} | Loss = {loss:.4f}")

        return w, b


if __name__ == "__main__":
    real_estate_valuation = fetch_ucirepo(id=477)

    X_raw = real_estate_valuation.data.features.values
    y_raw = real_estate_valuation.data.targets.values.ravel()

    print("Dataset name:", real_estate_valuation.metadata.name)
    print("Number of samples:", X_raw.shape[0])
    print("Number of features:", X_raw.shape[1])

    X = (X_raw - np.mean(X_raw, axis=0)) / np.std(X_raw, axis=0)

    loss = LinearRegression(X, y_raw)
    optimizer = GradientDescent(learning_rate=0.03, epochs=50000)
    w_opt, b_opt = optimizer.optimize(loss)

    print("=== Training Finished ===")
    print("Optimal weight w:", w_opt)
    print("Optimal bias b:", b_opt)
    final_loss = loss.compute_loss(w_opt, b_opt)
    print("Final Loss:", final_loss)
