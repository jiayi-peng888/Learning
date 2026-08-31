import numpy as np
from ucimlrepo import fetch_ucirepo


class LinearRegressionLoss:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.n = y.shape[0]

    def compute_loss(self, w, b):
        y_pred = self.X @ w + b
        mse = np.sum((y_pred - self.y) ** 2) / self.n
        return mse

    def compute_gradient(self, w, b):
        y_pred = self.X @ w + b
        dw = (2 / self.n) * self.X.T @ (y_pred - self.y)
        db = (2 / self.n) * np.sum(y_pred - self.y)
        return dw, db


class GradientDescentOptimizer:
    def __init__(self, learning_rate=0.001, epochs=20000):
        self.lr = learning_rate
        self.epochs = epochs

    def optimize(self, loss_func):
        feature_num = loss_func.X.shape[1]
        w = np.zeros(feature_num)
        b = 0.0

        for epoch in range(self.epochs):
            dw, db = loss_func.compute_gradient(w, b)
            w = w - self.lr * dw
            b = b - self.lr * db

            if epoch % 200 == 0:
                current_loss = loss_func.compute_loss(w, b)
                print(f"Epoch {epoch:5d} | Loss = {current_loss:.4f}")

        return w, b


if __name__ == "__main__":
    real_estate_valuation = fetch_ucirepo(id=477)

    X_raw = real_estate_valuation.data.features.values
    y_raw = real_estate_valuation.data.targets.values.ravel()

    print("Dataset name:", real_estate_valuation.metadata.name)
    print("Number of samples:", X_raw.shape[0], "Number of features:", X_raw.shape[1])

    X = (X_raw - np.mean(X_raw, axis=0)) / np.std(X_raw, axis=0)
    y = y_raw

    loss_obj = LinearRegressionLoss(X, y)
    optimizer = GradientDescentOptimizer(learning_rate=0.001, epochs=50000)
    w_opt, b_opt = optimizer.optimize(loss_obj)

    print("\n=== Training Finished ===")
    print("Optimal weight w:", w_opt)
    print("Optimal bias b:", b_opt)
    final_loss = loss_obj.compute_loss(w_opt, b_opt)
    print("Final Loss:", final_loss)
