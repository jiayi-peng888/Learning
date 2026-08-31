import numpy as np
from ucimlrepo import fetch_ucirepo

class LinearRegressionLoss:
    def compute_loss(self, X, y, w, b):
        n = y.shape[0]
        y_pred = X @ w + b
        mse = np.sum((y_pred - y) ** 2) / n
        return mse

    def compute_gradient(self, X, y, w, b):
        n = y.shape[0]
        y_pred = X @ w + b
        dw = (2 / n) * X.T @ (y_pred - y)
        db = (2 / n) * np.sum(y_pred - y)
        return dw, db

class GradientDescentOptimizer:
    def __init__(self, learning_rate=0.001, epochs=20000):
        self.lr = learning_rate
        self.epochs = epochs

    def optimize(self, X, y, loss_func):
        feature_num = X.shape[1]
        w = np.zeros(feature_num)
        b = 0.0
        loss_history = []

        for epoch in range(self.epochs):
            dw, db = loss_func.compute_gradient(X, y, w, b)
            w -= self.lr * dw
            b -= self.lr * db
            loss = loss_func.compute_loss(X, y, w, b)
            loss_history.append(loss)

            if epoch % 200 == 0:
                print(f"Epoch {epoch:5d} | Loss = {loss:.4f}")
        return w, b, loss_history


if __name__ == "__main__":
    real_estate_valuation = fetch_ucirepo(id=477)
    print(real_estate_valuation)
    print(type(real_estate_valuation))
    print(type(real_estate_valuation.data))
    print(type(real_estate_valuation.metadata))
    print(type(real_estate_valuation.data.features))

    X_raw = real_estate_valuation.data.features.values
    print(type(X_raw))
    y_raw = real_estate_valuation.data.targets.values.ravel()

    print("Dataset name:", real_estate_valuation.metadata.name)
    print("Number of samples:", X_raw.shape[0], "Number of features:", X_raw.shape[1])

    X = (X_raw - np.mean(X_raw, axis=0)) / np.std(X_raw, axis=0)
    y = y_raw

    loss_obj = LinearRegressionLoss()
    optimizer = GradientDescentOptimizer(learning_rate=0.001, epochs=50000)
    w_opt, b_opt, loss_record = optimizer.optimize(X, y, loss_obj)

    print("\n==== Training Finished ====")
    print("Optimal weight w:", w_opt)
    print("Optimal bias b:", b_opt)
    print("Final Loss:", loss_record[-1])

