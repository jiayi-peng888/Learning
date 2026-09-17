"""https://archive.ics.uci.edu/dataset/222/bank+marketing"""
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.special import expit


class LogisticRegression:
    def __init__(self, X, y, lam=0.01):
        self.X = X  # design matrix
        self.y = y  # target label 0/1
        self.n = y.shape[0]  # number of datapoints
        self.d = X.shape[1]  # number of feature dimensions
        self.lam = lam  # L2 regularization coefficient

    def compute_loss(self, w, b):
        y_pred_prob = expit(self.X @ w + b)
        cross_entropy = (-self.y * np.log(y_pred_prob + 1e-8) - (1 - self.y) * np.log(1 - y_pred_prob + 1e-8))
        loss = np.mean(cross_entropy) + self.lam * np.sum(np.square(w))
        return loss

    def compute_gradient(self, w, b):
        y_pred_prob = expit(self.X @ w + b)
        error = y_pred_prob - self.y
        dw = (2.0 / self.n) * self.X.T @ error + 2 * self.lam * w
        db = (2.0 / self.n) * np.sum(error)
        return dw, db


class GradientDescent:
    def __init__(self, learning_rate=0.03, epochs=2000):
        self.lr = learning_rate
        self.epochs = epochs

    def optimize(self, loss_func):
        w = np.zeros(loss_func.d)
        b = 0.0  
        for epoch in range(self.epochs):
            dw, db = loss_func.compute_gradient(w, b)
            w = w - self.lr * dw
            b = b - self.lr * db

            if epoch % 200 == 0:
                loss = loss_func.compute_loss(w, b)
                print(f"Epoch {epoch:5d} | Loss = {loss:.4f}")
        return w, b


if __name__ == "__main__":
    # load UCI Bank Marketing dataset id=222
    bank_marketing = fetch_ucirepo(id=222)
    X_raw = bank_marketing.data.features
    y_raw = bank_marketing.data.targets

    print("Dataset name:", bank_marketing.metadata.name)
    print("Number of samples:", X_raw.shape[0])
    print("Number of raw features:", X_raw.shape[1])

    # convert label: yes -> 1, no ->0
    y = (y_raw["y"] == "yes").astype(float).values

    # preprocess categorical & numerical features
    cat_cols = X_raw.select_dtypes(include=["object"]).columns
    num_cols = X_raw.select_dtypes(exclude=["object"]).columns

    ohe = OneHotEncoder(sparse_output=False, drop="first")
    X_cat = ohe.fit_transform(X_raw[cat_cols])
    X_num = X_raw[num_cols].values

    # standardization
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)
    X = np.hstack([X_num_scaled, X_cat])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Number of train samples:", X_train.shape[0])
    print("Number of features:", X_train.shape[1])

    # initialize logistic regression loss class, set lambda
    loss = LogisticRegression(X_train, y_train, lam=0.01)
    optimizer = GradientDescent(learning_rate=0.05, epochs=50000)
    w_opt, b_opt = optimizer.optimize(loss)

    print("Training Finished")
    print("Optimal weight w:", w_opt)
    print("Optimal bias b:", b_opt)
    final_loss = loss.compute_loss(w_opt, b_opt)
    print("Final Train Loss:", final_loss)

    # predict on test set
    def predict(X, w, b):
        prob = 1.0 / (1 + np.exp(-(X @ w + b)))
        return (prob >= 0.5).astype(int)

    y_pred = predict(X_test, w_opt, b_opt)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy = {acc:.4f}")
