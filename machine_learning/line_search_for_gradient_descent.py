"""https://blog.csdn.net/tsroad/article/details/50249393"""
import numpy as np
from ucimlrepo import fetch_ucirepo


def loss_function(w, X, y):
    y_pred = X @ w
    return np.mean(np.square(y_pred - y))


def gradient(w, X, y):
    n = X.shape[0]
    return 2 * X.T @ (X @ w - y) / n


def wolfe_line_search(w, grad, X, y, c1=1e-4, c2=0.9, beta=0.8):
    alpha = 1.0
    current_loss = loss_function(w, X, y)
    p = -grad
    grad_dot_p = grad.T @ p

    while True:
        w_new = w + alpha * p
        loss_new = loss_function(w_new, X, y)
        armijo_ok = loss_new <= current_loss + c1 * alpha * grad_dot_p
        grad_new = gradient(w_new, X, y)
        curvature_ok = grad_new.T @ p >= c2 * grad_dot_p

        if armijo_ok and curvature_ok:
            break
        alpha = alpha * beta
        if alpha < 1e-10:
            break
    return alpha


def gradient_descent_line_search(X, y, w_init, max_iter=200, tol=1e-6):
    w = w_init.copy()
    current_loss = 0.0
    for i in range(max_iter):
        grad = gradient(w, X, y)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            print(f"Converged at iteration: {i}")
            break
        alpha = wolfe_line_search(w, grad, X, y)
        w = w - alpha * grad
        current_loss = loss_function(w, X, y)
        print(f"Iter:{i:3d} | alpha = {alpha:.6f} | Loss = {current_loss:.4f}")
    return w, current_loss


real_estate_valuation = fetch_ucirepo(id=477)
X_raw = real_estate_valuation.data.features.values
y_raw = real_estate_valuation.data.targets.values.ravel()

X_scaled = (X_raw - np.mean(X_raw, axis=0)) / np.std(X_raw, axis=0)

# add bias term
n = X_scaled.shape[0]
X = np.hstack([np.ones((n, 1)), X_scaled])

w0 = np.zeros(X.shape[1])

w_opt, final_loss = gradient_descent_line_search(X, y_raw, w0)

print("Optimized weights: ", w_opt)
print("Final loss value: ", final_loss)
