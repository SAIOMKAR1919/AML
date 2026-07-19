import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error,mean_squared_error,r2_score)
model = None
X = None
y = None
y_pred = None
root = tk.Tk()
root.title("Multiple Linear Regression Visualizer")
root.geometry("900x750")
root.configure(bg="black")
title = tk.Label(
    root,
    text="Multiple Linear Regression Visualizer",
    font=("Arial", 20, "bold"),
    bg="black",
    fg="white"
)
title.pack(pady=20)
input_frame = ttk.LabelFrame(root, text="Training Data")
input_frame.pack(fill="x", padx=20, pady=10)
ttk.Label(input_frame, text="X1").grid(row=0, column=0, padx=10, pady=10)
entry_x1 = ttk.Entry(input_frame, width=60)
entry_x1.grid(row=0, column=1)
ttk.Label(input_frame, text="X2").grid(row=1, column=0, padx=10, pady=10)
entry_x2 = ttk.Entry(input_frame, width=60)
entry_x2.grid(row=1, column=1)
ttk.Label(input_frame, text="X3").grid(row=2, column=0, padx=10, pady=10)
entry_x3 = ttk.Entry(input_frame, width=60)
entry_x3.grid(row=2, column=1)
ttk.Label(input_frame, text="Y").grid(row=3, column=0, padx=10, pady=10)
entry_y = ttk.Entry(input_frame, width=60)
entry_y.grid(row=3, column=1)
predict_frame = ttk.LabelFrame(root, text="Prediction")
predict_frame.pack(fill="x", padx=20, pady=10)
ttk.Label(predict_frame, text="Predict X1").grid(row=0, column=0, padx=10, pady=10)
predict_x1 = ttk.Entry(predict_frame, width=20)
predict_x1.grid(row=0, column=1)
ttk.Label(predict_frame, text="Predict X2").grid(row=1, column=0, padx=10, pady=10)
predict_x2 = ttk.Entry(predict_frame, width=20)
predict_x2.grid(row=1, column=1)
ttk.Label(predict_frame, text="Predict X3").grid(row=2, column=0, padx=10, pady=10)
predict_x3 = ttk.Entry(predict_frame, width=20)
predict_x3.grid(row=2, column=1)
result_frame = ttk.LabelFrame(root, text="Output")
result_frame.pack(fill="both", expand=True, padx=20, pady=10)
result_label = tk.Label(
    result_frame,
    text="Result:",
    font=("Consolas", 12),
    justify="left",
    anchor="nw"
)
result_label.pack(anchor="w", padx=15, pady=15)
def train_model():
    global model, X, y, y_pred
    try:
        x1 = [float(i) for i in entry_x1.get().split(",")]
        x2 = [float(i) for i in entry_x2.get().split(",")]
        x3 = [float(i) for i in entry_x3.get().split(",")]
        y = [float(i) for i in entry_y.get().split(",")]
        if not (len(x1) == len(x2) == len(x3) == len(y)):
            result_label.config(
                text="Error:\nAll inputs must contain the same number of values."
            )
            return
        X = np.column_stack((x1, x2, x3))
        model = LinearRegression()
        model.fit(X, y)
        px1 = float(predict_x1.get())
        px2 = float(predict_x2.get())
        px3 = float(predict_x3.get())
        prediction = model.predict([[px1, px2, px3]])[0]
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        result = f"""
Intercept      : {model.intercept_:.4f}
Coefficient X1 : {model.coef_[0]:.4f}
Coefficient X2 : {model.coef_[1]:.4f}
Coefficient X3 : {model.coef_[2]:.4f}
Prediction     : {prediction:.4f}
R² Score       : {r2:.4f}
MAE            : {mae:.4f}
MSE            : {mse:.4f}
RMSE           : {rmse:.4f}
"""
        result_label.config(text=result)
    except ValueError:
        result_label.config(
            text="Please enter valid numeric values.\n\nExample:\n1,2,3,4"
        )
    except Exception as e:
        result_label.config(text=f"Error:\n{e}")
def actual_vs_predicted():
    if model is None:
        result_label.config(text="Please train the model first.")
        return
    plt.figure(figsize=(6,5))
    plt.scatter(y, y_pred, color="blue", s=60)
    minimum = min(min(y), min(y_pred))
    maximum = max(max(y), max(y_pred))
    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        color="red",
        linestyle="--",
        linewidth=2
    )
    plt.title("Actual vs Predicted")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.grid(True)
    plt.show()
def residual_plot():
    if model is None:
        result_label.config(text="Please train the model first.")
        return
    residuals = np.array(y) - y_pred
    plt.figure(figsize=(6,5))
    plt.scatter(y_pred, residuals, color="green", s=60)
    plt.axhline(
        y=0,
        color="red",
        linestyle="--"
    )
    plt.title("Residual Plot")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.grid(True)
    plt.show()
def coefficient_chart():
    if model is None:
        result_label.config(text="Please train the model first.")
        return
    labels = ["X1", "X2", "X3"]
    plt.figure(figsize=(6,5))
    plt.bar(labels, model.coef_)
    plt.title("Feature Coefficients")
    plt.xlabel("Features")
    plt.ylabel("Coefficient")
    plt.grid(axis="y")
    plt.show()
button_frame = ttk.Frame(root)
button_frame.pack(pady=10)
train_btn = ttk.Button(
    button_frame,
    text="Train & Predict",
    command=train_model
)
train_btn.grid(row=0, column=0, padx=10)
graph1 = ttk.Button(
    button_frame,
    text="Actual vs Predicted",
    command=actual_vs_predicted
)
graph1.grid(row=0, column=1, padx=10)
graph2 = ttk.Button(
    button_frame,
    text="Residual Plot",
    command=residual_plot
)
graph2.grid(row=0, column=2, padx=10)
graph3 = ttk.Button(
    button_frame,
    text="Coefficient Chart",
    command=coefficient_chart
)
graph3.grid(row=0, column=3, padx=10)
root.mainloop()
