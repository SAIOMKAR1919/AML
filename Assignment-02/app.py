import tkinter as tk
from tkinter import ttk
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
root = tk.Tk()
root.title("Multiple Linear Regression ")
root.geometry("900x700")
root.configure(bg="black")
title = tk.Label(
    root,
    text="Multiple Linear Regression ",
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
    text="Results will appear here.",
    font=("Arial", 12),
    justify="left",
    anchor="nw"
)
result_label.pack(anchor="w", padx=15, pady=15)

def train_model():
    try:
        x1 = [float(i) for i in entry_x1.get().split(",")]
        x2 = [float(i) for i in entry_x2.get().split(",")]
        x3 = [float(i) for i in entry_x3.get().split(",")]
        y = [float(i) for i in entry_y.get().split(",")]

        if not (len(x1) == len(x2) == len(x3) == len(y)):
            result_label.config(
                text="Error: All input lists must have the same number of values."
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
Intercept : {model.intercept_:.4f}

Coefficient X1 : {model.coef_[0]:.4f}
Coefficient X2 : {model.coef_[1]:.4f}
Coefficient X3 : {model.coef_[2]:.4f}

Prediction : {prediction:.4f}

R² Score : {r2:.4f}
MAE      : {mae:.4f}
MSE      : {mse:.4f}
RMSE     : {rmse:.4f}
"""

        result_label.config(text=result)

    except ValueError:
        result_label.config(
            text="Please enter valid numeric values.\nExample: 1,2,3,4"
        )

    except Exception as e:
        result_label.config(text=f"Error: {e}")

train_btn = ttk.Button(
    root,
    text="Train & Predict",
    command=train_model
)
train_btn.pack(pady=15)

root.mainloop()