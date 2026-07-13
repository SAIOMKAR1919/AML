from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import math
app = Flask(__name__)
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        x = [float(i) for i in request.form["x"].split(",")]
        y = [float(i) for i in request.form["y"].split(",")]
        predict_x = float(request.form["predict_x"])
        if len(x) != len(y):
            return "Error: X and Y must have the same number of values.
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(i * i for i in x)
        denominator = n * sum_x2 - sum_x ** 2
        if denominator == 0:
            return "Cannot calculate Linear Regression."
        m = (n * sum_xy - sum_x * sum_y) / denominator
        c = (sum_y - m * sum_x) / n
        prediction = m * predict_x + c
        predicted_y = [m * i + c for i in x]
        errors = [y[i] - predicted_y[i] for i in range(n)]
        mse = sum(error ** 2 for error in errors) / n
        rmse = math.sqrt(mse)
        mean_y = sum_y / n
        sst = sum((value - mean_y) ** 2 for value in y)
        sse = sum(error ** 2 for error in errors)
        r2 = 1 - (sse / sst)
        plt.figure(figsize=(6,4))
        plt.scatter(x, y, color="blue", label="Actual Data")
        plt.plot(x, predicted_y, color="red", linewidth=2, label="Regression Line")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Linear Regression")
        plt.grid(True)
        plt.legend()
        plt.savefig("static/graph.png")
        plt.close()
        table = []
        for i in range(n):
            table.append({
                "x": x[i],
                "y": y[i],
                "pred": round(predicted_y[i], 2),
                "error": round(errors[i], 2)
            })
        result = {
            "n": n,
            "sum_x": round(sum_x, 2),
            "sum_y": round(sum_y, 2),
            "sum_xy": round(sum_xy, 2),
            "sum_x2": round(sum_x2, 2),
            "m": round(m, 4),
            "c": round(c, 4),
            "prediction": round(prediction, 2),
            "predict_x": predict_x,
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
            "table": table
        }
    return render_template("Linear.html", result=result)
if __name__ == "__main__":
    app.run(debug=True)
