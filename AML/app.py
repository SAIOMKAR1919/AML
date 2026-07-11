from flask import Flask, render_template, request
import matplotlib.pyplot as plt
app = Flask(__name__)
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        x = [float(i) for i in request.form["x"].split(",")]
        y = [float(i) for i in request.form["y"].split(",")]
        predict_x = float(request.form["predict_x"])
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(i * i for i in x)
        m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        c = (sum_y - m * sum_x) / n
        prediction = m * predict_x + c
        line = [m * i + c for i in x]
        plt.figure(figsize=(6,4))
        plt.scatter(x, y, color="blue", label="Data")
        plt.plot(x, line, color="red", label="Regression Line")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Linear Regression")
        plt.legend()
        plt.grid(True)
        plt.savefig("static/graph.png")
        plt.close()
        result = {
            "sum_x": round(sum_x,2),
            "sum_y": round(sum_y,2),
            "sum_xy": round(sum_xy,2),
            "sum_x2": round(sum_x2,2),
            "m": round(m,2),
            "c": round(c,2),
            "prediction": round(prediction,2)
        }
    return render_template("Linear.html", result=result)
if __name__ == "__main__":
    app.run(debug=True)
