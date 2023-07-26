from flask import Flask, render_template, request
import model
import re

app = Flask(__name__)


# Home Page
@app.route("/")
def hello():
    return render_template("index.html")


# Form Submission Page
@app.route("/sub", methods=['POST'])
def submit():
    if request.method == "POST":
        # Get the text input from the form
        text = request.form["tarea"]

        # Call the 'predict_statement()' function from the 'model' module to make predictions
        res = model.predict_statement(text)

    # Render the "sub.html" template with the prediction result ('res')
    return render_template("sub.html", v=res)


if __name__ == "__main__":
    # Start the Flask development server
    app.run(debug=True)
