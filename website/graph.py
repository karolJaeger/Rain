import numpy as np

from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "Witaj na mojej stronie!"

if __name__ == '__main__':
    app.run(debug=True)
