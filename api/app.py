import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds

from flask import Flask, render_template

app = Flask(__name__, template_folder='./')

@app.route("/")
def hello_world():
    return render_template('index.html',data=['toto','tata',"lala"])
