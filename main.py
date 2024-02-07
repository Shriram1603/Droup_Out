from flask import Flask
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle


app= Flask(__name__)

@app.route('/',methods=['GET'])
def hello():
    return 0;