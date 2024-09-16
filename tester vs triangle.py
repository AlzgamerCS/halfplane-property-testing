from utility import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
import sys
from random import seed
from sys import argv


def main():
    filename = input("Give me a filename: ")
    epsilon = eval(input("Epsilon (sample): "))
    tests = int(input("Number of tests: "))
    n = int(filename.split("_")[0])
    matrix = [None for _ in range(n)]

    
    
