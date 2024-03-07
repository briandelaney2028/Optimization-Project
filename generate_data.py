import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_quad(coefs: list | np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Args:
        coefs (list or np.ndarray): coefficients of quadratic in form ax^2+bx+c
        n (int): number of data points to generate for curve
    Returns:
        x (np.ndarray): evenly spaced x values for quadratic
        y (np.ndarray): associated y values for quadratic
    """
    a, b, c = coefs
    # find vertex
    h = -b / (2*a)
    k = a*h**2 + b*h + c
    # find zero
    x2 = pow(-k/a, 1/2) + h
    # generate x data
    x = np.linspace(h, x2, num=n)
    # generate y data
    y = a*np.power(x-h, 2) + k
    y[-1] = 0.0
    return x, y

def generate_data(file: str, n:int=10) -> pd.DataFrame:
    """
    Args:
        file (str): filename for file containing simulation data
        n (int): number of data points to generate for curve
    Returns:
        df (pd.DataFrame): dataframe containing data for symbolic regressor
    """
    with open(file, 'r') as f:
        for i, line in enumerate(f.readlines()):
            items = line.strip().split(',')
            if i == 0:
                cols = ['x', 'y'] + items[3:]
                df = pd.DataFrame(columns=cols)
                continue
            items = [float(item) for item in items]
            x, y = generate_quad(items[0:3], n)
            for j in range(n):
                row = [x[j], y[j]] + items[3:]
                df.loc[len(df.index)] = row
    return df



if __name__=="__main__":
    df = generate_data('test.csv')
    print(df)