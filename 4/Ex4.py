import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import math


def predict_using_sklean():
    df = pd.read_csv("/Users/periyzat/Documents/Study/Programming/MLCodebasics/4/test_scores.csv")
    reg = LinearRegression()
    reg.fit(df[["math"]], df.cs)
    return reg.coef_, reg.intercept_


def test_scores(math, cs):
    m_curr = b_curr = 0
    iterations = 1000000
    n = len(math)
    learning_rate = 0.0002
    cost_previous = 0

    for i in range(iterations):
        cs_predicted = m_curr * math + b_curr
        cost = (1 / n) * sum([val ** 2 for val in (cs - cs_predicted)])
        md = -(2 / n) * sum(math * (cs - cs_predicted))
        bd = -(2 / n) * sum(cs - cs_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        if np.isclose(cost, cost_previous, rtol=1e-20):
            break
            cost_previous = cost
        print("m {}, b {}, cost {}, iteration {}".format(m_curr, b_curr, cost, i))

        return m_curr, b_curr


if __name__ == "__main__":
    df = pd.read_csv("/Users/periyzat/Documents/Study/Programming/MLCodebasics/4/test_scores.csv")
    math = np.array(df.math)
    cs = np.array(df.cs)

m, b = test_scores(math, cs)
print("Using gradient descent function: Coef {} Intercept {}".format(m, b))

m_sklearn, b_sklearn = predict_using_sklean()
print("Using sklearn: Coef {} Intercept {}".format(m_sklearn, b_sklearn))
