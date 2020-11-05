import numpy as np
from timeit import default_timer as timer

L = 1000
dm = 9

R = np.arange(dm**2).reshape((dm,dm))
P = np.arange((dm*L)**2).reshape((dm*L,dm*L))
Pc = P.copy()

Correct = P + np.kron(np.eye(L),R)


def evaluate(func):
    def wrapper(*args):
        global Pc
        Pc = P.copy()
        start = timer()
        Answer = func(*args)
        end = timer()

        print(f"{func.__name__} ellapsed time {(end - start):.5f}, ", end="")

        if not np.allclose(Answer, Correct):
            print("incorrect")
        else:
            print("correct")

    return wrapper

@evaluate
def use_kron(L,R):
    return Pc + np.kron(np.eye(L), R)

@evaluate
def use_loop(L,R):
    for k in range(0,dm*L,dm):
        Pc[k:(k+dm),k:(k+dm)] += R
    return Pc

use_loop(L,R)
use_kron(L,R)

