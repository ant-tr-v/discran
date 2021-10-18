import numpy as np
from enum import Enum


class VerboseLevel:
    NONE = 0
    ONLY_ERROR = 1
    ONLY_FALSE = 2
    BASE = 3
    DEBUG = 4


def i_log(n: int, base=2):
    """integer logarithm so long arithmetic is ok"""
    if n <= 0:
        raise RuntimeError("not positive argument of i_log")
    x = 1
    p = 0
    while x * base <= n:
        p += 1
        x *= base
    return p


def check_predicate_Pn(n: int):
    """:returns function that verifies the predicate in Pn"""
    n_dim = n

    def check_predicate(predicate, vector, verbose_level=VerboseLevel.ONLY_FALSE) -> bool:
        """verifies the predicate in Pn
            :param predicate  predicate as 2d array
            we assume that predicate is a set of columns i.e. as a 2d matrix it is a transposed predicate
            :param vector  function as as a vector with length l = n^N,
            where N is a number of arguments
            :param verbose_level if not NONE, prints to stdout
            :raises RuntimeError on incorrect inputs
        """

        P = np.array(predicate, dtype=np.int64)
        if type(vector) == type(""):
            vector = vector.replace(" ", "")
        f = np.array([int(i) for i in vector])
        if len(P.shape) == 1:
            P.reshape([-1, 1])
        l = f.size
        N = i_log(l, n_dim)
        if l != n_dim ** N:
            raise RuntimeError("Incorrect vector length")
        n_pred, pred_place = P.shape
        p_selection = np.zeros([N], dtype=np.int64)
        f_powers = np.array([n_dim ** (N - i - 1) for i in range(N)], dtype=np.int64)
        for i in range(n_pred ** N):
            # selecting predicates
            if verbose_level == VerboseLevel.DEBUG:
                print(f"predicate selection {p_selection}")
            tmp_P = P[p_selection].T
            if verbose_level == VerboseLevel.DEBUG:
                print(f"selected predicates:\n{tmp_P}")
            # computing functions
            tmp_idx = np.sum(tmp_P * f_powers, axis=1)
            if verbose_level == VerboseLevel.DEBUG:
                print(f"selected indexes: {tmp_idx}")
            tmp_f = f[tmp_idx]
            if verbose_level == VerboseLevel.DEBUG:
                print(f"function on selected predicates: {tmp_f}")

            flag = False
            for p in P:
                if np.array_equal(tmp_f, p):
                    flag = True
                    break
            if not flag:
                if verbose_level >= VerboseLevel.ONLY_FALSE:
                    print(f"failed while processing combination:\n{tmp_P}\n"
                          f"predicate is:\n{P.T}\n"
                          f"value of function is: {tmp_f}")
                return False
            # computing new selection
            carry = 1
            for j in range(-1, -N - 1, -1):
                p_selection[j] += carry
                if p_selection[j] >= n_pred:
                    p_selection[j] -= n_pred
                    carry = 1
                else:
                    break
        return True

    return check_predicate


class check_class_by_predicate:
    def __init__(self, n:int, class_name:str, predicate, verbose_level=VerboseLevel.ONLY_FALSE):
        check_pred = check_predicate_Pn(n)
        predicate = np.array(predicate)
        self.class_name = class_name
        self.predicate = predicate
        self.verbose_level = verbose_level

        def checker(vector, verb_level):
            ret = check_pred(predicate, vector, verb_level)
            if ret:
                print("PASSED")
                print(f"predicate was:\n{predicate.T}")
                print(f"OK {vector} is in {class_name}")
            else:
                print(f"NO {vector} is NOT in {class_name}")
            return ret
        self.checker = checker

    def __call__(self, vector, verbose_level=None):
        if verbose_level is None:
            verbose_level = self.verbose_level
        return self.checker(vector, verbose_level)


def check_classes(class_checkers, verbose_level=VerboseLevel.ONLY_FALSE):
    def run(vector):
        res = [class_checker(vector, verbose_level) for class_checker in class_checkers]
        print("====" * len(res))
        print("\t".join([c.class_name for c in class_checkers]))
        print("\t".join(["+" if b else "-" for b in res]))
        print("====" * len(res))
    return run


check_predicate_P2 = check_predicate_Pn(2)
check_predicate_P3 = check_predicate_Pn(3)

p2_t0 = check_class_by_predicate(2, 'T0', [[0]])
p2_t1 = check_class_by_predicate(2, 'T1', [[1]])
p2_l = check_class_by_predicate(2, 'L', [[0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0],
                                         [1, 0, 0, 1], [1, 0, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]])
p2_s = check_class_by_predicate(2, 'S', [[0, 1], [1, 0]])
p2_m = check_class_by_predicate(2, 'M', [[0, 0], [0, 1], [1, 1]])
p2 = check_classes([p2_t0, p2_t1, p2_l, p2_s, p2_m])

p3_m0 = check_class_by_predicate(3, 'M0', [[0, 0], [0, 2], [1, 1], [1, 0], [1, 2], [2, 2]])
p3_m1 = check_class_by_predicate(3, 'M1', [[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]])
p3_m2 = check_class_by_predicate(3, 'M2', [[0, 0], [0, 1], [0, 2], [1, 1], [2, 1], [2, 2]])
p3_u0 = check_class_by_predicate(3, 'U0', [[0, 0], [1, 1], [1, 2], [2, 1], [2, 2]])
p3_u1 = check_class_by_predicate(3, 'U1', [[0, 0], [0, 2], [1, 1], [2, 0], [2, 2]])
p3_u2 = check_class_by_predicate(3, 'U2', [[0, 0], [0, 1], [1, 0], [1, 1], [2, 2]])
p3_c0 = check_class_by_predicate(3, 'C0', [[0, 0], [1, 1], [2, 2], [0, 1], [0, 2], [1, 0], [2, 0]])
p3_c1 = check_class_by_predicate(3, 'C1', [[0, 0], [1, 1], [2, 2], [1, 0], [1, 2], [0, 1], [2, 1]])
p3_c2 = check_class_by_predicate(3, 'C2', [[0, 0], [1, 1], [2, 2], [2, 0], [2, 1], [0, 2], [1, 2]])
p3_t0 = check_class_by_predicate(3, 'T0', [[0]])
p3_t1 = check_class_by_predicate(3, 'T1', [[1]])
p3_t2 = check_class_by_predicate(3, 'T2', [[2]])
p3_t01 = check_class_by_predicate(3, 'T01', [[0], [1]])
p3_t02 = check_class_by_predicate(3, 'T02', [[0], [2]])
p3_t12 = check_class_by_predicate(3, 'T12', [[1], [2]])
p3_b = check_class_by_predicate(3, 'B',   [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 1, 1],
                                           [0, 2, 0], [0, 2, 2], [1, 0, 0], [1, 0, 1], [1, 1, 0],
                                           [1, 1, 1], [1, 1, 2], [1, 2, 1], [1, 2, 2], [2, 0, 0],
                                           [2, 0, 2], [2, 1, 1], [2, 1, 2], [2, 2, 0], [2, 2, 1], [2, 2, 2]])
p3_s = check_class_by_predicate(3, 'S', [[0, 1], [1, 2], [2, 0]])
p3_l = check_class_by_predicate(3, 'L', [[0, 0, 0, 0], [0, 0, 1, 2], [0, 0, 2, 1],
                                         [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 2, 2],
                                         [0, 2, 0, 2], [0, 2, 1, 1], [0, 2, 2, 0],
                                         [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 2, 2],
                                         [1, 1, 0, 2], [1, 1, 1, 1], [1, 1, 2, 0],
                                         [1, 2, 0, 0], [1, 2, 1, 2], [1, 2, 2, 1],
                                         [2, 0, 0, 2], [2, 0, 1, 1], [2, 0, 2, 0],
                                         [2, 1, 0, 0], [2, 1, 1, 2], [2, 1, 2, 1],
                                         [2, 2, 0, 1], [2, 2, 1, 0], [2, 2, 2, 2]])

p3 = check_classes([p3_m0, p3_m1, p3_m2, p3_u0, p3_u1, p3_u2, p3_c0, p3_c1, p3_c2,
                    p3_t0, p3_t1, p3_t2, p3_t12, p3_t02, p3_t01, p3_b, p3_s])
p3l = check_classes([p3_m0, p3_m1, p3_m2, p3_u0, p3_u1, p3_u2, p3_c0, p3_c1, p3_c2,
                    p3_t0, p3_t1, p3_t2, p3_t12, p3_t02, p3_t01, p3_b, p3_s, p3_l])

