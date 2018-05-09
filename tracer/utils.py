"""Utility functions module"""
import numpy as np


def distance(p1, p2):
    """Distance between two points"""
    return np.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))


def xamtfos(x, sig):
    aux = (1 / (np.sqrt(2 * np.pi * sig ** 2)))
    return -aux * (np.e ** -(x ** 2 / (2 * (sig ** 2)))) + aux + 1


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def is_in_box(point):
    return (
        point[0] >= 0 and
        point[0] <= 1 and
        point[1] >= 0 and
        point[1] <= 1
    )
