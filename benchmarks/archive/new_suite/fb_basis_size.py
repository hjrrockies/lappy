"""
Benchmark tests for the number of Fourier-Bessel basis functions
"""
from lappy import *
import numpy as np
from .domains import *
from .benchmarking import build_eigprob

def test_eq_tri():
    