#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This demo reproduces figures from
% Heinz et al. 2001 "Evaluating Auditory Performance Limits: I. One-Parameter Discrimination Using a Computational Model for the Auditory Nerve"

"""
from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt

import cochlea
from cochlea.stats import calc_psychophysical_performance_limit


def main():

    res = calc_psychophysical_performance_limit(
        model = cochlea.run_zilany2014_rate, 
        stim_fct = lambda delta : np.sin(2*np.pi*np.linspace(0,1,100e3)+delta),
        delta = 0.0001,
        cfs = np.linspace(125, 16e3, 10),
        model_pars = {'species': 'human'})
    print(res)

if __name__ == "__main__":
    main()
