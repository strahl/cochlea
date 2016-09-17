#!/usr/bin/env python
"""Run inner ear model from [Goldwyn2012]_.

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as dsp
import thorns as th
import cochlea


def main():

    fs = 1000e3

    # Make stimulus
    # Simulation time (micro sec)
    PulseRate = 5000
    PhaseDuration = 40
    PulseLevel = 0.462
    t_begin = 0
    t_end = 1e6
    dt = 1
    t = np.arange(t_begin, t_end + dt / 2, dt)
    I = np.zeros(len(t))
    I[t % (1E6 / PulseRate) < 2 * PhaseDuration] = -PulseLevel
    I[t % (1E6 / PulseRate) < PhaseDuration] = PulseLevel

    # Run model
    anf = cochlea.run_goldwyn2012(t, I, PulseRate, seed=0)

if __name__ == "__main__":
    main()
