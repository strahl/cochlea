#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

__author__ = "Marek Rudnicki"

import numpy as np
import pandas as pd
import logging

import marlib as mr
import marlib.thorns as th
import marlib.waves as wv

from bisection import find_zero


def calc_spont_threshold(model, model_pars=None):

    if model_pars is None:
        model_pars = {}

    pars = dict(model_pars)

    fs = pars.setdefault('fs', 100e3)
    pars.setdefault('cf', 1e3)

    tmax = 250e-3
    s = np.zeros(fs*tmax)

    anf = model(
        sound=s,
        anf_num=(10000,0,0),
        seed=0,
        **pars
    )

    rates = [th.calc_rate(train) for _,train in anf.iterrows()]
    rates = np.array(rates)

    threshold = rates.mean() + rates.std()

    logging.debug("mean + std rate = {}".format(threshold))

    return threshold



def error_func(dbspl, model, cf, spont_rate, model_pars):

    if model_pars is None:
        model_pars = {}

    pars = dict(model_pars)

    fs = pars.setdefault('fs', 100e3)

    tone_duration = 250e-3
    onset = 15e-3

    s = wv.make_ramped_tone(
        fs,
        cf,
        duration=tone_duration,
        ramp=2.5e-3,
        pad=0,
        dbspl=dbspl
    )

    anf = model(
        sound=s,
        anf_num=(1000, 0, 0),
        cf=cf,
        seed=0,
        **pars
    )

    trains = th.trim(anf, onset, None)
    rate = th.calc_rate(trains)

    error = rate - spont_rate

    logging.debug("{} {} {}".format(dbspl, rate, error))

    return error



def calc_hearing_threshold_rate(model, cf, spont_rate, model_pars=None):

    kwargs = {
        'model':model,
        'model_pars': model_pars,
        'cf': cf,
        'spont_rate': spont_rate
    }
    dbspl_opt = find_zero(
        error_func,
        kwargs=kwargs,
        x1=-100,
        x2=100,
        xtol=0.1
    )

    return dbspl_opt






def calc_hearing_thresholds_rate(
        model,
        cfs=None,
        **model_pars
):

    if cfs is None:
        cfs = np.logspace(np.log10(100), np.log10(16000), 32)


    spont_rate = mr.apply(
        calc_spont_threshold,
        model=model,
        model_pars=model_pars
    )

    space = [
        {
            'model': model,
            'model_pars': model_pars,
            'spont_rate': spont_rate,
            'cf': cf
        }
        for cf in cfs
    ]


    thresholds = mr.map(
        calc_hearing_threshold_rate,
        space,
        # backend='multiprocessing'
    )
    thresholds = list(thresholds)

    thresholds = pd.Series(thresholds, index=cfs)

    return thresholds






def main():
    import matplotlib.pyplot as plt
    import cochlea

    # spont_rate = calc_spont_threshold(
    #     cochlea.run_zilany2009,
    #     {}
    # )
    # print(spont_rate)


    # r = calc_hearing_threshold_rate(
    #     model=cochlea.run_zilany2009,
    #     cf=1e3,
    #     model_pars={},
    #     spont_rate=spont_rate
    # )

    # print(r)


    ths = calc_hearing_thresholds_rate(
        model=cochlea.run_zilany2009
    )
    print(ths)

    ths.plot(logx=True)
    plt.show()


if __name__ == "__main__":
    main()
