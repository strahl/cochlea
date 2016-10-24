import pandas as pd
import numpy as np
from . import goldwyn2012


def run_goldwyn2012(t, signal, pulserate, seed, threshold=0.852, relative_spread=0.0487, chronaxie=276, tau_sum=250, jitter=85.5, abs_ref=332, rel_ref=411, abs_relative_spread=199.0, rel_relative_spread=423.0, threshold_phase_duration=40):
    """Run the cochlear implant stimulation auditory nerve fibre model by [Goldwyn2012]_ arXiv:1201.5428.
    Point process model developed by Goldwyn, Rubinstein, Shea-Brown for response of auditory nerve fiber to cochlear implant stimulation

    This code is based on the original Matlab implementation provided by the authors [Last updated: June 2012 (JHG)]
    Converted to Python Aug 2016 (stef@nstrahl.de)

    TODO: refactor to use SI units
    TODO: Compare outputs of both implementation in unit tests (for now this was done manually...)

    Parameters
    ----------
    t  : array_like
        time carrier of signal (us).
    signal : array_like
        electric stimulus signal (mA).
    pulserate : int
        pulserate of the signal (Hz).
    seed : int
        Random seed.
    threshold : float
        (mA) default: 0.852
    relative_spread : float
        default: 0.0487
    chronaxie : float
        (us) default: 276
    tau_sum : float
        (us) de fault: 250  # summation time constant
    jitter : float
        (us) default: 85.5
    abs_ref : float
        (us) default: 332
    rel_ref : float
        (us) default: 411  # Time scale of relative refractory period
    abs_relative_spread : float
        default: 199.0
    rel_relative_spread : float
        default: 424.0

    Returns
    -------
    spike_train
        Auditory nerve spike train.

    References
    ----------
    If you are using results of this version of the model
    in your research, please cite [Goldwyn2012]_.

    .. [Goldwyn2012] JH Goldwyn, JT Rubinstein, E Shea-Brown (2012).
       A point process framework for modeling electrical stimulation
       of the auditory nerve. J Neurophysiol 108:1430-1452, 2012

    """
    assert signal.ndim == 1
    np.random.seed(seed)

    model = goldwyn2012.Goldwyn2012()
    model.init_neuron(threshold=0.852, relative_spread=0.0487, chronaxie=276, tau_sum=250, jitter=85.5, abs_ref=332,
                      rel_ref=411, abs_relative_spread=199.0, rel_relative_spread=423.0, threshold_phase_duration=40)
    spikes = model.simulate(t, signal, pulserate)

    trains = []
    trains.append({
        'spikes': spikes,
        'duration': t[-1] - t[0],
        'cf': 0,
        'type': 'toLookUp'
    })

    spike_trains = pd.DataFrame(list(trains))
    return spike_trains
