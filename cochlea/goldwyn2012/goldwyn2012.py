import sys
import time
import pylab
import numpy as np
import scipy.optimize
import scipy.special
import scipy.linalg


def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr = "{0:." + str(decimals) + "f}"
    percents = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = u'\u2588' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' %
                     (prefix, bar, percents, '%', suffix)),
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')
        sys.stdout.flush()


class Goldwyn2012(object):
    """Returns spike train using Goldwyn2012 et al. 2012 model"""

    def __init__(self):
        self.threshold = np.nan
        self.relative_spread = np.nan
        self.chronaxie = np.nan
        self.tau_sum = np.nan
        self.jitter = np.nan
        self.abs_relative_spread = np.nan
        self.rel_relative_spread = np.nan
        self.threshold_phase_duration = np.nan
        self.alpha_approx = np.nan

    def __jitter_fit(self, jitter, tau_j, phase_dur, tau_kappa_fit, beta_fit, alpha_fit, kappa_fit, thresh_val, dt):
        tend = 5000
        t = np.arange(0, tend + dt / 2, dt, dtype=np.float64)
        D = phase_dur  # phase duration
        w = np.zeros(len(t))
        w[0:int(D / dt)] = 1
        w[int(D / dt): int(2 * D / dt)] = -beta_fit
        # biphasic
        wFilter = np.convolve(1 / tau_kappa_fit *
                              np.exp(-t / tau_kappa_fit), w) * dt
        W = wFilter[0: len(t)] * (wFilter[0: len(t)] > 0)
        # output of stimulus filter, no kappa

        JWFilter = np.convolve(
            1 / tau_j * np.exp(-t / tau_j), W ** alpha_fit) * dt
        JW = JWFilter[0: len(t)] * (JWFilter[0: len(t)] > 0)
        lambda1 = (kappa_fit * thresh_val) ** alpha_fit * JW
        Lambda = np.hstack((0, scipy.integrate.cumtrapz(lambda1) * dt))
        f = 2 * lambda1 * np.exp(-Lambda)
        jj = np.sqrt(np.trapz(t ** 2.0 * f) - np.trapz(t * f) ** 2)

        JitterError = np.linalg.norm(jj - jitter)
        return JitterError

    def __summation_fit(self, beta, tau_sum, phase_dur, alpha, tau_kappa, dt, stimulus_type, ipi_all):
        # Total time to evaluate FE
        tend = 5000

        # Pulse Tain Parameters
        D = phase_dur

        # t
        t = np.arange(0, tend + dt / 2, dt)

        ThreshRatio = np.zeros(len(ipi_all))
        for i in np.arange(len(ipi_all)):
            IPI = ipi_all[i]

            # % PseudoMonophasic
            if stimulus_type == 'pseudo':
                tau = 5E6  # % Cartee time constant defined by hardware
                w1 = np.zeros(len(t))
                w1[0: int(D / dt)] = 1
                tt = t[int(D / dt): int(IPI / dt)] - D
                w1[int(D / dt): int(IPI / dt)] += - beta * D * np.exp(-(tt - D) / tau) / \
                    (tau * (1 - np.exp(-(IPI - D) / tau)))
                w2 = w1.copy()
                w2[int(IPI / dt): int(IPI / dt + D / dt)] += 1  # % 2nd pulse
                w2[int((IPI + D) / dt): int((2 * IPI) / dt)] += - beta * D * np.exp(-(tt - D) / tau) / \
                    (tau * (1 - np.exp(-(IPI - D) / tau)))

            elif stimulus_type == 'bi':  # % Biphasic
                w1 = np.zeros(len(t))
                w1[1: D / dt] = 1
                w1[D / dt: (2 * D) / dt] += - beta
                w2 = w1.copy()
                w2[IPI / dt: IPI / dt + D / dt] += 1  # % 2nd pulse
                w2[(IPI + D) / dt: (IPI + 2 * D) / dt] = - Beta  # 2nd pulse

            # Apply K filter
            w1Filter = np.convolve(
                1 / tau_kappa * np.exp(-t / tau_kappa), w1) * dt
            w2Filter = np.convolve(
                1 / tau_kappa * np.exp(-t / tau_kappa), w2) * dt
            # output of stimulus filter, no kappa because cancels out later
            W1 = w1Filter[0:len(t)] * (w1Filter[0:len(t)] > 0)
            # output of stimulus filter, no kappa because cancels out later
            W2 = w2Filter[0:len(t)] * (w2Filter[0:len(t)] > 0)

            # Apply nonlinearity and Integrate Responses
            intW1 = np.trapz(W1**alpha) * dt
            intW2 = np.trapz(W2**alpha) * dt

            ThreshRatio[i] = (intW1 / intW2) ** (1 / alpha)

        # Error Using Equation in Cartee et al
        SumError = np.linalg.norm(
            (1 - .5 * np.exp(-ipi_all / tau_sum)) - ThreshRatio)
        return SumError

    def __chronaxie_fit(self, tau_kappa, chronaxie, max_phase_dur, alpha, dt):
        """ TauKappa will be fit based on this function"""

        # Total time to evaluate FE
        tend = 5000

        # Phase durations
        D0 = chronaxie / dt
        Dmax = max_phase_dur / dt

        t = np.arange(0, tend + dt / 2, dt)

        # waveform functions, always monophasic
        w1 = np.zeros((len(t)))
        w1[0:int(D0 / dt)] = 1
        w2 = np.zeros((len(t)))
        w2[0:int(Dmax / dt)] = 1

        # Apply K filter
        w1Filter = np.convolve(1 / tau_kappa * np.exp(-t / tau_kappa), w1) * dt
        w2Filter = np.convolve(1 / tau_kappa * np.exp(-t / tau_kappa), w2) * dt
        # output of stimulus filter, no kappa because cancels out later
        W1 = w1Filter[0:len(t)] * (w1Filter[0:len(t)] > 0)
        # output of stimulus filter, no kappa because cancels out later
        W2 = w2Filter[0:len(t)] * (w2Filter[0:len(t)] > 0)

        # Apply nonlinearity and Integrate Responses
        intW1 = np.trapz(W1**alpha) * dt
        intW2 = np.trapz(W2**alpha) * dt

        # Error function that can be used to determine TauKappa
        CAerror = np.linalg.norm(2**alpha - (intW2 / intW1))
        return CAerror

    def __parameterize(self, relative_spread, chronaxie, tau_sum, threshold, jitter, known_input):
        """This function parameterizes the AN model presented in
         based on Parameterize.m by Joshua Goldwyn (Version 5/10/11)

        %%% Values used in simulations
        %[Alpha, AlphaApprox, TauKappa, Beta, Kappa, TauJ] = Parameterize(.0487, 276, 250, .852, 85.5,[0 0 0 0 0]);
        %[Alpha, AlphaApprox, TauKappa, Beta, TauJ] =   [25.6337   24.5196 325.3700 0.3330   9.3649 96.9336]
        %%%
        """
        dt = 1.0  # mu sec, for evaluating integrals and filters
        MaxPhaseDur = 2000.0   # Used in Chronaxie mu sec

        #%Pulses for Summation Experiment (Cartee et al 2006)
        SummationPhaseDur = 50.
        SummationPulseShape = 'pseudo'
        IPIall = np.array([100, 200, 300])

        # Pulses for Threshold Experiment (Miller et al 2001 biphasic
        # experiments)
        ThresholdPhaseDur = 40.
        ThresholdPulseShape = 'bi'

        # Pulses for Jitter Experiment (Miller et al 2001 biphasic experiments)
        JitterPhaseDur = 40
        JitterPulseShape = 'bi'

        # Approximation of Alpha vs RS
        AlphaApproximation = lambda rs: rs**(-1.0587)
        # %%%% NOTE THIS CODE GIVES THE APPROXIMATION:
        # RSfunc = @(a) sqrt(gamma(1+2./a)./(gamma(1+1./a)).^2 - 1);
        # Alpha = linspace(1,100,1E3);
        # p = fminsearch(@(p) norm(RSfunc(Alpha).^p-Alpha),-1);
        # %%%%%%%%%%%%%%%%

        if known_input[0] > 0:
            Alpha = known_input[0]
        else:
            # %%% First Compute Alpha for given RS
            # RS as a function of alpha using, using st dev / mean of Weibull
            # distribution
            RSfunc = lambda a: np.sqrt(scipy.special.gamma(
                1 + 2 / a) / (scipy.special.gamma(1 + 1 / a))**2 - 1)
            Alpha = scipy.optimize.fmin(func=lambda a: scipy.linalg.norm(
                RSfunc(a) - relative_spread, ord=2), x0=10)[0]
        AlphaApprox = AlphaApproximation(relative_spread)

        if known_input[1] > 0:
            TauKappa = known_input[1]
        else:
            # %%% Next Compute TauKappa for given Chronaxie
            # Maximum Phase Duration Used in Van den Honert and Stypulkowski
            # 1984
            TauKappa = scipy.optimize.fmin(func=lambda tk: self.__chronaxie_fit(
                tk, chronaxie, MaxPhaseDur, AlphaApprox, dt), x0=100)

        if known_input[2] > 0:
            Beta = known_input[2]
        else:
            # %%% Next Compute Beta for given Summation Time Constant
            Beta = scipy.optimize.fminbound(func=lambda b: self.__summation_fit(
                b, tau_sum, SummationPhaseDur, AlphaApprox, TauKappa, dt, SummationPulseShape, IPIall), x1=0, x2=1)

        if known_input[3] > 0:
            Kappa = known_input[3]
        else:
            # % % % Next Compute Kappa for given Threshold
            tend = 5000
            t = np.arange(0, tend + dt / 2, dt)
            D = ThresholdPhaseDur
            # % phase duration
            if ThresholdPulseShape == 'bi':
                w = np.zeros(len(t))
                w[0: int(D / dt)] = 1
                w[int(D / dt): int(2 * D / dt)] = -Beta  # % biphasic
            wFold = 0
            wold = 0
            wFilter = np.zeros(len(t))
            for i in np.arange(len(t)):
                # % conv(1 / TauKappa * exp(-t / TauKappa), w) * dt;
                wFilter[i] = wFold * np.exp(-dt / TauKappa) + (
                    dt / (2 * TauKappa)) * (wold * np.exp(-dt / TauKappa) + w[i])
                wFold = wFilter[i]
                wold = w[i]

            W = wFilter[0: len(t)] * (wFilter[0: len(t)] > 0)
            #% output of stimulus filter, no kappa
            intW = np.trapz(W ** AlphaApprox) * dt
            #% integrate response
            Kappa = (np.log(2) / intW) ** (1 / AlphaApprox) / threshold

        if known_input[4] > 0:
            TauJ = known_input[4]
        else:
            #%%% Last, compute TauJ for Jitter Filter
            # TauJ = fminsearch(@(tj) JitterFit(Jitter, tj, JitterPhaseDur,
            # TauKappa, Beta, AlphaApprox, Kappa, Threshold, dt), 100)
            TauJ = scipy.optimize.fmin(func=lambda tj: self.__jitter_fit(
                jitter, tj, JitterPhaseDur, TauKappa, Beta, AlphaApprox, Kappa, threshold, dt), x0=100)

        return (Alpha, AlphaApprox, TauKappa[0], Beta, Kappa, TauJ[0])

    def __current(self, t, PulseLevel, PhaseDuration, PulseRate):
        """ Define the current input for model"""

        # %%% Here is an example for biphasic pulse trains
        I = np.zeros(len(t))
        I[t % (1E6 / PulseRate) < 2 * PhaseDuration] = -PulseLevel
        I[t % (1E6 / PulseRate) < PhaseDuration] = PulseLevel
        return I

    def init_neuron(self, threshold=0.852, relative_spread=0.0487, chronaxie=276, tau_sum=250, jitter=85.5, abs_ref=332, rel_ref=411, abs_relative_spread=199.0, rel_relative_spread=423.0, threshold_phase_duration=40):
        """ set parameters of neuron and pre-caches values
        == Neural parameters ==
        threshold (mA) default: 0.852
        relative_spread
        chronaxie (us) default: 276
        tau_sum (us) default: 250  # summation time constant
        jitter (us) default: 85.5
        abs_ref (us) default: 332
        rel_ref (us) default: 411  # Time scale of relative refractory period
        abs_relative_spread default: 199.0
        rel_relative_spread default: 424.0
        # Phase duration used for stimulation that defines threshold
        threshold_phase_duration default: 40
        """
        self.threshold = threshold
        self.relative_spread = relative_spread
        self.chronaxie = chronaxie
        self.tau_sum = tau_sum
        self.jitter = jitter
        self.abs_ref = abs_ref
        self.rel_ref = rel_ref
        self.abs_relative_spread = abs_relative_spread
        self.rel_relative_spread = rel_relative_spread
        self.threshold_phase_duration = threshold_phase_duration

        # Compute associated model parameters
        (Alpha, self.alpha_approx, TauKappa, Beta, Kappa, TauJ) = self.__parameterize(
            relative_spread, chronaxie, tau_sum, threshold, jitter, np.array((0, 0, 0, 0, 0)))

        self.tau_kappa = TauKappa
        self.kappa = Kappa
        self.tau_j = TauJ
        self.beta = Beta

        # % Build look up table that will be used to determine kappa for varying values of alpha
        self.intW = np.zeros(5000)
        tend = 5000
        dt = 1.0
        tt = np.arange(0, tend + dt, dt)

        wFilter = np.zeros(len(tt))
        w = np.zeros(len(tt))
        w[0: int(self.threshold_phase_duration / dt)] = 1
        w[int(self.threshold_phase_duration / dt)          : int(2 * self.threshold_phase_duration / dt)] = -Beta
        # %biphasic
        wFold = 0
        wold = 0
        for ii in np.arange(len(tt)):
            wFilter[ii] = wFold * np.exp(-dt / TauKappa) + (
                dt / (2 * TauKappa)) * (wold * np.exp(-dt / TauKappa) + w[ii])
            wFold = wFilter[ii]
            wold = w[ii]

        # output of stimulus filter, no kappa
        W = wFilter[0: len(tt)] * (wFilter[0: len(tt)] > 0)

        for i in np.arange(len(self.intW)):
            ialpha = (i + 1) / 100.0
            self.intW[i] = np.trapz(W ** ialpha) * dt  # integrate response

    def __theta_func(self, theta, t):
        """
        # % Define Refractory Functions (time in these functions is in micro s)
        theta_func = @(theta,t) 99999.*(t<AbsRef) + (t>AbsRef)*theta / (1.0 - exp(-(t-AbsRef)/RelRef)) ;
        """
        if t <= self.abs_ref:
            return 99999.0
        else:
            return theta / (1.0 - np.exp(-(t - self.abs_ref) / self.rel_ref))

    def __rs_func(self, rs, t):
        """
        # % Define Refractory Functions (time in these functions is in micro s)
        rs_func = @(rs,t) min(.5, rs*(t<AbsRef) + (t>AbsRef)*rs / ( 1.0 - exp(-(t-AbsRS)/RelRS)));
        """
        if t <= self.abs_ref:
            ret = rs
        else:
            ret = rs / \
                (1.0 - np.exp(-(t - self.abs_relative_spread) / self.rel_relative_spread))
        return np.min((0.5, ret))

    def simulateMatlabExample(self, t_begin=0, t_end=1e6, dt=1.0, CurrentLevel=0.462, PhaseDuration=40, PulseRate=5000):
        """
        == Simulation Parameters ==
        Length of stimulus,
        pulse rate (pps) default: 5000
        phase duration (us) default:40
        current level per pulse (mA) default: 0.462
        == Output variables ==
        SpikeCount (number of spikes)
        SpikeTrain (list of spike times in  micro sec)
        """

        # Simulation time (micro sec)
        t = np.arange(t_begin, t_end + dt / 2, dt)

        # Stimulus (Here is an example of a train of constant current level,
        # biphasic pulses)
        # % Vector of Parameter Values passed in to Current.m
        # % Using Current.m to define stimulus
        I = self.__current(t, CurrentLevel, PhaseDuration, PulseRate)
        return self.simulate(t, I, PulseRate)

    def simulate(self, t, I, PulseRate):
        """
        == Simulation Parameters ==
        time carrier of stimulus,
        stimulus (mA) default
        Pulserate (Hz) needed to know onset of each pulse (TODO: this should derived from the signal...)
        == Output variables ==
        SpikeCount (number of spikes)
        SpikeTrain (list of spike times in  micro sec)
        """

        nt = len(t)
        dt = t[1] - t[0]

        # %%%%%%%%%%%%% Run Point Process Model And Record Spike Times %%%%%%%%
        # % Initial Values
        v = 0
        w = 0
        ci = 0
        Integrate_ci = 0
        TimeSinceSpike = 1E6  # Make it big if want no spike history at stimulus onset
        AlphaVal = self.alpha_approx
        ThresholdUpdate = self.threshold
        RelativeSpreadUpdate = self.relative_spread
        SpikeTrain = list()
        r = -np.log(np.random.rand())
        Iin = 0

        for i in np.arange(1, nt):
            if (i % 50000 == 0):
                printProgress(i, nt, prefix='Progress:',
                              suffix='Complete', barLength=50)

            TimeSinceSpike = TimeSinceSpike + dt

            Iin_old = Iin
            if I[i - 1] > 0:  # Positive phase
                Iin = I[i - 1]
            elif I[i - 1] < 0:  # Negative phase
                Iin = self.beta * I[i - 1]
            else:
                Iin = 0

            if (TimeSinceSpike < self.abs_ref):  # in absolute refractory period
                v = 0
            else:  # Update state variable
                # % Stimulus filter, convolution using Trapezoid method
                v = v * np.exp(-dt / self.tau_kappa) + (dt * self.kappa / (2.0 * self.tau_kappa)
                                                        ) * (Iin_old * np.exp(-dt / self.tau_kappa) + Iin)

            # Apply nonlinearity
            w_old = w  # Save this old value for later calculation
            w = np.max((0, v)) ** AlphaVal

            # Apply Jitter Filter
            ci_old = ci  # Conditional Intensity Value
            ci = ci * np.exp(-dt / self.tau_j) + dt / (2.0 * self.tau_j) * \
                (w + w_old * np.exp(-dt / self.tau_j))

            # Integrate Conditional Intensity with Trapezoid method
            Integrate_ci = Integrate_ci + dt * ci  # (dt * (ci + ci_old)) / 2

            # Check for spikes
            if Integrate_ci > r:  # SPIKE!
                # Record Spike
                SpikeTrain.append(t[i])

                # Reset dynamical variables
                v = 0
                w = 0
                ci = 0
                Integrate_ci = 0
                TimeSinceSpike = 0

                # Draw New Random Number
                r = -np.log(np.random.rand())

            # Update Spike Dependent Parameter Values at onset of next pulse
            if t[i] % (1E6 / (PulseRate * dt)) == 0:  # % new pulse
                RelativeSpreadUpdate = self.__rs_func(
                    self.relative_spread, TimeSinceSpike)
                if TimeSinceSpike == self.abs_ref:
                    ThresholdUpdate = 0
                else:
                    ThresholdUpdate = self.__theta_func(
                        self.threshold, TimeSinceSpike)
                AlphaVal = RelativeSpreadUpdate ** -1.0587
                # approximation to exact expression
                if TimeSinceSpike > self.abs_ref:
                    self.kappa = ((np.log(2) / self.intW[int(np.round(AlphaVal * 100)) - 1]
                                   ) ** (1 / AlphaVal)) / ThresholdUpdate
                else:
                    self.kappa = 0
        # % End loop over time steps
        print("Found {} spikes.".format(len(SpikeTrain)))
        return SpikeTrain

if __name__ == '__main__':
    print("Sample code to generate a spike train")
    timestamp_start = time.time()
    model = Goldwyn2012()
    model.init_neuron()
    timestamp_init = time.time()
    SpikeTrain = model.simulateMatlabExample()
    timestamp_stop = time.time()
    print("Init: {:1.1f} seconds".format(timestamp_init - timestamp_start))
    print("Simulate: {:1.1f} seconds".format(timestamp_stop - timestamp_init))
    print("Total: {:1.1f} seconds".format(timestamp_stop - timestamp_start))

    maxt = 25
    x, y = np.histogram(np.diff([0.001 * c for c in SpikeTrain]),
                        bins=np.arange(0, maxt, 0.05))
    pylab.plot(y[0:-1], x)
    pylab.xlabel("Interspike interval (ms)")
    pylab.ylabel("Number of occurences")
