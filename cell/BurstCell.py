from neuron import h


class BurstCell(object):
    def __init__(self, gid=-1):
        super().__init__()
        self.gid = gid
        self._setup_biophysics()
        self.ncstim = []

        self.spike_detector = h.NetCon(self.stim, None)
        self.spike_times = h.Vector()
        self.spike_detector.record(self.spike_times)

    def _setup_biophysics(self):
        self.soma = h.Section(name='soma', cell=self)
        self.stim = h.BurstStim2(sec=self.soma)
        self.stim.start = 0
        self.stim.number = 10000
        self.stim.interval = 10
        self.stim.noise = 0
        self.stim.burstint = 100
        self.stim.burstlen = 100

    def connect2target(self, target, delay=1, weight=0.04):
        self.ncstim.append(h.NetCon(self.stim, target))
        self.ncstim[-1].delay = delay
        self.ncstim[-1].weight[0] = weight
        return self.ncstim[-1]
