from neuron import h

class Septal(object):

    def __init__(self, gid=-1):
        super(Septal, self).__init__()
        self.gid = gid
        self._setup_biophysics()
        self.spike_detector = h.NetCon(self.stim, None)
        self.spike_times = h.Vector()
        self.spike_detector.record(self.spike_times)

    def _setup_biophysics(self):
        self.soma = h.Section(name='soma', cell=self)
        self.stim = h.BurstStim2(sec=self.soma)
        self.stim.number = 10000
        self.stim.start = 0
        self.stim.interval = 10
        self.stim.noise = 0
        self.stim.burstint = 100
        self.stim.burstlen = 100
