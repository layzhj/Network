from neuron import h, gui
from neuron.units import um


class OLMCell(object):
    def __init__(self, gid=-1):
        super().__init__()
        self.gid = gid

        self.pre_list = []
        self.synGroups = {'AMPA': {}, 'GABAA': {}, 'GABAB': {}}
        self.internal_netcons = []
        self.external_netcons = {}

        self._setup_morphology()
        self._setup_topology()
        self._create_lists()
        self._setup_geometry()
        self._setup_segments()
        self._setup_biophysics()
        self._setup_synapses()

        self.spike_threshold = -45.0

        self.spike_detector = h.NetCon(self.soma(0.5)._ref_v, None, sec=self.soma)
        self.spike_detector.threshold = self.spike_threshold
        self.spike_times = h.Vector()
        self.spike_detector.record(self.spike_times)

    def _setup_morphology(self):

        self.soma = h.Section(name='soma', cell=self)
        self.dend1 = h.Section(name='dend1', cell=self)
        self.dend2 = h.Section(name='dend2', cell=self)
        self.axon = h.Section(name='axon', cell=self)

    def _setup_topology(self):
        self.dend1.connect(self.soma(1))
        self.dend2.connect(self.soma(0))
        self.axon.connect(self.soma(1))

    def _create_lists(self):
        self.all = self.soma.wholetree()

    def _setup_geometry(self):
        self.soma.L, self.soma.diam = 20 * um, 10 * um
        self.dend1.L, self.dend1.diam = 250 * um, 3 * um
        self.dend2.L, self.dend2.diam = 250 * um, 3 * um
        self.axon.L, self.axon.diam = 150 * um, 1.5 * um

    def _setup_segments(self):
        d_lambda = 0.1
        frequency = 100
        for sec in self.all:
            sec.nseg = int((sec.L / (d_lambda * h.lambda_f(frequency, sec=sec)) + 0.9) / 2) * 2 + 1

    def _setup_biophysics(self):
        Rm = 20000 * 2
        c_m = 1.6

        for sec in self.all:
            sec.cm = c_m
            sec.Ra = 150

        self.soma.insert('IA')
        for seg in self.soma:
            seg.IA.gkAbar = 0.0165

        self.soma.insert('Ih')
        for seg in self.soma:
            seg.Ih.gkhbar = 0.00035 * 0.1

        self.soma.insert('Ksoma')
        for seg in self.soma:
            seg.Ksoma.gksoma = 0.0319 * 1.5

        self.soma.insert('Nasoma')
        for seg in self.soma:
            seg.Nasoma.gnasoma = 0.0107 * 1.2
            seg.Nasoma.gl = 1 / Rm
            seg.Nasoma.el = -67

        self.dend1.insert('IA')
        for seg in self.dend1:
            seg.IA.gkAbar = 0.004 * 1.2

        self.dend1.insert('Kdend')
        for seg in self.dend1:
            seg.Kdend.gkdend = 20 * 0.023

        self.dend1.insert('Nadend')
        for seg in self.dend1:
            seg.Nadend.gnadend = 2 * 0.0117
            seg.Nadend.gl = 1 / Rm
            seg.Nadend.el = -65

        self.dend2.insert('IA')
        for seg in self.dend2:
            seg.IA.gkAbar = 0.004 * 1.2

        self.dend2.insert('Kdend')
        for seg in self.dend2:
            seg.Kdend.gkdend = 20 * 0.023

        self.dend2.insert('Nadend')
        for seg in self.dend2:
            seg.Nadend.gnadend = 2 * 0.0117
            seg.Nadend.gl = 1 / Rm
            seg.Nadend.el = -65

        self.axon.insert('Kaxon')
        for seg in self.axon:
            seg.Kaxon.gkaxon = 0.05104

        self.axon.insert('Naaxon')
        for seg in self.axon:
            seg.Naaxon.gnaaxon = 0.01712
            seg.Naaxon.gl = 1 / Rm
            seg.Naaxon.el = -67

    def _setup_synapses(self):
        for sec in self.all:
            name = sec.name().split('.')[-1]
            self.synGroups['AMPA'][name] = {}
            self.synGroups['GABAA'][name] = {}
            self.synGroups['GABAB'][name] = {}
