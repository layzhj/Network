from neuron import h, gui
from neuron.units import um


class VIPCCKCell(object):
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

        self.radProx1 = h.Section(name='radProx1', cell=self)
        self.radMed1 = h.Section(name='radMed1', cell=self)
        self.radDist1 = h.Section(name='radDist1', cell=self)
        self.lmM1 = h.Section(name='lmM1', cell=self)
        self.lmt1 = h.Section(name='lmt1', cell=self)

        self.radProx2 = h.Section(name='radProx2', cell=self)
        self.radMed2 = h.Section(name='radMed2', cell=self)
        self.radDist2 = h.Section(name='radDist2', cell=self)
        self.lmM2 = h.Section(name='lmM2', cell=self)
        self.lmt2 = h.Section(name='lmt2', cell=self)

        self.oriProx1 = h.Section(name='oriProx1', cell=self)
        self.oriMed1 = h.Section(name='oriMed1', cell=self)
        self.oriDist1 = h.Section(name='oriDist1', cell=self)

        self.oriProx2 = h.Section(name='oriProx2', cell=self)
        self.oriMed2 = h.Section(name='oriMed2', cell=self)
        self.oriDist2 = h.Section(name='oriDist2', cell=self)

    def _setup_topology(self):
        self.radProx1.connect(self.soma(0))
        self.radMed1.connect(self.radProx1(1))
        self.radDist1.connect(self.radMed1(1))
        self.lmM1.connect(self.radDist1(1))
        self.lmt1.connect(self.lmM1(1))

        self.radProx2.connect(self.soma(1))
        self.radMed2.connect(self.radProx2(1))
        self.radDist2.connect(self.radMed2(1))
        self.lmM2.connect(self.radDist2(1))
        self.lmt2.connect(self.lmM2(1))

        self.oriProx1.connect(self.soma(0))
        self.oriMed1.connect(self.oriProx1(1))
        self.oriDist1.connect(self.oriMed1(1))

        self.oriProx2.connect(self.soma(1))
        self.oriMed2.connect(self.oriProx2(1))
        self.oriDist2.connect(self.oriMed2(1))

    def _create_lists(self):
        self.all = self.soma.wholetree()

    def _setup_geometry(self):
        self.soma.L, self.soma.diam = 20 * um, 10 * um
        self.radProx1.L, self.radProx1.diam = 100 * um, 4 * um
        self.radMed1.L, self.radMed1.diam = 100 * um, 3 * um
        self.radDist1.L, self.radDist1.diam = 200 * um, 2 * um
        self.lmM1.L, self.lmM1.diam = 100 * um, 1.5 * um
        self.lmt1.L, self.lmt1.diam = 100 * um, 1 * um
        self.radProx2.L, self.radProx2.diam = 100 * um, 4 * um
        self.radMed2.L, self.radMed2.diam = 100 * um, 3 * um
        self.radDist2.L, self.radDist2.diam = 200 * um, 2 * um
        self.lmM2.L, self.lmM2.diam = 100 * um, 1.5 * um
        self.lmt2.L, self.lmt2.diam = 100 * um, 1 * um
        self.oriProx1.L, self.oriProx1.diam = 100 * um, 2 * um
        self.oriMed1.L, self.oriMed1.diam = 100 * um, 1.5 * um
        self.oriDist1.L, self.oriDist1.diam = 100 * um, 1 * um
        self.oriProx2.L, self.oriProx2.diam = 100 * um, 2 * um
        self.oriMed2.L, self.oriMed2.diam = 100 * um, 1.5 * um
        self.oriDist2.L, self.oriDist2.diam = 100 * um, 1 * um

    def _setup_segments(self):
        d_lambda = 0.1
        frequency = 100
        for sec in self.all:
            sec.nseg = int((sec.L / (d_lambda * h.lambda_f(frequency, sec=sec)) + 0.9) / 2) * 2 + 1

    def _setup_biophysics(self):
        gna = 0.18
        gk = 0.013
        gleak = 0.00018 / 2
        c_m = 1.3

        for sec in self.all:
            sec.cm = c_m
            sec.Ra = 100
            sec.insert('ichan2vip')
            for seg in sec:
                seg.ichan2vip.gnatbar = gna
                seg.ichan2vip.gkfbar = gk
                seg.ichan2vip.gl = gleak
                seg.ichan2vip.el = -61.4
            sec.insert('ccanl')
            for seg in sec:
                seg.ccanl.catau = 10
                seg.ccanl.caiinf = 5.e-6
                # seg.ccanl.cao = 2
            sec.insert('borgka')
            for seg in sec:
                seg.borgka.gkabar = 0.00015 * 100
            sec.insert('nca')
            for seg in sec:
                seg.nca.gncabar = 0.0008
            sec.insert('lca')
            for seg in sec:
                seg.lca.glcabar = 0.005
            sec.insert('gskch')
            for seg in sec:
                seg.gskch.gskbar = 0.000002 * 2 * 0.01
            sec.insert('mykca')
            for seg in sec:
                seg.mykca.gkbar = 0.0002 * 10 * 10
            sec.enat = 55
            sec.ekf = -85
            sec.eks = -85
            sec.ek = -85
            sec.enca = 130
            sec.elca = 130

    def _setup_synapses(self):
        for sec in self.all:
            name = sec.name().split('.')[-1]
            self.synGroups['AMPA'][name] = {}
            self.synGroups['GABAA'][name] = {}
            self.synGroups['GABAB'][name] = {}
