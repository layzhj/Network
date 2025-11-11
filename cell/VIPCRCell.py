from neuron import h, gui
from neuron.units import um


class VIPCRCell(object):
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
        self.soma.L, self.soma.diam = 20*um, 10*um
        self.radProx1.L, self.radProx1.diam = 50*um, 4*um
        self.radMed1.L, self.radMed1.diam = 50*um, 3*um
        self.radDist1.L, self.radDist1.diam = 100*um, 2*um
        self.lmM1.L, self.lmM1.diam = 50*um, 1.5*um
        self.lmt1.L, self.lmt1.diam = 50*um, 1*um
        self.radProx2.L, self.radProx2.diam = 50*um, 4*um
        self.radMed2.L, self.radMed2.diam = 50*um, 3*um
        self.radDist2.L, self.radDist2.diam = 100*um, 2*um
        self.lmM2.L, self.lmM2.diam = 50 * um, 1.5 * um
        self.lmt2.L, self.lmt2.diam = 50 * um, 1 * um
        self.oriProx1.L, self.oriProx1.diam = 50*um, 2*um
        self.oriMed1.L, self.oriMed1.diam = 50*um, 1.5*um
        self.oriDist1.L, self.oriDist1.diam = 50*um, 1*um
        self.oriProx2.L, self.oriProx2.diam = 50*um, 2*um
        self.oriMed2.L, self.oriMed2.diam = 50*um, 1.5*um
        self.oriDist2.L, self.oriDist2.diam = 50*um, 1*um

    def _setup_segments(self):
        d_lambda = 0.1
        frequency = 100
        for sec in self.all:
            sec.nseg = int((sec.L/(d_lambda*h.lambda_f(frequency, sec=sec)) + 0.9) / 2) * 2 + 1

    def _setup_biophysics(self):
        soma_nafcr = 0.015
        soma_kdrcr = 0.018
        soma_Kslowcr = 0.000725
        soma_iCcr = 0.00003
        soma_kadcr = 0.003
        soma_cancr = 0.001
        v_initcr = -70

        for sec in self.all:
            sec.cm = 1.2
            sec.Ra = 150
            sec.insert('pas')
            for seg in sec:
                seg.pas.g = 1/20000
                seg.pas.e = v_initcr
            if sec == self.soma:
                sec.insert('Nafcr')
                for seg in sec:
                    seg.Nafcr.gnafbar = soma_nafcr
                sec.insert('kdrcr')
                for seg in sec:
                    seg.kdrcr.gkdrbar = soma_kdrcr
                sec.insert('IKscr')
                for seg in sec:
                    seg.IKscr.gKsbar = soma_Kslowcr
                sec.insert('iCcr')
                for seg in sec:
                    seg.iCcr.gkcbar = soma_iCcr
                sec.insert('kadcr')
                for seg in sec:
                    seg.kadcr.gkabar = soma_kadcr
                sec.insert('cancr')
                for seg in sec:
                    seg.cancr.gcabar = soma_cancr
                sec.insert('cadyn')
            else:
                sec.insert('Nafcr')
                for seg in sec:
                    seg.Nafcr.gnafbar = 0.018*5
                sec.insert('kdrcr')
                for seg in sec:
                    seg.kdrcr.gkdrbar = 0.018*0.5

            h.ko0_k_ion = 3.82
            h.ki0_k_ion = 140
            h.celsius = 23

    def _setup_synapses(self):
        for sec in self.all:
            name = sec.name().split('.')[-1]
            self.synGroups['AMPA'][name] = {}
            self.synGroups['GABAA'][name] = {}
            self.synGroups['GABAB'][name] = {}
