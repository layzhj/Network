from neuron import h, gui
from neuron.units import um

class PyramidalCell(object):

    def __init__(self, gid=-1, k_cal=1, k_kca=1):
        """
        系统参数的设计依据为
        :param gid: 用来表示在pc中的细胞编号
        :param k_cal: 当为1时，表示Control， 当为5/3时，表示AD
        :param k_kca: 当为1时，表示Control， 当为50时，表示AD
        """
        super().__init__()
        self.gid = gid
        self.k_cal = k_cal
        self.k_kca = k_kca

        self.nc = []
        self._setup_morphology()
        self._setup_topology()
        self._create_lists()
        self._setup_geometry()
        self._setup_biophysics()
        self._setup_segments()

        self.synGroups = {}
        self.generate_synapses()
        self.internal_netcons = []
        self.external_netcons = {}

        self.spike_threshold = -42.0

        self.spike_detector = h.NetCon(self.soma(0.5)._ref_v, None, sec=self.soma)
        self.spike_detector.threshold = self.spike_threshold
        self.spike_times = h.Vector()
        self.spike_detector.record(self.spike_times)

    def _setup_morphology(self):
        # apical trunk
        self.soma = h.Section(name='soma', cell=self)
        self.radTprox = h.Section(name='radTprox', cell=self)
        self.radTmed = h.Section(name='radTmed', cell=self)
        self.radTdist = h.Section(name='radTdist', cell=self)
        # slm1
        self.lm_thick1 = h.Section(name='lm_thick1', cell=self)
        self.lm_medium1 = h.Section(name='lm_medium1', cell=self)
        self.lm_thin1a = h.Section(name='lm_thin1a', cell=self)
        self.lm_thin1b = h.Section(name='lm_thin1b', cell=self)
        # slm2
        self.lm_thick2 = h.Section(name='lm_thick2', cell=self)
        self.lm_medium2 = h.Section(name='lm_medium2', cell=self)
        self.lm_thin2a = h.Section(name='lm_thin2a', cell=self)
        self.lm_thin2b = h.Section(name='lm_thin2b', cell=self)
        # rad1
        self.rad_thick1 = h.Section(name='rad_thick1', cell=self)
        self.rad_medium1 = h.Section(name='rad_medium1', cell=self)
        self.rad_thin1a = h.Section(name='rad_thin1a', cell=self)
        self.rad_thin1b = h.Section(name='rad_thin1b', cell=self)
        # rad2
        self.rad_thick2 = h.Section(name='rad_thick2', cell=self)
        self.rad_medium2 = h.Section(name='rad_medium2', cell=self)
        self.rad_thin2a = h.Section(name='rad_thin2a', cell=self)
        self.rad_thin2b = h.Section(name='rad_thin2b', cell=self)
        # basal dends
        self.oriprox1 = h.Section(name='oriprox1', cell=self)
        self.oridist1a = h.Section(name='oridist1a', cell=self)
        self.oridist1b = h.Section(name='oridist1b', cell=self)
        self.oriprox2 = h.Section(name='oriprox2', cell=self)
        self.oridist2a = h.Section(name='oridist2a', cell=self)
        self.oridist2b = h.Section(name='oridist2b', cell=self)

        self.axon = h.Section(name='axon', cell=self)

    def _setup_topology(self):
        # Connect sections
        # Apical trunk
        self.radTprox.connect(self.soma(1))
        self.radTmed.connect(self.radTprox(1))
        self.radTdist.connect(self.radTmed(1))
        # Apical oblique tree
        # Right
        self.rad_thick1.connect(self.radTprox(1))
        self.rad_medium1.connect(self.rad_thick1(1))
        self.rad_thin1a.connect(self.rad_medium1(1))
        self.rad_thin1b.connect(self.rad_medium1(1))
        # Left
        self.rad_thick2.connect(self.radTprox(1))
        self.rad_medium2.connect(self.rad_thick2(1))
        self.rad_thin2a.connect(self.rad_medium2(1))
        self.rad_thin2b.connect(self.rad_medium2(1))
        # Apical tuft tree
        # Right
        self.lm_thick1.connect(self.radTdist(1))
        self.lm_medium1.connect(self.lm_thick1(1))
        self.lm_thin1a.connect(self.lm_medium1(1))
        self.lm_thin1b.connect(self.lm_medium1(1))
        # Left
        self.lm_thick2.connect(self.radTdist(1))
        self.lm_medium2.connect(self.lm_thick2(1))
        self.lm_thin2a.connect(self.lm_medium2(1))
        self.lm_thin2b.connect(self.lm_medium2(1))
        # Basal tree
        # Right
        self.oriprox1.connect(self.soma(0))
        self.oridist1a.connect(self.oriprox1(1))
        self.oridist1b.connect(self.oriprox1(1))
        # Left
        self.oriprox2.connect(self.soma(0))
        self.oridist2a.connect(self.oriprox2(1))
        self.oridist2b.connect(self.oriprox2(1))
        # Axon
        self.axon.connect(self.soma(0))

    def _create_lists(self):
        self.all = self.soma.wholetree()
        self.ori = [self.oriprox1, self.oridist1a, self.oridist1b, self.oriprox2, self.oridist2a, self.oridist2b]
        self.rad = [self.rad_thick1, self.rad_medium1, self.rad_thin1a, self.rad_thin1b, self.rad_thick2,
                    self.rad_medium2, self.rad_thin2a, self.rad_thin2b]
        self.slm = [self.lm_thick1, self.lm_medium1, self.lm_thin1a, self.lm_thin1b, self.lm_thick2, self.lm_medium2,
                    self.lm_thin2a, self.lm_thin2b]
        self.trunk = [self.radTprox, self.radTmed, self.radTdist]

    def _setup_geometry(self):
        self.soma.L, self.soma.diam = 10 * um, 10 * um
        self.radTprox.L, self.radTprox.diam = 100 * um, 4 * um
        self.radTmed.L, self.radTmed.diam = 100 * um, 3 * um
        self.radTdist.L, self.radTdist.diam = 200 * um, 2 * um
        self.lm_thick1.L, self.lm_thick1.diam = 100 * um, 2 * um
        self.lm_medium1.L, self.lm_medium1.diam = 100 * um, 1.5 * um
        self.lm_thin1a.L, self.lm_thin1a.diam = 50 * um, 1 * um
        self.lm_thin1b.L, self.lm_thin1b.diam = 50 * um, 1 * um
        self.lm_thick2.L, self.lm_thick2.diam = 100 * um, 2 * um
        self.lm_medium2.L, self.lm_medium2.diam = 100 * um, 1.5 * um
        self.lm_thin2a.L, self.lm_thin2a.diam = 50 * um, 1.0 * um
        self.lm_thin2b.L, self.lm_thin2b.diam = 50 * um, 1.0 * um
        self.rad_thick1.L, self.rad_thick1.diam = 100 * um, 2.0 * um
        self.rad_medium1.L, self.rad_medium1.diam = 100 * um, 1.5 * um
        self.rad_thin1a.L, self.rad_thin1a.diam = 50 * um, 1.0 * um
        self.rad_thin1b.L, self.rad_thin1b.diam = 50 * um, 1.0 * um
        self.rad_thick2.L, self.rad_thick2.diam = 100 * um, 2.0 * um
        self.rad_medium2.L, self.rad_medium2.diam = 100 * um, 1.5 * um
        self.rad_thin2a.L, self.rad_thin2a.diam = 50 * um, 1.0 * um
        self.rad_thin2b.L, self.rad_thin2b.diam = 50 * um, 1.0 * um
        self.oriprox1.L, self.oriprox1.diam = 100 * um, 2.0 * um
        self.oridist1a.L, self.oridist1a.diam = 100 * um, 1.5 * um
        self.oridist1b.L, self.oridist1b.diam = 100 * um, 1.5 * um
        self.oriprox2.L, self.oriprox2.diam = 100 * um, 2.0 * um
        self.oridist2a.L, self.oridist2a.diam = 100 * um, 1.5 * um
        self.oridist2b.L, self.oridist2b.diam = 100 * um, 1.5 * um
        self.axon.L, self.axon.diam = 150 * um, 1 * um

    def _setup_segments(self):
        # Create segments based on `lambda_f`
        d_lambda = 0.1
        frequency = 100
        for sec in self.all:
            sec.nseg = int((sec.L / (d_lambda * h.lambda_f(frequency, sec=sec)) + 0.9) / 2) * 2 + 1

    def _setup_biophysics(self):

        Rm = 28000
        gh_soma = 0.00005
        gka_soma = 0.0075

        self.soma.insert('hha2')
        for seg in self.soma:
            seg.hha2.gnabar = 0.007
            seg.hha2.gkbar = 0.007 / 10
            seg.hha2.gl = 0
            seg.hha2.el = -70

        self.soma.insert('pas')
        for seg in self.soma:
            seg.pas.g = 1 / Rm

        self.soma.insert('h')
        for seg in self.soma:
            seg.h.ghdbar = gh_soma
            seg.h.vhalfl = -73

        self.soma.insert('kap')
        for seg in self.soma:
            seg.kap.gkabar = gka_soma

        self.soma.insert('km')
        for seg in self.soma:
            seg.km.gbar = 0.06

        self.soma.insert('cal')
        for seg in self.soma:
            seg.cal.gcalbar = 0.0014 / 2 * self.k_cal

        self.soma.insert('cat')
        for seg in self.soma:
            seg.cat.gcatbar = 0.0001 / 2

        self.soma.insert('somacar')
        for seg in self.soma:
            seg.somacar.gcabar = 0.0003

        self.soma.insert('kca')
        for seg in self.soma:
            seg.kca.gbar = 15 * 0.0001 * self.k_kca

        self.soma.insert('mykca')
        for seg in self.soma:
            seg.mykca.gkbar = 0.09075 * 5

        self.soma.insert('cad')

        radT_scale = [[2, 4, 7], [0.1, 10, 10], [5, 5, 0.5], [2, 2, 0.25], [2, 0, 0], [0, 4, 6]]
        ori_scale = [[1, 1.5, 2, 1, 1.5, 2]]

        for i, sec in enumerate(self.trunk):
            sec.insert('h')
            for seg in sec:
                seg.h.ghdbar = radT_scale[0][i] * gh_soma
                seg.h.vhalfl = -81
            sec.insert('car')
            for seg in sec:
                seg.car.gcabar = 0.1 * 0.0003
            sec.insert('calH')
            for seg in sec:
                seg.calH.gcalbar = radT_scale[1][i] * 0.00031635 * self.k_cal
            sec.insert('cat')
            for seg in sec:
                seg.cat.gcatbar = 0.0001
            sec.insert('cad')
            sec.insert('kca')
            for seg in sec:
                seg.kca.gbar = radT_scale[2][i] * 0.0001 * self.k_kca
            sec.insert('mykca')
            for seg in sec:
                seg.mykca.gkbar = radT_scale[3][i] * 0.0165
            sec.insert('km')
            for seg in sec:
                seg.km.gbar = 0.06
            sec.insert('kap')
            for seg in sec:
                seg.kap.gkabar = radT_scale[4][i] * gka_soma
            sec.insert('kad')
            for seg in sec:
                seg.kad.gkabar = radT_scale[5][i] * gka_soma
            sec.insert('hha_old')
            for seg in sec:
                sec.gnabar_hha_old = 0.007
                sec.gkbar_hha_old = 0.007 / 8.065
                sec.el_hha_old = -70
            sec.insert('pas')

        for i, sec in enumerate(self.rad):
            sec.insert('hha_old')
            for seg in sec:
                seg.gnabar_hha_old = 0.007
                seg.gkbar_hha_old = 0.007 / 8.065
                seg.el_hha_old = -70
                seg.gl_hha_old = 0
            sec.insert('pas')
            for seg in sec:
                seg.pas.g = 1 / 200000
            sec.insert('kad')
            for seg in sec:
                seg.kad.gkabar = 6.5 * gka_soma

        for i, sec in enumerate(self.slm):
            sec.insert('hha_old')
            for seg in sec:
                seg.gnabar_hha_old = 0.007
                seg.gkbar_hha_old = 0.007 / 8.065
                seg.el_hha_old = -70
                seg.gl_hha_old = 0
            sec.insert('pas')
            for seg in sec:
                seg.pas.g = 1 / 200000
            sec.insert('kad')
            for seg in sec:
                seg.kad.gkabar = 6.5 * gka_soma

        for i, sec in enumerate(self.ori):
            sec.insert('h')
            for seg in sec:
                seg.h.ghdbar = ori_scale[0][i] * gh_soma
                seg.h.vhalfl = -81
            sec.insert('car')
            for seg in sec:
                seg.car.gcabar = 0.1 * 0.0003
            sec.insert('calH')
            for seg in sec:
                seg.calH.gcalbar = 0.1 * 0.00031635 * self.k_cal
            sec.insert('cat')
            for seg in sec:
                seg.cat.gcatbar = 0.0001
            sec.insert('cad')
            sec.insert('kca')
            for seg in sec:
                seg.kca.gbar = 5 * 0.0001 * self.k_kca
            sec.insert('mykca')
            for seg in sec:
                seg.mykca.gkbar = 2 * 0.0165
            sec.insert('km')
            for seg in sec:
                seg.km.gbar = 0.06
            sec.insert('kap')
            for seg in sec:
                seg.kap.gkabar = gka_soma
            sec.insert('kad')
            for seg in sec:
                seg.kad.gkabar = 0
            sec.insert('hha_old')
            for seg in sec:
                sec.gnabar_hha_old = 0.007
                sec.gkbar_hha_old = 0.007 / 8.065
                sec.el_hha_old = -70
            sec.insert('pas')

        self.axon.insert('hha2')
        for seg in self.axon:
            seg.hha2.gnabar = 0.2
            seg.hha2.gkbar = 0.1 / 5
            seg.hha2.gl = 0.000002
            seg.hha2.el = -70
        self.axon.insert('pas')
        for seg in self.axon:
            seg.pas.g = 1 / Rm
        self.axon.insert('km')
        for seg in self.axon:
            seg.km.gbar = 0.5 * 0.06

        for sec in self.all:
            sec.ek = -80
            sec.ena = 50
            sec.e_pas = -70
            sec.g_pas = 1 / Rm
            sec.Ra = 150
            sec.cm = 1

    def generate_synapses(self):
        self.synGroups['AMPA'] = {}
        for sec in self.all:
            name = sec.name().split('.')[-1]
            self.synGroups['AMPA'][name] = {}

        self.synGroups['GABAA'] = {}
        for sec in self.all:
            name = sec.name().split('.')[-1]
            self.synGroups['GABAA'][name] = {}

        self.synGroups['GABAB'] = {}
        for sec in self.all:
            name = sec.name().split('.')[-1]
            self.synGroups['GABAB'][name] = {}

        self.synGroups['NMDA'] = {}
        for sec in self.all:
            name = sec.name().split('.')[-1]
            self.synGroups['NMDA'][name] = {}

        self.synGroups['STDPE2'] = {}
        for sec in self.all:
            name = sec.name().split('.')[-1]
            self.synGroups['STDPE2'][name] = {}

    def get_syn_parameters(self, sec_choice, syn_type):
        params = {}
        if syn_type == 'NMDA':
            tau1 = 2.3
            tau2 = 100.0
            gNMDAmax = 1.0
            params = {'tcon': tau1, 'tcoff': tau2, 'gNMDAmax': gNMDAmax}
        elif syn_type == 'STDPE2' or syn_type == 'AMPA':
            tau1 = 0.5
            tau2 = 3.0
            e = 0
            params = {'tau1': tau1, 'tau2': tau2, 'e': e}
        elif syn_type == 'GABAA':
            e = -75
            tau1 = 0.13
            tau2 = 11.0
            params = {'tau1': tau1, 'tau2': tau2, 'e': e}
        elif syn_type == 'GABAB':
            tau1 = 35
            tau2 = 100
            e = -75
            params = {'tau1': tau1, 'tau2': tau2, 'e': e}

        return params
