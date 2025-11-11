from neuron import h


class RandomStream(object):
    def __init__(self, seed, random_stream_offset=1000):
        super().__init__()
        self._seed = seed
        self.random_stream_offset = random_stream_offset
        self._setup_biophysics()
        self.start()

    def _setup_biophysics(self):
        self.stream = self._seed
        self.r = h.Random()

    def start(self):
        return self.r.MCellRan4(self.stream * self.random_stream_offset + 1)

    def re_pick(self):
        return self.r.repick()
