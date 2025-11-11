from neuron import h, load_mechanisms
import time
from neuron.units import ms, mV
import os
from mpi4py import MPI
from utils.SetupConnections import *
from utils.NeuronCircuit import Circuit
import matplotlib.pyplot as plt
import numpy as np

h.load_file('stdrun.hoc')
h.nrnmpi_init()
h.CVode().use_fast_imem(1)
load_mechanisms('./mechanisms/')


def get_ext_population_spikes(c,pop_id):
    spike_vec_dict = defaultdict(list)
    spike_times_vec = c.external_spike_time_recs[pop_id]
    spike_gids_vec  = c.external_spike_gid_recs[pop_id]
    for spike_t, spike_gid in zip(spike_times_vec, spike_gids_vec):
        spike_gid = int(spike_gid)
        spike_vec_dict[spike_gid].append(spike_t)
    return spike_vec_dict


def get_population_voltages(c,pop_id,rec_dt=0.1):
    v_vec_dict = {}
    for cid, cell in c.neurons[pop_id].items():
        v_vec = h.Vector()
        try:
            v_vec.record(cell.axon(0.5)._ref_v, rec_dt)
        except:
            v_vec.record(cell.soma(0.5)._ref_v, rec_dt)
        gid = c.ctype_offsets[pop_id] + cid
        v_vec_dict[gid] = v_vec
    return v_vec_dict

def pull_spike_times(population2info_dict):
    spike_times = {}
    gids = np.sort(list(population2info_dict.keys()))
    for gid in gids:
        gid_info = population2info_dict[gid]
        if 'spike times' in gid_info:
            spike_times[gid] = gid_info['spike times']
    return spike_times


delay = 500
dt = 0.1

pc = h.ParallelContext()
    
params_path = os.path.join('./params')


ar = Arena(os.path.join(params_path, 'areaparams.yaml'))
ar.generate_population_firing_rates()
ar.generate_cue_firing_rates('LEC', 1.0)

cued = True
    
fr = ar.cell_information['LEC']['cell info'][0]['firing rate']

edge  = 12.5
lp    = 1

arena_size = ar.params['Arena']['arena size']
bin_size   = ar.params['Arena']['bin size']
mouse_speed = ar.params['Arena']['mouse speed']
nlaps       = ar.params['Arena']['lap information']['nlaps']

arena_map  = np.arange(0, 200,step=0.1)
cued_positions  = np.linspace(edge, 200-edge, nlaps*lp)
random_cue_locs = np.arange(len(cued_positions))

if pc.id() == 0:
        np.random.shuffle(random_cue_locs)


time_for_single_lap = arena_size / mouse_speed * 1000.

frs_all = []
for i in range(nlaps):
    random_position = cued_positions[random_cue_locs[i]]
    to_roll = int( ( 100. - random_position) / 0.1 )
    fr_rolled = np.roll(fr, to_roll)
    frs_all.append(fr_rolled)

frs_all = np.asarray(frs_all)

place_information = {'place ids': [0], 'place fracs': [0.80]}

diagram = WiringDiagram(os.path.join(params_path, 'circuitparams_ee_ie_ca3_ec_lec.yaml'), place_information)

internal_kwargs = {}
internal_kwargs['place information'] = diagram.place_information
internal_kwargs['cue information'] = diagram.place_information
diagram.generate_internal_connectivity(**internal_kwargs)

external_kwargs = {'place information': diagram.place_information, 'external place ids': [100, 101, 102],
                   'cue information': diagram.place_information, 'external cue ids': [100, 101, 102]}

diagram.generate_external_connectivity(ar.cell_information, **external_kwargs)
diagram.generate_septal_connectivity()

ar.generate_spike_times('CA3', dt=dt, delay=delay)
ar.generate_spike_times('EC', dt=dt, delay=delay)
ar.generate_spike_times('LEC', dt=dt, delay=delay, cued=cued)
ar.generate_spike_times('Background', dt=dt, delay=delay)

place_ids = diagram.place_information[0]['place']
cue_ids = diagram.place_information[0]['not place']

mf_spike_times  = pull_spike_times(ar.cell_information['CA3']['cell info'])
mec_spike_times = pull_spike_times(ar.cell_information['EC']['cell info'])
lec_spike_times = pull_spike_times(ar.cell_information['LEC']['cell info'])
bk_spike_times  = pull_spike_times(ar.cell_information['Background']['cell info'])

circuit = Circuit(params_prefix=params_path,
                  params_filename='circuitparams_ee_ie_ca3_ec_lec.yaml',
                  arena_params_filename='areaparams.yaml',
                  internal_pop2id=diagram.pop2id,
                  external_pop2id=diagram.external_pop2id,
                  external_spike_times = {100: mf_spike_times,
                                          101: mec_spike_times,
                                          102: lec_spike_times,
                                          103: bk_spike_times})

circuit.build_cells()
circuit.build_internal_netcons(diagram.internal_con, diagram.internal_ws)
circuit.build_external_netcons(100, diagram.external_con[100], diagram.external_ws[100])
circuit.build_external_netcons(101, diagram.external_con[101], diagram.external_ws[101])
circuit.build_external_netcons(102, diagram.external_con[102], diagram.external_ws[102])
circuit.build_external_netcons(103, diagram.external_con[103], diagram.external_ws[103])
# circuit.build_septal_netcons(diagram.septal_con)
pc = circuit.pc
    
exc_v_vecs = get_population_voltages(circuit, 0)
t_vec = h.Vector()  # Time stamp vector
t_vec.record(h._ref_t)

tic = time.time()
h.dt = 0.1
h.celsius = 37.
mindelay = pc.set_maxstep(10 * ms)
h.finitialize(-65 * mV)
h.tstop = time_for_single_lap*(10 + 1) + delay
print('So far, there are no problems')
pc.set_maxstep(10 * ms)
print('So far, there are no problems')
pc.psolve(h.tstop - mindelay)
print('So far, there are no problems')
elapsed = time.time() - tic
pc.barrier()


start_time = time_for_single_lap * 0 + 400
end_time = time_for_single_lap * 5 + 400
spike_list=[]
gids=[]
for i in range(130):
    if i in place_ids:
        data = circuit.neurons[0][i].spike_times
        data = data.as_numpy()
        data = [i for i in data if start_time<i<end_time]
        time = np.ones_like(data) * i
        time = time.tolist()
        if type(data) == float or type(data) == int:
            spike_list.append(data)
            gids.append(time)
        else:
            spike_list += data
            gids += time

spike_list = np.array(spike_list)
gids = np.array(gids)

plt.figure()
plt.title('nlaps = 10')
plt.scatter(spike_list, gids, marker='|', color='black', linewidths=0.5, s=100)
plt.xlabel('Time (ms)')
plt.ylabel('Pyramidal Place Cells')
plt.show()
