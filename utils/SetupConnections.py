import numpy as np
import yaml
from copy import deepcopy
from collections import defaultdict
import math
import random
from neuron import h
import brian2
from utils.make_grid_like_inputs import get_grid_like_inputs
from utils.sp_make_place_inputs import get_place_inputs
from utils.glim_shuf_new_noisy import get_spike_times
from utils.generate_position import generate_path
from scipy.interpolate import Akima1DInterpolator


def generate_soma_positions(ncells, maxpos=1.):
    positions = np.linspace(0, maxpos, ncells)
    return { i: positions[i] for i in range(ncells) }

def soma_positions_to_field_center(volume_positions, arena_size, maxpos=1.):

    positions = np.linspace(0, maxpos, 100)
    centers = np.linspace(0, arena_size, 100)
    spatial_ip = Akima1DInterpolator(positions, centers)
    
    spatial_positions = {}
    for gid in volume_positions.keys():
        gid_pos     = volume_positions[gid]
        spatial_positions[gid] = spatial_ip(gid_pos)
    return spatial_positions

def generate_place_firing_maps(field_centers, peak_rates, peak_rate_probs, min_rate, diameter, arena_map, rnd):
    firing_rates = {}
    for gid in list(field_centers.keys()):
        max_rate = rnd.choice(peak_rates, p=peak_rate_probs)
        current_center = field_centers[gid]
        fr = np.asarray(place_field_fr(current_center, arena_map, max_rate, diameter),
                        dtype='float32')
        fr[fr<=min_rate] = min_rate
        firing_rates[gid] = fr
    return firing_rates

def generate_grid_firing_maps(field_centers, peak_rates, peak_rate_probs, min_rate, diameter, gap, arena_map, rnd):
    firing_rates = {}
    arena_min, arena_max = np.min(arena_map), np.max(arena_map)
    for gid in list(field_centers.keys()):
        
        max_rate = rnd.choice(peak_rates, p=peak_rate_probs)
        current_center = field_centers[gid]
        current_firing_rate    = np.asarray(place_field_fr(current_center, arena_map, max_rate, 
                                                           diameter), dtype='float32')
        current_pos = current_center - gap
        while (current_pos >= arena_min):
            hopped_fr = np.asarray(place_field_fr(current_pos, arena_map, max_rate, 
                                                  diameter), dtype='float32')
            current_firing_rate += hopped_fr
            current_pos -= gap
        current_pos = current_center + gap
        while (current_pos <= arena_max):

            hopped_fr = np.asarray(place_field_fr(current_pos, arena_map, max_rate, 
                                                  diameter), dtype='float32')
            current_firing_rate += hopped_fr
            current_pos += gap
        current_firing_rate[current_firing_rate <= min_rate] = min_rate
        firing_rates[gid] = current_firing_rate
    return firing_rates

def place_field_fr(center, spatial_bins, max_fr, diameter):
    c      = diameter / 4.3
    fnc = max_fr * np.exp(-( (spatial_bins-center) / (c)) ** 2.)
    return fnc


def get_inhom_poisson_spike_times_by_thinning(rate, t, dt=0.02, delay=0.0, refractory=3., generator=None):
    """
    Given a time series of instantaneous spike rates in Hz, produce a spike train consistent with an inhomogeneous
    Poisson process with a refractory period after each spike.
    :param rate: instantaneous rates in time (Hz)
    :param t: corresponding time values (ms)
    :param dt: temporal resolution for spike times (ms)
    :param refractory: absolute deadtime following a spike (ms)
    :param generator: :class:'np.random.RandomState()'
    :return: list of m spike times (ms)
    """
    tt = deepcopy(t)
#     fr_dt = tt[1]-tt[0]
#     tt += delay
#     expanded_t    = np.arange(0,delay,step=fr_dt)
#     expanded_rate = [1.0 for _ in range(len(expanded_t))]
#     rate = np.asarray(expanded_rate + list(rate), dtype='float32')
#     tt    = np.asarray(list(expanded_t) + list(tt), dtype='float32')
    min_fr = np.min(rate)
    if generator is None:
        generator = random
    interp_t = np.arange(tt[0], tt[-1], dt)
    
    try:
        rate_ip = Akima1DInterpolator(tt, rate)
        interp_rate = rate_ip(interp_t)
    except Exception:
        print('t shape: %s rate shape: %s' % (str(tt.shape), str(rate.shape)))
        raise
    
    delay_t = np.arange(-delay, 0, dt)
    delay_r = np.ones(len(delay_t))*2.
    
    interp_t = np.concatenate((delay_t, interp_t))
    interp_t += delay
    interp_rate = np.concatenate((delay_r, interp_rate)) 
    
    interp_rate /= 1000.
    spike_times = []
    non_zero = np.where(interp_rate > 1.e-100)[0]
    if len(non_zero) == 0:
        return spike_times
    interp_rate[non_zero] = 1. / (1. / interp_rate[non_zero] - refractory)
    max_rate = np.max(interp_rate)
    if not max_rate > 0.:
        return spike_times
    i = 0
    ISI_memory = 0.
    while i < len(interp_t):
        x = generator.uniform(0.0, 1.0)
        if x > 0.:
            ISI = -np.log(x) / max_rate
            i += int(ISI // dt)
            ISI_memory += ISI
            if (i < len(interp_t)) and (generator.uniform(0.0, 1.0) <= interp_rate[i] / max_rate) and ISI_memory >= 0.:
                spike_times.append(interp_t[i])
                ISI_memory = -refractory
    return np.asarray(spike_times, dtype='float32')


class Arena(object):

    def __init__(self, params_filepath, theta_modulation=8):
        self.pc = h.ParallelContext()
        self.params = {}
        self.params_filepath = params_filepath
        self._read_arena_params()  # self.params['Arena']
        self._read_arena_cell_params()  # self.params['Spatial']

        self.theta_modulation = theta_modulation

        arena_rnd = None
        if int(self.pc.id()) == 0:
            arena_rnd = np.random.RandomState(seed=self.params['Arena']['random seed'])
        self.arena_rnd = self.pc.py_broadcast(arena_rnd, 0)

        self.arena_size = self.params['Arena']['arena size']
        self.mouse_speed = self.params['Arena']['mouse speed'] / 1000  # in cm/ms
        self.bin_size = self.params['Arena']['bin size']
        self.arena_map = np.arange(0, self.arena_size, step=self.bin_size)
        self.cell_information = {}

    def _read_arena_params(self):
        # 获得文件中小鼠的移动信息
        self.params['Arena'] = {}
        arena_params = None
        if self.pc.id() == 0:
            with open(self.params_filepath, 'r') as f:
                fparams = yaml.load(f, Loader=yaml.FullLoader)
                arena_params = fparams['Arena']
                self.params['Arena'] = arena_params
        self.pc.barrier()
        self.params['Arena'] = self.pc.py_broadcast(arena_params, 0)

    def _read_arena_cell_params(self):
        # 获得文件中输入细胞的信息
        self.params['Spatial'] = {}
        arena_cell_params = None
        if self.pc.id() == 0:
            with open(self.params_filepath, 'r') as f:
                fparams = yaml.load(f, Loader=yaml.FullLoader)
                arena_cell_params = fparams['Arena Cells']
                self.params['Spatial'] = arena_cell_params
        self.pc.barrier()
        self.params['Spatial'] = self.pc.py_broadcast(arena_cell_params, 0)

    def generate_population_firing_rates(self, seed=2e9):

        for population_name in self.params['Spatial'].keys():
            self.cell_information[population_name] = {}
            current_population = self.params['Spatial'][population_name]

            self.cell_information[population_name]['id']     = current_population['id']
            self.cell_information[population_name]['ncells'] = current_population['ncells']
            somatic_positions = generate_soma_positions(current_population['ncells'])

            ncells = current_population['ncells']
            self.cell_information[population_name]['cell info'] = {}
            ctype_offset = 0

            for idx in range(ncells):
                gid = idx + ctype_offset
                self.cell_information[population_name]['cell info'][gid] = {}
                self.cell_information[population_name]['cell info'][gid]['soma position'] = somatic_positions[idx]
                
            if 'place' in current_population:
                self.cell_information[population_name]['spatial type'] = 'place'
                field_centers = soma_positions_to_field_center(somatic_positions, self.arena_size)
                for idx in range(ncells):
                    gid = idx + ctype_offset
                    self.cell_information[population_name]['cell info'][gid]['field center'] = field_centers[idx]

                peak_rates = current_population['place']['peak rates']
                peak_rate_probs = current_population['place']['peak rate probabilities']
                peak_rate_prob_sum = np.sum(peak_rate_probs)
                peak_rate_probs_nrm  = peak_rate_probs / peak_rate_prob_sum
                min_rate  = current_population['place']['min rate']
                diameter  = current_population['place']['diameter']
                place_firing_rates = generate_place_firing_maps(field_centers, peak_rates, peak_rate_probs_nrm, min_rate, diameter, self.arena_map, self.arena_rnd)
                for idx in range(ncells):
                    gid = idx + ctype_offset                    
                    self.cell_information[population_name]['cell info'][gid]['firing rate'] = place_firing_rates[idx]
                    
            elif 'grid' in current_population:
                self.cell_information[population_name]['spatial type'] = 'grid'
                field_centers = soma_positions_to_field_center(somatic_positions, self.arena_size)
                for idx in range(ncells):
                    gid = idx + ctype_offset                    
                    self.cell_information[population_name]['cell info'][gid]['field center'] = field_centers[idx]

                peak_rates = current_population['grid']['peak rates']
                peak_rate_probs = current_population['grid']['peak rate probabilities']
                peak_rate_prob_sum = np.sum(peak_rate_probs)
                peak_rate_probs_nrm  = peak_rate_probs / peak_rate_prob_sum
                min_rate  = current_population['grid']['min rate']
                diameter  = current_population['grid']['diameter']
                gap       = current_population['grid']['gap']

                grid_firing_rates = generate_grid_firing_maps(field_centers, peak_rates, peak_rate_probs_nrm,
                                                              min_rate, diameter, gap, self.arena_map, self.arena_rnd)
                for idx in range(ncells):
                    gid = idx + ctype_offset                    
                    self.cell_information[population_name]['cell info'][gid]['firing rate'] = grid_firing_rates[idx]

            ctype_offset += ncells

    def generate_cue_firing_rates(self, population, percent_cue):

            noise_fr    = self.params['Spatial'][population]['noise']['mean rate']
            ncells      = self.params['Spatial'][population]['ncells']
            
            ncue_cells = int(ncells*percent_cue)
            cells_cued = self.arena_rnd.choice(np.arange(ncells), size=(ncue_cells,), replace=False)
            min_fr=self.params['Spatial'][population]['cue']['min rate']
            peak_rates=self.params['Spatial'][population]['cue']['peak rates']
            peak_rate_probs = self.params['Spatial'][population]['cue']['peak rate probabilities']
            peak_rate_prob_sum = np.sum(peak_rate_probs)
            peak_rate_probs_nrm  = peak_rate_probs / peak_rate_prob_sum
            diameter=self.params['Spatial'][population]['cue']['diameter']
            cue_firing_rates = []
            for i in range(ncells):
                cue_fr = None
                if i in cells_cued:
                    max_fr = self.arena_rnd.choice(peak_rates, p=peak_rate_probs_nrm)
                    cue_fr = place_field_fr(int(self.arena_size/2), self.arena_map, max_fr, diameter)
                else:
                    cue_fr = [noise_fr for _ in range(len(self.arena_map))]
                cue_fr = np.asarray(cue_fr, dtype='float32')
                cue_fr[cue_fr<=min_fr] = min_fr
                cue_firing_rates.append(cue_fr)
            self.cell_information[population] = {}
            self.cell_information[population]['ncells'] = ncells
            self.cell_information[population]['id'] = self.params['Spatial'][population]['id']
            self.cell_information[population]['cell info'] = {}
            for (idx,cfr) in enumerate(cue_firing_rates):
                self.cell_information[population]['cell info'][idx] = {}
                self.cell_information[population]['cell info'][idx]['firing rate'] = cfr

    def generate_spike_times(self, population, dt=0.05, delay=0, cued=False):
         
        mouse_speed = self.params['Arena']['mouse speed']
        lap_information = self.params['Arena']['lap information']
        nlaps           = lap_information['nlaps']
        is_spatial      = lap_information['is spatial']
        up_state        = lap_information.get('up state', None)
        run_step_dur    = self.bin_size/mouse_speed
        arena_length    = len(self.arena_map)
        start_time = 0
        end_time  = nlaps * self.arena_size / mouse_speed
        bin2times = np.arange(0, end_time, step=run_step_dur, dtype='float32') * 1000.

        population_info = self.cell_information[population]
        ncells = population_info['ncells']
        gids   = range(int(self.pc.id()), ncells, int(self.pc.nhost()))

        
        nfr      = np.clip(self.arena_rnd.normal(self.params['Spatial'][population]['noise']['mean rate'],
                                                 scale=self.params['Spatial'][population]['noise']['scale'],
                                                 size=ncells),
                           0.0, None)
        noise_fr = np.vstack([nfr for _ in range(arena_length)])

        up_state_fr = None
        if 'up state' in self.params['Spatial'][population]:
            usfr      = np.clip(self.arena_rnd.normal(self.params['Spatial'][population]['up state']['mean rate'],
                                                      scale=self.params['Spatial'][population]['up state']['scale'],
                                                      size=ncells),
                                0.0, None)
            usfr_on_duration  = self.params['Spatial'][population]['up state'].get("on duration", None)
            usfr_off_duration  = self.params['Spatial'][population]['up state'].get("off duration", None)
            if usfr_off_duration is None:
                usfr_off_duration = usfr_on_duration
            up_state_fr = None 
            if usfr_on_duration is not None:
                up_state_on_length = int(round(usfr_on_duration / run_step_dur))
                up_state_off_length = int(round(usfr_off_duration / run_step_dur))
                n_up_state_fr = 0
                up_state_fr_list = []
                while n_up_state_fr < arena_length:
                    up_state_fr_list.extend([usfr for _ in range(up_state_on_length)])
                    n_up_state_fr += up_state_on_length
                    up_state_fr_list.extend([nfr for _ in range(up_state_off_length)])
                    n_up_state_fr += up_state_off_length
                up_state_fr = np.vstack(up_state_fr_list[:arena_length])
            else:
                up_state_fr = np.vstack([usfr for _ in range(arena_length)])

        firing_rates = {}
        for gid in gids:
            try:
                fr = population_info['cell info'][gid]['firing rate']
            except:
                fr = noise_fr[:,gid]
            firing_rates[gid] = fr

        if cued:
            self.cued_positions  = np.linspace(12.5, self.arena_size-12.5, np.sum(is_spatial))
            self.random_cue_locs = np.arange(len(self.cued_positions))
            self.arena_rnd.shuffle(self.random_cue_locs)
            print(self.random_cue_locs)
        for (gid, fr) in firing_rates.items():
            current_full_fr = []
            online_number = 0
            for n in range(nlaps):
                if not is_spatial[n]:
                    if (up_state_fr is not None) and (up_state is not None) and (up_state[n] > 0):
                        this_fr = up_state_fr[:,gid]
                        
                        current_full_fr.extend(this_fr)
                    else:
                        if population == 'MF': current_full_fr.extend(np.multiply(noise_fr[:,gid], 1.0))
                        else: current_full_fr.extend(noise_fr[:,gid])
                else: 
                    if cued:
                        random_position = self.cued_positions[self.random_cue_locs[online_number]]
                        to_roll = int( ( self.arena_size/2 - random_position) / (self.arena_map[1]-self.arena_map[0]) )
                        current_full_fr.extend(np.roll(fr, to_roll))
                    else:
                        current_full_fr.extend(fr)
                    online_number += 1
            current_full_fr = np.asarray(current_full_fr, dtype='float32')
            if (bin2times.shape[0] > current_full_fr.shape[0]): bin2times = bin2times[:-1]
            spike_times = np.asarray(get_inhom_poisson_spike_times_by_thinning(current_full_fr, bin2times, dt=dt, delay=delay),
                                     dtype='float32')
            self.cell_information[population]['cell info'][gid]['spike times'] = spike_times

    def generate_population_position(self, population, delay=400, n_dend=8):

        self.cell_information[population] = {}
        current_population = self.params['Spatial'][population]

        self.cell_information[population]['id'] = current_population['id']
        self.cell_information[population]['ncells'] = current_population['ncells']

        mouse_position = generate_path(max_x=self.arena_size, n_laps=self.params['Arena']['lap information']['nlaps'])
        
        ncells = current_population['ncells']
        soma_positions = generate_soma_positions(ncells)
        self.cell_information[population]['cell info'] = {}


        for idx in range(ncells):
            gid = idx
            self.cell_information[population]['cell info'][gid] = {}
            self.cell_information[population]['cell info'][gid]['soma position'] = soma_positions[gid]

        if 'place' in current_population:
            self.cell_information[population]['spatial type'] = 'place'
            place_field = get_grid_like_inputs(mouse_positions=mouse_position, n_laps=self.params['Arena']['lap information']['nlaps'], 
                                               freq=self.theta_modulation)
            place_inputs = get_place_inputs(place_field=place_field, arena_size=self.arena_size, mouse_positions=mouse_position,
                                            n_laps=self.params['Arena']['lap information']['nlaps'])
            spike_times = get_spike_times(time_delay=delay, n_dend=n_dend, inputs=place_inputs, n_place_field=41,
                                          n_laps=self.params['Arena']['lap information']['nlaps'], shuffle=False)
            # path = './trace_inputs/place.npy'
            # # np.save(path, spike_times)
            # spike_times = np.load(path, allow_pickle=True).item()
            for idx in range(ncells):
                gid = idx
                spike_time = spike_times[gid]
                spike_time = h.Vector(spike_time)
                # 1-6对应位置0-10，中心位置为5
                self.cell_information[population]['cell info'][gid]['spike times'] = spike_time

        elif 'grid' in current_population:
            self.cell_information[population]['spatial type'] = 'grid'
            place_field = get_grid_like_inputs(mouse_positions=mouse_position, n_laps=self.params['Arena']['lap information']['nlaps'], 
                                               freq=self.theta_modulation)
            spike_times = get_spike_times(time_delay=delay, n_dend=n_dend, inputs=place_field, n_place_field=41,
                                          n_laps=self.params['Arena']['lap information']['nlaps'], shuffle=False)
            # path = './trace_inputs/grid.npy'
            # # np.save(path, spike_times)
            # spike_times = np.load(path, allow_pickle=True).item()
            for idx in range(ncells):
                gid = idx
                spike_time = spike_times[gid]
                spike_time = h.Vector(spike_time)
                self.cell_information[population]['cell info'][gid]['spike times'] = spike_time

        elif 'cue' in current_population:
            self.cell_information[population]['spatial type'] = 'cue'
            place_field = get_grid_like_inputs(mouse_positions=mouse_position, n_laps=self.params['Arena']['lap information']['nlaps'], 
                                               freq=self.theta_modulation, n_dend=3)
            spike_times = get_spike_times(time_delay=delay, n_dend=n_dend, inputs=place_field, n_place_field=41,
                                          n_laps=self.params['Arena']['lap information']['nlaps'], shuffle=True)
            # path = './trace_inputs/grid.npy'
            # # np.save(path, spike_times)
            # spike_times = np.load(path, allow_pickle=True).item()
            for idx in range(ncells):
                gid = idx
                spike_time = spike_times[gid]
                spike_time = h.Vector(spike_time)
                self.cell_information[population]['cell info'][gid]['spike times'] = spike_time

        

    def generate_possion_inputs(self, population, seed=1):

        self.cell_information[population] = {}
        current_population = self.params['Spatial'][population]

        self.cell_information[population]['id'] = current_population['id']
        self.cell_information[population]['ncells'] = current_population['ncells']
        self.cell_information[population]['cell info'] = {}

        brian2.seed(seed=seed)
        freq = self.params['Spatial'][population]['noise']['mean rate']
        ncells = self.params['Spatial'][population]['ncells']
        gids = range(ncells)

        TINIT    = 500
        STARTDEL = 500
        THETA    = 125  # (8 Hz)
        GAMMA    = 25  # (40 Hz)

        duration = 115
        time_per_lap = TINIT + (THETA*duration) + STARTDEL/10

        time_input = time_per_lap*10 * brian2.ms
        rate = freq * brian2.Hz
        p = brian2.PoissonGroup(ncells, rates=rate)
        s = brian2.SpikeMonitor(p)
        brian2.run(time_input, report='text', report_period=10 * brian2.second)
        for gid in gids:
            self.cell_information[population]['cell info'][gid] = {}
            spike_times = [round(x/brian2.ms, 1) for x in list(s.spike_trains()[gid])]
            spike_times = np.array(spike_times, dtype=np.float32)
            # path = f'./trace_inputs/noise/noise_{gid}.npy'
            # np.save(path, spike_times)
            # spike_times = np.load(path, allow_pickle=True)
            spike_times = h.Vector(spike_times)
            self.cell_information[population]['cell info'][gid]['spike times'] = spike_times


class WiringDiagram(object):

    def __init__(self, params_filepath, place_information):
        self.pc = h.ParallelContext()
        self.params = None
        self._read_params_filepath(params_filepath)

        internal_con_rnd = None
        external_con_rnd = None
        septal_con_rnd = None

        if self.pc.id() == 0:
            internal_con_rnd = np.random.RandomState(seed=self.params['internal seed'])
            external_con_rnd = np.random.RandomState(seed=self.params['external seed'])
            septal_con_rnd = np.random.RandomState(seed=self.params['septal seed'])

        self.internal_con_rnd = self.pc.py_broadcast(internal_con_rnd, 0)
        self.external_con_rnd = self.pc.py_broadcast(external_con_rnd, 0)
        self.septal_con_rnd = self.pc.py_broadcast(septal_con_rnd, 0)

        place_ids = place_information.get('place ids', [])
        place_fracs = place_information.get('place fracs', [])

        self.wiring_information = {}
        pops, ncells = self.params['cells'], self.params['ncells']
        self.pop2id = {pops[i]: i for i in range(len(pops))}
        self.pops = pops
        self.place_information = {}

        ctype_offset = 0
        for (i, pop) in enumerate(pops):
            self.wiring_information[pop] = {}
            self.wiring_information[pop]['ncells'] = ncells[i]
            self.wiring_information[pop]['cell info'] = {}
            self.wiring_information[pop]['ctype offset'] = ctype_offset

            for idx in range(ncells[i]):
                gid = idx + ctype_offset
                self.wiring_information[pop]['cell info'][gid] = {}

            place_gids = []
            if i in place_ids:
                frac_place = place_fracs[place_ids.index(i)]
                is_place = None
                if self.pc.id() == 0:
                    is_place = self.internal_con_rnd.choice([0, 1], p=[1.0 - frac_place, frac_place], size=(ncells[i],))
                is_place = self.pc.py_broadcast(is_place, 0)
                for (idx, ip) in enumerate(is_place):
                    gid = idx + ctype_offset
                    self.wiring_information[pop]['cell info'][gid]['place'] = ip
                    if ip:
                        place_gids.append(gid)
            else:
                place_gids = list([idx + ctype_offset for idx in range(ncells[i])])

            not_place_gids = set([idx + ctype_offset for idx in range(ncells[i])]) - set(place_gids)

            if i in place_ids:  # place_information中只有PYR存在
                self.place_information[i] = {}
                self.place_information[i]['place'] = list(sorted(place_gids))
                self.place_information[i]['not place'] = list(sorted(not_place_gids))

            soma_coordinates = np.linspace(0., 1.0, ncells[i])
            for gid in place_gids:
                d_idx = gid - ctype_offset
                self.wiring_information[pop]['cell info'][gid]['soma position'] = soma_coordinates[d_idx]

            for gid in not_place_gids:
                d_idx = gid - ctype_offset
                self.wiring_information[pop]['cell info'][gid]['soma position'] = soma_coordinates[d_idx]

            ctype_offset += ncells[i]

    def _read_params_filepath(self, params_filepath):
        self.params = {}
        circuit_params = None
        if self.pc.id() == 0:
            with open(params_filepath, 'r') as f:
                fparams = yaml.load(f, Loader=yaml.FullLoader)
                circuit_params = fparams['Circuit']
                self.params = circuit_params
        self.pc.barrier()
        self.params = self.pc.py_broadcast(circuit_params, 0)

    def generate_internal_connectivity(self, **kwargs):

        place_information = kwargs['place information']
        cue_information = kwargs['cue information']

        self.internal_adj_matrices = {}
        self.internal_con = {}
        self.internal_ws = {}
        for popA in self.pops:
            popA_id = self.pop2id[popA]
            self.internal_adj_matrices[popA_id] = {}
            self.internal_con[popA_id] = {}
            self.internal_ws[popA_id] = {}
            for popB in self.pops:
                popB_id = self.pop2id[popB]
                params = self.params['internal connectivity'][popA_id].get(popB_id, None)
                if params is None: continue
                popA_pos = self._get_soma_coordinates(self.wiring_information[popA]['cell info'])  # [i/n_cells]
                popB_pos = self._get_soma_coordinates(self.wiring_information[popB]['cell info'])  # [i/n_cells]
                nA, nB = self.wiring_information[popA]['ncells'], self.wiring_information[popB]['ncells']

                same_pop = False

                if popA == popB: same_pop = True

                place_gid_set = None
                cue_gid_set = None

                if popB_id in place_information:
                    place_gids = place_information[popB_id]['place']
                    place_gid_set = set(place_gids)

                    cue_gids = cue_information[popB_id]['not place']
                    cue_gid_set = set(cue_gids)

                cue_weights = None
                place_weights = None
                weight_scale_func = None
                gradient = self.params['internal connectivity'][popA_id][popB_id].get('gradient', None)

                if gradient is not None:
                    cue_weight_mean = gradient['cue']['mean']
                    cue_weight_scale = gradient['cue']['scale']
                    place_weight_mean = gradient['place']['mean']
                    place_weight_scale = gradient['place']['scale']
                    cue_weights = np.clip(self.external_con_rnd.normal(cue_weight_mean,
                                                                       cue_weight_scale,
                                                                       nB), 0.0, None)
                    place_weights = np.clip(self.external_con_rnd.normal(place_weight_mean,
                                                                         place_weight_scale,
                                                                         nB), 0.0, None)

                    if same_pop:
                        weight_scale_func = lambda dst_gid, src_gid: cue_weights[dst_gid] if src_gid in cue_gid_set else \
                        place_weights[dst_gid]
                    else:
                        weight_scale_func = lambda dst_gid, src_gid: cue_weights[dst_gid] if dst_gid in cue_gid_set else \
                        place_weights[dst_gid]

                try:
                    convergence = self.params['internal connectivity'][popA_id][popB_id]['probability']  # list
                    compartments = self.params['internal connectivity'][popA_id][popB_id]['connection']  # list
                    if popA_id == 0 and popB_id == 0:
                        alpha = 0.0125
                        am, ws, pre_gids = self.create_adjacency_matrix(popA_pos, popB_pos, nA, nB,
                                                                        convergence, compartments,
                                                                        self.internal_con_rnd,
                                                              inv_func=('exp', alpha), same_pop=same_pop,
                                                              src_id=popA_id,
                                                              weight_scale_func=weight_scale_func)
                    else:
                        am, ws, pre_gids = self.create_adjacency_matrix(None, None, nA, nB,
                                                                        convergence, compartments,
                                                                        self.internal_con_rnd, same_pop=same_pop,
                                                              weight_scale_func=weight_scale_func)
                    self.internal_adj_matrices[popA_id][popB_id] = am
                    self.internal_con[popA_id][popB_id] = pre_gids
                    self.internal_ws[popA_id][popB_id] = ws
                except:
                    print('no connection between src %s and dst %s' % (popA, popB) )
                    continue

    def generate_external_connectivity(self, external_information, **kwargs):

        place_information = kwargs['place information']
        external_place_ids = kwargs['external place ids']

        cue_information = kwargs['cue information']
        external_cue_ids = kwargs['external cue ids']

        external_pops = list(external_information.keys())
        external_ids = [external_information[pop]['id'] for pop in external_pops]

        ctype_offset = 0
        for pop in self.wiring_information.keys():
            if ctype_offset < self.wiring_information[pop]['ctype offset']:
                ctype_offset = self.wiring_information[pop]['ctype offset'] + self.wiring_information[pop]['ncells']

        self.external_pop2id = {pop: i for (pop, i) in list(zip(external_pops, external_ids))}

        self.external_information = external_information
        self.external_adj_matrices = {}
        self.external_con = {}
        self.external_ws = {}
        for src_id, src_pop in list(zip(external_ids, external_pops)):
            self.external_information[src_pop]['ctype offset'] = ctype_offset
            ctype_offset += self.external_information[src_pop]['ncells']
            self.external_adj_matrices[src_id] = {}
            self.external_con[src_id] = {}
            self.external_ws[src_id] = {}
            for dst_pop in self.pops:
                dst_pop_id = self.pop2id[dst_pop]
                try:
                    src_pos = self._get_soma_coordinates(external_information[src_pop]['cell info'])
                except:
                    src_pos = None

                dst_pos = self._get_soma_coordinates(self.wiring_information[dst_pop]['cell info'])
                nsrc, ndst = external_information[src_pop]['ncells'], self.wiring_information[dst_pop]['ncells']

                if dst_pop_id not in self.params['external connectivity'][src_id]: continue
                convergence = self.params['external connectivity'][src_id][dst_pop_id]['probability']
                compartments = self.params['external connectivity'][src_id][dst_pop_id]['connection']

                place_connection_flag = dst_pop_id in place_information and src_id in external_place_ids
                cue_connection_flag = dst_pop_id in cue_information and src_id in external_cue_ids

                dst_gids_to_connect_to = []

                place_gids = None
                cue_gids = None

                if place_connection_flag:
                    place_gids = place_information[dst_pop_id]['place']
                    place_gid_set = set(place_gids)
                    dst_gids_to_connect_to.append(place_gids)

                if cue_connection_flag:
                    cue_gids = cue_information[dst_pop_id]['not place']
                    cue_gid_set = set(cue_gids)
                    dst_gids_to_connect_to.append(cue_gids)

                if not (place_connection_flag or cue_connection_flag):
                    dst_gids_to_connect_to.append(np.arange(ndst))

                dst_gids_to_connect_to = np.concatenate(dst_gids_to_connect_to)

                cue_weights = None
                place_weights = None
                gradient = self.params['external connectivity'][src_id][dst_pop_id].get('gradient', None)

                if gradient is not None:
                    cue_weight_mean = gradient['cue']['mean']
                    cue_weight_scale = gradient['cue']['scale']
                    place_weight_mean = gradient['place']['mean']
                    place_weight_scale = gradient['place']['scale']
                    cue_weights = np.clip(self.external_con_rnd.normal(cue_weight_mean,
                                                                       cue_weight_scale,
                                                                       ndst), 0.0, None)
                    place_weights = np.clip(self.external_con_rnd.normal(place_weight_mean,
                                                                         place_weight_scale,
                                                                         ndst), 0.0, None)

                weight_scale_func = None
                if cue_weights is not None and place_weights is not None:
                    weight_scale_func = lambda dst_gid, src_gid: cue_weights[dst_gid] if dst_gid in cue_gid_set else \
                    place_weights[dst_gid]

                if dst_pop_id == 0 and src_id < 102:
                    alpha = 0.00075
                    am, ws, pre_gids = self.create_adjacency_matrix(src_pos, dst_pos, nsrc, ndst,
                                                          convergence, compartments,
                                                          self.external_con_rnd, inv_func=('exp', alpha),
                                                          valid_gids=dst_gids_to_connect_to,
                                                          src_id=src_id,
                                                          weight_scale_func=weight_scale_func)
                else:
                    am, ws, pre_gids = self.create_adjacency_matrix(None, None, nsrc, ndst,
                                                          convergence, compartments,
                                                          self.external_con_rnd,
                                                          valid_gids=dst_gids_to_connect_to,
                                                          weight_scale_func=weight_scale_func)

                self.external_adj_matrices[src_id][dst_pop_id] = am
                self.external_ws[src_id][dst_pop_id] = ws
                self.external_con[src_id][dst_pop_id] = pre_gids

    def generate_septal_connectivity(self):
        id2pop = {v: k for (k, v) in self.pop2id.items()}
        self.septal_adj_matrices = {}
        self.septal_ws = {}
        self.septal_con = {}
        dst_ids = list(self.params['Septal']['connectivity'].keys())
        for dst_id in dst_ids:
            if dst_id not in list(id2pop.keys()): continue

            n_src, n_dst = self.params['Septal']['ncells'], self.wiring_information[id2pop[dst_id]]['ncells']
            convergence = self.params['Septal']['connectivity'][dst_id]['probability']
            compartments = self.params['Septal']['connectivity'][dst_id]['connection']
            try:
                am, ws, pre_gids = self.create_adjacency_matrix(None, None, n_src, n_dst,
                                                                convergence, compartments,
                                                                self.septal_con_rnd, same_pop=False)
                self.septal_adj_matrices[dst_id] = am
                self.septal_ws[dst_id] = ws
                self.septal_con[dst_id] = pre_gids
            except:
                print('Septal error...', dst_id, convergence)

    def _get_soma_coordinates(self, gid_dict):
        pos = []
        for gid in np.sort(list(gid_dict.keys())):
            pos.append(gid_dict[gid]['soma position'])
        return pos

    def create_adjacency_matrix(self, src_coordinates, dst_coordinates, nsrc, ndst, convergence, compartment, rnd,
                                inv_func=None, same_pop=False, valid_gids=None, src_id=None,
                                weight_scale_func=None):
        """
        实现对与hoc版本对应关系的描述
        :param src_coordinates: 没有对应，用于描述位置
        :param dst_coordinates: 没有对应，用于描述位置
        :param nsrc: 源细胞个数 $3
        :param ndst: 靶细胞个数 $1
        :param convergence: 细胞间的连接 $5
        :param rnd: 随机方程
        :param inv_func:
        :param same_pop:
        :param valid_gids:
        :param src_id:
        :param weight_scale_func:
        :return:
        """

        if valid_gids is None: valid_gids = np.arange(ndst)
        adj_mat = np.zeros((nsrc, ndst), dtype='uint16')
        weight_scales = defaultdict(lambda: 1.0)
        pre_synaptic_gids = []
        compartment = np.max(compartment)

        for d in range(ndst):
            if d not in valid_gids: continue
            if src_coordinates is not None and dst_coordinates is not None:

                assert len(src_coordinates) == nsrc
                assert len(dst_coordinates) == ndst

                dst_coord = dst_coordinates[d]
                distances = np.asarray([(dst_coord - src_coord) ** 2 for src_coord in src_coordinates])
                if inv_func[0] == 'inv':
                    inv_dist = 1. / (distances + 1.0e-5) ** inv_func[1]
                elif inv_func[0] == 'exp':
                    inv_dist = np.exp(-distances / inv_func[1])
                if same_pop:
                    inv_dist[d] = 0.0

                p_con = inv_dist / (np.sum(inv_dist) + 1.0e-10)
                effective_convergence = np.max(convergence)
                if (src_id == 0) and (dst_coord < 0.1 or dst_coord > 0.90) and (
                        d in self.place_information[0]['place']):
                    effective_convergence = math.ceil(effective_convergence * 0.75)
                elif (src_id == 100 or src_id == 101) and (dst_coord < 0.1 or dst_coord > 0.90):
                    effective_convergence = math.ceil(effective_convergence * 0.75)
                dst_place_field = int(dst_coord * 200 / 5)
                if src_id == 100:
                    if d in self.place_information[0]['place']:
                        pre_synaptic_gid = rnd.choice(range(dst_place_field*6, dst_place_field*6+6), replace=True,
                                                      size=2)
                        chosen_compartments = rnd.randint(0, compartment, size=(2,))
                    else:
                        pre_synaptic_gid = rnd.choice(np.arange(nsrc), replace=True,
                                                      size=effective_convergence)
                        chosen_compartments = rnd.randint(0, compartment, size=(effective_convergence,))

                elif src_id == 101:
                    if d in self.place_information[0]['place']:
                        pre_synaptic_gid = list(range(dst_place_field*8, dst_place_field*8+8))
                        chosen_compartments = rnd.randint(0, compartment, size=(8,))
                    else:
                        pre_synaptic_gid = rnd.choice(np.arange(nsrc), replace=True,
                                                      size=effective_convergence)
                        chosen_compartments = rnd.randint(0, compartment, size=(effective_convergence,))
                else:
                    pre_synaptic_gid = rnd.choice(np.arange(nsrc), p=p_con, replace=True,
                                                  size=effective_convergence)
                    chosen_compartments = rnd.randint(0, compartment, size=(effective_convergence,))
            else:
                inv_dist = np.ones(nsrc)
                if same_pop:
                    inv_dist[d] = 0
                p_con = inv_dist / (np.sum(inv_dist) + 1.0e-10)
                # for con in convergence:
                #    effective_convergence = con
                #    pre_synaptic_gid = rnd.choice(np.arange(nsrc), p=p_con, replace=True, size=effective_convergence)
                effective_convergence = np.max(convergence)
                compartment = np.max(compartment)
                pre_synaptic_gid = rnd.choice(np.arange(nsrc), p=p_con, replace=True, size=effective_convergence)
                chosen_compartments = rnd.randint(0, compartment, size=(effective_convergence,))

            for idx, gid in enumerate(pre_synaptic_gid):
                adj_mat[gid, d] += 1
                pre_synaptic_gids.append((gid, d, chosen_compartments[idx]))
                if weight_scale_func is not None:
                    weight_scales[(gid, d)] = weight_scale_func(d, gid)

        return adj_mat, weight_scales, pre_synaptic_gids