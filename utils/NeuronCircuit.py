import numpy as np
import copy
import yaml
import os
import sys
import pickle
from collections import Counter

from cell.PyramidalCell import PyramidalCell
from cell.AACell import AACell
from cell.BasketCell import BasketCell
from cell.BistratifiedCell import BistratifiedCell
from cell.OLMCell import OLMCell
from cell.VIPCCKCell import VIPCCKCell
from cell.VIPCRCell import VIPCRCell
from cell.septal import Septal
from cell.RandomStream import RandomStream

from neuron import h
from utils.synapse_utils import create_netcon


class Circuit(object):

    def __init__(self, arena_params_filename, params_filename, internal_pop2id, external_pop2id, external_spike_times,
                 params_prefix='.'):
        self.pc = h.ParallelContext()
        self.params_prefix = params_prefix
        self.cell_params = None
        self.arena_cells = None
        self._read_params(params_prefix, params_filename, arena_params_filename)
        self.cell_params_path = os.path.join(params_prefix, params_filename)

        self.internal_pop2id = internal_pop2id
        self.external_pop2id = external_pop2id

        self.ctype_offsets = {}
        self.neurons = {}
        self.v_trace = {}
        self.spike_times = {}
        self.external_spike_times = {}
        self.external_spike_time_recs = {}
        self.external_spike_gid_recs = {}
        self.nclist = []

        self.external_spike_times = external_spike_times
        self.stimecell_vecs = {}

    def _read_params(self, params_prefix, params_filename, arena_params_filename):

        if self.pc.id() == 0:
            with open(os.path.join(params_prefix, params_filename), 'r') as f:
                fparams = yaml.load(f, Loader=yaml.FullLoader)
                self.cell_params = fparams['Circuit']
            with open(os.path.join(params_prefix, arena_params_filename), 'r') as f:
                fparams = yaml.load(f, Loader=yaml.FullLoader)
                self.arena_cells = fparams['Arena Cells']
        self.pc.barrier()
        self.cell_params = self.pc.py_broadcast(self.cell_params, 0)
        self.arena_cells = self.pc.py_broadcast(self.arena_cells, 0)

    def build_cells(self):
        ext_types = list(self.arena_cells.keys())
        cell_types = self.cell_params['cells']
        cell_numbers = self.cell_params['ncells']

        ctype_offset = 0
        for (i, ctype) in enumerate(cell_types):
            cidx = self.internal_pop2id[ctype]
            ncells = cell_numbers[i]
            self.ctype_offsets[cidx] = ctype_offset
            self.neurons[cidx] = {}
            for ctype_id in range(ncells):
                gid = ctype_id + ctype_offset
                if ctype == 'PyramidalCell':
                    cell = PyramidalCell(gid)
                elif ctype == 'AACell':
                    cell = AACell(gid)
                elif ctype == 'BasketCell':
                    cell = BasketCell(gid)
                elif ctype == 'BistratifiedCell':
                    cell = BistratifiedCell(gid)
                elif ctype == 'OLMCell':
                    cell = OLMCell(gid)
                elif ctype == 'VIPCCKCell':
                    cell = VIPCCKCell(gid)
                elif ctype == 'VIPCRCell':
                    cell = VIPCRCell(gid)
                else:
                    raise RuntimeError(f"Unknown cell type {ctype}")

                self.neurons[cidx][ctype_id] = cell
                self.pc.set_gid2node(gid, self.pc.id())
                self.pc.cell(gid, cell.spike_detector)

            ctype_offset += ncells

        for (i, ctype) in enumerate(ext_types):
            cidx = self.external_pop2id[ctype]
            ncells = self.arena_cells[ctype]['ncells']
            self.ctype_offsets[cidx] = ctype_offset
            self.neurons[cidx] = {}
            self.external_spike_time_recs[cidx] = h.Vector()
            self.external_spike_gid_recs[cidx] = h.Vector()
            for ctype_id in range(ncells):
                gid = ctype_id + ctype_offset
                if ctype == 'CA3':
                    stim_cell = h.VecStim()
                    vec = h.Vector(self.external_spike_times[cidx][ctype_id])
                    stim_cell.play(vec)
                elif ctype == 'EC':
                    stim_cell = h.VecStim()
                    vec = h.Vector(self.external_spike_times[cidx][ctype_id])
                    stim_cell.play(vec)
                elif ctype == 'LEC':
                    stim_cell = h.VecStim()
                    vec = h.Vector(self.external_spike_times[cidx][ctype_id])
                    stim_cell.play(vec)
                elif ctype == 'Background':
                    stim_cell = h.VecStim()
                    vec = h.Vector(self.external_spike_times[cidx][ctype_id])
                    stim_cell.play(vec)
                else:
                    raise RuntimeError(f"Unknown cell type {ctype}")
                self.neurons[cidx][ctype_id] = stim_cell
                self.pc.set_gid2node(gid, self.pc.id())
                spike_detector = h.NetCon(stim_cell, None)
                self.pc.cell(gid, spike_detector)  # Associate the cell with this host and gid
                self.pc.spike_record(gid,
                                     self.external_spike_time_recs[cidx],
                                     self.external_spike_gid_recs[cidx])

            ctype_offset += ncells

        if 'Septal' in self.cell_params.keys():
            sep_params = self.cell_params['Septal']['parameters']
            ncells = self.cell_params['Septal']['ncells']
            self.neurons['Septal'] = {}
            self.ctype_offsets['Septal'] = ctype_offset
            burst_int = sep_params['THETA'] * 2 / 3
            burst_len = sep_params['THETA'] / 3
            sep_num = sep_params['number']
            for ctype_id in range(ncells):
                gid = ctype_id + ctype_offset
                r_obj = RandomStream(seed=int(ctype_id + ctype_offset))
                r_obj.start()
                r_obj.r.discunif(0, sep_num)

                sc = Septal(gid)
                stim = sc.stim
                stim.number = r_obj.re_pick()
                stim.start = sep_params['start']
                stim.interval = sep_params['interval']
                stim.noise = sep_params['noise']
                stim.burstint = burst_int
                stim.burstlen = burst_len

                self.neurons['Septal'][ctype_id] = sc
                self.pc.set_gid2node(gid, self.pc.id())
                self.pc.cell(gid, sc.spike_detector)

            ctype_offset += ncells

    def build_internal_netcons(self, internal_connection, internal_ws, seed=1e6):
        rnd = np.random.RandomState(seed=int(seed))
        src_population_ids = list(internal_connection.keys())
        for src_pop_id in src_population_ids:
            src_offset = self.ctype_offsets[src_pop_id]
            dst_population_ids = list(internal_connection[src_pop_id].keys())
            for dst_pop_id in dst_population_ids:
                con_matrix = internal_connection[src_pop_id][dst_pop_id]  # [(src， gid)]
                if con_matrix is []: continue
                this_external_ws = internal_ws[src_pop_id][dst_pop_id]
                dst_neurons = self.neurons[dst_pop_id]
                connection_information = self.cell_params['internal connectivity'][src_pop_id][dst_pop_id]
                synapse_information = connection_information['synapse']
                n_synapse = connection_information['connection']
                n_connection = connection_information['probability']
                for idx, connection in enumerate(n_connection):
                    scales = int(len(con_matrix) * connection / np.max(n_connection))
                    sub_matrix = con_matrix[:scales]
                    dst_ids = set([con_matrix[i][1] for i in range(scales)])
                    for dst_id in dst_ids:
                        synapses = n_synapse[idx]
                        src_ids = [sub_matrix[i][0] for i in range(scales) if sub_matrix[i][1] == dst_id]
                        synapse_info = synapse_information[idx]
                        compartments = synapse_info['compartments']
                        rnd_compartments = [sub_matrix[i][2] if sub_matrix[i][1] == dst_id and sub_matrix[i][2]<synapses
                                            else rnd.randint(0, synapses) for i in range(scales)]

                        chosen_compartments = [compartments[r_idx] for r_idx in rnd_compartments]
                        for src_id, compartment in zip(src_ids, chosen_compartments):
                            ws = this_external_ws[(src_id, dst_id)]
                            src_gid = src_id + src_offset
                            ncs = create_netcon(self.pc, src_pop_id, dst_pop_id, src_gid, dst_neurons[dst_id],
                                                synapse_info, compartment, self.cell_params, weight_scale=ws)
                            self.nclist.extend(ncs)
                            dst_neurons[dst_id].internal_netcons.append((src_gid, ncs, compartment,
                                                                         synapse_info['type']))

    def build_external_netcons(self, src_pop_id, external_connection, external_ws, seed=1e7):
        seed += src_pop_id
        rnd = np.random.RandomState(seed=int(seed))

        src_offset = self.ctype_offsets[src_pop_id]
        dst_population_ids = list(external_connection.keys())
        for dst_pop_id in dst_population_ids:
            con_matrix = external_connection[dst_pop_id]  # [(src， gid)]
            if con_matrix is []: continue
            this_external_ws = external_ws[dst_pop_id]
            dst_neurons = self.neurons[dst_pop_id]
            connection_information = self.cell_params['external connectivity'][src_pop_id][dst_pop_id]
            cell_connection_information = connection_information['probability']
            synapse_information = connection_information['synapse']
            n_synapse = connection_information['connection']
            n_connection = connection_information['probability']
            if src_pop_id < 102 and dst_pop_id == 0:
                dst_ids = set([con_matrix[i][1] for i in range(len(con_matrix))])
                for dst_id in dst_ids:
                    src_ids = [con_matrix[i][0] for i in range(len(con_matrix)) if con_matrix[i][1] == dst_id]
                    for idx, synapses in enumerate(n_synapse):
                        synapse_info = synapse_information[idx]
                        compartments = synapse_info['compartments']
                        if n_synapse[idx] > len(compartments):
                            rnd_compartment = rnd.randint(0, len(compartments),
                                                          size=(n_synapse[idx] - len(compartments),))
                            rnd_compartments = list(range(len(compartments))) + list(rnd_compartment)
                        else:
                            rnd_compartments = list(range(len(compartments)))
                        chosen_compartments = [compartments[r_idx] for r_idx in rnd_compartments]
                        for src_id, compartment in zip(src_ids, chosen_compartments):
                            ws = this_external_ws[(src_id, dst_id)]
                            src_gid = src_id + src_offset
                            ncs = create_netcon(self.pc, src_pop_id, dst_pop_id, src_gid, dst_neurons[dst_id],
                                                synapse_info, compartment, self.cell_params, weight_scale=ws)
                            self.nclist.extend(ncs)
                            if src_pop_id not in dst_neurons[dst_id].external_netcons:
                                dst_neurons[dst_id].external_netcons[src_pop_id] = []
                            dst_neurons[dst_id].external_netcons[src_pop_id].append((src_gid, ncs, compartment,
                                                                                     synapse_info['type']))
            else:
                dst_ids = set([con_matrix[i][1] for i in range(len(con_matrix))])
                for dst_id in dst_ids:
                    src_ids = [con_matrix[i][0] for i in range(len(con_matrix)) if
                               con_matrix[i][1] == dst_id]  # r = rs.r.discunif($4, $4+$3-1)
                    for idx, synapses in enumerate(n_synapse):
                        synapse_info = synapse_information[idx]
                        compartments = synapse_info['compartments']
                        if len(src_ids) > cell_connection_information[idx]: src_ids = src_ids[
                                                                                      0:cell_connection_information[
                                                                                          idx]]
                        rnd_compartments = rnd.randint(0, len(compartments),
                                                       size=(len(src_ids),))  # j = rs.r.discunif($6, $7)
                        chosen_compartments = [compartments[r_idx] for r_idx in
                                               rnd_compartments]  # syn = cells.object(i).pre_list.object(j)
                        for src_id, compartment in zip(src_ids, chosen_compartments):
                            ws = this_external_ws[(src_id, dst_id)]
                            src_gid = src_id + src_offset
                            ncs = create_netcon(self.pc, src_pop_id, dst_pop_id, src_gid, dst_neurons[dst_id],
                                                synapse_info, compartment, self.cell_params, weight_scale=ws)
                            self.nclist.extend(ncs)
                            if src_pop_id not in dst_neurons[dst_id].external_netcons:
                                dst_neurons[dst_id].external_netcons[src_pop_id] = []
                            dst_neurons[dst_id].external_netcons[src_pop_id].append((src_gid, ncs, compartment,
                                                                                     synapse_info['type']))

    def build_septal_netcons(self, septal_connection, seed=1e8):
        rnd = np.random.RandomState(seed=int(seed))
        src_offset = self.ctype_offsets['Septal']
        dst_population_ids = list(septal_connection.keys())
        for dst_pop_id in dst_population_ids:
            con_matrix = septal_connection[dst_pop_id]
            if con_matrix is []: continue
            dst_neurons = self.neurons[dst_pop_id]
            connection_information = self.cell_params['Septal']['connectivity'][dst_pop_id]
            synapse_information = connection_information['synapse']
            n_synapse = connection_information['connection']
            dst_ids = set([con_matrix[i][1] for i in range(len(con_matrix))])
            for dst_id in dst_ids:
                src_ids = [con_matrix[i][0] for i in range(len(con_matrix)) if con_matrix[i][1] == dst_id]
                for idx, synapses in enumerate(n_synapse):
                    synapse_info = synapse_information[idx]
                    compartments = synapse_info['compartments']
                    rnd_compartments = rnd.randint(0, len(compartments), size=(len(src_ids),))
                    chosen_compartments = [compartments[r_idx] for r_idx in rnd_compartments]
                    for src_id, compartment in zip(src_ids, chosen_compartments):
                        src_gid = src_id + src_offset
                        ncs = create_netcon(self.pc, 'Septal', dst_pop_id, src_gid, dst_neurons[dst_id],
                                            synapse_info, compartment, self.cell_params)
                        self.nclist.extend(ncs)
                        dst_neurons[dst_id].internal_netcons.append((src_gid, ncs, compartment,
                                                                     synapse_info['type']))

    def record_lfp(self, population_ids):  # [0, 2]
        neurons = self.neurons
        self.lfp = []
        for pop_id in population_ids:
            current_neural_population = neurons[pop_id]
            for gid in current_neural_population.keys():
                current_neuron = current_neural_population[gid]
                for syntype in current_neuron.synGroups:
                    for synlocation in current_neuron.synGroups[syntype]:
                        if (synlocation == 'soma' or synlocation == 'radTprox' or synlocation == 'oriprox1'
                                or synlocation == 'oriprox2'):
                            synapses = current_neuron.synGroups[syntype][synlocation]
                            for pid in synapses.keys():
                                syns = synapses[pid]
                                for syn in syns:
                                    curr = h.Vector()
                                    curr.record(syn._ref_i, 0.1)
                                    self.lfp.append(curr)

    def save_netcon_data(self, save_filepath):
        complete_weights = {}
        for population_id in self.neurons.keys():
            if population_id == 'Septal': continue
            ctype_offset = self.ctype_offsets[population_id]
            population_info = self.neurons[population_id]
            for cell_id in population_info.keys():
                cell_gid = ctype_offset + int(cell_id)
                connection_weights = []
                connection_weights_upd = []
                src_gids = []
                compartments = []
                synapse_types = []
                cell_info = population_info[cell_id]
                if not hasattr(cell_info, 'internal_netcons'):
                    continue
                for (presynaptic_id, ncs, compartment, syn_type) in cell_info.internal_netcons:
                    for netcon in ncs:
                        compartments.append(compartment)
                        src_gids.append(int(netcon.srcgid()))
                        synapse_types.append(syn_type)
                        connection_weights.append(netcon.weight[0])
                        if len(netcon.weight) == 3:
                            connection_weights_upd.append(netcon.weight[1])
                        else:
                            connection_weights_upd.append(np.nan)
                for external_id in cell_info.external_netcons.keys():
                    external_cell_info = cell_info.external_netcons[external_id]
                    for (idx, (presynaptic_gid, ncs, compartment, syn_type)) in enumerate(external_cell_info):
                        for netcon in ncs:
                            compartments.append(compartment)
                            synapse_types.append(syn_type)
                            src_gids.append(int(netcon.srcgid()))
                            connection_weights.append(netcon.weight[0])
                            if len(netcon.weight) == 3:
                                connection_weights_upd.append(netcon.weight[1])
                            else:
                                connection_weights_upd.append(np.nan)
                connection_info = np.core.records.fromarrays((np.asarray(connection_weights, dtype=np.float32),
                                                              np.asarray(connection_weights_upd, dtype=np.float32),
                                                              np.asarray(src_gids, dtype=np.uint32),
                                                              np.asarray(compartments),
                                                              np.asarray(synapse_types)),
                                                             names='weights,weights_upd,src_gids,compartments, synapse_types')
                complete_weights[str(cell_gid)] = connection_info

        all_complete_weights = self.pc.py_gather(complete_weights, 0)

        if self.pc.id() == 0:
            # np.savez(save_filepath, **all_complete_weights)
            complete_weights = {}
            for d in all_complete_weights:
                complete_weights.update(d)
            np.savez(save_filepath, **complete_weights)

        self.pc.barrier()

    def save_v_vecs(self, save_filepath, v_vecs):

        v_vec_dict = {k: np.asarray(v, dtype=np.float32) for k, v in v_vecs.items()}
        all_v_vecs = self.pc.py_gather(v_vec_dict, 0)

        if self.pc.id() == 0:
            v_vecs = {}
            for d in all_v_vecs:
                v_vecs.update([(str(k), v) for (k, v) in d.items()])
            np.savez(save_filepath, **v_vecs)

        self.pc.barrier()

    def save_spike_vecs(self, save_filepath, *spike_time_dicts):

        all_spike_dicts = self.pc.py_gather(spike_time_dicts, 0)

        if self.pc.id() == 0:
            spike_dict = {}
            for ds in all_spike_dicts:
                for d in ds:
                    spike_dict.update([(str(k), v) for (k, v) in d.items()])
            np.savez(save_filepath, **spike_dict)

        self.pc.barrier()
