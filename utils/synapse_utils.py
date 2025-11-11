import numpy as np
import copy
from neuron import h


def _make_synapse(stype, compartment, weight, synapse_information, srcid, dstid, dst_cell, c, params):
    syn_params = {}
    if stype == 'AMPA':
        if dstid == 0:
            syn_params = dst_cell.get_syn_parameters(compartment, stype)
            if srcid == 0:
                syn_params['tau1'] = 0.10
                syn_params['tau2'] = 1.50
        else:
            syn_params = {'e': synapse_information['e'], 'tau1': synapse_information['tau1'],
                          'tau2': synapse_information['tau2']}
    elif stype == 'NMDA':
        syn_params = dst_cell.get_syn_parameters(compartment, stype)
    elif stype == 'GABAA':
        syn_params = {'e': synapse_information['e'], 'tau1': synapse_information['tau1'],
                      'tau2': synapse_information['tau2']}
    elif stype == 'GABAB':
        syn_params['e'] = -75.0
        syn_params['tau1'] = 35.0
        syn_params['tau2'] = 100.0
    elif stype == 'STDPE2':
        if dstid == 0:
            syn_params = dst_cell.get_syn_parameters(compartment, stype)
        else:
            syn_params = {'e': synapse_information['e'], 'tau1': synapse_information['tau1'],
                          'tau2': synapse_information['tau2']}
        if dstid == 0 and syn_params['tau1'] is None:
            syn_params = {'e': synapse_information['e'], 'tau1': synapse_information['tau1'],
                          'tau2': synapse_information['tau2']}

        pparams = params['symplasticity']


        syn_params['p'] = pparams.get('potentiation', 1.0)
        syn_params['d'] = pparams.get('depression', 0.4)
        syn_params['thresh'] = pparams.get('thresh', -10.)
        wmax_scaler = synapse_information.get('wmax_scaler', pparams.get('wmax_scaler', 1.0))
        wmin = pparams.get('wmin', 0.0005)
        syn_params['wmax'] = wmax_scaler * wmin
        syn_params['wmin'] = wmin
        syn_params['dtau'] = pparams.get('dtau', 34.0)
        syn_params['ptau'] = pparams.get('ptau', 17.0)
        syn_params['p'] = synapse_information.get('depression', syn_params['d']) / params['scale']
        syn_params['d'] = synapse_information.get('depression', syn_params['d']) / params['scale']
        syn_params['dtau'] = synapse_information.get('dtau', syn_params['dtau'])
        syn_params['ptau'] = synapse_information.get('dtau', syn_params['dtau'])

    syn_ = None
    if stype == 'STDPE2':
        syn_ = h.STDPE2(c(0.5))
    elif stype == 'STDPE2ASYM':
        syn_ = h.STDPE2ASYM(c(0.5))
    elif stype == 'NMDA':
        syn_ = h.NMDA(c(0.5))
    else:
        if dstid == 4:
            syn_ = h.Exp2Syn(c(0.5))
        else:
            syn_ = h.MyExp2Syn(c(0.5))
    for k in syn_params.keys(): setattr(syn_, k, syn_params[k])
    return syn_


def create_netcon(pc, srcid, dstid, src_gid, dst_cell, synapse_information, compartment, params, **netcon_params):
    ncs = []

    if not pc.gid_exists(dst_cell.gid):
        return ncs

    synapse_type = netcon_params.get('synapse_type', synapse_information['type'])
    weight1 = synapse_information['weight1']  # 突触连接权值
    weight = weight1
    if params['scale'] > 1:
        weight /= (float(params['scale']))
    ws = netcon_params.get('weight_scale', 1.0)  # 获得权重比例

    weight *= float(ws)

    syn_params = copy.deepcopy(params)
    if synapse_type == 'STDPE2':
        if 'weight_scale' in netcon_params:
            syn_params['symplasticity']['wmax_scaler'] = params['symplasticity']['wmax_scaler'] * ws
    elif synapse_type == 'STDPE2ASYM':
        if 'weight_scale' in netcon_params:
            syn_params['asymplasticity']['wmax_scaler'] = params['asymplasticity']['wmax_scaler'] * ws


    c = None
    if srcid not in dst_cell.synGroups[synapse_type][compartment]:
        c = getattr(dst_cell, compartment)
        dst_cell.synGroups[synapse_type][compartment][srcid] = []
    if c is not None:  # if synape doesn't exist, build it
        syn_ = _make_synapse(synapse_type, compartment, weight, synapse_information,
                             srcid, dstid, dst_cell, c, syn_params)
        dst_cell.synGroups[synapse_type][compartment][srcid].append(syn_)
    else:
        syn_ = dst_cell.synGroups[synapse_type][compartment][srcid][0]

    nc = pc.gid_connect(src_gid, syn_)
    nc.delay = 1.
    nc.weight[0] = weight
    if synapse_type == 'STDPE2' or synapse_type == 'STDPE2ASYM':
        # nc.weight[0] = 0.0005
        nc.weight[1] = 0.

    ncs.append(nc)

    return ncs
