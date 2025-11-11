import numpy as np


def get_spike_times(time_delay, n_dend, inputs, n_laps, n_place_field=41, shuffle=False):
	
	list_dirs = []
	spike_times = {}
	count_id = 0
	for plf in range(n_place_field):
		list_dirs.append(f'place_field_{plf+1}')
	list_shuffle_dirs = list(np.random.permutation(list_dirs)) if shuffle else list_dirs
	for list_dir in list_shuffle_dirs:
		lap_delay = time_delay
		vec = list(np.random.permutation(range(n_dend))) if shuffle else list(range(n_dend))
		
		for i in range(n_dend):
			lines_noisy = []
			for lap in range(n_laps):
				np.random.seed(lap)
				lines = inputs[lap][list_dir][vec[i]]
				lines = [int(x)+lap_delay for x in lines]
				lines = sorted(lines)
			
				max_mum = lines[-1]

				for i_line in lines:
					if np.random.rand() <= 0.05:
						lines_noisy.append(int(max_mum * np.random.rand()))
				else:
					lines_noisy.append(i_line)
					
				lines_noisy = list(set(lines_noisy))
				lines_noisy = sorted(lines_noisy)
				lines_noisy = [x for x in lines_noisy if x > 0]
				lap_delay = lines_noisy[-1]
			
			spike_times[count_id] = lines_noisy
			count_id += 1
	return spike_times


def get_cue_spike_times(positions, n_dend, inputs, time_delay, seed):
    '''
        position: {gid:{'type': 'cue' or 'normal', 'position': [position]}} [position]的长度为n_laps, 即每个元素的索引即为其所处的lap
        inputs: {'place_field_n': {dend: [(spike_time, lap)]}}
    '''
    np.random.seed(seed)

    gids = positions.keys()
    spike_times = {}
    
    for gid in gids:
        info = positions[gid]
        if info['type'] == 'normal':
            field_id = gid // n_dend
            dend_id = gid - field_id * n_dend
            spikes_and_laps = inputs[f'place_field_{field_id+1}'][dend_id]

            lines = [int(x[0])+time_delay for x in spikes_and_laps]
            lines = sorted(lines)
        else:
            lines = []
            for lap, position in enumerate(info['position']):
                field_id = int(position / 5)
                dend_id = gid % n_dend
                spikes_and_laps = inputs[f'place_field_{field_id+1}'][dend_id]

                line = [int(x[0])+time_delay for x in spikes_and_laps if x[1] == lap]
                lines.extend(line)
            lines = sorted(lines)
        
        max_mum = lines[-1]
        lines_noisy = []

        for i_line in lines:
            if np.random.rand() <= 0.10:
                lines_noisy.append(int(max_mum * np.random.rand()))
            else:
                lines_noisy.append(i_line)

        lines_noisy = list(set(lines_noisy))
        lines_noisy = sorted(lines_noisy)
        lines_noisy = [x for x in lines_noisy if x > 0]

        spike_times[gid] = lines_noisy

    return spike_times   
