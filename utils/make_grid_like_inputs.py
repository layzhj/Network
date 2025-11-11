import numpy as np
from utils.gridfield import gridfield

import warnings

warnings.filterwarnings("ignore")


def get_grid_like_inputs(mouse_positions, freq, n_laps, n_dend=8, phase=0):
	# place_field: {n_lap: {'place_field_0': {0: spikes}}}
	
	x_array = range(0, 201, 5)
	y_array = [1]
	place_field = {}
	for lap in range(n_laps):
		np.random.seed(lap)
		loc_path = mouse_positions[lap]
		place_field[lap] = {}
		
		n_field = 0
		
		for xxx in x_array:
			for yyy in y_array:

				n_field += 1
				folder = f'place_field_{n_field}'

				d = np.zeros((n_dend, 201, 1))
				dd = np.zeros((201, 1))

				angle = 0.0
				lambda_var = 3.0
				for ni in range(n_dend):
					lambda_var += 0.5
					angle += 0.4
					for x in range(201):
						# d is the point x,y of grid field of dend ni
						for y in range(1):
							d[ni, x, y] = gridfield(angle, lambda_var, xxx, yyy, x, y)
					
				for ni in range(n_dend):
					dd += d[ni, :, :]
				dict_spike = {}
				for ni in range(n_dend):
					spikes = []
					for i in range(len(loc_path)):  # 表示放电时间
						current_loc = loc_path[i, :]

						probability = d[ni, current_loc[0] - 1, current_loc[1] - 1]
						probability *= (np.sin(2.0 * np.pi * freq *
											   i / 1000.0 + phase) + 1.0) / 2.0

						r_ = np.random.rand(1)
						if (probability > 0.7) and (r_ < probability / 2.0):
							spikes.append(i)
							
					dict_spike[ni] = spikes
				place_field[lap][folder] = dict_spike
	return place_field
