import numpy as np


def get_place_inputs(place_field, arena_size, mouse_positions, n_laps, sigma=0.002, mu_in=0.16, mu_out=0.016):

    n_EC = 8
    n_CA3 = 6
    half_size = 6
    n_place = 41

    place_inputs = {}
    x_array = range(0, arena_size+1, 5)
	
    for lap in range(n_laps):
        np.random.seed(lap)
        place_inputs[lap] = {}

        path = mouse_positions[lap]
        path = path[:, 0]

        time_array = np.bincount(path)[1:]
        csum = np.cumsum(time_array)
        csum = np.insert(csum, 0, 0)

        for plf in range(1, n_place+1):
            folder = f'place_field_{plf}'
            place_inputs[lap][folder] = {}

            spikemap_sall = []
            for dend in range(n_EC):
                spike_time = place_field[lap][folder][dend]
                spikemap_sall += spike_time

            vector = sorted(spikemap_sall)
            peak = x_array[plf-1]

            initial = peak-half_size
            final = peak+half_size

            if initial < 0:
                initial = 0

            if final > arena_size:
                final = arena_size

            inplf_ca3_input = []
            outplf_ca3_input = []

            for spiketime in vector:
                if csum[initial] <= spiketime <= csum[final]:
                    inplf_ca3_input.append(spiketime)
                else:
                    outplf_ca3_input.append(spiketime)

            for i in range(0, n_CA3):

                z_in = (mu_in + np.random.randn() * sigma)
                z_out = (mu_out + np.random.randn() * sigma)

                inplf_ca3_input = list(np.random.permutation(inplf_ca3_input))
                outplf_ca3_input = list(np.random.permutation(outplf_ca3_input))

                count_in = int(len(inplf_ca3_input) * z_in)
                count_out = int(len(outplf_ca3_input) * z_out)

                if count_in == 0:
                    count_in = -1
                if count_out == 0:
                    count_out = -1

                shuf_vec_in = inplf_ca3_input[-count_in:]
                shuf_vec_out = outplf_ca3_input[-count_out:]

                ca3_input = shuf_vec_in + shuf_vec_out

                ca3_input = sorted(set(ca3_input))
                ca3_input = list(ca3_input)

                place_inputs[lap][folder][i] = ca3_input
    return place_inputs
