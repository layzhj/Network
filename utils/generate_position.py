import numpy as np

def generate_path(max_x, max_y=1, n_laps=10):
    paths = {}
    for lap in range(n_laps):
        np.random.seed(lap)
        nx, ny = 1, 1

        p0 = [nx, ny]
        k = 1

        zold = 1
        if max_y == 1:
            npoints = max_x*max_y
        else:
            npoints = max_x*(max_y-1)

        nstops = np.random.randint(0, 10, 1).item()
        rand_stops = sorted(list(set(np.random.randint(1, max_x-1, nstops))))
        rand_stops_minus = [i-1 for i in rand_stops]
        rand_stops_plus = [i+1 for i in rand_stops]
        path = []
        for i in range(1, npoints+1):
            if (ny > max_y):
                break

            if (i in rand_stops) or (i in rand_stops_minus) or (i in rand_stops_plus):
                sigma = 20
                mu = 50
            else:
                sigma = 10
                mu = 30
            z = int(mu + np.random.randn(1)*sigma)

            while z < 1:
                z = int(mu + np.random.randn(1)*sigma)

            time_at_the_same_point = z
            path += [p0]*int(time_at_the_same_point)

            nx = nx + k
            p0 = [nx, ny]

            if (nx > max_x) and (ny % 2) != 0:
                ny = ny+1
                nx = max_x-1
                p0 = [nx, ny]
                k = -1

            if (nx < 1) and (ny % 2) == 0:
                ny = ny+1
                nx = 1
                p0 = [nx, ny]
                k = 1

            zold += time_at_the_same_point

        paths[lap] = np.array(path, dtype=int)
    return paths
