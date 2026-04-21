#중복조건 제거&연산순서 정리

import numpy as np
import time
from mpi4py import MPI

comm = MPI.COMM_WORLD

rank = comm.Get_rank()  
size = comm.Get_size()  

N = 1000
T = 20 
infection_prob = 0.3 
recovery_time = 5 
death_prob = 0.1
num_initial_infected = 200 

rows_per_proc_list = [N // size + (1 if i < N % size else 0) for i in range(size)]
start_row = sum(rows_per_proc_list[:rank])
current_rows = rows_per_proc_list[rank]
end_row = start_row + current_rows

grid = np.zeros((current_rows + 2, N), dtype=np.int32)
infection_duration = np.zeros_like(grid, dtype=np.int32)

if rank == 0:
    city_map = np.zeros((N, N), dtype=np.int32)
    city_map[20:25, :] = 1
    city_map[:, 45:48] = 1
    for i in range(0, N):
        city_map[i, int(20 + 10 * np.sin(i / 10))] = 1
    city_map[60:68, 10:18] = 2
    city_map[30:40, 70:80] = 2
    city_map[50:52, 50:70] = 2
else:
    city_map = None

city_map = comm.bcast(city_map, root=0)

if rank == 0:
    possible_positions = np.argwhere(city_map == 1)
    if len(possible_positions) < num_initial_infected:
        num_initial_infected = len(possible_positions)
    indices = np.random.choice(possible_positions.shape[0], num_initial_infected, replace=False)
    init_coords = [tuple(pos) for pos in possible_positions[indices]]
else:
    init_coords = None

init_coords = comm.bcast(init_coords, root=0)

for i_global, j in init_coords:
    if start_row <= i_global < end_row:
        local_i = i_global - start_row + 1
        if city_map[i_global, j] != 2:
            grid[local_i, j] = 1
            infection_duration[local_i, j] = 1

def exchange_borders(grid):
    top_border = grid[1, :].copy()
    bottom_border = grid[-2, :].copy()
    requests = []

    if rank > 0:
        req = comm.Isend(top_border, dest=rank - 1, tag=22)
        requests.append(req)
        req = comm.Irecv(grid[0, :], source=rank - 1, tag=11)
        requests.append(req)

    if rank < size - 1:
        req = comm.Isend(bottom_border, dest=rank + 1, tag=11)
        requests.append(req)
        req = comm.Irecv(grid[-1, :], source=rank + 1, tag=22)
        requests.append(req)

    MPI.Request.Waitall(requests)

def step(grid, infection_duration, city_map):
    new_grid = grid.copy()
    new_duration = infection_duration.copy()

    # 회복/사망 로직(벡터화)
    infected_mask = (grid[1:-1, :] == 1)
    new_duration[1:-1, :][infected_mask] += 1

    ready_to_change_mask = (new_duration[1:-1, :] >= recovery_time) & infected_mask

    if np.any(ready_to_change_mask):
        dies_mask = np.random.rand(*ready_to_change_mask.shape) < death_prob
        
        death_cells_mask = ready_to_change_mask & dies_mask
        new_grid[1:-1, :][death_cells_mask] = 3
        new_duration[1:-1, :][death_cells_mask] = 0

        recover_cells_mask = ready_to_change_mask & ~dies_mask
        new_grid[1:-1, :][recover_cells_mask] = 2
        new_duration[1:-1, :][recover_cells_mask] = 0

    # 감염 확산 로직(벡터화)
    temp_city_map_padded = np.zeros_like(grid, dtype=np.int32)
    temp_city_map_padded[1:-1, :] = city_map[start_row:end_row, :]
    if rank > 0:
        temp_city_map_padded[0, :] = city_map[start_row - 1, :]
    if rank < size - 1:
        temp_city_map_padded[-1, :] = city_map[end_row, :]

    susceptible_mask_padded = (grid == 0) & (temp_city_map_padded == 1)
    infector_mask_padded = (grid == 1) & (temp_city_map_padded == 1)

    has_infected_neighbor_mask = np.zeros_like(grid, dtype=bool)
    has_infected_neighbor_mask[1:, :] |= infector_mask_padded[:-1, :]
    has_infected_neighbor_mask[:-1, :] |= infector_mask_padded[1:, :]
    has_infected_neighbor_mask[:, 1:] |= infector_mask_padded[:, :-1]
    has_infected_neighbor_mask[:, :-1] |= infector_mask_padded[:, 1:]
    potential_infection_mask = susceptible_mask_padded[1:-1, :] & has_infected_neighbor_mask[1:-1, :]
    random_prob_matrix = np.random.rand(current_rows, N)
    actually_infected_mask = potential_infection_mask & (random_prob_matrix < infection_prob)

    new_grid[1:-1, :][actually_infected_mask] = 1
    new_duration[1:-1, :][actually_infected_mask] = 1

    return new_grid, new_duration

if rank == 0:
    recvcounts = [r * N for r in rows_per_proc_list]
    displs = [sum(rows_per_proc_list[:i]) * N for i in range(size)]
else:
    recvcounts = None
    displs = None

comm.Barrier()
start_time = time.time()

for t in range(T):
    exchange_borders(grid)
    grid, infection_duration = step(grid, infection_duration, city_map)

    final_grid = None
    if rank == 0:
        final_grid = np.zeros((N, N), dtype=np.int32)

    sendbuf = grid[1:-1, :].copy()
    comm.Gatherv(sendbuf, (final_grid, recvcounts, displs, MPI.INT), root=0)

    if rank == 0:
        np.save(f"frame_{t:02d}_road.npy", final_grid.copy())
        if t == 0:
            np.save("city_map.npy", city_map)

comm.Barrier()
end_time = time.time()
elapsed = end_time - start_time

total_elapsed_time = comm.reduce(elapsed, op=MPI.SUM, root=0)

if rank == 0:
    print(f"총 시뮬레이션 시간 (Rank 0): {elapsed:.4f}초")
    print(f"총 시뮬레이션 시간 (평균): {total_elapsed_time:.4f}초")

local_infected = np.sum(grid[1:-1, :] == 1)
local_dead = np.sum(grid[1:-1, :] == 3)

total_infected = comm.reduce(local_infected, op=MPI.SUM, root=0)
total_dead = comm.reduce(local_dead, op=MPI.SUM, root=0)

if rank == 0:
    print(f"최종 감염자 수: {total_infected}")
    print(f"최종 사망자 수: {total_dead}")
    print(f"최종 감염 맵 (frame {T-1})이 저장되었습니다.")
