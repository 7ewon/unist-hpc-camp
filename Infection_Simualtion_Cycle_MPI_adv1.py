import numpy as np
import time
from mpi4py import MPI
from PIL import Image

comm = MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()

N = 100
T = 20
infection_prob = 0.3
recovery_time = 5
death_prob = 0.1
num_initial_infected = 20
movement_speed_road = 2
movement_speed_general = 1

rows_per_proc = N // size
start_row = rank * rows_per_proc
end_row = (rank + 1) * rows_per_proc

grid = np.zeros((rows_per_proc + 2, N), dtype=np.int32)
infection_duration = np.zeros_like(grid, dtype=np.int32)

if rank == 0:
    IMAGE_PATH = "/Users/kywn07/Desktop/Cycle/Cycle_map_Image.png"

    try:
        print(f"'{IMAGE_PATH}' 이미지를 불러오는 중입니다...")
        img = Image.open(IMAGE_PATH).convert("RGB")
        img = img.resize((N, N))
        img_np = np.array(img)
        print("이미지를 성공적으로 불러왔습니다.")

        city_map = np.zeros((N, N), dtype=np.int32)

        print("이미지 픽셀을 분석하여 지도를 생성하는 중입니다...")
        for i in range(img_np.shape[0]):
            for j in range(img_np.shape[1]):
                r, g, b = img_np[i, j]

                if r > 200 and g > 200 and b > 200 and abs(r - g) < 20 and abs(g - b) < 20:
                    city_map[i, j] = 0

                elif g > 150 and g > r and g > b + 20: 
                    city_map[i, j] = 2
                else:
                    city_map[i, j] = 1

        print("지도 생성이 완료되었습니다.")
        np.save("city_map.npy", city_map)

    except FileNotFoundError:
        print(f"[오류] Rank {rank}에서 '{IMAGE_PATH}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        comm.Abort()
        city_map = None
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
        comm.Abort()
        city_map = None
else:
    city_map = None

city_map = comm.bcast(city_map, root=0)

if rank == 0:
    movable_positions = np.argwhere(city_map != 2)
    if len(movable_positions) < num_initial_infected:
        print("경고: 초기 감염자 수보다 이동 가능한 위치의 수가 적습니다.")
        num_initial_infected = len(movable_positions)

    indices = np.random.choice(movable_positions.shape[0], num_initial_infected, replace=False)
    init_coords = [tuple(pos) for pos in movable_positions[indices]]
else:
    init_coords = None

init_coords = comm.bcast(init_coords, root=0)

for i_global, j in init_coords:
    if start_row <= i_global < end_row:
        local_i = i_global - start_row + 1
        if city_map[i_global, j] != 2:
            grid[local_i, j] = 1
            infection_duration[local_i, j] = 1

def exchange_borders(grid_to_exchange):
    top = grid_to_exchange[1, :].copy()
    bottom = grid_to_exchange[-2, :].copy()

    if rank > 0:
        comm.Sendrecv(
            sendbuf=top, dest=rank-1, sendtag=11,
            recvbuf=grid_to_exchange[0, :], source=rank-1, recvtag=22
        )
    if rank < size - 1:
        comm.Sendrecv(
            sendbuf=bottom, dest=rank+1, sendtag=22,
            recvbuf=grid_to_exchange[-1, :], source=rank+1, recvtag=11
        )


def step(grid, infection_duration, city_map):
    new_grid = np.zeros_like(grid)
    new_infection_duration = np.zeros_like(infection_duration)

    moved_cells_temp = np.zeros_like(grid)
    moved_durations_temp = np.zeros_like(grid)

    outgoing_top = np.zeros(N, dtype=np.int32)
    outgoing_bottom = np.zeros(N, dtype=np.int32)
    outgoing_top_duration = np.zeros(N, dtype=np.int32)
    outgoing_bottom_duration = np.zeros(N, dtype=np.int32)

    for i in range(1, rows_per_proc + 1):
        for j in range(N):
            gi = start_row + i - 1

            if grid[i, j] == 1:
                current_speed = movement_speed_road if city_map[gi, j] == 0 else movement_speed_general
                
                possible_moves = []
                for dx in range(-current_speed, current_speed + 1):
                    for dy in range(-current_speed, current_speed + 1):
                        if dx == 0 and dy == 0: continue
                        
                        ni = i + dx
                        nj = j + dy
                        ngi = gi + dx

                        if 0 <= nj < N and 0 <= ngi < N and city_map[ngi, nj] != 2:
                            possible_moves.append((ni, nj))

                if possible_moves:
                    next_i, next_j = possible_moves[np.random.choice(len(possible_moves))]
                    
                    if 1 <= next_i <= rows_per_proc:
                        moved_cells_temp[next_i, next_j] = 1
                        moved_durations_temp[next_i, next_j] = infection_duration[i, j]
                        moved_cells_temp[i, j] = 0
                        moved_durations_temp[i, j] = 0
                    elif next_i == 0:
                        outgoing_top[next_j] = 1
                        outgoing_top_duration[next_j] = infection_duration[i, j]
                        moved_cells_temp[i, j] = 0
                        moved_durations_temp[i, j] = 0
                    elif next_i == rows_per_proc + 1:
                        outgoing_bottom[next_j] = 1
                        outgoing_bottom_duration[next_j] = infection_duration[i, j]
                        moved_cells_temp[i, j] = 0
                        moved_durations_temp[i, j] = 0
                else:
                    moved_cells_temp[i, j] = 1
                    moved_durations_temp[i, j] = infection_duration[i, j]
            else:
                moved_cells_temp[i, j] = grid[i, j]
                moved_durations_temp[i, j] = infection_duration[i, j]
    
    for i in range(1, rows_per_proc + 1):
        for j in range(N):
            if moved_cells_temp[i, j] == 0 and grid[i,j] != 0:
                pass

    incoming_top = np.zeros(N, dtype=np.int32)
    incoming_bottom = np.zeros(N, dtype=np.int32)
    incoming_top_duration = np.zeros(N, dtype=np.int32)
    incoming_bottom_duration = np.zeros(N, dtype=np.int32)

    reqs = []
    if rank < size - 1:
        reqs.append(comm.Isend(outgoing_bottom, dest=rank+1, tag=33))
        reqs.append(comm.Irecv(incoming_bottom, source=rank+1, tag=44))
        reqs.append(comm.Isend(outgoing_bottom_duration, dest=rank+1, tag=55))
        reqs.append(comm.Irecv(incoming_bottom_duration, source=rank+1, tag=66))
    
    if rank > 0:
        reqs.append(comm.Isend(outgoing_top, dest=rank-1, tag=44))
        reqs.append(comm.Irecv(incoming_top, source=rank-1, tag=33))
        reqs.append(comm.Isend(outgoing_top_duration, dest=rank-1, tag=66))
        reqs.append(comm.Irecv(incoming_top_duration, source=rank-1, tag=55))

    MPI.Request.Waitall(reqs)

    new_grid[1:rows_per_proc+1, :] = moved_cells_temp[1:rows_per_proc+1, :]
    new_infection_duration[1:rows_per_proc+1, :] = moved_durations_temp[1:rows_per_proc+1, :]

    for j in range(N):
        if incoming_top[j] == 1:
            new_grid[1, j] = 1
            new_infection_duration[1, j] = max(new_infection_duration[1, j], incoming_top_duration[j])
        if incoming_bottom[j] == 1:
            new_grid[rows_per_proc, j] = 1
            new_infection_duration[rows_per_proc, j] = max(new_infection_duration[rows_per_proc, j], incoming_bottom_duration[j])

    exchange_borders(grid)
    
    temp_grid_for_spread = new_grid.copy()
    
    for i in range(1, rows_per_proc + 1):
        for j in range(N):
            gi = start_row + i - 1
            
            if temp_grid_for_spread[i, j] == 0 and city_map[gi, j] != 2:
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + dx, j + dy
                    ngi = gi + dx

                    if (0 <= nj < N and 0 <= ni < rows_per_proc + 2 and 0 <= ngi < N and
                        city_map[ngi, nj] != 2):
                        
                        if grid[ni, nj] == 1:
                            if np.random.rand() < infection_prob:
                                new_grid[i, j] = 1
                                new_infection_duration[i, j] = 1
                                break

    temp_grid = grid[1:-1,:].copy()
    temp_duration = infection_duration[1:-1,:].copy()

    target_grid = np.zeros_like(temp_grid)
    target_duration = np.zeros_like(temp_duration)
    
    outgoing_agents_top = []
    outgoing_agents_bottom = []

    for i_local in range(rows_per_proc): 
        for j in range(N):
            i_padded = i_local + 1 
            gi = start_row + i_local 
            if grid[i_padded, j] == 1:
                current_speed = movement_speed_road if city_map[gi, j] == 0 else movement_speed_general
                possible_moves = []
                for dx in range(-current_speed, current_speed + 1):
                    for dy in range(-current_speed, current_speed + 1):
                        if dx == 0 and dy == 0: continue
                        ni = i_padded + dx 
                        nj = j + dy
                        ngi = gi + dx
                        if 0 <= nj < N and 0 <= ngi < N and city_map[ngi, nj] != 2:
                            possible_moves.append((ni, nj))
                if possible_moves:
                    next_i_padded, next_j = possible_moves[np.random.choice(len(possible_moves))]
                    if 1 <= next_i_padded <= rows_per_proc: 
                        if target_grid[next_i_padded-1, next_j] == 0:
                             target_grid[next_i_padded-1, next_j] = 1
                             target_duration[next_i_padded-1, next_j] = infection_duration[i_padded, j]
                        else: 
                             target_grid[next_i_padded-1, next_j] = 1
                             target_duration[next_i_padded-1, next_j] = max(target_duration[next_i_padded-1, next_j], infection_duration[i_padded, j])
                    elif next_i_padded == 0:
                        outgoing_agents_top.append((next_j, infection_duration[i_padded, j]))
                    elif next_i_padded == rows_per_proc + 1:
                        outgoing_agents_bottom.append((next_j, infection_duration[i_padded, j]))
                else: 
                    if target_grid[i_local, j] == 0:
                        target_grid[i_local, j] = 1
                        target_duration[i_local, j] = infection_duration[i_padded, j]
                    else: 
                        target_grid[i_local, j] = 1
                        target_duration[i_local, j] = max(target_duration[i_local, j], infection_duration[i_padded, j])

            else: 
                if target_grid[i_local, j] == 0:
                    target_grid[i_local, j] = grid[i_padded, j]
                    target_duration[i_local, j] = infection_duration[i_padded, j]


    send_top_indices = np.array([item[0] for item in outgoing_agents_top], dtype=np.int32)
    send_top_durations = np.array([item[1] for item in outgoing_agents_top], dtype=np.int32)
    send_bottom_indices = np.array([item[0] for item in outgoing_agents_bottom], dtype=np.int32)
    send_bottom_durations = np.array([item[1] for item in outgoing_agents_bottom], dtype=np.int32)

    send_counts = {
        'top_len': len(outgoing_agents_top),
        'bottom_len': len(outgoing_agents_bottom)
    }
    recv_counts = {
        'top_len': 0, 'bottom_len': 0,
        'neighbor_top_len': 0, 'neighbor_bottom_len': 0
    }
    
    reqs = []
    if rank > 0:
        reqs.append(comm.Isend(np.array([send_counts['top_len']]), dest=rank-1, tag=101))
        reqs.append(comm.Irecv(np.array([recv_counts['neighbor_top_len']]), source=rank-1, tag=102))
    if rank < size - 1:
        reqs.append(comm.Isend(np.array([send_counts['bottom_len']]), dest=rank+1, tag=102))
        reqs.append(comm.Irecv(np.array([recv_counts['neighbor_bottom_len']]), source=rank+1, tag=101))
    MPI.Request.Waitall(reqs)

    recv_top_indices = np.zeros(recv_counts['neighbor_top_len'], dtype=np.int32)
    recv_top_durations = np.zeros(recv_counts['neighbor_top_len'], dtype=np.int32)
    recv_bottom_indices = np.zeros(recv_counts['neighbor_bottom_len'], dtype=np.int32)
    recv_bottom_durations = np.zeros(recv_counts['neighbor_bottom_len'], dtype=np.int32)

    reqs = []
    if rank > 0:
        reqs.append(comm.Isend(send_top_indices, dest=rank-1, tag=201))
        reqs.append(comm.Irecv(recv_top_indices, source=rank-1, tag=202))
        reqs.append(comm.Isend(send_top_durations, dest=rank-1, tag=203))
        reqs.append(comm.Irecv(recv_top_durations, source=rank-1, tag=204))
    if rank < size - 1:
        reqs.append(comm.Isend(send_bottom_indices, dest=rank+1, tag=202))
        reqs.append(comm.Irecv(recv_bottom_indices, source=rank+1, tag=201))
        reqs.append(comm.Isend(send_bottom_durations, dest=rank+1, tag=204))
        reqs.append(comm.Irecv(recv_bottom_durations, source=rank+1, tag=203))
    MPI.Request.Waitall(reqs)

    for j, duration in zip(recv_top_indices, recv_top_durations):
        gi = start_row
        if city_map[gi, j] != 2:
            target_grid[0, j] = 1
            target_duration[0, j] = max(target_duration[0, j], duration)

    for j, duration in zip(recv_bottom_indices, recv_bottom_durations):
        gi = start_row + rows_per_proc -1
        if city_map[gi, j] != 2:
            target_grid[rows_per_proc - 1, j] = 1
            target_duration[rows_per_proc - 1, j] = max(target_duration[rows_per_proc - 1, j], duration)

    new_grid[1:-1, :] = target_grid.copy()
    new_infection_duration[1:-1, :] = target_duration.copy()


    exchange_borders(new_grid) 
    temp_grid_for_spread_check = new_grid.copy() 

    for i in range(1, rows_per_proc + 1):
        for j in range(N):
            gi = start_row + i - 1
            if new_grid[i, j] == 0 and city_map[gi, j] != 2:
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + dx, j + dy
                    ngi = gi + dx

                    if (0 <= nj < N and 0 <= ni < rows_per_proc + 2 and 0 <= ngi < N and
                        city_map[ngi, nj] != 2):
                        
                        if temp_grid_for_spread_check[ni, nj] == 1:
                            if np.random.rand() < infection_prob:
                                new_grid[i, j] = 1
                                new_infection_duration[i, j] = 1
                                break

    for i in range(1, rows_per_proc + 1):
        for j in range(N):
            if new_grid[i, j] == 1:
                new_infection_duration[i, j] += 1
                if new_infection_duration[i, j] >= recovery_time:
                    if np.random.rand() < death_prob:
                        new_grid[i, j] = 3
                    else:
                        new_grid[i, j] = 2
                    new_infection_duration[i, j] = 0

    return new_grid, new_infection_duration


for t in range(T):
    grid, infection_duration = step(grid, infection_duration, city_map)

    final_grid = None
    if rank == 0:
        final_grid = np.zeros((N, N), dtype=np.int32)
    comm.Gather(grid[1:-1, :], recvbuf=final_grid, root=0)

    if rank == 0:
        np.save(f"frame_{t:02d}_road.npy", final_grid.copy())
        if t == 0:
            np.save("city_map.npy", city_map)

local_infected = np.sum(grid[1:-1, :] == 1)
local_dead = np.sum(grid[1:-1, :] == 3)

total_infected = comm.reduce(local_infected, op=MPI.SUM, root=0)
total_dead = comm.reduce(local_dead, op=MPI.SUM, root=0)

if rank == 0:
    print(f"total infected: {total_infected}")
    print(f"total dead: {total_dead}")
    print(f"final_infection_map.npy (frame {T-1}) has been saved.")
start_time = time.time()

grid = np.zeros((rows_per_proc + 2, N), dtype=np.int32)
infection_duration = np.zeros_like(grid, dtype=np.int32)

for i_global, j in init_coords:
    if start_row <= i_global < end_row:
        local_i = i_global - start_row + 1
        if city_map[i_global, j] != 2:
            grid[local_i, j] = 1
            infection_duration[local_i, j] = 1

for t in range(T):
    grid, infection_duration = step(grid, infection_duration, city_map)
end_time = time.time()
elapsed = end_time - start_time
print(f"[Rank {rank}] 처리 시간: {elapsed:.4f}초")

total_time = comm.reduce(elapsed, op=MPI.SUM, root=0)
if rank == 0:
    print(f"총 처리 시간: {total_time:.4f}초")