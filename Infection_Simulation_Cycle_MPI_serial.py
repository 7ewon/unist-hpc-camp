import numpy as np
import time

# 파라미터 설정
T = 20
N = 1000
infection_prob = 0.3
recovery_time = 5
death_prob = 0.1
num_initial_infected = 200

# 배열 초기화
grid = np.zeros((N, N), dtype=np.int32)
infection_duration = np.zeros((N, N), dtype=np.int32)

# 도시 지도 생성
city_map = np.zeros((N, N), dtype=np.int32)

# 수평 도로
city_map[20:25, :] = 1

# 수직 도로
city_map[:, 45:48] = 1

# 곡선 도로
for i in range(N):
    j = int(N//-3 + (N//6) * np.sin(i /30))
    city_map[i, j] = 1
    city_map[i, j+1] = 1
    city_map[i, j+2] = 1

# 건물
city_map[40:50,10:20] = 2
city_map[25:45,60:70] = 2
city_map[70:80,60:70] = 2


# 도시 저장
np.save("city_map.npy", city_map)

# 초기 감염자 무작위 배치
init_coords = [(np.random.randint(0, N), np.random.randint(0, N)) for _ in range(20)]
for i, j in init_coords:
    if city_map[i, j] != 2:
        grid[i, j] = 1

# 시뮬레이션 단계 함수
def step(grid, infection_duration, city_map):
    new = grid.copy()
    duration = infection_duration.copy()
    for i in range(N):
        for j in range(N):
            # 건강인
            if grid[i, j] == 0:
                for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < N and 0 <= nj < N:
                        if grid[ni, nj] == 1 and city_map[i, j] == 1:
                            if np.random.rand() < infection_prob:
                                new[i, j] = 1
                                duration[i, j] = 1
                            break
            # 감염자
            elif grid[i, j] == 1:
                duration[i, j] += 1
                if duration[i, j] >= recovery_time:
                    new[i, j] = 3 if np.random.rand() < death_prob else 2
                    duration[i, j] = 0

    return new, duration

# 시뮬레이션 실행
for t in range(T):
    grid, infection_duration = step(grid, infection_duration, city_map)
    np.save(f"frame_{t:02d}_road.npy", grid.copy())

# 감염/사망 집계
total_infected = np.sum(grid == 1)
total_dead = np.sum(grid == 3)
print(f"Total infected: {total_infected}")
print(f"Total dead: {total_dead}")
np.save("final_infection_map.npy", grid)

# 성능 측정 루프
start_time = time.time()
infection_duration = np.zeros((N, N), dtype=np.int32)
for t in range(T):
    grid, infection_duration = step(grid, infection_duration, city_map)
end_time = time.time()
print(f"총 처리 시간: {end_time - start_time:.4f}초")