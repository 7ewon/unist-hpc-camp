import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

state_labels = ['Infected', 'Recovered', 'Dead']
colors = ['red', 'green', 'black']


T=20
N=100
infected, recovered, dead = [], [], []

# 저장된 frame_XX_road.npy 파일 읽기
for t in range(T):
    frame = np.load(f"frame_{t:02d}_road.npy")
    infected.append(np.sum(frame == 1))
    recovered.append(np.sum(frame == 2))
    dead.append(np.sum(frame == 3))

infected = np.array(infected)
recovered = np.array(recovered)
dead = np.array(dead)
all_infected = infected + recovered + dead

# 감염 상태 컬러맵: 0=미감염, 1=감염, 2=회복, 3=사망
state_cmap = ListedColormap(['white', 'red', 'green', 'black'])
state_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], state_cmap.N)

# 도시 배경 컬러맵: 0=일반, 1=도로, 2=건물
terrain_cmap = ListedColormap(['yellow', 'blue', 'red'])

# 도시 마스크 로드
city_map = np.load("city_map.npy")

# 전체 시각화 레이아웃 구성 (1행 4열)
fig, (ax_grid, ax_bar, ax_text, ax_line) = plt.subplots(1, 4, figsize=(22, 6))

# --- 막대그래프 초기화 ---
bars = ax_bar.bar(state_labels, [0, 0, 0], color=colors)
ax_bar.set_ylim(0, np.max(all_infected) * 1.3)
ax_bar.set_ylabel("People")
ax_bar.set_title("Current Status")

# --- 누적 감염자 텍스트 초기화 ---
ax_text.axis('off')
text_handle = ax_text.text(0.5, 0.5, '', fontsize=14, ha='center', va='center')

# --- 선 그래프 초기화 ---
ax_line.set_xlim(0, T)
ax_line.set_ylim(0, np.max(all_infected) * 1.3)
ax_line.set_xlabel("Time Step")
ax_line.set_ylabel("People")
ax_line.set_title("Status Transition Over Time")
ax_line.grid(True)

line_inf, = ax_line.plot([], [], color='red', label='Infected')
line_rec, = ax_line.plot([], [], color='green', label='Recovered')
line_dead, = ax_line.plot([], [], color='black', label='Dead')
ax_line.legend(loc='upper left')

# --- 지형 범례 ---
terrain_legend = [
    mpatches.Patch(color='yellow', label='general'),
    mpatches.Patch(color='blue', label='road'),
    mpatches.Patch(color='red', label='building'),
]

# 애니메이션 업데이트 함수
def update(frame):
    # --- 상태 격자 시각화 ---
    ax_grid.clear()
    data = np.load(f"frame_{frame:02d}_road.npy")
    ax_grid.imshow(city_map, cmap=terrain_cmap, alpha=0.5, origin='upper')
    ax_grid.imshow(data, cmap=state_cmap, norm=state_norm, alpha=0.8, origin='upper')
    ax_grid.set_title(f"Time Step {frame}")
    ax_grid.set_xlim(0, N)
    ax_grid.set_ylim(N, 0)
    ax_grid.set_aspect('equal')
    ax_grid.axis('off')
    ax_grid.legend(handles=terrain_legend, loc='upper right', fontsize='small')

    # --- 막대 그래프 업데이트 ---
    bars[0].set_height(infected[frame])
    bars[1].set_height(recovered[frame])
    bars[2].set_height(dead[frame])
    ax_bar.set_title(f"Time Step {frame}")

    # --- 누적 감염자 텍스트 ---
    text_handle.set_text(f"Cumulative Infected:\n{all_infected[frame]}")

    # --- 선 그래프 업데이트 ---
    x = np.arange(frame + 1)
    line_inf.set_data(x, infected[:frame + 1])
    line_rec.set_data(x, recovered[:frame + 1])
    line_dead.set_data(x, dead[:frame + 1])

    return bars + (text_handle, line_inf, line_rec, line_dead)

# 애니메이션 실행
ani = animation.FuncAnimation(fig, update, frames=T, interval=500, blit=False)

plt.tight_layout()
plt.show()