import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import random
from collections import defaultdict

# Simulation parameters
radius = 5.0
ball_radius = 0.2
hole_angle = np.pi / 6
hole_start = np.pi / 4
gravity = 0.05
dt = 0.05
revolutions = 3
frame_number = 1000
rotation_speed = 2 * revolutions * np.pi / frame_number
trail_length = 25
cell_size = 2 * ball_radius + 0.1

# Figure setup
fig, ax = plt.subplots()
fig.patch.set_facecolor('black')
ax.set_facecolor('black')
ax.set_aspect('equal')
ax.set_xlim(-radius - 1, radius + 1)
ax.set_ylim(-radius - 1, radius + 1)
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

# Objects
ball_positions, ball_velocities, ball_colors = [], [], []
ball_patches, ball_trails, ball_trail_collections = [], [], []
ball_escaped_flags = []

circle_edge, = ax.plot([], [], 'b-')
hole_patch, = ax.plot([], [], 'k', lw=4)
ball_count_text = ax.text(0, radius + 0.5, '', ha='center', va='bottom', fontsize=12, color='white', weight='bold')

def random_color():
    return (random.random(), random.random(), random.random())

def rotate_point(p, theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.dot(np.array([[c, -s], [s, c]]), p)

def is_in_hole(pos, angle):
    ball_angle = np.arctan2(pos[1], pos[0]) - angle
    ball_angle = (ball_angle + 2*np.pi) % (2*np.pi)
    hole_start_norm = (hole_start + 2*np.pi) % (2*np.pi)
    hole_end = (hole_start + hole_angle) % (2*np.pi)
    if hole_start_norm < hole_end:
        return hole_start_norm <= ball_angle <= hole_end
    else:
        return ball_angle >= hole_start_norm or ball_angle <= hole_end

def add_ball(pos, vel):
    ball_positions.append(np.array(pos))
    ball_velocities.append(np.array(vel))
    color = random_color()
    ball_colors.append(color)
    ball_patches.append(ax.plot([], [], 'o', ms=8, color=color)[0])
    ball_escaped_flags.append(False)
    ball_trails.append([])

    trail_collection = LineCollection([], linewidths=[], colors=[], alpha=1.0)
    ax.add_collection(trail_collection)
    ball_trail_collections.append(trail_collection)

def spawn_two_balls():
    offset = 0.5
    vel = np.array([1.5, 2.0])
    for dx in [-offset, offset]:
        random_y = np.random.uniform(-0.2, 0.2)
        random_vel = vel + np.random.uniform(-0.5, 0.5, size=2)
        add_ball([dx, random_y], random_vel)

def handle_ball_collisions():
    grid = defaultdict(list)
    for idx, pos in enumerate(ball_positions):
        cell = tuple((pos // cell_size).astype(int))
        grid[cell].append(idx)

    checked_pairs = set()
    for cell, indices in grid.items():
        neighbors = [(i, j) for dx in [-1, 0, 1] for dy in [-1, 0, 1]
                     for i in grid.get((cell[0] + dx, cell[1] + dy), [])
                     for j in indices if i < j]
        for i, j in neighbors:
            if (i, j) in checked_pairs:
                continue
            checked_pairs.add((i, j))

            pos_i, pos_j = ball_positions[i], ball_positions[j]
            diff = pos_j - pos_i
            dist = np.linalg.norm(diff)
            if dist < 2 * ball_radius and dist > 1e-10:
                normal = diff / dist
                overlap = 2 * ball_radius - dist
                correction = normal * overlap / 2
                ball_positions[i] -= correction
                ball_positions[j] += correction

                vi, vj = ball_velocities[i], ball_velocities[j]
                vi_n, vj_n = np.dot(vi, normal), np.dot(vj, normal)
                if vi_n - vj_n < 0:
                    restitution = 0.85
                    vi_t = vi - vi_n * normal
                    vj_t = vj - vj_n * normal
                    ball_velocities[i] = (vj_n * restitution) * normal + vi_t
                    ball_velocities[j] = (vi_n * restitution) * normal + vj_t

# Initialize
add_ball([0.0, 0.0], [1.5, 2.0])

def update(frame):
    theta = rotation_speed * frame
    handle_ball_collisions()
    to_remove = []
    limit = radius + 1

    for i in range(len(ball_positions)):
        pos, vel = ball_positions[i], ball_velocities[i]
        vel[1] -= gravity
        pos += vel * dt

        dist = np.linalg.norm(pos)
        if not ball_escaped_flags[i] and dist + ball_radius >= radius:
            edge_point = pos * ((radius - ball_radius) / dist)
            if is_in_hole(edge_point, theta):
                ball_escaped_flags[i] = True
            else:
                normal = pos / dist
                vel -= 2 * np.dot(vel, normal) * normal
                pos = normal * (radius - ball_radius - 0.001)

        if abs(pos[0]) > limit or abs(pos[1]) > limit:
            to_remove.append(i)

        ball_positions[i] = pos
        ball_velocities[i] = vel

        # Trail update
        ball_trails[i].append(pos.copy())
        if len(ball_trails[i]) > trail_length:
            ball_trails[i].pop(0)
        trail = np.array(ball_trails[i])
        if len(trail) >= 2:
            segments = [[trail[j], trail[j+1]] for j in range(len(trail)-1)]
            widths = np.linspace(3.5, 0.5, len(segments))
            alphas = np.linspace(0.1, 0.7, len(segments))
            rgba = [(*ball_colors[i], a) for a in alphas]
            ball_trail_collections[i].set_segments(segments)
            ball_trail_collections[i].set_linewidths(widths)
            ball_trail_collections[i].set_colors(rgba)
        else:
            ball_trail_collections[i].set_segments([])

    # Remove dead balls
    for i in reversed(to_remove):
        ball_patches[i].remove()
        ball_trail_collections[i].remove()
        del ball_positions[i]
        del ball_velocities[i]
        del ball_colors[i]
        del ball_patches[i]
        del ball_escaped_flags[i]
        del ball_trails[i]
        del ball_trail_collections[i]
        spawn_two_balls()

    # Draw everything
    for i, pos in enumerate(ball_positions):
        ball_patches[i].set_data([pos[0]], [pos[1]])

    angles = np.linspace(0, 2*np.pi, 200)
    circle_edge.set_data(*rotate_point(np.array([radius*np.cos(angles), radius*np.sin(angles)]), theta))
    hole_angles = np.linspace(hole_start, hole_start + hole_angle, 10)
    hole_coords = rotate_point(np.array([radius*np.cos(hole_angles), radius*np.sin(hole_angles)]), theta)
    hole_patch.set_data(*hole_coords)

    ball_count_text.set_text(f"Balls: {len(ball_positions)}")

    return [circle_edge, hole_patch, ball_count_text] + ball_patches + ball_trail_collections

ani = FuncAnimation(fig, update, frames=frame_number, interval=20, blit=False)
plt.title("Circle Escape", color='white')
plt.show()
