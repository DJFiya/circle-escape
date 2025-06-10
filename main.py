import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import random

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
trail_length = 25  # length of comet trail

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
ball_positions = []
ball_velocities = []
ball_patches = []
ball_colors = []
ball_escaped_flags = []
ball_collision_cooldowns = []
ball_trails = []
ball_trail_collections = []

circle_edge, = ax.plot([], [], 'b-')
hole_patch, = ax.plot([], [], 'k', lw=4)

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
    ball_collision_cooldowns.append({})
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
    n = len(ball_positions)
    if n < 2:
        return
    for i in range(n):
        cooldown_dict = ball_collision_cooldowns[i]
        keys_to_remove = [k for k, v in cooldown_dict.items() if v <= 0]
        for k in keys_to_remove:
            del cooldown_dict[k]
        for k in cooldown_dict:
            cooldown_dict[k] -= 1

    min_separation = 2 * ball_radius + 0.03
    cooldown_frames = 10

    for i in range(n):
        for j in range(i + 1, n):
            if j in ball_collision_cooldowns[i] or i in ball_collision_cooldowns[j]:
                continue
            pos_i, pos_j = ball_positions[i], ball_positions[j]
            diff = pos_j - pos_i
            distance = np.linalg.norm(diff)
            if distance < min_separation and distance > 1e-10:
                normal = diff / distance
                overlap = min_separation - distance
                separation = overlap / 2 + 0.02
                ball_positions[i] -= normal * separation
                ball_positions[j] += normal * separation

                vi, vj = ball_velocities[i], ball_velocities[j]
                vi_n, vj_n = np.dot(vi, normal), np.dot(vj, normal)
                rel_velocity = vi_n - vj_n
                if rel_velocity < 0:
                    vi_t = vi - vi_n * normal
                    vj_t = vj - vj_n * normal
                    restitution = 0.85
                    new_vi_n = vj_n * restitution
                    new_vj_n = vi_n * restitution
                    ball_velocities[i] = new_vi_n * normal + vi_t
                    ball_velocities[j] = new_vj_n * normal + vj_t

                    perturb = 0.05 * np.random.randn(2)
                    ball_velocities[i] += perturb
                    ball_velocities[j] -= perturb

                    ball_collision_cooldowns[i][j] = cooldown_frames
                    ball_collision_cooldowns[j][i] = cooldown_frames

# Initialize first ball
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

        distance = np.linalg.norm(pos)
        escaped = ball_escaped_flags[i]
        if not escaped and distance + ball_radius >= radius:
            edge_point = pos * ((radius - ball_radius) / distance)
            if is_in_hole(edge_point, theta):
                ball_escaped_flags[i] = True
            else:
                normal = pos / distance
                vel_n = np.dot(vel, normal)
                vel -= 2 * vel_n * normal
                pos = normal * (radius - ball_radius - 0.001)

        if abs(pos[0]) > limit or abs(pos[1]) > limit:
            to_remove.append(i)

        ball_positions[i], ball_velocities[i] = pos, vel

        ball_trails[i].append(pos.copy())
        if len(ball_trails[i]) > trail_length:
            ball_trails[i].pop(0)

        trail = np.array(ball_trails[i])
        if len(trail) >= 2:
            segments = [[trail[j], trail[j+1]] for j in range(len(trail)-1)]
            widths = np.linspace(3.5, 0.5, len(segments))
            alphas = np.linspace(0.1, 0.7, len(segments))
            rgba = [(*ball_colors[i], alpha) for alpha in alphas]
            ball_trail_collections[i].set_segments(segments)
            ball_trail_collections[i].set_linewidths(widths)
            ball_trail_collections[i].set_colors(rgba)
        else:
            ball_trail_collections[i].set_segments([])

    for i in reversed(to_remove):
        ball_patches[i].remove()
        ball_trail_collections[i].remove()
        del ball_positions[i]
        del ball_velocities[i]
        del ball_patches[i]
        del ball_colors[i]
        del ball_escaped_flags[i]
        del ball_collision_cooldowns[i]
        del ball_trails[i]
        del ball_trail_collections[i]
        for cooldown in ball_collision_cooldowns:
            keys = list(cooldown.keys())
            for key in keys:
                if key == i:
                    del cooldown[key]
                elif key > i:
                    cooldown[key - 1] = cooldown.pop(key)
        spawn_two_balls()

    for i, pos in enumerate(ball_positions):
        ball_patches[i].set_data([pos[0]], [pos[1]])

    angles = np.linspace(0, 2*np.pi, 200)
    circle_edge.set_data(*rotate_point(np.array([radius*np.cos(angles), radius*np.sin(angles)]), theta))
    hole_angles = np.linspace(hole_start, hole_start + hole_angle, 10)
    hole_coords = rotate_point(np.array([radius*np.cos(hole_angles), radius*np.sin(hole_angles)]), theta)
    hole_patch.set_data(*hole_coords)

    return [circle_edge, hole_patch] + ball_patches + ball_trail_collections

ani = FuncAnimation(fig, update, frames=frame_number, interval=20, blit=False)
plt.title("Circle Escape", color='white')
plt.show()
