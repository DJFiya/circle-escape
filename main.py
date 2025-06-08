import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

# Parameters
radius = 5.0
ball_radius = 0.2
hole_angle = np.pi / 6  
hole_start = np.pi / 4  
gravity = 0.05
dt = 0.05
rotation_speed = 0.02  

fig, ax = plt.subplots()
fig.patch.set_facecolor('black')
ax.set_facecolor('black')

ax.set_aspect('equal')
ax.set_xlim(-radius - 1, radius + 1)
ax.set_ylim(-radius - 1, radius + 1)

ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

ball_patches = []
ball_positions = []
ball_velocities = []
ball_escaped_flags = []
ball_colors = []

circle_edge, = ax.plot([], [], 'b-')
hole_patch, = ax.plot([], [], 'k', lw=4)

def random_color():
    return (random.random(), random.random(), random.random())

ball_positions.append(np.array([0.0, 0.0]))
ball_velocities.append(np.array([1.5, 2.0]))
ball_colors.append(random_color())
ball_patches.append(ax.plot([], [], 'o', ms=8, color=ball_colors[-1])[0])
ball_escaped_flags.append(False)

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

def spawn_two_balls():
    offset = 0.3
    vel = np.array([1.5, 2.0])
    for dx in [-offset, offset]:
        ball_positions.append(np.array([dx, 0.0]))
        ball_velocities.append(vel.copy())
        ball_colors.append(random_color())
        ball_patches.append(ax.plot([], [], 'o', ms=8, color=ball_colors[-1])[0])
        ball_escaped_flags.append(False)

def handle_ball_collisions():
    n = len(ball_positions)
    for _ in range(3):  
        for i in range(n):
            for j in range(i + 1, n):
                pos_i = ball_positions[i]
                pos_j = ball_positions[j]
                diff = pos_j - pos_i
                dist = np.linalg.norm(diff)
                min_dist = 2 * ball_radius
                
                if dist < min_dist and dist > 1e-8:
                    normal = diff / dist
                    
                    rel_vel = ball_velocities[i] - ball_velocities[j]
                    vel_along_normal = np.dot(rel_vel, normal)
                    
                    if vel_along_normal < 0: 
                        impulse = -vel_along_normal
                        ball_velocities[i] += impulse * normal
                        ball_velocities[j] -= impulse * normal
                        
                    overlap = min_dist - dist
                    correction = overlap * normal
                    ball_positions[i] -= correction  
                    ball_positions[j] += correction  


def update(frame):
    theta = rotation_speed * frame

    handle_ball_collisions()

    to_remove = []
    x_limit = radius + 1
    y_limit = radius + 1

    for i in range(len(ball_positions)):
        pos = ball_positions[i]
        vel = ball_velocities[i]
        escaped = ball_escaped_flags[i]

        vel[1] -= gravity
        pos += vel * dt

        distance_from_center = np.linalg.norm(pos)

        if not escaped:
            if distance_from_center + ball_radius >= radius:
                edge_point = pos * ((radius - ball_radius) / distance_from_center)
                if is_in_hole(edge_point, theta):
                    ball_escaped_flags[i] = True
                else:
                    normal = pos / distance_from_center
                    vel_parallel = normal * np.dot(vel, normal)
                    vel_perpendicular = vel - vel_parallel
                    vel = vel_perpendicular - vel_parallel
                    vel *= 1.0
                    pos = normal * (radius - ball_radius - 0.001)

        if (pos[0] < -x_limit or pos[0] > x_limit or
            pos[1] < -y_limit or pos[1] > y_limit):
            to_remove.append(i)

        ball_positions[i] = pos
        ball_velocities[i] = vel
        ball_patches[i].set_data([pos[0]], [pos[1]])

    for i in reversed(to_remove):
        ball_patches[i].remove()
        del ball_positions[i]
        del ball_velocities[i]
        del ball_escaped_flags[i]
        del ball_patches[i]
        del ball_colors[i]
        spawn_two_balls()

    angles = np.linspace(0, 2*np.pi, 200)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    xy = np.vstack((x, y))
    rot_xy = rotate_point(xy, theta)
    circle_edge.set_data(rot_xy[0], rot_xy[1])

    hole_angles = np.linspace(hole_start, hole_start + hole_angle, 10)
    hole_x = radius * np.cos(hole_angles)
    hole_y = radius * np.sin(hole_angles)
    hole_xy = rotate_point(np.vstack((hole_x, hole_y)), theta)
    hole_patch.set_data(hole_xy[0], hole_xy[1])

    return [circle_edge, hole_patch] + ball_patches


ani = FuncAnimation(fig, update, frames=1000, interval=20, blit=True)
plt.title("Circle Escape", color='white')
plt.show()
