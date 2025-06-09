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
revolutions = 3
frame_number = 1000
rotation_speed = 2 * revolutions * np.pi / frame_number

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
ball_collision_cooldowns = []  # New: collision cooldown timers

circle_edge, = ax.plot([], [], 'b-')
hole_patch, = ax.plot([], [], 'k', lw=4)

def random_color():
    return (random.random(), random.random(), random.random())

ball_positions.append(np.array([0.0, 0.0]))
ball_velocities.append(np.array([1.5, 2.0]))
ball_colors.append(random_color())
ball_patches.append(ax.plot([], [], 'o', ms=8, color=ball_colors[-1])[0])
ball_escaped_flags.append(False)
ball_collision_cooldowns.append({})

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
    offset = 0.5  # Increased separation to reduce immediate collisions
    vel = np.array([1.5, 2.0])
    for i, dx in enumerate([-offset, offset]):
        # Add some random variation to initial positions and velocities
        random_y = np.random.uniform(-0.2, 0.2)
        random_vel_x = np.random.uniform(-0.5, 0.5)
        random_vel_y = np.random.uniform(-0.5, 0.5)
        
        ball_positions.append(np.array([dx, random_y]))
        ball_velocities.append(vel.copy() + np.array([random_vel_x, random_vel_y]))
        ball_colors.append(random_color())
        ball_patches.append(ax.plot([], [], 'o', ms=8, color=ball_colors[-1])[0])
        ball_escaped_flags.append(False)
        ball_collision_cooldowns.append({})

def handle_ball_collisions():
    """Boundary box collision with cooldown system to prevent sticking"""
    n = len(ball_positions)
    if n < 2:
        return
    
    # Update cooldown timers
    for i in range(n):
        cooldown_dict = ball_collision_cooldowns[i]
        keys_to_remove = []
        for other_ball, timer in cooldown_dict.items():
            if timer > 0:
                cooldown_dict[other_ball] = timer - 1
            else:
                keys_to_remove.append(other_ball)
        for key in keys_to_remove:
            del cooldown_dict[key]
    
    min_separation = 2 * ball_radius + 0.03  # Buffer to prevent immediate re-collision
    cooldown_frames = 10  # Prevent collision between same pair for 10 frames
    
    for i in range(n):
        for j in range(i + 1, n):
            # Check if these balls are in cooldown
            if j in ball_collision_cooldowns[i] or i in ball_collision_cooldowns[j]:
                continue
            
            pos_i = ball_positions[i]
            pos_j = ball_positions[j]
            
            # Calculate distance between centers
            diff = pos_j - pos_i
            distance = np.linalg.norm(diff)
            
            # Check if balls are overlapping
            if distance < min_separation and distance > 1e-10:
                # Calculate collision normal (from ball i to ball j)
                collision_normal = diff / distance
                
                # Calculate overlap and separate balls immediately
                overlap = min_separation - distance
                separation_distance = overlap / 2 + 0.02  # Extra buffer
                
                # Teleport balls apart along collision normal
                ball_positions[i] -= collision_normal * separation_distance
                ball_positions[j] += collision_normal * separation_distance
                
                # Now handle the physics properly
                vel_i = ball_velocities[i]
                vel_j = ball_velocities[j]
                
                # Project velocities onto collision normal
                vel_i_normal = np.dot(vel_i, collision_normal)
                vel_j_normal = np.dot(vel_j, collision_normal)
                
                # Only resolve if balls are moving towards each other
                relative_velocity = vel_i_normal - vel_j_normal
                if relative_velocity < 0:  # Moving towards each other
                    # Calculate tangential velocities (perpendicular to collision)
                    vel_i_tangent = vel_i - vel_i_normal * collision_normal
                    vel_j_tangent = vel_j - vel_j_normal * collision_normal
                    
                    # For elastic collision between equal masses, normal velocities are exchanged
                    # Add restitution factor for some energy loss
                    restitution = 0.85
                    
                    # New normal velocities after collision
                    new_vel_i_normal = vel_j_normal * restitution
                    new_vel_j_normal = vel_i_normal * restitution
                    
                    # Combine normal and tangential components
                    ball_velocities[i] = new_vel_i_normal * collision_normal + vel_i_tangent
                    ball_velocities[j] = new_vel_j_normal * collision_normal + vel_j_tangent
                    
                    # Add small random perturbation to prevent perfect stacking
                    perturbation = 0.05
                    random_angle = np.random.random() * 2 * np.pi
                    random_perturb = np.array([np.cos(random_angle), np.sin(random_angle)]) * perturbation
                    
                    ball_velocities[i] += random_perturb
                    ball_velocities[j] -= random_perturb
                    
                    # Set cooldown to prevent immediate re-collision
                    ball_collision_cooldowns[i][j] = cooldown_frames
                    ball_collision_cooldowns[j][i] = cooldown_frames  


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

    # Clear data for balls that will be removed to prevent visual artifacts
    for i in to_remove:
        ball_patches[i].set_data([], [])

    # Update positions for remaining balls
    for i in range(len(ball_positions)):
        if i not in to_remove:
            pos = ball_positions[i]
            ball_patches[i].set_data([pos[0]], [pos[1]])

    # Remove balls in reverse order to maintain correct indices
    for i in reversed(to_remove):
        ball_patches[i].remove()
        del ball_positions[i]
        del ball_velocities[i]
        del ball_escaped_flags[i]
        del ball_patches[i]
        del ball_colors[i]
        del ball_collision_cooldowns[i]  # Remove cooldown data
        
        # Update collision cooldown indices for remaining balls
        for k in range(len(ball_collision_cooldowns)):
            cooldown_dict = ball_collision_cooldowns[k]
            new_cooldown_dict = {}
            for other_ball, timer in cooldown_dict.items():
                if other_ball < i:
                    new_cooldown_dict[other_ball] = timer
                elif other_ball > i:
                    new_cooldown_dict[other_ball - 1] = timer
                # Skip other_ball == i as that ball is being removed
            ball_collision_cooldowns[k] = new_cooldown_dict
        
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


ani = FuncAnimation(fig, update, frames=frame_number, interval=20, blit=False)
plt.title("Circle Escape", color='white')
plt.show()