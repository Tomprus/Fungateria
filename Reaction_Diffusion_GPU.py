import torch
import matplotlib.pyplot as plt

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
grid_size = 200  # Start with a smaller grid size
total_iterations = 1000

# Reaction-Diffusion parameters
pa = 0.6
pb = 0.8
pe = 4.5
d = 50
dt = 1e-1
threshold = 1
gamma = 625
ph = 1
amax = 40
smax = 80
width = 1.5

# Initialize activator, suppressor, and mycelium matrices as PyTorch tensors
u = torch.zeros((grid_size, grid_size), device=device)
v = torch.zeros((grid_size, grid_size), device=device)
c = torch.zeros((grid_size, grid_size), device=device)

u_new = torch.zeros_like(u)
v_new = torch.zeros_like(v)
c_new = torch.zeros_like(c)

# Initialize nutrient landscape with nutrient islands
n = torch.ones((grid_size, grid_size), device=device) * 0.05
num_islands = 100
island_radius = 3
high_nutrient_value = 0.9

# Randomly place nutrient islands
for _ in range(num_islands):
    center_x = torch.randint(island_radius, grid_size - island_radius, (1,)).item()
    center_y = torch.randint(island_radius, grid_size - island_radius, (1,)).item()

    for i in range(grid_size):
        for j in range(grid_size):
            if (i - center_x) ** 2 + (j - center_y) ** 2 <= island_radius ** 2:
                n[i, j] = high_nutrient_value

# Initialize rocks (obstacles)
num_rocks = 100
rock_radius = 3
for _ in range(num_rocks):
    rock_x = torch.randint(rock_radius, grid_size - rock_radius, (1,)).item()
    rock_y = torch.randint(rock_radius, grid_size - rock_radius, (1,)).item()

    for i in range(grid_size):
        for j in range(grid_size):
            if (i - rock_x) ** 2 + (j - rock_y) ** 2 <= rock_radius ** 2:
                n[i, j] = 0  # Set nutrient level to zero for rocks

# Initialize mycelium as a circle
radius = 5
center_x, center_y = grid_size // 2, grid_size // 2
for i in range(grid_size):
    for j in range(grid_size):
        distance = ((i - center_x) ** 2 + (j - center_y) ** 2) ** 0.5
        if distance <= radius:
            u[i, j] = 0.5 + torch.rand(1).item() / 100
            v[i, j] = 0.5 + torch.rand(1).item() / 100
            c[i, j] = 0.5  # Initialize mycelium state continuously

# Matrix to track active regions
ij_mat = torch.zeros((grid_size, grid_size), device=device)

def update_visualization(step, u, v, c, n):
    fig = plt.figure(figsize=(15, 15))

    # Activator u
    ax1 = fig.add_subplot(221)
    img1 = ax1.imshow(u.cpu().numpy(), cmap='jet', vmin=0, vmax=amax)
    ax1.set_title(f'Activator u at iteration {step}')
    fig.colorbar(img1, ax=ax1)

    # Suppressor v
    ax2 = fig.add_subplot(222)
    img2 = ax2.imshow(v.cpu().numpy(), cmap='jet', vmin=0, vmax=smax)
    ax2.set_title(f'Suppressor v at iteration {step}')
    fig.colorbar(img2, ax=ax2)

    # Mycelium c
    ax3 = fig.add_subplot(223)
    img3 = ax3.imshow(c.cpu().numpy(), cmap='jet', vmin=0, vmax=1)
    ax3.contour(c.cpu().numpy(), levels=10, colors='black', linewidths=0.5)
    ax3.set_title(f'Mycelium at iteration {step}')
    fig.colorbar(img3, ax=ax3)

    # Substrate n
    ax4 = fig.add_subplot(224)
    img4 = ax4.imshow(n.cpu().numpy(), cmap='jet')
    ax4.set_title('Substrate')
    fig.colorbar(img4, ax=ax4)

    plt.tight_layout()
    plt.show()

# Main simulation loop
base_consumption_rate = 0.002
for step in range(total_iterations + 1):
    # Reaction terms controlling chemical behavior
    f_uv = width * (pa * u + u ** 2 - pb * u * v) * n
    g_uv = pe * u ** 3 - v  # Enhanced suppressor response

    # Incorporate the radius of the plasmatic membrane
    ij_mat = torch.zeros_like(ij_mat)
    ij_mat[c > 0.5] = 1  # Fill ij_mat based on the mycelium matrix

    # Update activator and suppressor
    v_new = v + dt * (d * (0.05 * (torch.roll(v, 1, 0) + torch.roll(v, -1, 0) +
                                    torch.roll(v, 1, 1) + torch.roll(v, -1, 1)) - 0.2 * v) + gamma * g_uv)
    u_new = u + dt * (0.05 * (torch.roll(u, 1, 0) + torch.roll(u, -1, 0) +
                              torch.roll(u, 1, 1) + torch.roll(u, -1, 1)) - 0.2 * u + gamma * f_uv)

    # Apply threshold and adjust alpha (continuous growth)
    alpha = torch.where(u <= threshold, 0.49, 0.49 - 2.5 * (u - threshold))
    v_new[u > threshold] = 0

    # Update c with continuous values (allow gradual change)
    c_growth = gamma * ph * c * (alpha - c) * (c - 1)
    c_new = c + dt * c_growth
    c_new = torch.clamp(c_new, 0, 1)  # Ensure values stay within 0 to 1

    # Gradual growth where u exceeds threshold
    c_new = torch.where(u > threshold, c_new + 0.01 * u, c_new)

    # Prevent growth where nutrients are zero (rocks), but no reset to -1
    c_new[n == 0] = c_new[n == 0]

    # Limiters of activator and suppressor to avoid exponential growth
    u_new = torch.clamp(u_new, 0, amax)
    v_new = torch.clamp(v_new, 0, smax)

    # Proportional Nutrient Consumption: Mycelium consumes nutrients
    n = n - base_consumption_rate * c_new * n
    n = torch.clamp(n, min=0)  # Prevent negative nutrient values

    # Update variables
    u, v, c = u_new, v_new, c_new

    # Visualization every 100 steps
    if step % 100 == 0:
        update_visualization(step, u, v, c, n)

print("Simulation completed.")
