# %%
import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch.nn.functional as F

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# %%
def export_tensor_to_csv_numpy(tensor, filename):
    array = tensor.detach().cpu().numpy()

    # Ensure the array is of floating-point type
    if not np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float64)
    
    # Save with double precision and '.' as decimal delimiter
    np.savetxt(filename, array, delimiter=',', fmt='%.15f')
    print(f"Exported tensor to {filename} using NumPy with double precision.")

# %%
def load_and_process_image(image_path, threshold=0.5):
    # Step 1: Load the RGB image
    rgb_image = mpimg.imread(image_path)
    
    # Step 2: Use only the first channel for simplicity
    first_channel = rgb_image[:, :, 0]  # Extract the first channel
    skeleton = np.where(first_channel < threshold, 1, -1)  # Create bipolar image directly
    
    # Step 3: Extract dimensions
    dimensions = skeleton.shape
    print(f"Dimensions of the bipolar image: {dimensions}")
    
    return skeleton

# %%
def initialize_simulation(skeleton, device, grid_size, substrate_val=0.3):
    # Initialize activator, suppressor, and mycelium matrices as PyTorch tensors
    u = torch.zeros((grid_size, grid_size), device=device)
    v = torch.zeros((grid_size, grid_size), device=device)
    c = torch.zeros((grid_size, grid_size), device=device)
    
    u_new = torch.zeros_like(u)
    v_new = torch.zeros_like(v)
    c_new = torch.zeros_like(c)
    
    # Initialize nutrient landscape
    mid = grid_size // 2
    
    n = np.linspace(substrate_val, 1, grid_size)
    n = np.tile(n, (grid_size, 1))
    
    # Step 2: Apply the skeleton mask
    n[skeleton < 0] = -1
    print(f"Mid index: {mid}")
    print(f"n shape: {n.shape}")
    print(f"u shape: {u.shape}")
    print(f"v shape: {v.shape}")
    print(f"c shape: {c.shape}")
    
    # plt.imshow(n, cmap='viridis')  # Commented out to disable plotting
    # plt.colorbar()
    # plt.title("Modified Substrate Matrix with Skeleton Applied")
    # plt.show()
    
    n = torch.tensor(n, dtype=torch.float32, device=device)
    
    # Initialize matrices
    u = torch.zeros((grid_size, grid_size), device=device)
    v = torch.zeros((grid_size, grid_size), device=device)
    c = torch.zeros((grid_size, grid_size), device=device)
    
    # Initial activator and suppressor states
    for k in range(-6, 5):  # This ranges from -6 to 4, inclusive
        size = 2 * abs(k) + 1  # This computes the size of the square
        start_idx = mid + k if k < 0 else mid - k
        if start_idx + size <= grid_size:  # Check to ensure indices are within bounds
            random_tensor = torch.rand((size, size), device=device) * 0.5 
            u[start_idx:start_idx+size, start_idx:start_idx+size] = 0.5 + random_tensor / (0.5 * n[start_idx:start_idx+size, start_idx:start_idx+size])
            v[start_idx:start_idx+size, start_idx:start_idx+size] = 0.1 + random_tensor / (0.5 * n[start_idx:start_idx+size, start_idx:start_idx+size])
            c[start_idx:start_idx+size, start_idx:start_idx+size] = 1
    
    return u, v, c, n

# %%
def update_visualization(step, u, v, c, n, amax=100, smax=35):
    # Commented out to disable plotting
    """
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # Wider figure to accommodate colorbars
    
    # Define limits for zooming into the center
    mid = u.shape[0] // 2
    x_limits = (mid-11, mid+10)
    y_limits = (mid-11, mid+10)
    
    # Plot u matrix
    im_u = axs[0].imshow(u.cpu().numpy(), cmap='viridis')
    axs[0].set_title('Activator Matrix (u)')
    axs[0].axis('off')
    axs[0].set_xlim(x_limits)
    axs[0].set_ylim(y_limits)
    fig.colorbar(im_u, ax=axs[0], fraction=0.046, pad=0.04)  # Add colorbar to the plot of u
    
    # Plot v matrix
    im_v = axs[1].imshow(v.cpu().numpy(), cmap='viridis')
    axs[1].set_title('Suppressor Matrix (v)')
    axs[1].axis('off')
    axs[1].set_xlim(x_limits)
    axs[1].set_ylim(y_limits)
    fig.colorbar(im_v, ax=axs[1], fraction=0.046, pad=0.04)  # Add colorbar to the plot of v
    
    # Plot c matrix
    im_c = axs[2].imshow(c.cpu().numpy(), cmap='viridis')
    axs[2].set_title('Substrate Matrix (c)')
    axs[2].axis('off')
    axs[2].set_xlim(x_limits)
    axs[2].set_ylim(y_limits)
    fig.colorbar(im_c, ax=axs[2], fraction=0.046, pad=0.04)  # Add colorbar to the plot of c
    
    plt.tight_layout()
    plt.show()
    """
    pass  # No operation since plotting is disabled

# %%
def initialize_laplacian(grid_size, lap_side=0.35, lap_diag=0.1, lap=1/9, depth=2):
    # Define the 3x3 Laplacian kernel L
    L = torch.tensor([
        [lap_diag, lap_side, lap_diag],
        [lap_side, -lap,     lap_side],
        [lap_diag, lap_side, lap_diag]
    ], dtype=torch.float32, device=device)
    
    print("Laplacian Kernel L:")
    print(L)
    
    return L

# %%
def conv2_same(input_tensor, kernel):
    """
    Performs a 2D convolution with 'same' padding.
    
    Args:
        input_tensor (torch.Tensor): Input tensor of shape [H, W].
        kernel (torch.Tensor): Convolution kernel of shape [kH, kW] or [1, 1, kH, kW].
    
    Returns:
        torch.Tensor: Convolved tensor of shape [H, W].
    """
    # Ensure the kernel has 4 dimensions
    if kernel.dim() == 2:
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, kH, kW]
    elif kernel.dim() == 4:
        pass  # Already in the correct shape
    else:
        raise ValueError("Kernel must be a 2D or 4D tensor.")
    
    # Extract kernel dimensions
    _, _, kH, kW = kernel.shape
    
    # Calculate padding for 'same' convolution
    pad_h = (kH - 1) // 2
    pad_w = (kW - 1) // 2
    
    # Reshape input to [N, C, H, W]
    input_reshaped = input_tensor.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]
    
    # Perform convolution with calculated padding
    conv_result = F.conv2d(input_reshaped, kernel, padding=(pad_h, pad_w))
    
    # Reshape back to [H, W]
    return conv_result.squeeze(0).squeeze(0)

# %%
def run_simulation(skeleton, output_dir, device):
    # Step 3: Extract dimensions
    grid_size = skeleton.shape[0]
    print(f"Grid size: {grid_size}")
    
    # Hyperparameters
    total_iterations = 500
    
    # Reaction-Diffusion parameters
    pa = 0.5
    pb = 0.8
    pc = 0.16
    pe = 2.6
    d = 30
    dt = 1e-1
    threshold = 0.5
    pk = 0.05
    gamma = 625
    ph = 1
    alpha_init = 1
    amax = 100
    smax = 35
    substrate_val = 0.3
    
    # Initialize simulation
    u, v, c, n = initialize_simulation(skeleton, device, grid_size, substrate_val)
    
    # Define Laplacian Kernel
    L = initialize_laplacian(grid_size)
    
    # Laplacian (Diffusion) Kernel Weights
    lap_side = 0.35
    lap_diag = 0.1
    lap = 1/9
    
    # Depth (Size) of Kernel
    depth = 2
    
    # Plasma Membrane of Mycelium
    ij_mat = torch.zeros((grid_size, grid_size), device=device)
    
    # Pre-calculate the Laplacian kernel indices and weights if they remain constant
    lap_kernel = torch.zeros((2*depth+1, 2*depth+1), device=device)
    for dx in range(-depth, depth + 1):
        for dy in range(-depth, depth + 1):
            if dx == 0 and dy == 0:
                lap_kernel[depth, depth] = lap
            elif abs(dx) == abs(dy):
                lap_kernel[depth + dx, depth + dy] = lap_diag
            else:
                lap_kernel[depth + dx, depth + dy] = lap_side
    
    print("Laplacian Kernel:")
    print(lap_kernel)
    print(f"Laplacian Kernel shape: {lap_kernel.shape}")
    
    repeat_factor = grid_size // 5
    D = lap_kernel.repeat(repeat_factor, repeat_factor)
    
    print(f"D shape: {D.shape}")  # Should be (grid_size, grid_size)
    
    # Generate a random integer between 1 and 10, and divide by 10000
    noise_u = torch.randint(1, 11, u.size(), device=device).float() / 10000
    noise_c = torch.randint(1, 1001, c.size(), device=device).float() / 10000
    
    ones_kernel = torch.ones((1, 1, lap_kernel.shape[0], lap_kernel.shape[1]), dtype=torch.float32, device=device)
    
    # Main simulation loop
    for step in range(total_iterations + 1):
        # Reaction Terms
        f_uv = (pa * u + u ** 2 - pb * u * v) * n
        g_uv = pe * u ** 3 - v

        # Calculate ij_mat - the plasma membrane in which the mycelium can expand each iteration
        c_positive = (c > 0).float()

        input_tensor = c_positive.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, gridSize, gridSize]

        conv_result = F.conv2d(input_tensor, ones_kernel, padding=depth)
        conv_result = conv_result.squeeze(0).squeeze(0)  # Shape: [gridSize, gridSize]
        
        conv_min = torch.minimum(conv_result, torch.ones_like(conv_result))

        # ij_mat = conv_min * D
        ij_mat = torch.mul(conv_min, D)

        # Compute v_new and u_new
        conv_v = conv2_same(v, L)  
        term_v = d * conv_v + gamma * g_uv 
        update_v = ij_mat * term_v 
        v_new = v + dt * update_v

        conv_u = conv2_same(u, L)
        term_u = conv_u + gamma * f_uv 
        update_u = ij_mat * term_u
        u_new = u + dt * update_u 

        # Apply threshold and adjust alpha
        u_new = torch.where(n == -1, u - noise_u, u_new)
        alpha = torch.where(u <= threshold, 0.49, 0.49 - 2.5 * (u - threshold))
        v_new = torch.where(u <= threshold, v_new, torch.zeros_like(v_new))

        # Update c using alpha and apply limits
        c_new = c + dt * gamma * ph * c * (alpha - c) * (c - 1)
        c_new = torch.where(alpha < 0, c + noise_c, c_new)

        # Limiters of activator and suppressor to avoid exponential growth
        u_new = torch.clamp(u_new, min=0, max=amax)
        v_new = torch.where(v_new > smax, torch.full_like(v_new, smax), v_new)
        c_new = torch.where(c_new > 1, torch.ones_like(c_new), c_new)
        c_new = torch.where(c_new < 0, c + noise_c, c_new)
        c_new = torch.where(c_new == 1, c - noise_c, c_new)
        c_new = torch.where(c_new > 1, torch.ones_like(c_new), c_new)
        
        # Update variables
        u, v, c = u_new, v_new, c_new

        # Visualization every 100 steps
        if step % 100 == 0:
            # Define filenames with step number
            filename_c = os.path.join(output_dir, f'c_step_{step}.csv')
            filename_u = os.path.join(output_dir, f'u_step_{step}.csv')
            filename_v = os.path.join(output_dir, f'v_step_{step}.csv')
            filename_n = os.path.join(output_dir, f'n_step_{step}.csv')
            
            # Export tensors to CSV using NumPy
            export_tensor_to_csv_numpy(c, filename_c)
            export_tensor_to_csv_numpy(u, filename_u)
            export_tensor_to_csv_numpy(v, filename_v)
            export_tensor_to_csv_numpy(n, filename_n)

            # update_visualization(step, u, v, c, n)  # Commented out to disable plotting

    print("Simulation completed.")

    # Optionally, export final tensors
    filename_final_c = os.path.join(output_dir, f'c_final.csv')
    filename_final_u = os.path.join(output_dir, f'u_final.csv')
    filename_final_v = os.path.join(output_dir, f'v_final.csv')
    filename_final_n = os.path.join(output_dir, f'n_final.csv')
    
    export_tensor_to_csv_numpy(c, filename_final_c)
    export_tensor_to_csv_numpy(u, filename_final_u)
    export_tensor_to_csv_numpy(v, filename_final_v)
    export_tensor_to_csv_numpy(n, filename_final_n)

    # Final statistics
    print(f"Mean of u: {torch.mean(u).item()}")
    print(f"Mean of v: {torch.mean(v).item()}")
    print(f"Max of c: {torch.max(c).item()}")

# %%
def main():
    # Directory containing the skeleton images
    image_dir = "D:\\Fungateria\\Code\\Mycelium_Ver_2\\Fitting\\"  # Adjust as needed
    
    # Base output directory
    output_base_dir = 'D:/Fungateria/Github/Fungateria/Mycelium_Model/Validation/Python'
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Find all png and jpg files in the image directory
    image_patterns = ["*.png", "*.jpg", "*.jpeg"]
    image_files = []
    for pattern in image_patterns:
        image_files.extend(glob.glob(os.path.join(image_dir, pattern)))
    
    if not image_files:
        print("No image files found in the specified directory.")
        return
    
    print(f"Found {len(image_files)} image(s) to process.")
    
    for image_path in image_files:
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        print(f"\nProcessing image: {image_name}")
        
        # Load and process the image into a skeleton
        skeleton = load_and_process_image(image_path, threshold=0.5)
        
        # Define output directory for this skeleton
        skeleton_output_dir = os.path.join(output_base_dir, image_name)
        os.makedirs(skeleton_output_dir, exist_ok=True)
        
        # Run the simulation for this skeleton
        run_simulation(skeleton, skeleton_output_dir, device)

# %%
if __name__ == "__main__":
    main()

