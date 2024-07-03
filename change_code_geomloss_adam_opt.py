import os
import os
import subprocess
import json
# Find the conda root directory
try:
    conda_info = subprocess.check_output(["conda", "info", "--json"]).decode("utf-8")
    conda_root = json.loads(conda_info)["root_prefix"]
except (subprocess.CalledProcessError, json.JSONDecodeError):
    print("Failed to find conda root directory.")
    conda_root = None

# Find a specific environment
env_name = "deliverable10"
env_path = None

if conda_root:
    envs_dir = os.path.join(conda_root, "envs")
    for entry in os.listdir(envs_dir):
        env_dir = os.path.join(envs_dir, entry)
        if os.path.isdir(env_dir) and entry == env_name:
            env_path = env_dir
            break

    if env_path:
        print(f"Conda root directory: {conda_root}")
        print(f"Environment '{env_name}' path: {env_path}")
    else:
        print(f"Environment '{env_name}' not found.")



# Relative path to the conda environment

# Path to the file within the conda environment
file_path = os.path.join(env_path, "lib", "python3.8", "site-packages", "geomloss", "sinkhorn_divergence.py")


print (file_path)

# New line to replace line 159
new_line = "    diameter = max_diameter(x.reshape(-1, D), y.reshape(-1, D))"

# Read the file contents
with open(file_path, "r") as file:
    lines = file.readlines()

# Modify line 159
lines[158] = new_line + "\n"

# Write the modified contents back to the file
with open(file_path, "w") as file:
    file.writelines(lines)

print(f"Line 159 in {file_path} has been modified.")


if conda_root:
    envs_dir = os.path.join(conda_root, "envs")
    for entry in os.listdir(envs_dir):
        env_dir = os.path.join(envs_dir, entry)
        if os.path.isdir(env_dir) and entry == env_name:
            env_path = env_dir
            break

    if env_path:
        # Path to the adamw.py file
        file_path = os.path.join(env_path, "lib", "python3.8", "site-packages", "torch", "optim", "adamw.py")
        print (file_path)
        
        # New line to replace line 496
        new_line = "        if weight_decay not in [None]:  torch._foreach_mul_(device_params, 1 - lr * weight_decay)"

        # Read the file contents
        with open(file_path, "r") as file:
            lines = file.readlines()

        # Modify line 496
        lines[495] = new_line + "\n"

        # Write the modified contents back to the file
        with open(file_path, "w") as file:
            file.writelines(lines)

        print(f"Line 496 in {file_path} has been modified.")
    else:
        print(f"Environment '{env_name}' not found.")
else:
    print("Conda root directory not found.")