import os
import subprocess

# Path to the GFPGAN folder
path_to_inference = "./inference"

# Change the current directory to Inference
os.chdir(path_to_inference)

# Run setup.py
# try:
#     subprocess.check_call(["python", "setup.py", "develop"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
# except subprocess.CalledProcessError as e:
#     print("Output:", e.output)
#     print("Error:", e.stderr)

# Remove any existing results
# print("Cleaning up old results ...")
# subprocess.check_call(["rm", "-rf", "results"])

# Run inference
print("Running inference ...")
subprocess.check_call(["python", "inference.py", "-i", "inputs/upload", "-o", "results", "-v", "1.3", "-s", "2", "--bg_upsampler", "realesrgan"])

# Change the directory back to the original
os.chdir("..")

print("Done")