import gpustat
gpu_stats = gpustat.GPUStatCollection.new_query()
## Print GPU information
for gpu in gpu_stats.gpus:
    print(f"GPU {gpu.index}: {gpu.name}, Utilization: {gpu.utilization}%")
