"""GPU load test utilities for stress testing and monitoring GPU performance."""

import subprocess
import time
import tensorflow as tf


def get_gpu_memory_info(gpu_idx: int) -> tuple[float, int, int]:
    """Get GPU memory info: usage percent, total MB, and used MB.
    
    Args:
        gpu_idx: Index of the GPU to query.
        
    Returns:
        Tuple of (percent_used, total_mb, used_mb).
    """
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.total,memory.used', 
         '--format=csv,noheader,nounits', f'--id={gpu_idx}'],
        capture_output=True, text=True
    )

    total, used = map(int, result.stdout.strip().split(', '))
    percent = (used / total) * 100

    return percent, total, used


def get_gpu_temp(gpu_idx: int) -> int:
    """Get GPU temperature (C) using nvidia-smi.
    
    Args:
        gpu_idx: Index of the GPU to query.
        
    Returns:
        Temperature in degrees Celsius.
    """
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=temperature.gpu', 
         '--format=csv,noheader,nounits', f'--id={gpu_idx}'],
        capture_output=True, text=True
    )

    return int(result.stdout.strip())


def enable_memory_growth():
    """Enable memory growth for all available GPUs.
    
    This prevents TensorFlow from allocating all GPU memory at startup.
    
    Returns:
        List of available GPU devices.
    """
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f'Memory growth enabled for {len(gpus)} GPU(s)')
        except RuntimeError as e:
            print(f'Could not set memory growth: {e}')
    
    return gpus


def scale_matrix_to_memory_target(gpu_idx: int, target_memory_percent: float = 75.0,
                                   initial_size: int = 1000, step_size: int = 1000):
    """Scale matrix size to reach target GPU memory usage.
    
    Args:
        gpu_idx: Index of the GPU to use.
        target_memory_percent: Target percentage of GPU memory to use.
        initial_size: Starting matrix size.
        step_size: Amount to increase matrix size each iteration.
        
    Returns:
        Tuple of (matrix_a, matrix_b, final_matrix_size).
    """
    _, total_mem, initial_used = get_gpu_memory_info(gpu_idx)
    
    matrix_size = initial_size
    last_successful_size = matrix_size

    print(f'Total GPU memory: {total_mem} MB')
    print(f'Initial memory used: {initial_used} MB ({initial_used/total_mem*100:.1f}%)')
    print(f'Target: {target_memory_percent}% memory usage\n')
    
    with tf.device(f'/GPU:{gpu_idx}'):
        while True:
            tf.keras.backend.clear_session()
            
            print(f'Trying matrix size: {matrix_size}x{matrix_size}...', end=' ', flush=True)
            
            try:
                a = tf.random.normal([matrix_size, matrix_size])
                b = tf.random.normal([matrix_size, matrix_size])
                
                c = tf.matmul(a, b)
                _ = c.numpy()
                
                mem_percent, _, mem_used = get_gpu_memory_info(gpu_idx)
                print(f'Memory: {mem_used} MB ({mem_percent:.1f}%)')
                
                last_successful_size = matrix_size
                
                if mem_percent >= target_memory_percent:
                    print(f'\nReached {mem_percent:.1f}% memory usage with {matrix_size}x{matrix_size} matrices')
                    break
                
                matrix_size += step_size
                
            except (tf.errors.ResourceExhaustedError, tf.errors.InternalError, 
                    RuntimeError, Exception) as e:
                print(f'\nOOM! ({type(e).__name__})')
                print(f'Using last successful size: {last_successful_size}x{last_successful_size}')

                tf.keras.backend.clear_session()

                matrix_size = last_successful_size
                a = tf.random.normal([matrix_size, matrix_size])
                b = tf.random.normal([matrix_size, matrix_size])
                c = tf.matmul(a, b)
                _ = c.numpy()
                break
    
    return a, b, matrix_size


def run_stress_test(gpu_idx: int, matrix_a, matrix_b, matrix_size: int,
                    duration_seconds: int = 600, temp_record_interval: int = 5,
                    progress_interval: int = 30) -> dict:
    """Run GPU stress test for specified duration.
    
    Args:
        gpu_idx: Index of the GPU to use.
        matrix_a: First matrix for multiplication.
        matrix_b: Second matrix for multiplication.
        matrix_size: Size of the matrices.
        duration_seconds: Duration of the stress test in seconds.
        temp_record_interval: Interval in seconds for recording temperature.
        progress_interval: Interval in seconds for printing progress.
        
    Returns:
        Dictionary containing test results and temperature data.
    """
    initial_temp = get_gpu_temp(gpu_idx)
    
    temp_data = {'times': [0], 'temps': [initial_temp]}
    
    print(f'\nRunning stress test for {duration_seconds // 60} minutes at {matrix_size}x{matrix_size}...')
    print(f'Initial temperature: {initial_temp}°C')
    
    start_time = time.time()
    iteration_count = 0
    last_update_time = start_time
    last_temp_record_time = start_time
    
    with tf.device(f'/GPU:{gpu_idx}'):
        while (time.time() - start_time) < duration_seconds:
            c = tf.matmul(matrix_a, matrix_b)
            _ = c.numpy()
            iteration_count += 1
            
            current_time = time.time()

            if current_time - last_temp_record_time >= temp_record_interval:
                current_temp = get_gpu_temp(gpu_idx)
                elapsed = current_time - start_time
                temp_data['times'].append(elapsed)
                temp_data['temps'].append(current_temp)
                last_temp_record_time = current_time
            
            if current_time - last_update_time >= progress_interval:
                current_temp = get_gpu_temp(gpu_idx)
                elapsed = current_time - start_time
                remaining = duration_seconds - elapsed
                print(f'  {elapsed:.0f}s elapsed, {remaining:.0f}s remaining - '
                      f'Temp: {current_temp}°C, Iterations: {iteration_count}')
                last_update_time = current_time
    
    elapsed_time = time.time() - start_time
    final_temp = get_gpu_temp(gpu_idx)
    
    temp_data['times'].append(elapsed_time)
    temp_data['temps'].append(final_temp)
    
    final_mem_percent, _, final_mem_used = get_gpu_memory_info(gpu_idx)
    
    return {
        'matrix_size': matrix_size,
        'final_mem_used': final_mem_used,
        'final_mem_percent': final_mem_percent,
        'iteration_count': iteration_count,
        'elapsed_time': elapsed_time,
        'initial_temp': initial_temp,
        'final_temp': final_temp,
        'temp_data': temp_data
    }


def run_gpu_load_test(target_memory_percent: float = 75.0,
                       test_duration_seconds: int = 600) -> dict:
    """Run complete GPU load test on all available GPUs.
    
    Args:
        target_memory_percent: Target percentage of GPU memory to use.
        test_duration_seconds: Duration of the stress test in seconds.
        
    Returns:
        Dictionary mapping GPU index to test results including temperature data.
    """
    gpus = enable_memory_growth()
    num_gpus = len(gpus)
    
    gpu_results = {}
    
    for gpu_idx, gpu in enumerate(gpus):
        print(f'{"="*60}')
        print(f'Testing GPU {gpu_idx}: {gpu.name}')
        print(f'{"="*60}')
        
        matrix_a, matrix_b, matrix_size = scale_matrix_to_memory_target(
            gpu_idx, target_memory_percent
        )
        
        results = run_stress_test(
            gpu_idx, matrix_a, matrix_b, matrix_size, test_duration_seconds
        )
        results['name'] = gpu.name
        
        gpu_results[gpu_idx] = results
        
        print(f'\nGPU {gpu_idx} stress test completed!')
        print(f'  - Final matrix size: {results["matrix_size"]}x{results["matrix_size"]}')
        print(f'  - Peak memory usage: {results["final_mem_used"]} MB ({results["final_mem_percent"]:.1f}%)')
        print(f'  - Total iterations: {results["iteration_count"]}')
        print(f'  - Total time: {results["elapsed_time"]:.1f} seconds ({results["elapsed_time"]/60:.1f} minutes)')
        print(f'  - Avg time per operation: {results["elapsed_time"]/results["iteration_count"]:.4f} seconds')
        print(f'  - Temperature: {results["initial_temp"]}°C → {results["final_temp"]}°C '
              f'(Δ{results["final_temp"] - results["initial_temp"]:+d}°C)\n')

    print(f'{"="*60}')
    print(f'All {num_gpus} GPU(s) stress tested successfully!')
    print(f'{"="*60}')
    
    return gpu_results


def plot_temperature_results(gpu_results: dict):
    """Plot temperature over time for each GPU.
    
    Args:
        gpu_results: Dictionary of GPU test results from run_gpu_load_test.
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(8, 4))
    plt.title('GPU temperature over time during stress test')

    for gpu_idx, results in gpu_results.items():
        temp_data = results['temp_data']
        times_minutes = [t / 60 for t in temp_data['times']]
        plt.plot(times_minutes, temp_data['temps'], label=f'GPU {gpu_idx}')

    plt.xlabel('Time (minutes)')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.tight_layout()
    plt.show()
