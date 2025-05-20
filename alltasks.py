import argparse
import os
import numpy as np
import pandas as pd
from mpi4py import MPI
from PIL import Image, ImageFilter
import csv
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def timing_wrapper(func, *args, **kwargs):
    start = MPI.Wtime()
    result = func(*args, **kwargs)
    end = MPI.Wtime()
    elapsed = end - start
    times = comm.gather(elapsed, root=0)
    avg_time = None
    if rank == 0:
        avg_time = sum(times) / len(times)
        print(f"[Performance] Task executed in {avg_time:.6f} seconds (average across {size} processes).")
    return result, avg_time

def sorting_task(nums):
    numbers = list(map(int, nums.split(',')))
    chunk_size = len(numbers) // size
    start_idx = rank * chunk_size
    if rank == size - 1:
        local_nums = numbers[start_idx:]
    else:
        local_nums = numbers[start_idx:start_idx + chunk_size]
    local_sorted = sorted(local_nums)
    gathered = comm.gather(local_sorted, root=0)
    if rank == 0:
        merged = sorted([num for sublist in gathered for num in sublist])
        return f"Sorted Result: {merged}"
    return None

def file_processing_task(filepath):
    if not os.path.exists(filepath):
        if rank == 0:
            return f"[X] Uploaded file not found: {filepath}"
        else:
            return None

    with open(filepath, 'r') as file:
        lines = file.readlines()

    chunk_size = len(lines) // size
    start_idx = rank * chunk_size
    if rank == size - 1:
        local_lines = lines[start_idx:]
    else:
        local_lines = lines[start_idx:start_idx + chunk_size]
    local_word_count = sum(len(line.split()) for line in local_lines)
    total = comm.reduce(local_word_count, op=MPI.SUM, root=0)

    if rank == 0:
        return f"Total Word Count: {total}"
    return None

def image_processing_task(filepath, filter_type):
    if rank == 0:
        if not os.path.exists(filepath):
            return f"[X] Uploaded file not found: {filepath}"

        img = Image.open(filepath)
        if filter_type.lower() == "blur":
            img = img.filter(ImageFilter.GaussianBlur(radius=5))
        elif filter_type.lower() == "greyscale":
            img = img.convert("L")
        else:
            return f"[X] Invalid filter type: {filter_type}"

        output_path = "processed_image.png"
        img.save(output_path)
        return f"[✓] Image processed and saved as {output_path}"
    return None

def matrix_multiplication_task(args):
    A = np.fromstring(args.a_values, sep=",").reshape((int(args.a_rows), int(args.a_cols)))
    B = np.fromstring(args.b_values, sep=",").reshape((int(args.b_rows), int(args.b_cols)))

    if A.shape[1] != B.shape[0]:
        if rank == 0:
            return "[X] A's columns must equal B's rows for multiplication."
        return None

    rows_per_proc = A.shape[0] // size
    start_idx = rank * rows_per_proc
    if rank == size - 1:
        local_A = A[start_idx:]
    else:
        local_A = A[start_idx:start_idx + rows_per_proc]
    local_result = np.dot(local_A, B)
    gathered = comm.gather(local_result, root=0)

    if rank == 0:
        result = np.vstack(gathered)
        return f"Matrix A:\n{A}\n\nMatrix B:\n{B}\n\nResult of A * B:\n{result}"
    return None

def text_search_task(filepath, keyword):
    if not os.path.exists(filepath):
        if rank == 0:
            return f"[X] Uploaded file not found: {filepath}"
        return None

    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    chunk_size = len(text) // size
    start = rank * chunk_size
    if rank == size - 1:
        end = len(text)
    else:
        end = start + chunk_size
    local_text = text[start:end]

    local_positions = []
    pos = local_text.find(keyword)
    while pos != -1:
        global_pos = start + pos
        local_positions.append(global_pos)
        pos = local_text.find(keyword, pos + 1)

    all_positions = comm.gather(local_positions, root=0)

    if rank == 0:
        positions = [pos for sublist in all_positions for pos in sublist]
        positions.sort()
        return f"Total occurrences: {len(positions)}\nPositions: {positions}"
    return None

def statistics_analysis_task(filepath):
    if not os.path.exists(filepath):
        if rank == 0:
            return f"[X] Uploaded file not found: {filepath}"
        return None

    data = None
    if rank == 0:
        df = pd.read_csv(filepath)
        df = df.select_dtypes(include='number')
        data = df.to_dict(orient='list')
    data = comm.bcast(data, root=0)

    df = pd.DataFrame(data)
    cols = list(df.columns)
    num_cols = len(cols)
    cols_per_proc = num_cols // size
    start = rank * cols_per_proc
    if rank == size - 1:
        end = num_cols
    else:
        end = start + cols_per_proc
    local_cols = cols[start:end]

    local_stats = {}
    for col in local_cols:
        col_data = df[col]
        local_stats[col] = {
            "mean": col_data.mean(),
            "median": col_data.median(),
            "mode": col_data.mode().iloc[0] if not col_data.mode().empty else None,
            "min": col_data.min(),
            "max": col_data.max(),
            "std_dev": col_data.std()
        }

    gathered_stats = comm.gather(local_stats, root=0)

    if rank == 0:
        final_stats = {}
        for d in gathered_stats:
            final_stats.update(d)
        output = "Statistics Analysis Results:\n"
        for col, stats_ in final_stats.items():
            output += f"\nColumn: {col}\n"
            for stat_name, stat_val in stats_.items():
                output += f"  {stat_name}: {stat_val}\n"
        return output
    return None

def save_performance_data(task_num, num_procs, elapsed_time):
    if rank == 0:
        file_exists = os.path.isfile('performance_times.csv')
        with open('performance_times.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(['Task', 'Num_Processes', 'Time_Seconds'])
            writer.writerow([task_num, num_procs, elapsed_time])

def plot_performance_chart():
    if rank == 0:
        if not os.path.exists('performance_times.csv'):
            print("[X] No performance data found to plot. Run some tasks first.")
            return

        df = pd.read_csv('performance_times.csv')
        tasks = df['Task'].unique()

        for task in tasks:
            task_data = df[df['Task'] == task]
            plt.plot(task_data['Num_Processes'], task_data['Time_Seconds'], marker='o', label=f"Task {task}")

        plt.xlabel("Number of Processes")
        plt.ylabel("Execution Time (seconds)")
        plt.title("MPI Tasks Performance Comparison")
        plt.legend()
        plt.grid(True)
        plt.show()

def main():
    if rank == 0:
        print(f"[MPI] Number of processes: {size}")

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True, help="Task to execute (e.g., 1, 2, 3, 5, 6, 7).")
    parser.add_argument('--nums', help="Comma-separated numbers for sorting.")
    parser.add_argument('--file', help="Path to the input file.")
    parser.add_argument('--filter', help="Filter type for image processing (blur, greyscale).")
    parser.add_argument('--a_rows', help="Number of rows in matrix A.")
    parser.add_argument('--a_cols', help="Number of columns in matrix A.")
    parser.add_argument('--b_rows', help="Number of rows in matrix B.")
    parser.add_argument('--b_cols', help="Number of columns in matrix B.")
    parser.add_argument('--a_values', help="Comma-separated values of matrix A.")
    parser.add_argument('--b_values', help="Comma-separated values of matrix B.")
    parser.add_argument('--keyword', help="Keyword for text search.")
    parser.add_argument('--show_chart', action='store_true', help="Show performance comparison chart.")
    args = parser.parse_args()

    if args.show_chart:
        plot_performance_chart()
        return

    result = None
    elapsed = None

    if args.task == '1':
        if args.nums:
            num_items = len(args.nums.split(','))
            if size > num_items and rank == 0:
                print(f"[Warning] Number of processes ({size}) is greater than number of elements to sort ({num_items}).")
        result, elapsed = timing_wrapper(sorting_task, args.nums)

    elif args.task == '2':
        result, elapsed = timing_wrapper(file_processing_task, args.file)

    elif args.task == '3':
        result, elapsed = timing_wrapper(image_processing_task, args.file, args.filter)

    elif args.task == '5':
        if not args.keyword:
            if rank == 0:
                print("[X] --keyword argument is required for task 5.")
            return
        result, elapsed = timing_wrapper(text_search_task, args.file, args.keyword)

    elif args.task == '6':
        result, elapsed = timing_wrapper(statistics_analysis_task, args.file)

    elif args.task == '7':
        result, elapsed = timing_wrapper(matrix_multiplication_task, args)

    else:
        if rank == 0:
            print(f"[X] Invalid task number: {args.task}")
        return

    if rank == 0:
        if result is not None:
            print(result)
        if elapsed is not None:
            # Save performance data to CSV for future plotting
            save_performance_data(args.task, size, elapsed)

if __name__ == '__main__':
    main()

