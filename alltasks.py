import argparse
import os
import numpy as np
from mpi4py import MPI
from PIL import Image, ImageFilter

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def sorting_task(nums):
    numbers = list(map(int, nums.split(',')))
    chunk_size = len(numbers) // size
    local_nums = numbers[rank * chunk_size:(rank + 1) * chunk_size] if rank != size - 1 else numbers[rank * chunk_size:]
    local_sorted = sorted(local_nums)
    gathered = comm.gather(local_sorted, root=0)
    if rank == 0:
        merged = sorted([num for sublist in gathered for num in sublist])
        return f"Sorted Result: {merged}"


def file_processing_task(filepath):
    if not os.path.exists(filepath):
        return f"[X] Uploaded file not found: {filepath}"

    with open(filepath, 'r') as file:
        lines = file.readlines()

    chunk_size = len(lines) // size
    local_lines = lines[rank * chunk_size:(rank + 1) * chunk_size] if rank != size - 1 else lines[rank * chunk_size:]
    local_word_count = sum(len(line.split()) for line in local_lines)
    total = comm.reduce(local_word_count, op=MPI.SUM, root=0)

    if rank == 0:
        return f"Total Word Count: {total}"


def image_processing_task(filepath, filter_type):
    if rank == 0:
        if not os.path.exists(filepath):
            return f"[X] Uploaded file not found: {filepath}"

        img = Image.open(filepath)
        if filter_type == "BLUR":
            img = img.filter(ImageFilter.BLUR)
        elif filter_type == "CONTOUR":
            img = img.filter(ImageFilter.CONTOUR)
        elif filter_type == "DETAIL":
            img = img.filter(ImageFilter.DETAIL)
        elif filter_type == "SHARPEN":
            img = img.filter(ImageFilter.SHARPEN)
        elif filter_type == "EDGE":
            img = img.filter(ImageFilter.FIND_EDGES)
        else:
            return f"[X] Invalid filter type: {filter_type}"

        output_path = "processed_image.png"
        img.save(output_path)
        return f"[✓] Image processed and saved as {output_path}"


def matrix_multiplication_task(args):
    A = np.fromstring(args.a_values, sep=",").reshape((int(args.a_rows), int(args.a_cols)))
    B = np.fromstring(args.b_values, sep=",").reshape((int(args.b_rows), int(args.b_cols)))

    if A.shape[1] != B.shape[0]:
        if rank == 0:
            return "[X] A's columns must equal B's rows for multiplication."
        return

    rows_per_proc = A.shape[0] // size
    local_A = A[rank * rows_per_proc:(rank + 1) * rows_per_proc] if rank != size - 1 else A[rank * rows_per_proc:]
    local_result = np.dot(local_A, B)
    gathered = comm.gather(local_result, root=0)

    if rank == 0:
        result = np.vstack(gathered)
        return f"Matrix A:\n{A}\n\nMatrix B:\n{B}\n\nResult of A * B:\n{result}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True, help="Task to execute.")
    parser.add_argument('--nums', help="Comma-separated numbers for sorting.")
    parser.add_argument('--file', help="Path to the input file.")
    parser.add_argument('--filter', help="Filter type for image processing.")
    parser.add_argument('--a_rows', help="Number of rows in matrix A.")
    parser.add_argument('--a_cols', help="Number of columns in matrix A.")
    parser.add_argument('--b_rows', help="Number of rows in matrix B.")
    parser.add_argument('--b_cols', help="Number of columns in matrix B.")
    parser.add_argument('--a_values', help="Comma-separated values of matrix A.")
    parser.add_argument('--b_values', help="Comma-separated values of matrix B.")
    args = parser.parse_args()

    if args.task == '1':
        result = sorting_task(args.nums)
    elif args.task == '2':
        result = file_processing_task(args.file)
    elif args.task == '3':
        result = image_processing_task(args.file, args.filter)
    elif args.task == '7':
        result = matrix_multiplication_task(args)
    else:
        result = f"[X] Invalid task number: {args.task}"

    if rank == 0:
        print(result)


if __name__ == '__main__':
    main()
