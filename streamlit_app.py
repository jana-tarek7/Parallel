import streamlit as st
from mpi4py import MPI
import subprocess
import os
import numpy as np
from PIL import Image, ImageFilter

# Streamlit Sidebar
st.sidebar.title("Parallel Data Processing")
task = st.sidebar.selectbox(
    "Choose a Task",
    [
        "Select Task",
        "Sorting Numbers",
        "File Processing",
        "Image Filtering",
        "Machine Learning Training",
        "Text Search",
        "Statistics Analysis",
        "Matrix Multiplication"
    ]
)
process_count = st.sidebar.number_input("Number of Processes", min_value=1, value=4)

# Task-specific Inputs
if task == "Sorting Numbers":
    numbers = st.text_input("Enter numbers (comma-separated):", "3,2,1,4,5")
elif task == "File Processing":
    uploaded_file = st.file_uploader("Upload a file:", type=["txt"])
elif task == "Image Filtering":
    filter_type = st.selectbox("Choose Filter", ["Blur", "Sharpen", "Edge Detection"])
    image_file = st.file_uploader("Upload an image:", type=["png", "jpg", "jpeg"])
elif task == "Machine Learning Training":
    st.write("No additional inputs required for this task.")
elif task == "Text Search":
    keyword = st.text_input("Enter keyword to search:")
    text_file = st.file_uploader("Upload a file:", type=["txt"])
elif task == "Statistics Analysis":
    stats_file = st.file_uploader("Upload a numerical data file (CSV/TXT):", type=["csv", "txt"])
elif task == "Matrix Multiplication":
    a_rows = st.number_input("A Matrix Rows", min_value=1, value=2)
    a_cols = st.number_input("A Matrix Columns", min_value=1, value=2)
    b_rows = st.number_input("B Matrix Rows", min_value=1, value=2)
    b_cols = st.number_input("B Matrix Columns", min_value=1, value=2)
    a_values = st.text_input("A Matrix Values (comma-separated):", "1,2,3,4")
    b_values = st.text_input("B Matrix Values (comma-separated):", "5,6,7,8")

# Function to Run MPI Command
def run_mpi(task_number, args):
    cmd = [
        "mpirun", "-n", str(process_count),
        "python", "C:\StreamlitParallel\alltasks.py",
        "--task", str(task_number)
    ]
    cmd.extend(args)
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.stdout, result.stderr

# Run Task
if st.sidebar.button("Run Task"):
    if task == "Sorting Numbers":
        stdout, stderr = run_mpi(1, ["--nums", numbers])
    elif task == "File Processing" and uploaded_file:
        file_path = f"uploads/{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        stdout, stderr = run_mpi(2, ["--file", file_path])
    elif task == "Image Filtering" and image_file:
        image_path = f"uploads/{image_file.name}"
        with open(image_path, "wb") as f:
            f.write(image_file.getbuffer())
        stdout, stderr = run_mpi(3, ["--file", image_path, "--filter", filter_type])
    elif task == "Machine Learning Training":
        stdout, stderr = run_mpi(4, [])
    elif task == "Text Search" and text_file:
        text_path = f"uploads/{text_file.name}"
        with open(text_path, "wb") as f:
            f.write(text_file.getbuffer())
        stdout, stderr = run_mpi(5, ["--file", text_path, "--keyword", keyword])
    elif task == "Statistics Analysis" and stats_file:
        stats_path = f"uploads/{stats_file.name}"
        with open(stats_path, "wb") as f:
            f.write(stats_file.getbuffer())
        stdout, stderr = run_mpi(6, ["--file", stats_path])
    elif task == "Matrix Multiplication":
        stdout, stderr = run_mpi(7, [
            "--a_rows", str(a_rows), "--a_cols", str(a_cols),
            "--b_rows", str(b_rows), "--b_cols", str(b_cols),
            "--a_values", a_values, "--b_values", b_values
        ])
    else:
        st.error("Please provide the required inputs for the selected task.")
        stdout, stderr = "", ""
    
    # Display Results
    st.subheader("Output:")
    st.text(stdout)
    st.text(stderr)
