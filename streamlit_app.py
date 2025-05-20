import streamlit as st
from mpi4py import MPI
import subprocess
import os
import sys
from PIL import Image, ImageFilter

# Directory for uploaded files
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Helper function to save uploaded files safely
def save_uploaded_file(uploaded_file):
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

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
        "Matrix Multiplication",
    ],
)
process_count = st.sidebar.number_input("Number of Processes", min_value=1, value=4)

# Task-specific Inputs
if task == "Sorting Numbers":
    numbers = st.text_input("Enter numbers (comma-separated):", "3,2,1,4,5")
elif task == "File Processing":
    uploaded_file = st.file_uploader("Upload a file:", type=["txt"])
elif task == "Image Filtering":
    filter_type = st.selectbox("Choose Filter", ["Blur", "Greyscale"])
    image_file = st.file_uploader("Upload an image:", type=["png", "jpg", "jpeg"])

    # Show preview immediately in Streamlit
    if image_file:
        try:
            img = Image.open(image_file)
            if filter_type == "Blur":
                processed_img = img.filter(ImageFilter.GaussianBlur(radius=5))
            elif filter_type == "Greyscale":
                processed_img = img.convert("L")
            else:
                st.error(f"Invalid filter type: {filter_type}")
                processed_img = None

            if processed_img:
                st.image(processed_img, caption=f"Image after {filter_type} filter", use_column_width=True)
        except Exception as e:
            st.error(f"Error processing image: {e}")

elif task == "Machine Learning Training":
    ml_file = st.file_uploader("Upload a numeric data file (CSV):", type=["csv"])
    if ml_file:
        try:
            import pandas as pd
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LinearRegression

            data = pd.read_csv(ml_file)
            st.write("Preview of Uploaded Data:")
            st.dataframe(data.head())

            target_column = st.selectbox("Select Target Column", options=data.columns)

            if target_column:
                X = data.drop(columns=[target_column])
                y = data[target_column]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = LinearRegression()
                model.fit(X_train, y_train)

                st.write("Model Trained Successfully!")
                st.write("Coefficients:", model.coef_)
                st.write("Intercept:", model.intercept_)
        except Exception as e:
            st.error(f"Error during training: {e}")

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
def run_mpi(task_num, extra_args):
    mpi_script = r"C:\StreamlitParallel\alltasks.py"  # Update this to your actual path
    n_procs = int(process_count)

    cmd = [
        "mpiexec",
        "-n", str(n_procs),
        sys.executable,
        mpi_script,
        "--task", str(task_num),
    ] + extra_args

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.stdout, result.stderr

# Run Task
if st.sidebar.button("Run Task"):
    stdout, stderr = "", ""
    try:
        if task == "Sorting Numbers":
            stdout, stderr = run_mpi(1, ["--nums", numbers])
        elif task == "File Processing" and uploaded_file:
            file_path = save_uploaded_file(uploaded_file)
            stdout, stderr = run_mpi(2, ["--file", file_path])
        elif task == "Image Filtering" and image_file:
            # Note: Your image filtering is done directly in Streamlit preview,
            # If you want MPI processing, save and send to MPI here:
            image_path = save_uploaded_file(image_file)
            stdout, stderr = run_mpi(3, ["--file", image_path, "--filter", filter_type])
        elif task == "Machine Learning Training" and ml_file:
            # Currently ML training done in Streamlit, but if you want MPI:
            # save file and send to MPI script
            ml_path = save_uploaded_file(ml_file)
            stdout, stderr = run_mpi(4, ["--file", ml_path])
        elif task == "Text Search" and text_file and keyword.strip() != "":
            text_path = save_uploaded_file(text_file)
            stdout, stderr = run_mpi(5, ["--file", text_path, "--keyword", keyword])
        elif task == "Statistics Analysis" and stats_file:
            stats_path = save_uploaded_file(stats_file)
            stdout, stderr = run_mpi(6, ["--file", stats_path])
        elif task == "Matrix Multiplication":
            stdout, stderr = run_mpi(7, [
                "--a_rows", str(a_rows), "--a_cols", str(a_cols),
                "--b_rows", str(b_rows), "--b_cols", str(b_cols),
                "--a_values", a_values, "--b_values", b_values,
            ])
        else:
            st.error("Please provide the required inputs for the selected task.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

    # Display Results
    st.subheader("Output:")
    st.text(stdout if stdout else "No output available.")
    st.text(stderr if stderr else "No errors encountered.")
