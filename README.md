# Optimization Algorithms Comparison for Image Classification

## Overview
This project compares different optimization algorithms by observing how they perform in an image classification task. It was part of an academic assignment on optimization methods, using both C and Python for implementation and visualization.

## Project Structure
The project has two main parts:

### Part A: Image Classification with Optimization Algorithms
- Created a grayscale image dataset with two classes (at least 100 samples per class).
- Normalized pixel values to [0,1] range.
- Converted images into N×N vectors (size: N*N, where N > 20).
- Split the dataset into 80% training and 20% testing.
- Implemented a classification model using the formula:
output = tanh(w * x)
where `w` are the learnable parameters and `x` is the input vector.
- Compared the performance of three optimization algorithms:
- Gradient Descent (GD)
- Stochastic Gradient Descent (SGD)
- Adam Optimizer
- Trained models using five different initial weights and recorded:
- Training/Test accuracy
- Training time
- Number of updates
- Generated Epoch vs. Loss and Time vs. Loss plots.

### Part B: Visualization of Optimization Trajectories
- Tracked parameter updates during optimization.
- Applied t-SNE to reduce the weight trajectory dimensions to 2D.
- Analyzed how different starting points affected the optimization paths.

## Technologies Used
- **C Programming Language** (for model training and optimization implementation)
- **Python** (for data visualization and dimensionality reduction with t-SNE)
- **Matplotlib**, **NumPy**, **Pandas** (Python libraries for plotting and data handling)

## How to Run
1. **Clone the repository:**
git clone <https://github.com/burcukdl/Project-Optimization>
2. **Download the MNIST dataset** (if not included in the repo).
3. **Run the C code** to generate the dataset and train the model:
- Navigate to the project folder and run:
  ```
   gcc -o data_prep_opt_cmp data_prep_opt_cmp.c
   data_prep_opt_cmp
  ```
4. **Run the Python code** to visualize results:
- Install required Python libraries:
  ```
   pip install matplotlib numpy pandas scikit-learn
  ```
- Run the Python script:
  ```
   python algorithmComparison.py
   python tsneFigure.py
  ```
5. **View the results** in the generated plots and CSV files.

## Sample Results

### Training Loss Curve
![Training Loss Curve](/results/2all_optimizers_results.png)

### t-SNE Figure
![t-SNE](/results/2all_tsne_results.png)

## License
This project is for educational purposes only.
