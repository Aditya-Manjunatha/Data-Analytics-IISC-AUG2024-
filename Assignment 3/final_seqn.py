import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f
from scipy.linalg import pinv

def load_and_preprocess_gene_data(filepath):

    raw_data = pd.read_csv(filepath, sep="\t", header=0)  # Assuming the first row is the header

    # Deleting the Go column
    raw_data.drop(columns=['Go'], inplace=True)

    # Deleting rows with NaN in GeneSymbol and EntrezGeneID
    raw_data.dropna(subset=['GeneSymbol', 'EntrezGeneID'], inplace=True)

    # Delete all rows which have at least one 0
    #raw_data = raw_data[(raw_data.iloc[:, 1:49] != 0).all(axis=1)]

    # Average values of rows with the same GeneSymbol
    # Group by GeneSymbol and calculate the mean for numeric columns only
    numeric_cols = raw_data.columns[1:-2]  # Exclude ProbeName, GeneSymbol, and EntrezGeneID
    # Exponentiate every value in the dataset (excluding GeneSymbol and EntrezGeneID and ProbeName)
    for col in numeric_cols:
        raw_data[col] = 2 ** raw_data[col]
    #print(raw_data.columns.tolist())
    # Remove the ProbeName column, the GeneSymbol column, and the EntrezGeneID column while converting to numpy array
    numpy_array = raw_data.iloc[:, 1:-2].to_numpy()
    return numpy_array


def creating_matrices():
    groups = 4
    samples = 12
    A = np.zeros((groups * samples, 4))
    B = np.zeros((groups * samples, 4))

    # Use numpy broadcasting to assign patterns
    A_patterns = np.array([
        [1, 0, 0, 0],  # Male Non-Smoker
        [0, 1, 0, 0],  # Female Non-Smoker
        [0, 0, 1, 0],  # Male Smoker
        [0, 0, 0, 1]   # Female Smoker
    ])

    B_patterns = np.array([
        [1, 0, 1, 0],  # Male Non-Smoker
        [1, 0, 0, 1],  # Female Non-Smoker
        [0, 1, 1, 0],  # Male Smoker
        [0, 1, 0, 1]   # Female Smoker
    ])

    # Assign patterns to A and B using repetition for each group
    for i in range(groups):
        A[i * samples:(i + 1) * samples] = A_patterns[i]
        B[i * samples:(i + 1) * samples] = B_patterns[i]

    return A, B

# Function to calculate F-statistic for a given probe
def calculate_f_statistic(probe, A_mat, B_mat, df_numerator, df_denominator):
    identity_matrix = np.eye(48)
    num = probe.T @ (A_mat - B_mat) @ probe
    den = probe.T @ (identity_matrix - A_mat) @ probe

    if den != 0:
        f_statistic = (num * df_numerator) / (den * df_denominator)
        return f_statistic
    return None


def calculate_p_value(f_statistic, den, num):
    if f_statistic is not None:
        return 1 - f.cdf(f_statistic, den, num)
    return 1.0


def compute_pvalues(data, A, B):
    A_mat = A @ pinv(A.T @ A) @ A.T
    B_mat = B @ pinv(B.T @ B) @ B.T

    df_num = 48 - np.linalg.matrix_rank(A)
    df_den = np.linalg.matrix_rank(A) - np.linalg.matrix_rank(B)

    pvalues = []

    for probe in data:
        f_stat = calculate_f_statistic(probe, A_mat, B_mat, df_num, df_den)
        p_value = calculate_p_value(f_stat, df_den, df_num)
        pvalues.append(p_value)

    return np.array(pvalues)


def plot_pvalue_histogram(pvalues):
    plt.figure(figsize=(10, 6))
    plt.hist(pvalues, bins=100)
    plt.title('Distribution of the interaction P-values with 100 bins')
    plt.xlabel('P-values')
    plt.ylabel('Frequencies')
    plt.show()


if __name__ == "__main__":
    file_path = "/Users/adityamanjunatha/Library/CloudStorage/OneDrive-IndianInstituteofScience/IISc Semester/5th Semester/Data Analytics/Assignments/Assignment 3/Raw Data_GeneSpring.txt"

    data = load_and_preprocess_gene_data(file_path)
    print("Shape of gene data:", data.shape)
    print("Length of gene data:", len(data))
    # print first 5 rows of the gene data
    #print("First 5 rows of the gene data:")
    #print(gene_data[:7])
    
    A, B = creating_matrices()
    print("Shape of A:", A.shape)
    print("Shape of B:", B.shape)
    p_values = compute_pvalues(data, A, B)

    # Print the largest and smalles p value
    print("Largest p-value:", np.max(p_values))
    print("Smallest p-value:", np.min(p_values))
    # Print the first 10 p values
    print("First 10 p-values:", p_values[:10])
    #print("P-values:", p_values)

    plot_pvalue_histogram(p_values)