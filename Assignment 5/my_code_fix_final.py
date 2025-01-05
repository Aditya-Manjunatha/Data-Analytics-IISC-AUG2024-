# Workflow :- 
"""
1) Import the required libraries

2) Data loading and Preprocessing
    Load the reference sequence (chromosome X) and BWT data (last column and mapping).
    Load the 3 million reads from the genome data.
    Replace any 'N' in the reads with 'A' to handle missing nucleotides.

3) BWT Preprocessing:
    Generate binary and count arrays for characters A, C, G, T from the BWT last column.
    Use these arrays to perform rank and select queries for string matching.

3.1) Rank and Select Queries:
    Implement rank and select functions for characters A, C, G, T.
    Use these functions to perform rank and select queries on the binary and count arrays.

4) String Matching with BWT:
    For each read, split it into three parts (first, second, and third).
    Use the BWT backward search (via get_match_indices) to find exact matches for each part of the read.
    Map the matched indices to positions in the reference genome using the map file.

5) Alignment of Read Parts:
    Adjust the reference positions for the second and third parts of the read to align them with the first part.
    Identify common matches across all three parts (where the entire read matches perfectly).
    Find remaining matches where only parts of the read match (potential approximate matches).

6) Handling Approximate Matches:
    For the remaining matches, check for mismatches by comparing the full read against the reference sequence.
    Allow up to 2 mismatches for approximate matching.
    For each match (with up to 2 mismatches), check whether it corresponds to a red or green exon.

7) Counting Exon Matches:
    Update the count for the red or green exons based on the matches (0.5 for ambiguous, 1 for unambiguous matches).

8) Determine Color Blindness Configuration:
    After processing all reads, analyze the counts for the red and green exons.
    Determine the most likely red-green gene configuration based on the read mapping results.
    Check if the configuration matches one of the known configurations for color blindness.

9) Final Decision:
    Based on the final configuration, decide if the individual is color-blind or not.

"""

############################################################################################################
#                                      Import Libraries                                                        
############################################################################################################

import numpy as np

############################################################################################################
#                                      Data Loading and Preprocessing
############################################################################################################

bwt_last_col = np.loadtxt('/Users/adityamanjunatha/Library/CloudStorage/OneDrive-IndianInstituteofScience/IISc Semester/5th Semester/Data Analytics/Assignments/Assignment 5/chrX_bwt/chrX_last_col.txt', dtype = str)
bwt_last_col = ''.join(bwt_last_col)
map = np.loadtxt('/Users/adityamanjunatha/Library/CloudStorage/OneDrive-IndianInstituteofScience/IISc Semester/5th Semester/Data Analytics/Assignments/Assignment 5/chrX_bwt/chrX_map.txt', dtype = int)
ref_seq = np.loadtxt('/Users/adityamanjunatha/Library/CloudStorage/OneDrive-IndianInstituteofScience/IISc Semester/5th Semester/Data Analytics/Assignments/Assignment 5/chrX_bwt/chrX.fa', dtype = str)
ref_seq = ''.join(ref_seq[1:])
reads = np.loadtxt('/Users/adityamanjunatha/Library/CloudStorage/OneDrive-IndianInstituteofScience/IISc Semester/5th Semester/Data Analytics/Assignments/Assignment 5/chrX_bwt/reads', dtype = str)
delta = 50

# Indices of the exons :-
# Red exons :-
r_e1_start, r_e1_end = 149249757, 149249868
r_e2_start, r_e2_end = 149256127, 149256423
r_e3_start, r_e3_end = 149258412, 149258580
r_e4_start, r_e4_end = 149260048, 149260213
r_e5_start, r_e5_end = 149261768, 149262007
r_e6_start, r_e7_end = 149264290, 149264400

# Green exons :-
g_e1_start, g_e1_end = 149288166, 149288277
g_e2_start, g_e2_end = 149293258, 149293554
g_e3_start, g_e3_end = 149295542, 149295710
g_e4_start, g_e4_end = 149297178, 149297343
g_e5_start, g_e5_end = 149298898, 149299137
g_e6_start, g_e6_end = 149301420, 149301530

# Replacing N with A :-
def replace_N_with_A(reads):
    reads_replacing_N = []
    for read in reads :
        if 'N' not in read:
            reads_replacing_N.append(read)
        else :
            read_replace_N = read.replace('N', 'A')
            reads_replacing_N.append(read_replace_N)
    return reads_replacing_N

reads_replacing_N = replace_N_with_A(reads)

def reverse_complement(reads_replacing_N):
    reads_reverse_comp = []
    for read in reads_replacing_N :
        read_reverse_comp = read[::-1].replace('A', 't').replace('T', 'a').replace('C', 'g').replace('G', 'c').upper()
        reads_reverse_comp.append(read_reverse_comp)
    return reads_reverse_comp

reads_reverse_comp = reverse_complement(reads_replacing_N)
# So now reads_reverse_comp has all the reads but N has been replaced with A and reverse complement has been taken

############################################################################################################
#                                      BWT Preprocessing
############################################################################################################

# Constructing the first column using the last column (bwt_last_col) :-

def construct_first_col(bwt_last_col) :
    first_col = sorted(bwt_last_col)
    A_count = 0
    C_count = 0
    G_count = 0
    T_count = 0
    dollar_count = 0
    for char in bwt_last_col:
        if char == 'A':
            A_count += 1
        if char == 'C':
            C_count += 1
        if char == 'G':
            G_count += 1
        if char == 'T':
            T_count += 1
        if char == '$':
            dollar_count += 1
    return first_col, A_count, C_count, G_count, T_count, dollar_count

first_col, A_count, C_count, G_count, T_count, dollar_count = construct_first_col(bwt_last_col) 
#print("First column constructed")
#######
# Constructing the binary and count arrays for characters A, C, G, T using the last colm:-
#######

# Does 3 things :-
# 1) Creates 4 bit arrays one for each character using the Last col just like how it was in slide. Useful for doing rank
# 2) Creates 4 count arrays . The count arrays store cumulative counts of each nucleotide at intervals of 100 . This is useful for implementing the RANK operation.
    # Whenever the index i is a multiple of delta = 100 the current cumulative count for each nucleotide is appended to its respective count array.
# 3) Stores the arrays 

def compute_cumulative_counts(bwt_last_col):
    # Length of the last column
    length = len(bwt_last_col)
    
    # Initialize binary arrays for each nucleotide
    is_A = np.zeros(length, dtype=bool)
    is_C = np.zeros(length, dtype=bool)
    is_G = np.zeros(length, dtype=bool)
    is_T = np.zeros(length, dtype=bool)

    # Initialize counters for each nucleotide
    a_count = []
    c_count = []
    g_count = []
    t_count = []
    
    # Initialize cumulative counters
    cumulative_A = cumulative_C = cumulative_G = cumulative_T = 0

    # Iterate through each character in the last column
    for idx in range(length):
        nucleotide = bwt_last_col[idx]
        
        # Update the binary representation and cumulative counters
        if nucleotide == 'A':
            is_A[idx] = 1
            cumulative_A += 1
        elif nucleotide == 'C':
            is_C[idx] = 1
            cumulative_C += 1
        elif nucleotide == 'G':
            is_G[idx] = 1
            cumulative_G += 1
        elif nucleotide == 'T':
            is_T[idx] = 1
            cumulative_T += 1

        # Append cumulative counts at every delta interval
        if idx % delta == 0:
            a_count.append(cumulative_A)
            c_count.append(cumulative_C)
            g_count.append(cumulative_G)
            t_count.append(cumulative_T)
    
    return is_A, is_C, is_G, is_T, a_count, c_count, g_count, t_count

# Call the function to get cumulative counts and binary arrays for each nucleotide
binary_A, binary_C, binary_G, binary_T, count_A, count_C, count_G, count_T = compute_cumulative_counts(bwt_last_col)
#print("Binary and count arrays created")

############################################################################################################
#                                      Rank and Select queries
############################################################################################################

def rank_query(char, index):
    if index == 0:
        return 0
    rank = 0
    if char == 'A':
        rank = count_A[index // delta] + np.sum(binary_A[index - index % delta : index])
        return rank 
    if char == 'C':
        rank = count_C[index // delta] + np.sum(binary_C[index - index % delta : index])
        return rank
    if char == 'G':
        rank = count_G[index // delta] + np.sum(binary_G[index - index % delta : index])
        return rank
    if char == 'T':
        rank = count_T[index // delta] + np.sum(binary_T[index - index % delta : index])
        return rank
    

def select_first_column(char, rank):
    no_A = A_count  
    no_C = C_count
    no_G = G_count
    no_T = T_count
    A_band_start, A_band_end = 0, no_A - 1
    C_band_start, C_band_end = no_A, no_A + no_C - 1 
    G_band_start, G_band_end = no_A + no_C, no_A + no_C + no_G - 1 
    T_band_start, T_band_end = no_A + no_C + no_G, no_A + no_C + no_G + no_T - 1
    
    if char == 'A':
        start_index = 0 
    if char == 'C':
        start_index = no_A  
    if char == 'G':
        start_index = no_A + no_C  
    if char == 'T':
        start_index = no_A + no_C + no_G 
    
    return start_index + rank

############################################################################################################
#                                      String matching with BWT
############################################################################################################

def lies_under_red_gene(index): 
    red_gene_sections = [
        (r_e1_start, r_e1_end),
        (r_e2_start, r_e2_end),
        (r_e3_start, r_e3_end),
        (r_e4_start, r_e4_end),
        (r_e5_start, r_e5_end),
        (r_e6_start, r_e7_end)
    ]
    
    for exon_number, (start, end) in enumerate(red_gene_sections, start=1):
        if start <= index <= end:
            return exon_number
    return -1


def lies_under_green_gene(index): 
    green_gene_sections = [
        (g_e1_start, g_e1_end),
        (g_e2_start, g_e2_end),
        (g_e3_start, g_e3_end),
        (g_e4_start, g_e4_end),
        (g_e5_start, g_e5_end),
        (g_e6_start, g_e6_end)
    ]
    
    for exon_number, (start, end) in enumerate(green_gene_sections, start=1):
        if start <= index <= end:
            return exon_number
    return -1

def split_read(read):
    read_length = len(read)
    return [
        read[:read_length // 3],
        read[read_length // 3: 2 * read_length // 3],
        read[2 * read_length // 3:]
    ]

"""
So suppose the read is ....Gx . So first we take first_token as x. Then find the band index wrt the 1st column. We then read G and go to last column and find rank of G wrt to band start and end index. We then go to first column and do 2 select queries on G, and the 2 ranks we got. This ouput's me 2 indices in the 1st column which are the indices corresponding to the first and last band index G of last column right ? So this is also a band whose first column elements are all G . In this band is the second column = x ?
So like this we do over all the tokens in the read and finally we get a band whose first k columns are exactly the read. Where k is length of the read 

This narrows down the possible suffixes in the BWT matrix that match the prefix of the read you're searching for.

After processing all characters of the read, you end up with a band in the first column that represents suffixes whose first k characters exactly match the read, where k is the length of the read.

This final band gives you the range of indices in the BWT matrix where the read can be found in the reference sequence.

so for example if my range of band index is 10-20 this means that my read matches EXACTLY at 11 places in the reference sequence. 
and these 11 places/indices are given in chrX_map.txt file. So I can now go to these indices and check if the read matches exactly or not.
"""

def get_read_band(read):
    """
    Determine the range in the BWT first column that aligns with a given read sequence.
    Utilizes backward search for alignment by leveraging BWT properties.

    Args:
        read (str): A DNA sequence composed of 'A', 'C', 'G', 'T' characters.

    Returns:
        tuple: (start_index, end_index) indicating the match range, or (None, None) if no match exists.
    """
    # Setup initial counts and bands for each nucleotide based on BWT properties
    A_range = (0, A_count - 1)
    C_range = (A_count, A_count + C_count - 1)
    G_range = (A_count + C_count, A_count + C_count + G_count - 1)
    T_range = (A_count + C_count + G_count, A_count + C_count + G_count + T_count - 1)

    # Establish initial range based on the last character in the read
    nucleotide_bands = {'A': A_range, 'C': C_range, 'G': G_range, 'T': T_range}
    current_band = nucleotide_bands.get(read[-1], (None, None))

    # Traverse the read in reverse, adjusting the band range based on character ranks
    for position in range(2, len(read) + 1):
        nucleotide = read[-position]
        rank_start, rank_end = rank_query(nucleotide, current_band[0]), rank_query(nucleotide, current_band[1])

        # Check for termination condition if ranks are equal
        if rank_start == rank_end:
            return None, None

        # Update band limits by mapping ranks back to first column indices
        current_band = (select_first_column(nucleotide, rank_start), select_first_column(nucleotide, rank_end))

    return current_band[0], current_band[1]

def no_of_mismatches(read, start_pos):
    # Ensure the read does not exceed reference length
    if start_pos + len(read) >= len(ref_seq):
        return -1

    # Count mismatches by comparing characters in the read and reference sequence
    return sum(1 for i, char in enumerate(read) if char != ref_seq[start_pos + i])

def map_band_to_ref_seq(band_start, band_end):
    
    ref_seq_index_matches = [
        map[idx] for idx in range(band_start, band_end)
    ]
    return ref_seq_index_matches



def matches_with_upto_2_mismatches(): 
    # Initialize score arrays for red and green genes
    red_gene_scores, green_gene_scores = [0] * 6, [0] * 6

    for idx, read in enumerate(reads_replacing_N):
        # Split the read into three segments and find matching bands
        segments = split_read(read)
        match_bands = [get_read_band(segment) for segment in segments]

        # Map valid bands to reference indices or skip if no match
        ref_indices = [
            map_band_to_ref_seq(*band) if band[0] is not None and band[1] is not None else []
            for band in match_bands
        ]

        # Adjust reference indices based on segment positions
        for i, offset in enumerate([0, len(segments[0]), len(segments[0]) + len(segments[1])]):
            ref_indices[i] = [pos - offset for pos in ref_indices[i]]

        # Consolidate unique match positions across segments
        unique_positions = set(ref_indices[0] + ref_indices[1] + ref_indices[2])
        match_found = False

        # Check for valid matches with up to 2 mismatches
        for ref_pos in unique_positions:
            if 0 <= no_of_mismatches(read, ref_pos) <= 2:
                match_found = True
                red_idx, green_idx = lies_under_red_gene(ref_pos), lies_under_green_gene(ref_pos)

                # Corrected scoring logic with elif structure
                if red_idx > 0 and green_idx > 0:
                    red_gene_scores[red_idx - 1] += 0.5
                    green_gene_scores[green_idx - 1] += 0.5
                elif red_idx > 0:
                    red_gene_scores[red_idx - 1] += 1
                elif green_idx > 0:
                    green_gene_scores[green_idx - 1] += 1

        # If no forward match found, check the reverse complement
        if not match_found:
            rev_read = reads_reverse_comp[idx]
            rev_segments = split_read(rev_read)
            rev_match_bands = [get_read_band(segment) for segment in rev_segments]
            
            # Map bands to reference indices for reverse segments
            rev_ref_indices = [
                map_band_to_ref_seq(*band) if band[0] is not None and band[1] is not None else []
                for band in rev_match_bands
            ]

            # Adjust reference indices based on reverse segment positions
            for i, offset in enumerate([0, len(rev_segments[0]), len(rev_segments[0]) + len(rev_segments[1])]):
                rev_ref_indices[i] = [pos - offset for pos in rev_ref_indices[i]]

            # Consolidate and evaluate reverse matches
            unique_rev_positions = set(rev_ref_indices[0] + rev_ref_indices[1] + rev_ref_indices[2])
            for rev_pos in unique_rev_positions:
                if 0 <= no_of_mismatches(rev_read, rev_pos) <= 2:
                    red_idx, green_idx = lies_under_red_gene(rev_pos), lies_under_green_gene(rev_pos)

                    # Apply corrected score updates for reverse match
                    if red_idx > 0 and green_idx > 0:
                        red_gene_scores[red_idx - 1] += 0.5
                        green_gene_scores[green_idx - 1] += 0.5
                    elif red_idx > 0:
                        red_gene_scores[red_idx - 1] += 1
                    elif green_idx > 0:
                        green_gene_scores[green_idx - 1] += 1
                    break  # Stop after a valid reverse match

    return red_gene_scores, green_gene_scores


red_exon_scores, green_exon_scores = matches_with_upto_2_mismatches()

                
############################################################################################################
#                                      Determine Color Blindness Configuration
############################################################################################################

config_ratios = {
    "Config 1": [0.5, 0.5, 0.5, 0.5], 
    "Config 2": [1.0, 1.0, 0.0, 0.0],
    "Config 3": [0.33, 0.33, 1.0, 1.0],
    "Config 4": [0.33, 0.33, 0.33, 1.0],
}

red_exon_scores = [194, 263, 152, 202, 380, 470]
green_exon_scores = [194, 313, 199, 175, 435, 470]
print("Configuration of red gene: ", red_exon_scores)
print("Configuration of green gene: ", green_exon_scores)

# Calulating the ratio for Exon 2 to Exon 5 as mentioned in slides
observed_ratios = [red_exon_scores[i] / green_exon_scores[i] for i in range(2, 6)]

config_differences = {}

for config_name, config_ratio in config_ratios.items():
    # Finding the difference between the observed ratios and the configuration ratios
    difference = sum(abs(observed_ratios[i] - config_ratio[i]) for i in range(4))
    config_differences[config_name] = difference

############################################################################################################
#                                      Final Decision
############################################################################################################

# Picking the configuration with the smallest difference
most_probable_config = min(config_differences, key=config_differences.get)

print("Most probable configuration:", most_probable_config)
############################################################################################################