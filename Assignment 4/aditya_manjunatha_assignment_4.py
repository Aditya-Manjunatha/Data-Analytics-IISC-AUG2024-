#                                          Loading the modules                                                
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize
###############################################################################################################################

#                                           Data - Preprocessing

df = pd.read_csv('../data/')
len_df = len(df)
# Looking at datapoints only in the 1st innings
df = df[df['Innings'] == 1]
df.reset_index(drop=True, inplace=True)
# Remove rows whose Runs.Remaining is negative
df = df[df['Runs.Remaining'] >= 0]
df.reset_index(drop=True, inplace=True)

# Removing all the useless columns
col = [col for col in df.columns if col not in ['Match', 'Date', 'Innings', 'Over', 'Runs', 'Total.Runs', 'Innings.Total', 'Runs.Remaining', 'Total.Out', 'Innings.Total.Out', 'Outs.Remaining', 'Wickets.in.Hand', 'Innings.Total.Runs', 'Total.Overs']]
df.drop(col, axis=1, inplace=True)

# Adding Overs Remaining columns (u from slides) :
df['Overs.Remaining'] = 50 - df['Over']

###############################################################################################################################

#                                           Creating 10 dataframes for each wicket 

# So df_i contains all the overs where 'i' wickets are left

df_10 = df[df['Wickets.in.Hand'] == 10].reset_index(drop=True)
df_9 = df[df['Wickets.in.Hand'] == 9].reset_index(drop=True)
df_8 = df[df['Wickets.in.Hand'] == 8].reset_index(drop=True)
df_7 = df[df['Wickets.in.Hand'] == 7].reset_index(drop=True)
df_6 = df[df['Wickets.in.Hand'] == 6].reset_index(drop=True)
df_5 = df[df['Wickets.in.Hand'] == 5].reset_index(drop=True)
df_4 = df[df['Wickets.in.Hand'] == 4].reset_index(drop=True)
df_3 = df[df['Wickets.in.Hand'] == 3].reset_index(drop=True)
df_2 = df[df['Wickets.in.Hand'] == 2].reset_index(drop=True)
df_1 = df[df['Wickets.in.Hand'] == 1].reset_index(drop=True)
df_0 = df[df['Wickets.in.Hand'] == 0].reset_index(drop=True)


# Again filtering out the usefull columns

col = ['Match', 'Date', 'Innings.Total.Runs','Runs.Remaining', 'Overs.Remaining' ,'Total.Overs'] # Columns to be used for the model
df_10 = df_10[col]
df_9 = df_9[col]
df_8 = df_8[col]
df_7 = df_7[col]
df_6 = df_6[col]
df_5 = df_5[col]
df_4 = df_4[col]
df_3 = df_3[col]
df_2 = df_2[col]
df_1 = df_1[col]
df_0 = df_0[col]


# Sorting the dataframes wrt Overs.Remaining so that rows with same overs left are together

df_10 = df_10.sort_values(by=['Overs.Remaining'], ascending=False).reset_index(drop=True)
df_10['Wickets.in.Hand'] = 10
df_9 = df_9.sort_values(by=['Overs.Remaining'], ascending=False).reset_index(drop=True)
df_9['Wickets.in.Hand'] = 9
df_8 = df_8.sort_values(by=['Overs.Remaining'], ascending=False).reset_index(drop=True)
df_8['Wickets.in.Hand'] = 8
df_7 = df_7.sort_values(by=['Overs.Remaining'], ascending=False).reset_index(drop=True)
df_7['Wickets.in.Hand'] = 7
df_6 = df_6.sort_values(by=['Overs.Remaining'], ascending=False).reset_index(drop=True)
df_6['Wickets.in.Hand'] = 6
df_5 = df_5.sort_values(by=['Overs.Remaining'], ascending=False).reset_index(drop=True)
df_5['Wickets.in.Hand'] = 5
df_4 = df_4.sort_values(by=['Overs.Remaining'], ascending=False).reset_index(drop=True)
df_4['Wickets.in.Hand'] = 4
df_3 = df_3.sort_values(by=['Overs.Remaining'], ascending=False).reset_index(drop=True)
df_3['Wickets.in.Hand'] = 3
df_2 = df_2.sort_values(by=['Overs.Remaining'], ascending=False).reset_index(drop=True)
df_2['Wickets.in.Hand'] = 2
df_1 = df_1.sort_values(by=['Overs.Remaining'], ascending=False).reset_index(drop=True)
df_1['Wickets.in.Hand'] = 1
df_0 = df_0.sort_values(by=['Overs.Remaining'], ascending=False).reset_index(drop=True)
df_0['Wickets.in.Hand'] = 0

############################################################################################################################### 

#  Now for every data frame we need to add rows where Overs.Remaining is 50, And Runs Remaning is made to be Innings.Total.Runs for every  row which has 49 Overs Remaining


df_10_copy = df_10.copy()
# Filter rows where Overs.Remaining = 49
rows_with_49_overs = df_10[df_10['Overs.Remaining'] == 49]

# Create new rows with required modifications
new_rows = rows_with_49_overs.copy()
new_rows['Overs.Remaining'] = 50
new_rows['Runs.Remaining'] = new_rows['Innings.Total.Runs']
new_rows['Wickets.in.Hand'] = 10
new_rows['Total.Overs'] = 50

# Append the new rows to the original DataFrame
df_10_with_new_rows = pd.concat([df_10_copy, new_rows], ignore_index=True)

# Sorting the DataFrame wrt match and remaining overs
df_10_with_new_rows.sort_values(by=['Match', 'Overs.Remaining'], ascending = False,inplace=True)

df_10_with_new_rows.reset_index(drop=True, inplace=True)

df_10 = df_10_with_new_rows

df_9_copy = df_9.copy()
rows_with_49_overs = df_9[df_9['Overs.Remaining'] == 49]
new_rows = rows_with_49_overs.copy()
new_rows['Overs.Remaining'] = 50
new_rows['Runs.Remaining'] = new_rows['Innings.Total.Runs']
new_rows['Wickets.in.Hand'] = 9
new_rows['Total.Overs'] = 50
df_9_with_new_rows = pd.concat([df_9_copy, new_rows], ignore_index=True)
df_9_with_new_rows.sort_values(by=['Match', 'Overs.Remaining'], ascending = False,inplace=True)
df_9_with_new_rows.reset_index(drop=True, inplace=True)
df_9 = df_9_with_new_rows

df_8_copy = df_8.copy()
rows_with_49_overs = df_8[df_8['Overs.Remaining'] == 49]
new_rows = rows_with_49_overs.copy()
new_rows['Overs.Remaining'] = 50
new_rows['Runs.Remaining'] = new_rows['Innings.Total.Runs']
new_rows['Wickets.in.Hand'] = 8
new_rows['Total.Overs'] = 50
df_8_with_new_rows = pd.concat([df_8_copy, new_rows], ignore_index=True)
df_8_with_new_rows.sort_values(by=['Match', 'Overs.Remaining'], ascending = False,inplace=True)
df_8_with_new_rows.reset_index(drop=True, inplace=True)
df_8 = df_8_with_new_rows

df_7_copy = df_7.copy()
rows_with_49_overs = df_7[df_7['Overs.Remaining'] == 49]
new_rows = rows_with_49_overs.copy()
new_rows['Overs.Remaining'] = 50
new_rows['Runs.Remaining'] = new_rows['Innings.Total.Runs']
new_rows['Wickets.in.Hand'] = 7
new_rows['Total.Overs'] = 50
df_7_with_new_rows = pd.concat([df_7_copy, new_rows], ignore_index=True)
df_7_with_new_rows.sort_values(by=['Match', 'Overs.Remaining'], ascending = False,inplace=True)
df_7_with_new_rows.reset_index(drop=True, inplace=True)
df_7 = df_7_with_new_rows

df_6_copy = df_6.copy()
rows_with_49_overs = df_6[df_6['Overs.Remaining'] == 49]
new_rows = rows_with_49_overs.copy()
new_rows['Overs.Remaining'] = 50
new_rows['Runs.Remaining'] = new_rows['Innings.Total.Runs']
new_rows['Wickets.in.Hand'] = 6
new_rows['Total.Overs'] = 50
df_6_with_new_rows = pd.concat([df_6_copy, new_rows], ignore_index=True)
df_6_with_new_rows.sort_values(by=['Match', 'Overs.Remaining'], ascending = False,inplace=True)
df_6_with_new_rows.reset_index(drop=True, inplace=True)
df_6 = df_6_with_new_rows

df_5_copy = df_5.copy()
rows_with_49_overs = df_5[df_5['Overs.Remaining'] == 49]
new_rows = rows_with_49_overs.copy()
new_rows['Overs.Remaining'] = 50
new_rows['Runs.Remaining'] = new_rows['Innings.Total.Runs']
new_rows['Wickets.in.Hand'] = 5
new_rows['Total.Overs'] = 50
df_5_with_new_rows = pd.concat([df_5_copy, new_rows], ignore_index=True)
df_5_with_new_rows.sort_values(by=['Match', 'Overs.Remaining'], ascending = False,inplace=True)
df_5_with_new_rows.reset_index(drop=True, inplace=True)
df_5 = df_5_with_new_rows

df_4_copy = df_4.copy()
rows_with_49_overs = df_4[df_4['Overs.Remaining'] == 49]
new_rows = rows_with_49_overs.copy()
new_rows['Overs.Remaining'] = 50
new_rows['Runs.Remaining'] = new_rows['Innings.Total.Runs']
new_rows['Wickets.in.Hand'] = 4
new_rows['Total.Overs'] = 50
df_4_with_new_rows = pd.concat([df_4_copy, new_rows], ignore_index=True)
df_4_with_new_rows.sort_values(by=['Match', 'Overs.Remaining'], ascending = False,inplace=True)
df_4_with_new_rows.reset_index(drop=True, inplace=True)
df_4 = df_4_with_new_rows

df_3_copy = df_3.copy()
rows_with_49_overs = df_3[df_3['Overs.Remaining'] == 49]
new_rows = rows_with_49_overs.copy()
new_rows['Overs.Remaining'] = 50
new_rows['Runs.Remaining'] = new_rows['Innings.Total.Runs']
new_rows['Wickets.in.Hand'] = 3
new_rows['Total.Overs'] = 50
df_3_with_new_rows = pd.concat([df_3_copy, new_rows], ignore_index=True)
df_3_with_new_rows.sort_values(by=['Match', 'Overs.Remaining'], ascending = False,inplace=True)
df_3_with_new_rows.reset_index(drop=True, inplace=True)
df_3 = df_3_with_new_rows

df_2_copy = df_2.copy()
rows_with_49_overs = df_2[df_2['Overs.Remaining'] == 49]
new_rows = rows_with_49_overs.copy()
new_rows['Overs.Remaining'] = 50
new_rows['Runs.Remaining'] = new_rows['Innings.Total.Runs']
new_rows['Wickets.in.Hand'] = 2
new_rows['Total.Overs'] = 50
df_2_with_new_rows = pd.concat([df_2_copy, new_rows], ignore_index=True)
df_2_with_new_rows.sort_values(by=['Match', 'Overs.Remaining'], ascending = False,inplace=True)
df_2_with_new_rows.reset_index(drop=True, inplace=True)
df_2 = df_2_with_new_rows

df_1_copy = df_1.copy()
rows_with_49_overs = df_1[df_1['Overs.Remaining'] == 49]
new_rows = rows_with_49_overs.copy()
new_rows['Overs.Remaining'] = 50
new_rows['Runs.Remaining'] = new_rows['Innings.Total.Runs']
new_rows['Wickets.in.Hand'] = 1
new_rows['Total.Overs'] = 50
df_1_with_new_rows = pd.concat([df_1_copy, new_rows], ignore_index=True)
df_1_with_new_rows.sort_values(by=['Match', 'Overs.Remaining'], ascending = False,inplace=True)
df_1_with_new_rows.reset_index(drop=True, inplace=True)
df_1 = df_1_with_new_rows

df_0_copy = df_0.copy()
rows_with_49_overs = df_0[df_0['Overs.Remaining'] == 49]
new_rows = rows_with_49_overs.copy()
new_rows['Overs.Remaining'] = 50
new_rows['Runs.Remaining'] = new_rows['Innings.Total.Runs']
new_rows['Wickets.in.Hand'] = 0
new_rows['Total.Overs'] = 50
df_0_with_new_rows = pd.concat([df_0_copy, new_rows], ignore_index=True)
df_0_with_new_rows.sort_values(by=['Match', 'Overs.Remaining'], ascending = False,inplace=True)
df_0_with_new_rows.reset_index(drop=True, inplace=True)
df_0 = df_0_with_new_rows

###############################################################################################################################

#                                           The Important Functions

def Z_val(w, df):
    # Average number of runs scored with w wickets over all overs
    z0 = df['Runs.Remaining'].mean()
    return z0

#print(f"The initial guesses for Z are {[0, Z_val(1, df_1), Z_val(2, df_2), Z_val(3, df_3), Z_val(4, df_4), Z_val(5, df_5), Z_val(6, df_6), Z_val(7, df_7), Z_val(8, df_8), Z_val(9, df_9), Z_val(10, df_10)]}")

 # Intial guesses for Z_0
def Z_0(w):
    z_0 = [0, Z_val(1, df_1), Z_val(2, df_2), Z_val(3, df_3), Z_val(4, df_4), Z_val(5, df_5), Z_val(6, df_6), Z_val(7, df_7), Z_val(8, df_8), Z_val(9, df_9), Z_val(10, df_10)]
    x = z_0[w]
    #print(f"Z_0 initial guess is {z_0}")
    return x 
    
# Initial Guesses for L_0
def L_0(w, df): 
    l0 = df['Runs.Remaining'].sum()/df['Overs.Remaining'].sum()
    return l0

#print(f"The initial guesses for L are {[0, L_0(1, df_1), L_0(2, df_2), L_0(3, df_3), L_0(4, df_4), L_0(5, df_5), L_0(6, df_6), L_0(7, df_7), L_0(8, df_8), L_0(9, df_9), L_0(10, df_10)]}")

# Initial Guesses for L_0
def L_arr(w): 
    L = [0, L_0(1, df_1), L_0(2, df_2), L_0(3, df_3), L_0(4, df_4), L_0(5, df_5), L_0(6, df_6), L_0(7, df_7), L_0(8, df_8), L_0(9, df_9), L_0(10, df_10)]
    return L[w]

# The score function as described in the assignment
def Z(u, w, z_0, L_0):
    if z_0 !=0 and u  !=0 :
        z = z_0 * (1 - math.exp(-L_0 * u/z_0))
        return z
    if u == 0 or z_0 == 0:
        return 0
    

# The loss function for a single datapoint :- 
def loss(z, y):
    loss = (z+1)*math.log((z+1)/(y+1)) - z + y
    return loss

# The total loss function :- Accounts for averge loss over all the datapoints :-
def total_loss(df, w, z_0, L_0):
    total_loss = 0
    #z_0 = Z_0(w)
    #L_0 = L_0(w, df)
    #print(f"DataFrame length: {len(df)}")
    for i in range(len(df)):
        u = df['Overs.Remaining'][i]
        z_ = Z(u, w, z_0, L_0)# Best parameters
        y = df['Runs.Remaining'][i]
        total_loss = total_loss + loss(z_,y)
    return total_loss/len(df)

#total_loss = total_loss(df_10, 50, 10, L_arr(10))
#print(total_loss)
#L_0_10 = L_0(10, df_10)
#print(L_0_10)

# Wrapper function needed for Optimzer in Q1
def total_loss_wrapper(params, df, w):
    z_0, l_0 = params
    return total_loss(df, w, z_0, l_0)

# Wrapper function needed for Optimzer in Q2
def total_loss_wrapper_1(params, df, w, l_0):
    z_0 = params
    return total_loss(df, w, z_0, l_0)

###############################################################################################################################

#                                           Plotting the graph for Q1
                                                       
dfs = {0:df_0, 1: df_1, 2: df_2, 3: df_3, 4: df_4, 5: df_5, 6: df_6, 7: df_7, 8: df_8, 9: df_9, 10: df_10}

def plot_graph_Q1():
    plt.figure(figsize=(10, 7))
    z_opt_arr_1 = []
    L_opt_arr_1 = []

    for w in range(0, 11):
        df = dfs[w]
        z_0 = Z_0(w)
        l_0 = L_arr(w)
        initial_guess = [z_0, l_0]
        result = minimize(total_loss_wrapper, initial_guess, args=(df, w), method='BFGS')
        z_opt, L_opt = result.x
        L_opt_arr_1.append(L_opt)
        z_opt_arr_1.append(z_opt)
        # Calculate Z for U = [1, 2, ..., 50]
        U = list(range(0, 51))
        Z_values = []
        #Z_values.append(0)
        for u in U:
            Z_values.append(Z(u, w, z_opt, L_opt))
        
        
        # Plot Z vs U for this w
        plt.plot(U, Z_values, label=f'w={w}')

    # Configure the overall plot
    plt.title('Expected Runs vs Overs Remaining for Different Wickets Q1 (w)')
    plt.xlabel('Overs Remaining (U)')
    plt.ylabel('Expected Runs (Z)')
    plt.grid(True)
    plt.legend(title="Wickets in Hand (w)")
    #plt.show()
    print(f"The optimized z_0 array for problem 1 is {z_opt_arr_1}")
    print(f"The optimized L_0 array for problem 1 is {L_opt_arr_1}")

    return z_opt_arr_1, L_opt_arr_1
    
###############################################################################################################################

#                                           Plotting the graph for Q2
                                                   
def plot_graph_Q2(L_opt_arr_2):
    L = np.array(L_opt_arr_2)
    Wt = np.array([len(df_0)/len_df, len(df_1)/len_df, len(df_2)/len_df, len(df_3)/len_df, len(df_4)/len_df, len(df_5)/len_df, len(df_6)/len_df, len(df_7)/len_df, len(df_8)/len_df, len(df_9)/len_df, len(df_10)/len_df])
    L_avg = np.dot(L, Wt)  # Fixed l_0 for Q2 

    plt.figure(figsize=(10, 7))
    z_opt_arr_2 = []
    for w in range(0, 11):
        df = dfs[w]
        z_0 = Z_0(w)
        l_0 = L_avg

        initial_guess = [z_0]
        
        result = minimize(total_loss_wrapper_1, initial_guess, args=(df, w, l_0), method='BFGS')
        z_opt = result.x[0]

        z_opt_arr_2.append(z_opt)

        # Plotting for all overs remaining
        U = list(range(0, 51))
        Z_values = []

        #Z_values.append(0)
        for u in U:
            Z_values.append(Z(u, w, z_opt, L_avg))
        
        # Plot Z vs U for this w
        plt.plot(U, Z_values, label=f'w={w}')

    # Finally plotting
    plt.title('Expected Runs vs Overs Remaining for Different Wickets Q2 (w)')
    plt.xlabel('Overs Remaining (U)')
    plt.ylabel('Expected Runs (Z)')
    plt.grid(True)
    plt.legend(title="Wickets in Hand (w)")
    #plt.show()
    print(f"The optimized z_0 array for problem 2 is {z_opt_arr_2}")
    print(f"The optimized l_0 value for problem 2 is {L_avg}")

    return z_opt_arr_2, L_avg

###############################################################################################################################

#                                           Calling the Functions

if __name__ == '__main__':

    z_opt_arr_1, L_opt_arr_1 = plot_graph_Q1()
    z_opt_arr_2, L_avg = plot_graph_Q2(L_opt_arr_1)

###############################################################################################################################