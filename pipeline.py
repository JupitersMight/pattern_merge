import numpy as np
import pandas as pd
import math
from scipy.spatial import distance
from merge_patterns import merge_patterns

f = open("chess.txt", "r")
supports = [0.99, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10] #,0.90]
#min_sup = 0.30
nr_lines = 0
variables = []

files = ["chess.txt"]#["chess.txt", "accidents.txt", "connect.txt", "mushroom.txt", "pumsb.txt"]

'''
# Read file
for line in f:
    curr_transaction = line.split(" ")
    for value in curr_transaction:
        processed_value = value.replace("\n","")
        if processed_value not in variables:
            variables.append(processed_value)
    nr_lines += 1
f.seek(0)
# Create dataframe
df = pd.DataFrame(data=np.zeros((nr_lines, len(variables)), dtype=int), columns=variables)
curr_row_pointer = 0
for line in f:
    curr_transaction = line.split(" ")
    for variable in curr_transaction:
        processed_value = variable.replace("\n", "")
        df.at[curr_row_pointer, processed_value] = 1
    curr_row_pointer += 1
'''

for ficheiro in files:
    print(ficheiro)
    f = open(ficheiro, "r")
    nr_lines = 0
    variables = []
    # Read file
    for line in f:
        curr_transaction = line.split(" ")
        for value in curr_transaction:
            processed_value = value.replace("\n","")
            if processed_value not in variables:
                variables.append(processed_value)
        nr_lines += 1
    f.seek(0)
    # Create dataframe
    df = pd.DataFrame(data=np.zeros((nr_lines, len(variables)), dtype=int), columns=variables)
    curr_row_pointer = 0
    for line in f:
        curr_transaction = line.split(" ")
        for variable in curr_transaction:
            processed_value = variable.replace("\n", "")
            df.at[curr_row_pointer, processed_value] = 1
        curr_row_pointer += 1

    for sup in supports:
        print("#########################################################################################")
        print("Current support: " + str(sup))
        # Filter out columns with support less than threshold
        columns_to_use = []
        for col in df.columns:
            counter = 0
            for value in df[col]:
                if value == 1:
                    counter += 1
            if counter/nr_lines > sup:
                columns_to_use.append(col)

        copy_df = df[columns_to_use]
        print(len(copy_df.columns))

        # Calculate similarity matrix
        similarity_matrix = []
        for col in copy_df.columns:
            similarity_matrix.append([])

        for pointer_1 in range(0,len(copy_df.columns)):
            for pointer_2 in range(0, len(copy_df.columns)):
                dice_dist = distance.jaccard(copy_df[copy_df.columns[pointer_1]], copy_df[copy_df.columns[pointer_2]])
                similarity_matrix[pointer_1].append(dice_dist)

        for i in range(0, len(similarity_matrix)):
            for j in range(0, len(similarity_matrix)):
                if similarity_matrix[i][j] == 0.0:
                    similarity_matrix[i][j] = math.inf
                if similarity_matrix[i][j] == 1.0:
                    similarity_matrix[i][j] = -math.inf

        #pattern_mining(copy_df, sup)
        '''
        K = len(similarity_matrix)
        a = int(round(math.log(K, 2), 0))
        clusters = cost_optimal_clustering(similarity_matrix)
        print("Number of clusters: "+str(len(clusters)))
        print(clusters)
        if len(clusters) == 1:
            continue
        s = False
        for c in clusters:
            if len(c) > 11:
                s = True
                break
        if s:
            continue
        patterns = []
        individual_clusters = []
        counter = 0
        for cluster in clusters:
            columns_to_use = []
            for value in cluster:
                columns_to_use.append(copy_df.columns[value])
            df_copy = copy_df[columns_to_use].copy()
            p_result = pattern_mining(df_copy, sup)
            print(len(p_result))
            individual_clusters.append(p_result)
            patterns.extend(p_result)
            counter += 1

        min_size_of_vocab_per_cluster = 1.0
        min_vob_in_cluster = 2
        patterns_to_merge = []
        new_individual_clusters = []
        for i in range(0, len(individual_clusters)):
            size_of_vocab_in_cluster = len(clusters[i])
            i_c = []
            if size_of_vocab_in_cluster >= min_vob_in_cluster:
                for val in individual_clusters[i]:
                    if len(val["columns"]) / size_of_vocab_in_cluster >= min_size_of_vocab_per_cluster:
                        patterns_to_merge.append(val)
                        i_c.append(val)
                new_individual_clusters.append(i_c)

        combined_patterns = merge_patterns(patterns_to_merge, new_individual_clusters, nr_lines, sup, patterns)
        #print(len(combined_patterns))
        '''


'''
# Filter out columns with support less than threshold
columns_to_use = []
for col in df.columns:
    counter = 0
    for value in df[col]:
        if value == 1:
            counter += 1
    if counter/nr_lines > min_sup:
        columns_to_use.append(col)

df = df[columns_to_use]
print(len(df.columns))

# Calculate similarity matrix
similarity_matrix = []
for col in df.columns:
    similarity_matrix.append([])

for pointer_1 in range(0,len(df.columns)):
    for pointer_2 in range(0, len(df.columns)):
        dice_dist = distance.jaccard(df[df.columns[pointer_1]], df[df.columns[pointer_2]])
        similarity_matrix[pointer_1].append(dice_dist)

for i in range(0, len(similarity_matrix)):
    for j in range(0, len(similarity_matrix)):
        if similarity_matrix[i][j]==0.0:
            similarity_matrix[i][j] = math.inf
        if similarity_matrix[i][j]==1.0:
            similarity_matrix[i][j] = -math.inf

#isolated_patterns = pattern_mining(df, min_sup)
#print(isolated_patterns)
#print("Number of patterns found in isolated pattern mining: "+str(len(isolated_patterns)))


K = len(similarity_matrix)
a = int(round(math.log(K, 2), 0))
clusters = cost_optimal_clustering(similarity_matrix)
print("Number of clusters: "+str(len(clusters)))
#print(clusters)
patterns = []
individual_clusters = []
counter = 0
for cluster in clusters:
    columns_to_use = []
    for value in cluster:
        columns_to_use.append(df.columns[value])
    df_copy = df[columns_to_use].copy()
    p_result = pattern_mining(df_copy, min_sup)
    print(len(p_result))
    individual_clusters.append(p_result)
    patterns.extend(p_result)
    counter += 1

min_size_of_vocab_per_cluster = 1.0
patterns_to_merge = []
new_individual_clusters = []
for i in range(0, len(individual_clusters)):
    size_of_vocab_in_cluster = len(clusters[i])
    i_c = []
    for val in individual_clusters[i]:
        if len(val["columns"]) / size_of_vocab_in_cluster >= min_size_of_vocab_per_cluster:
            patterns_to_merge.append(val)
            i_c.append(val)
    new_individual_clusters.append(i_c)


combined_patterns = merge_patterns(patterns_to_merge, new_individual_clusters, nr_lines, min_sup, patterns)
print(len(combined_patterns))


#isolated_patterns = pattern_mining(df, min_sup)
#print("Number of patterns found in isolated pattern mining: "+str(len(isolated_patterns)))

'''
