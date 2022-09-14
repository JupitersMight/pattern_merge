import numpy as np
import pandas
import pandas as pd
import math
from apriori_numerical import runApriori
import time
from sklearn.cluster import AgglomerativeClustering
from merge_patterns import merge_patterns
from merge_patterns import merge_patterns_a
import random
from scipy.stats import binom
from scipy.special import betainc
from scipy.special import betaincinv

#x = betaincinv(100, 1300, 0.33*0.33*0.33)
#x = binom.ppf(q=0.05, n=1300, p=0.33*0.33*0.33)
#y = betainc(100, 1300, 0.33)
#Y = 1 - binom.cdf(10, 1300, 0.33*0.33*0.33)
#y = 0


def average_link(active_set, sim_matrix, max_num_per_cluster, broken_down):
    g1 = 0
    g2 = 0
    curr_best = math.inf
    for key1 in active_set.keys():
        for key2 in active_set.keys():
            if key1 == key2:
                continue
            x = active_set[key1]
            y = active_set[key2]
            if len(x)+len(y) > max_num_per_cluster:
                continue
            skip = False
            for b in broken_down:
                if key1 in b.keys() and key2 in b.keys():
                    if set(b[key1]) == set(x) and set(b[key2]) == set(y):
                        skip = True
                        break
            if skip:
                continue
            sumation = 0
            for val1 in x:
                for val2 in y:
                    sumation += sim_matrix[val1][val2]
            sumation /= (len(x)*len(y))
            if sumation < curr_best:
                curr_best = sumation
                g1 = key1
                g2 = key2
    return g1, g2, curr_best


def agglomerative_clustering(sim_matrix, max_num_per_cluster):
    active_set = {}
    finalized_clusters = {}
    broken_down = []
    combined_clusters = []
    for n in range(len(sim_matrix)):
        active_set[n] = [n]

    while len(active_set.keys()) > 1:
        g1, g2, dist = average_link(active_set, sim_matrix, max_num_per_cluster, broken_down)
        if dist == math.inf:
            #break down last item formed
            inverted_list = combined_clusters[::-1]
            for c in inverted_list:
                skip = False
                for val in c.keys():
                    if val in finalized_clusters.keys():
                        skip = True
                        break
                if skip:
                    continue
                for val in c.keys():
                    if val in active_set.keys():
                        active_set.pop(val)
                    active_set[val] = c[val]
                list_keys = list(c.keys())
                broken_down.append({list_keys[0]: c[list_keys[0]], list_keys[1]: c[list_keys[1]]})

                for i in range(len(combined_clusters)):
                    val = combined_clusters[i]
                    if set(list(val.keys())) == set(list(c.keys())):
                        keys = list(val.keys())
                        if set(val[keys[0]]) == set(c[keys[0]]) and set(val[keys[1]]) == set(c[keys[1]]):
                            del combined_clusters[i]
                            break
                break

        else:
            #remove g1 and g2 from active and add
            combined = np.concatenate((active_set[g1], active_set[g2]))
            if len(combined) < max_num_per_cluster:
                combined_clusters.append({g1:active_set[g1],g2:active_set[g2]})
                active_set[g1] = combined
            else:
                active_set.pop(g1)
                finalized_clusters[g1] = combined

            active_set.pop(g2)

    return finalized_clusters


def construct_distance_matrix(dataset, type):
    similarity_matrix = []
    for _ in dataset.columns:
        similarity_matrix.append([])

    for pointer_1 in range(0, len(dataset.columns)):
        for pointer_2 in range(0, len(dataset.columns)):
            if type == "order-preserving":
                distance = order_preserving(dataset[dataset.columns[pointer_1]], dataset[dataset.columns[pointer_2]])
            else:
                distance = constant(np.array(dataset[dataset.columns[pointer_1]]), np.array(dataset[dataset.columns[pointer_2]]))

            similarity_matrix[pointer_1].append(distance)

    for i in range(0, len(similarity_matrix)):
        for j in range(0, len(similarity_matrix)):
            if i == j:
                similarity_matrix[i][j] = 0
            #    similarity_matrix[i][j] = math.inf
            #if similarity_matrix[i][j] == 0.0:
            #    similarity_matrix[i][j] = math.inf
            #if similarity_matrix[i][j] == 1.0:
            #    similarity_matrix[i][j] = -math.inf

    return similarity_matrix


def constant(v1, v2):
    if len(v1) != len(v2):
        return 0
    unique_pairs = {}

    for i in range(len(v1)):
        if v1[i] == "?" or v2[i] == "?":
            continue
        pair = (v1[i], v2[i])
        if pair in unique_pairs.keys():
            unique_pairs[pair] += 1
        else:
            unique_pairs[pair] = 1

    if len(unique_pairs) == 0:
        return 1

    #argmax = max(unique_pairs.values())/len(v1)

    somation = 0
    u_v1 = {}
    max_entropy = 0
    for val in v1:
        max_entropy += 1/len(v1) * math.log(1/len(v1), 2)
        if val != "?":
            if val not in u_v1.keys():
                u_v1[val] = 1
            else:
                u_v1[val] += 1
    max_entropy *= -1

    u_v2 = {}
    for val in v2:
        if val != "?":
            if val not in u_v2.keys():
                u_v2[val] = 1
            else:
                u_v2[val] += 1


    #u_v1 = np.unique(np.array([i for i in v1 if i != "?"]), return_counts=True)
    #u_v2 = np.unique(np.array([i for i in v2 if i != "?"]), return_counts=True)
    entropy = 0
    for pair in unique_pairs.keys():
        count_1 = u_v1[pair[0]]
        count_2 = u_v2[pair[1]]
        somation += unique_pairs[pair] / max(count_1, count_2)
        entropy += (unique_pairs[pair]/len(v1)) * math.log(unique_pairs[pair]/len(v1), 2)
    entropy *= -1

    return 1 - ((1 - entropy / max_entropy) * (somation / len(unique_pairs)))


def order_preserving(v1, v2):
    if len(v1) != len(v2):
        return 0

    count_v1_v2 = 0
    count_v2_v1 = 0
    for i in range(len(v1)):
        if v1[i] == np.nan or v2[i] == np.nan:
            continue
        if v1[i] > v2[i]:
            count_v1_v2 += 1
        else:
            count_v2_v1 += 1
    return max(count_v1_v2, count_v2_v1) / len(v1)




a1 = [1, 1, 1, 1, 1, 1, 1]
a2 = [1, 1, 1, 1, 1, 1, 1]
1
print(constant(a1, a2))

a1 = [1, 1, 1, 1, 1, 1, 2]
a2 = [1, 1, 1, 1, 1, 1, 2]
0.78
print(constant(a1, a2))

a1 = [1, 1, 1, 1, 2, 2, 2]
a2 = [1, 1, 1, 1, 2, 2, 2]
0.64
print(constant(a1, a2))

a1 = [1, 1, 1, 2, 2, 3, 3]
a2 = [1, 1, 1, 4, 4, 3, 3]
0.44
print(constant(a1, a2))

a1 = [1, 1, 1, 1, 2, 3, 4]
a2 = [1, 1, 1, 1, 2, 3, 4]
0.40
print(constant(a1, a2))

a1 = [1, 1, 1, 1, 1, 2, 2]
a2 = [1, 1, 1, 2, 2, 3, 3]

print(constant(a1, a2))

a1 = [1, 2, 3, 4, 5, 6, 7]
a2 = [1, 2, 3, 4, 5, 6, 7]

print(constant(a1, a2))



d = pandas.DataFrame(data=[
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 2],
    [1, 1, 1, 1, 1, 1, 2],
    [1, 1, 1, 1, 2, 2, 2],
    [1, 1, 1, 1, 2, 2, 2],
    [1, 1, 1, 2, 2, 3, 3],
    [1, 1, 1, 4, 4, 3, 3],
    [1, 1, 1, 1, 2, 3, 4],
    [1, 1, 1, 1, 2, 3, 4],
    [1, 1, 1, 1, 1, 2, 2],
    [1, 1, 1, 2, 2, 3, 3],
    [1, 2, 3, 4, 5, 6, 7],
    [1, 2, 3, 4, 5, 6, 7]
]).transpose()

similarity_m = construct_distance_matrix(d, "constant")


datasets = ["datasets/pumsb.txt"] #, "datasets/accidents.txt", "datasets/pumsb.txt"]

for file in datasets:
    print("Current file: " + file)
    f = open(file, "r")
    nr_lines = 0
    variables = []

    # Read file
    for line in f:
        curr_transaction = line.split(" ")
        for value in curr_transaction:
            processed_value = value.replace("\n", "")
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

    df = df.drop(columns=[""])

    ones = 0
    for col in df.columns:
        ones += len(df[df[col] == 1])

    number_of_variables = [4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72]
    max_per_cluster = 4
    total_entries = df.shape[0] * df.shape[1]
    p = ones/total_entries
    n = len(df[df.columns[0]])
    p = pow(p, max_per_cluster)
    min_support = binom.ppf(q=0.05, n=n, p=p)/n

    print("Minimum support : "+str(min_support))

    for N in number_of_variables:
        print("#########################################################################################")
        print("Number of variables: " + str(N))

        print()
        print()
        print("#########################################################################################")
        print("#########################################################################################")
        print("Similar")
        print("#########################################################################################")

        similarity_m_times = []
        hierarchical_times = []
        apriori_clusters_times = []
        merging_times = []
        intersections = []
        variables_subsets = []
        variables_per_pattern = []
        observations_per_pattern = []

        #hierarchichal clustering properties
        patterns_discovered = []

        stored_dfs = []
        for counter in range(10):
            columns_to_use = random.sample(list(df.columns), N)
            copy_df = df[columns_to_use]
            stored_dfs.append(copy_df)

            start = time.time_ns()
            copy_df = copy_df.replace(0, "?")
            similarity_m = construct_distance_matrix(copy_df, "constant")

            s2 = time.time_ns()
            similarity_m_times.append((s2 - start) / 1000000000)
            print("similarity matrix: " + str((s2 - start) / 1000000000) + " seconds")

            aglo_start = time.time_ns()

            clusters = agglomerative_clustering(similarity_m, max_per_cluster)
            print(clusters)


            aglo_end = time.time_ns()
            hierarchical_times.append((aglo_end - aglo_start) / 1000000000)
            print("Hierarchical clustering: " + str((aglo_end - aglo_start) / 1000000000) + " seconds")

            s1 = time.time_ns()
            clusters_apriori = []
            print("Number of clusters : " + str(len(clusters.keys())))

            for cluster in clusters.keys():
                columns_to_use = []
                for value in clusters[cluster]:
                    columns_to_use.append(copy_df.columns[value])
                df_copy = copy_df[columns_to_use].copy()
                y = runApriori(data_iter=df_copy, class_vector=0, minSupport=min_support, minLift=1.3)
                y = merge_patterns([y], nr_lines, min_support)
                clusters_apriori.append(y)

            end = time.time_ns()
            apriori_clusters_times.append((end - s1) / 1000000000)
            print("Apriori with clusters: " + str((end - s1) / 1000000000) + " seconds")
            print("#########################################################################################")
            print("Intermediate analysis")
            nr_patterns_bef = []
            size_obs_bef = []
            size_vars = []
            for cluster in clusters_apriori:
                nr_patterns_bef.append(len(cluster))
                variables_p = []
                obs_p = []
                for pattern in cluster:
                    variables_p.append(len(pattern["columns"]))
                    obs_p.append(len(pattern["indexes"]))
                size_obs_bef.append(round((sum(obs_p)/(len(cluster) if len(cluster) > 0 else 1)), 0))
                size_vars.append(round(sum(variables_p)/(len(cluster) if len(cluster) > 0 else 1), 2))

            observations_per_pattern.append(sum(size_obs_bef)/len(clusters_apriori))
            variables_per_pattern.append(sum(size_vars)/len(clusters_apriori))
            print("Number of patterns per cluster: "+ str(nr_patterns_bef))
            print("Average number of observations per cluster: "+ str(size_obs_bef))
            print("Average number of variables per cluster: "+ str(size_vars))
            print("End of intermediate analysis")
            print("#########################################################################################")
            start_merge = time.time_ns()
            intermediate_intersections = []
            intermediate_variables = []
            result = merge_patterns_a(clusters_apriori, nr_lines, min_support, intermediate_intersections, intermediate_variables)
            intersections.append(sum(intermediate_intersections))
            variables_subsets.append(sum(intermediate_variables))
            end_merge = time.time_ns()
            merging_times.append((end_merge - start_merge) / 1000000000)

            patterns_discovered.append(len(result))
            print("Merging with clusters: " + str((end_merge - start_merge) / 1000000000) + " seconds")
            print("Combined time: " + str((end_merge - start) / 1000000000) + " seconds")

        print("#########################################################################################")
        print("#########################################################################################")
        print("#########################################################################################")
        print("For "+str(N)+" variables the time were:")
        print("Similarity Matrix")
        print(str(similarity_m_times))
        print("Hierarchical clustering")
        print(str(hierarchical_times))
        print("Apriori with clusters")
        print(str(apriori_clusters_times))
        print("Merging of patterns")
        print(str(merging_times))

        print("Number of intersections")
        print(str(intersections))
        print("Number of variables tested for subsets")
        print(str(variables_subsets))
        print("Number of observations per pattern")
        print(str(observations_per_pattern))
        print("Number of variables per pattern")
        print(str(variables_per_pattern))

        print("Number of patterns discovered")
        print(str(patterns_discovered))
        print("#########################################################################################")
        print("#########################################################################################")
        print("#########################################################################################")

        print("#########################################################################################")
        print("Dissimilar")
        print("#########################################################################################")
        similarity_m_times = []
        hierarchical_times = []
        apriori_clusters_times = []
        merging_times = []
        intersections = []
        variables_subsets = []
        variables_per_pattern = []
        observations_per_pattern = []


        #hierarchichal clustering properties
        patterns_discovered = []

        for s_df in stored_dfs:
            copy_df = s_df

            start = time.time_ns()
            copy_df = copy_df.replace(0, "?")
            similarity_m = construct_distance_matrix(copy_df, "constant")

            for i in range(0, len(similarity_m)):
                for j in range(0, len(similarity_m)):
                    if i == j:
                        similarity_m[i][j] = 0
                    else:
                        similarity_m[i][j] = 1 - similarity_m[i][j]

            s2 = time.time_ns()
            similarity_m_times.append((s2 - start) / 1000000000)
            print("similarity matrix: " + str((s2 - start) / 1000000000) + " seconds")

            aglo_start = time.time_ns()

            clusters = agglomerative_clustering(similarity_m, max_per_cluster)
            print(clusters)

            aglo_end = time.time_ns()
            hierarchical_times.append((aglo_end - aglo_start) / 1000000000)
            print("Hierarchical clustering: " + str((aglo_end - aglo_start) / 1000000000) + " seconds")

            s1 = time.time_ns()
            clusters_apriori = []
            print("Number of clusters : " + str(len(clusters.keys())))
            for cluster in clusters.keys():
                columns_to_use = []
                for value in clusters[cluster]:
                    columns_to_use.append(copy_df.columns[value])
                df_copy = copy_df[columns_to_use].copy()
                y = runApriori(data_iter=df_copy, class_vector=0, minSupport=min_support, minLift=1.3)
                y = merge_patterns([y], nr_lines, min_support)
                clusters_apriori.append(y)

                #print(len(y))
            end = time.time_ns()
            apriori_clusters_times.append((end - s1) / 1000000000)
            print("Apriori with clusters: " + str((end - s1) / 1000000000) + " seconds")

            print("Intermediate analysis")
            nr_patterns_bef = []
            size_obs_bef = []
            size_vars = []
            for cluster in clusters_apriori:
                nr_patterns_bef.append(len(cluster))
                variables_p = []
                obs_p = []
                for pattern in cluster:
                    variables_p.append(len(pattern["columns"]))
                    obs_p.append(len(pattern["indexes"]))
                size_obs_bef.append(round((sum(obs_p)/(len(cluster) if len(cluster) > 0 else 1)), 0))
                size_vars.append(round(sum(variables_p)/(len(cluster) if len(cluster) > 0 else 1), 2))


            observations_per_pattern.append(sum(size_obs_bef)/len(clusters_apriori))
            variables_per_pattern.append(sum(size_vars)/len(clusters_apriori))
            print("Number of patterns per cluster: "+ str(nr_patterns_bef))
            print("Average number of observations per cluster: "+ str(size_obs_bef))
            print("Average number of variables per cluster: "+ str(size_vars))
            print("End of intermediate analysis")


            start_merge = time.time_ns()
            intermediate_intersections = []
            intermediate_variables = []
            result = merge_patterns_a(clusters_apriori, nr_lines, min_support, intermediate_intersections, intermediate_variables)
            intersections.append(sum(intermediate_intersections))
            variables_subsets.append(sum(intermediate_variables))
            end_merge = time.time_ns()
            merging_times.append((end_merge - start_merge) / 1000000000)

            patterns_discovered.append(len(result))
            print("Merging with clusters: " + str((end_merge - start_merge) / 1000000000) + " seconds")
            print("Combined time: " + str((end_merge - start) / 1000000000) + " seconds")

        print("#########################################################################################")
        print("#########################################################################################")
        print("#########################################################################################")
        print("For "+str(N)+" variables the time were:")
        print("Similarity Matrix")
        print(str(similarity_m_times))
        print("Hierarchical clustering")
        print(str(hierarchical_times))
        print("Apriori with clusters")
        print(str(apriori_clusters_times))
        print("Merging of patterns")
        print(str(merging_times))

        print("Number of intersections")
        print(str(intersections))
        print("Number of variables tested for subsets")
        print(str(variables_subsets))
        print("Number of observations per pattern")
        print(str(observations_per_pattern))
        print("Number of variables per pattern")
        print(str(variables_per_pattern))

        print("Number of patterns discovered")
        print(str(patterns_discovered))
        print("#########################################################################################")
        print("#########################################################################################")
        print("#########################################################################################")

        print("#########################################################################################")
        print("Random")
        print("#########################################################################################")

        similarity_m_times = []
        hierarchical_times = []
        apriori_clusters_times = []
        merging_times = []
        intersections = []
        variables_subsets = []
        variables_per_pattern = []
        observations_per_pattern = []


        #hierarchichal clustering properties
        patterns_discovered = []

        for s_df in stored_dfs:

            copy_df = s_df

            start = time.time_ns()
            copy_df = copy_df.replace(0, "?")
            similarity_m = construct_distance_matrix(copy_df, "constant")

            for i in range(0, len(similarity_m)):
                for j in range(0, len(similarity_m)):
                    if i == j:
                        similarity_m[i][j] = 0
                    else:
                        similarity_m[i][j] = 1 - similarity_m[i][j]

            s2 = time.time_ns()
            similarity_m_times.append((s2 - start) / 1000000000)
            print("similarity matrix: " + str((s2 - start) / 1000000000) + " seconds")

            aglo_start = time.time_ns()

            #clusters = agglomerative_clustering(similarity_m, 4)
            columns_to_use = random.sample(list(copy_df.columns), N)
            clusters = {}
            next = -1
            for i in range(N):
                if i % max_per_cluster == 0:
                    next += 1
                if next not in clusters.keys():
                    clusters[next] = [columns_to_use[i]]
                else:
                    clusters[next].append(columns_to_use[i])

            print(clusters)

            aglo_end = time.time_ns()
            hierarchical_times.append((aglo_end - aglo_start) / 1000000000)
            print("Hierarchical clustering: " + str((aglo_end - aglo_start) / 1000000000) + " seconds")

            s1 = time.time_ns()
            clusters_apriori = []
            print("Number of clusters : " + str(len(clusters.keys())))
            for cluster in clusters.keys():
                columns_to_use = clusters[cluster]
                df_copy = copy_df[columns_to_use].copy()
                y = runApriori(data_iter=df_copy, class_vector=0, minSupport=min_support, minLift=1.3)
                y = merge_patterns([y], nr_lines, min_support)
                clusters_apriori.append(y)

                #print(len(y))
            end = time.time_ns()
            apriori_clusters_times.append((end - s1) / 1000000000)
            print("Apriori with clusters: " + str((end - s1) / 1000000000) + " seconds")

            print("Intermediate analysis")
            nr_patterns_bef = []
            size_obs_bef = []
            size_vars = []
            for cluster in clusters_apriori:
                nr_patterns_bef.append(len(cluster))
                variables_p = []
                obs_p = []
                for pattern in cluster:
                    variables_p.append(len(pattern["columns"]))
                    obs_p.append(len(pattern["indexes"]))
                size_obs_bef.append(round((sum(obs_p)/(len(cluster) if len(cluster) > 0 else 1)), 0))
                size_vars.append(round(sum(variables_p)/(len(cluster) if len(cluster) > 0 else 1), 2))


            observations_per_pattern.append(sum(size_obs_bef)/len(clusters_apriori))
            variables_per_pattern.append(sum(size_vars)/len(clusters_apriori))
            print("Number of patterns per cluster: "+ str(nr_patterns_bef))
            print("Average number of observations per cluster: "+ str(size_obs_bef))
            print("Average number of variables per cluster: "+ str(size_vars))
            print("End of intermediate analysis")

            start_merge = time.time_ns()
            intermediate_intersections = []
            intermediate_variables = []
            result = merge_patterns_a(clusters_apriori, nr_lines, min_support, intermediate_intersections, intermediate_variables)
            intersections.append(sum(intermediate_intersections))
            variables_subsets.append(sum(intermediate_variables))
            end_merge = time.time_ns()
            merging_times.append((end_merge - start_merge) / 1000000000)

            patterns_discovered.append(len(result))
            print("Merging with clusters: " + str((end_merge - start_merge) / 1000000000) + " seconds")
            print("Combined time: " + str((end_merge - start) / 1000000000) + " seconds")

        print("#########################################################################################")
        print("#########################################################################################")
        print("#########################################################################################")
        print("For "+str(N)+" variables the time were:")
        print("Similarity Matrix")
        print(str(similarity_m_times))
        print("Hierarchical clustering")
        print(str(hierarchical_times))
        print("Apriori with clusters")
        print(str(apriori_clusters_times))
        print("Merging of patterns")
        print(str(merging_times))

        print("Number of intersections")
        print(str(intersections))
        print("Number of variables tested for subsets")
        print(str(variables_subsets))
        print("Number of observations per pattern")
        print(str(observations_per_pattern))
        print("Number of variables per pattern")
        print(str(variables_per_pattern))

        print("Number of patterns discovered")
        print(str(patterns_discovered))
        print("#########################################################################################")
        print("#########################################################################################")
        print("#########################################################################################")



