import matplotlib.pyplot as plt
import numpy as np
import os


directory = os.getcwd()

number_of_variables = [4,8,12,16,20]#,24,28,32,36,40,44,48]

'''
files = [
    str(directory)+"\\output\\4\\accident.txt",
    str(directory)+"\\output\\4\\chess.txt",
    str(directory)+"\\output\\4\\connect.txt",
    str(directory)+"\\output\\4\\mushroom.txt",
    str(directory)+"\\output\\4\\pumsb.txt"
]
file_output_names = [
    str(directory)+"\\output\\4\\accident",
    str(directory)+"\\output\\4\\chess",
    str(directory)+"\\output\\4\\connect",
    str(directory)+"\\output\\4\\mushroom",
    str(directory)+"\\output\\4\\pumsb"
]
'''

files = [
    str(directory)+"\\output\\4\\accident.txt",
    str(directory)+"\\output\\4\\chess.txt",
    str(directory)+"\\output\\4\\connect.txt",
    str(directory)+"\\output\\4\\mushroom.txt",
    str(directory)+"\\output\\4\\pumsb.txt"
]
file_output_names = [
    str(directory)+"\\output\\4\\accident",
    str(directory)+"\\output\\4\\chess",
    str(directory)+"\\output\\4\\connect",
    str(directory)+"\\output\\4\\mushroom",
    str(directory)+"\\output\\4\\pumsb"
]

for i in range(len(files)):
    file = files[i]
    counter = 0

    s_apriori_times = []
    d_apriori_times = []
    r_apriori_times = []

    s_merging_times = []
    d_merging_times = []
    r_merging_times = []


    "Number of variables:"
    #vezes 3
    "the time were:"
    "Apriori with clusters"
    "Merging of patterns"
    with open(file) as f:
        now = False
        apri = False
        merg = False
        for line in f:
            line = line.replace("\n","")
            if "the time were:" in line:
                now = True
            elif now:
                if "Apriori with clusters" in line:
                    apri = True
                elif apri:
                    line = line.replace("[","")
                    line = line.replace("]","")
                    line_splited = line.split(",")
                    a = []
                    for val in line_splited:
                        a.append(float(val))
                    if counter == 0:
                        s_apriori_times.append(sum(a)/len(a))
                    elif counter == 1:
                        d_apriori_times.append(sum(a)/len(a))
                    elif counter == 2:
                        r_apriori_times.append(sum(a)/len(a))
                    apri = False
                elif "Merging of patterns" in line:
                    merg = True
                elif merg:
                    line = line.replace("[","")
                    line = line.replace("]","")
                    line_splited = line.split(",")
                    a = []
                    for val in line_splited:
                        a.append(float(val))
                    if counter == 0:
                        s_merging_times.append(sum(a)/len(a))
                    elif counter == 1:
                        d_merging_times.append(sum(a)/len(a))
                    elif counter == 2:
                        r_merging_times.append(sum(a)/len(a))
                        counter = -1
                    counter += 1
                    merg = False
                    now = False

    plt.figure()
    plt.plot(number_of_variables, d_apriori_times, color='red', linewidth=1.5, label="Dissimilarity")
    plt.plot(number_of_variables, s_apriori_times, color='green', linewidth=1.5, label="Similarity")
    plt.plot(number_of_variables, r_apriori_times, color='yellow', linewidth=1.5, label="Random")
    plt.legend(loc='upper left')
    #print(d_apriori_times)
    #print(s_apriori_times)
    #print(r_apriori_times)
    #print("###########################################################")

    plt.xlabel("Number of items")
    plt.ylabel("Execution time (Seconds)")
    plt.title("Execution time of apriori")
    plt.savefig(file_output_names[i] + "_apriori.png", bbox_inches='tight')
    plt.show()

    plt.figure()
    plt.plot(number_of_variables, d_merging_times, color='red', linewidth=1.5, label="Dissimilarity")
    plt.plot(number_of_variables, s_merging_times, color='green', linewidth=1.5, label="Similarity")
    plt.plot(number_of_variables, r_merging_times, color='yellow', linewidth=1.5, label="Random")
    plt.legend(loc='upper left')

    #print(d_merging_times)
    #print(s_merging_times)
    #print(r_merging_times)

    plt.xlabel("Number of items")
    plt.ylabel("Execution time (Seconds)")
    plt.title("Execution time of Merging")
    plt.savefig(file_output_names[i] + "_merge.png", bbox_inches='tight')
    plt.show()

    plt.figure()
    plt.plot(number_of_variables, np.add(d_apriori_times, d_merging_times), color='red', linewidth=1.5, label="Dissimilarity")
    plt.plot(number_of_variables, np.add(s_apriori_times, s_merging_times), color='green', linewidth=1.5, label="Similarity")
    plt.plot(number_of_variables, np.add(r_apriori_times, r_merging_times), color='yellow', linewidth=1.5, label="Random")
    plt.legend(loc='upper left')

    #print(d_merging_times)
    #print(s_merging_times)
    #print(r_merging_times)

    plt.xlabel("Number of items")
    plt.ylabel("Execution time (Seconds)")
    plt.title("Execution time combined")
    plt.savefig(file_output_names[i] + "_combined.png", bbox_inches='tight')
    plt.show()

