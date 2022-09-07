import sys
from itertools import chain, combinations
from collections import defaultdict
from optparse import OptionParser
import itertools
import numpy as np
import pandas as pd
from scipy.special import betainc


def Tsig(self, i):
        """ Calculates the statistical significance of a given subspace
        https://doi.org/10.1007/s10618-017-0521-2
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            Statistical significance of a subspace
        """
        p = 1.0
        # Constant
        if self.patterns[i]["type"] == "Constant":
            col_pos = 0
            for column in self.patterns[i]["columns"]:
                column_value = self.patterns[i]["column_values"][col_pos]
                counts = self.data[column]
                counter = 0
                for item in counts:
                    if self.border_values and (float(column_value) == item or float(column_value)+0.5 == item or float(column_value)-0.5 == item):
                        counter += 1
                    elif not self.border_values and ((float(column_value) - self.patterns[i]["noise"][col_pos]) <= item <= (float(column_value) + self.patterns[i]["noise"][col_pos])):
                        counter += 1

                p = p * (counter/self.size_of_dataset)
                col_pos += 1
        '''
        elif self.patterns[i]["type"] == "Additive" or self.patterns[i]["type"] == "Multiplicative":
            for column in self.patterns[i]["columns"]:
                uniques_col = self.patterns[i]["x_data"][column].unique()
                counts = self.data[column]
                counter = 0
                for item in counts:
                    if self.border_values:
                        for unique_value in uniques_col:
                            if float(unique_value) == item or float(unique_value) + 0.5 == item or float(
                                    unique_value) - 0.5 == item:
                                counter += 1
                    elif not self.border_values:
                        for unique_value in uniques_col:
                            if float(unique_value) == item:
                                counter += 1

                p = p * (counter / self.size_of_dataset)
        elif self.patterns[i]["type"] == "Order-Preserving":
            p = 0 #float(Decimal(1.0) / Decimal(math.factorial(self.patterns[i]["nr_cols"])))
            counter = 0
            for column in self.data.columns:
                for row in self.data[column]:
                    if is_number(row):
                        counter += 1
            row_counter = 0
            for row in range(self.size_of_dataset):
                aux_counter = 0
                for values in self.data.iloc[row,:]:
                    if is_number(values):
                        aux_counter += 1
                if aux_counter > self.patterns[i]["nr_cols"]:
                    row_counter += 1
            percentage_missings = counter // (len(self.data.columns)*self.size_of_dataset)
        '''
        return betainc(self.patterns[i]["Cx"], self.size_of_dataset, p)

'''
def handle_numerical_outcome(vector_mean, vector_std, subspace):
    """ Calculated the interception point between the gaussian curve of outcome variable and outcome variable described by the subspace

    Parameters
    ----------
    x_space : list
        Oucome variable described by the subspace.
    Returns
    -------
    metric : list
        [0] : first point of interception
        [1] : second point of interceptiion
    """
    m1 = subspace.map(float).mean()
    m2 = vector_mean
    std1 = subspace.map(float).std()
    std2 = vector_std

    # Solve for a gaussian
    a= -1/(std1**2) + 1/(std2**2)
    b= 2*(-m2/(std2**2) + m1/(std1**2))
    c= (m2**2)/(std2**2) - (m1**2)/(std1**2) + np.log((std2**2)/(std1**2))
    intercep_points = np.roots([a, b, c])
    idxs = sorted(intercep_points)
    return [idxs[0], idxs[1]]


def lift(vector, subspace):
    Pxy = len(subspace[subspace == 1]) / len(vector)
    Px = len(subspace) / len(vector)
    Py = len(vector[vector == 1]) / len(vector)
    return Pxy / (Px * Py)
'''

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet, indexList):
    """calculates the support for items in the itemSet and returns a subset
    of the itemSet each of whose elements satisfies the minimum support"""
    _itemSet = set()
    localSet = defaultdict(int)

    for item in itemSet:
        for i in range(len(transactionList)):
            transaction = transactionList[i]
            if item.issubset(transaction):
                if item not in list(freqSet.keys()):
                    freqSet[item] = []
                freqSet[item].append(indexList[i])
                localSet[item] += 1

    for item, count in localSet.items():
        support = float(count) / len(transactionList)

        if support >= minSupport:
            _itemSet.add(item)

    return _itemSet


def joinSet(itemSet, length):
    """Join a set with itself and returns the n-element itemsets"""
    return set(
        [i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length]
    )

# If we need to remove missing it should be here
def getItemSetTransactionList(data_iterator):
    transactionList = list()
    itemSet = set()
    indexList = list()
    for index, row in data_iterator.iterrows():
    #for record in data_iterator:
        # Remove missings
        index_to_drop = []
        for s_index, s_row in row.items():
            if "?" in s_row:
                index_to_drop.append(s_index)
        row = row.drop(index=index_to_drop)
        #if row.size == 0:
        #    continue
        # Adiciona cada transação a uma lista
        transaction = frozenset(row)
        transactionList.append(transaction)
        indexList.append(index)
        # Adiciona todos os items unicos ao conjunto de items
        for item in transaction:
            itemSet.add(frozenset([item]))  # Generate 1-itemSets
    return itemSet, transactionList, indexList


def runApriori(data_iter, class_vector, minSupport, minLift):
    """
    run the apriori algorithm. data_iter is a record iterator
    Return both:
     - items (tuple, support)
     - rules ((pretuple, posttuple), confidence)
    """

    '''
    O itemset tem que ser transformado em algo como:
    a ,  b ,  c ,  d
    a_1, b_0, c_1, d_2
    a_2, b_0, c_1, d_3
    a_1, b_0, c_1, d_4
    '''
    data_iter = data_iter.applymap(lambda x: str(x))
    for column in data_iter.columns:
        i = 0
        for element in data_iter[column]:
            data_iter.at[i, column] = column + "_" + str(element)
            i += 1
    #print(data_iter)
    # Rename to class and create usefull variables
    #class_vector = class_vector.rename("class")
    #class_vector_mean = class_vector.map(float).mean()
    #class_vector_std = class_vector.map(float).std()

    #itemset  = a uma lista de items unicos
    #transactionList = ao dataset (conjunto de transações)
    itemSet, transactionList, indexList = getItemSetTransactionList(data_iter)
    freqSet = defaultdict(int)
    largeSet = dict()
    oneCSet = returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet, indexList)

    currentLSet = oneCSet
    k = 2
    while currentLSet != set([]):
        largeSet[k - 1] = currentLSet
        currentLSet = joinSet(currentLSet, k)
        currentCSet = returnItemsWithMinSupport(
            currentLSet, transactionList, minSupport, freqSet, indexList
        )
        currentLSet = currentCSet
        k = k + 1

    toRetItems = []
    for key, value in largeSet.items():
        temp = []
        for item in value:
            columns_of_p = []
            for val in item:
                columns_of_p.append(val.split("_")[0])
            temp.append({"columns": frozenset(columns_of_p), "indexes": freqSet[item]})
        toRetItems.extend(temp)
        #toRetItems.extend([(tuple(item), getSupport(item)) for item in value])

    return toRetItems







