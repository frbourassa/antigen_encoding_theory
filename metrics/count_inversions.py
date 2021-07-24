import pandas as pd
import time
import random

def time_func_sort(func, tosort, tries=100):
    start = time.process_time()
    for i in range(tries):
        answer = func(tosort)
    stop = time.process_time()
    delay = stop - start
    return delay/tries, answer


# Adjacent swaps sort: complexity O(N^2), counting inversions
# which may need more moves than the version above.
def count_inversions_standard(lisp):
    longueur = len(lisp)
    lis = list(lisp)  # Copy the list
    swaps = 0
    for i in range(longueur):
        elem = lisp[i]
        for j in range(i-1, -1, -1):  # Swap back to earlier position until sorted
            if elem < lis[j]:
                lis[j+1], lis[j] = lis[j], lis[j+1]
                swaps += 1
            else:
                break  # All elements before were already sorted, so skip.
    return lis, swaps


# Merge sort: complexity O(N log N)
def count_inversions_mergesort(lisp):
    lis = list(lisp)
    if len(lisp) > 1:
        lis, inversions = mergesort_count_internal(lis, 0)
    else:
        inversions = 0
    return lis, inversions

def mergesort_count_internal(lis, inverts):
    if len(lis) == 1: # Nothing to do,  no inversion
        pass

    elif len(lis) == 2:  # Switch the two elements if needed
        if lis[0] > lis[1]:  # No inversions if equal
            lis = lis[::-1]
            inverts += 1

    else:  # Sort two sub-lists and then recombine them
        midpoint = len(lis) // 2
        half1, inverts = mergesort_count_internal(lis[:midpoint], inverts)
        half2, inverts = mergesort_count_internal(lis[midpoint:], inverts)
        # Each time an element in the right half (half2) is larger than all
        # elements in the left half, count len(half1) inversions.
        for i in range(len(lis)):
            if len(half1) > 0:
                if len(half2) > 0:  # Can choose between both lists
                    if half1[0] <= half2[0]:  # No inversion if equal
                        lis[i] = half1.pop(0)
                    else:
                        lis[i] = half2.pop(0)
                        inverts += len(half1)  # Since we pop elements,
                        # the only elements left in half1 are those over which
                        # we jump to put this half2 element in place.
                else:  # half2 is empty
                    lis[i] = half1.pop(0)
                    # All inversions have already been counted when adding the
                    # elements of half2 in front of half1
            elif len(half2) > 0:  # half1 is empty but not half2
                lis[i] = half2.pop(0)
                # No inversions needed to put those elements back
            else: # Both lists are empty before filling lis
                raise RuntimeError('Both halves somehow lost an element of lisp!')
    return lis, inverts

def count_swaps(rawList, orderDict = {}, swapType='merge'):
    #Replace categorical labels with numerical orderings if an orderDict passed
    if len(orderDict) != 0:
        rawList = [orderDict[x] for x in rawList]
    #Grab both true ordered swap number and worse case swap number for each type of swap
    if swapType == 'merge':
        orderedList,inverts = count_inversions_mergesort(rawList)
        # Analytical expression if N1 times value 1, N2 times value 2, ... :
        # N1*(N-N1) + N2*(N-N1-N2) + ... = N^2 - sum(l, i<l) Ni*Nl
        # Might as well use the counting algorithm again.
        # Reduces to N(N-1)/2 if all elements are unique.
        orderedListWorseCase,worseCaseInverts = count_inversions_mergesort(orderedList[::-1])
    else:
        orderedList,inverts = count_inversions_standard(rawList)
        orderedListWorseCase,worseCaseInverts = count_inversions_standard(orderedList[::-1])

    #This metric normalizes swap number by the worse case number of swaps possible with the given categories
    orderAccuracy = 100 - (inverts/worseCaseInverts)*100
    # Each pair of distinct elements has probability 1/2 of being inverted
    # in a random list, so by linearity of the expectation value, there is on
    # average 1/2 * number of distinct pairs = 1/2 * worstCase swaps.
    # randomCaseInverts = worseCaseInverts / 2
    # randomOrderAccuracy = 50  # Not even worth returning: it's always 50 %
    return inverts, orderAccuracy

def returnSwapDf(df,swapType='merge',swapLevel='Peptide',orderDict={'N4':7,'A2':6,'Y3':5,'Q4':4,'T4':3,'V4':2,'G4':1,'E1':0}):
    nonPeptideLevels = [x for x in df.index.names if x != swapLevel]
    parsingDf = df.groupby(nonPeptideLevels).sum()
    parsingDf.columns.name = 'Feature'
    swapDf,normalizedSwapDf = parsingDf.copy(),parsingDf.copy()
    for row in range(parsingDf.shape[0]):
        for col in range(parsingDf.shape[1]):
            parameter = parsingDf.columns.tolist()[col]
            indexingTuple = list(parsingDf.iloc[row,:].name)
            parsingTuple = []
            i=0
            for level in df.index.names:
                if level in nonPeptideLevels:
                    parsingTuple.append(indexingTuple[i])
                    i+=1
                else:
                    parsingTuple.append(slice(None))
            subsetDf = df.loc[tuple(parsingTuple),:]
            parameterOrder = subsetDf[parameter].sort_values().index.unique(swapLevel).tolist()
            swaps,normalizedSwaps = count_swaps(parameterOrder,orderDict = orderDict, swapType = swapType)
            swapDf.iloc[row,col] = swaps
            normalizedSwapDf.iloc[row,col] = normalizedSwaps
    fullSwapDf = pd.concat([swapDf,normalizedSwapDf],axis=1,keys=['Swaps','Order Accuracy'],names=['Statistic']).stack('Feature')
    return fullSwapDf

if __name__ == "__main__":
    # First, check that the methods sort lists properly
    assert count_inversions_standard([4, 3, 2, 1])[0] == [1, 2, 3, 4]
    #assert count_inversions_mergesort([4, 3, 2, 1])[0] == [1, 2, 3, 4]

    # Second, check that the number of inversions is correct in different cases
    # No inversion
    ok_list = [1, 2, 3, 4]
    assert count_inversions_standard(ok_list) == (ok_list, 0)
    assert count_inversions_mergesort(ok_list) == (ok_list, 0)

    # One inversion in the left half (also using number of elements != 2^k)
    one_left = [2, 1, 3, 4, 5]
    assert count_inversions_standard(one_left) == (sorted(one_left), 1)
    assert count_inversions_mergesort(one_left) == (sorted(one_left), 1)

    # One inversion in the right half
    one_right = [-1, 1, 3, 7, 5]
    assert count_inversions_standard(one_right) == (sorted(one_right), 1)
    assert count_inversions_mergesort(one_right) == (sorted(one_right), 1)

    # One inversion in each half, none across halves
    one_each = [-5, -1, -3, 1, 5, 3]
    assert count_inversions_standard(one_each) == (sorted(one_each), 2)
    assert count_inversions_mergesort(one_each) == (sorted(one_each), 2)

    # One inversion across both halves: happens when we combine halves
    one_across = [1, 2, 0, 4]  # That's two inversions, 2 swaps to put 0 first
    assert count_inversions_standard(one_across) == (sorted(one_across), 2)
    assert count_inversions_mergesort(one_across) == (sorted(one_across), 2)

    # More inversions when we combine halves
    two_across = [2, 3, 0, 1]
    assert count_inversions_standard(two_across) == (sorted(two_across), 4)
    assert count_inversions_mergesort(two_across) == (sorted(two_across), 4)

    # Inversions across and within
    one_across_one_within = [2, 1, 0, 3]  # 3 in total
    assert count_inversions_standard(one_across_one_within) == (sorted(one_across_one_within), 3)
    assert count_inversions_mergesort(one_across_one_within) == (sorted(one_across_one_within), 3)

    # More complicated case
    complicated = [3, 1, 0, 4, -3, 2, 5]  # Total of 4 + 2 + 1 + 2 = 9 invs
    # To see why, count for each element how many elements larger than itself
    # are on its left
    assert count_inversions_standard(complicated) == (sorted(complicated), 9)
    assert count_inversions_mergesort(complicated) == (sorted(complicated), 9)

    ### New (April 2021): cases with equal elements.
    # Case with ties (equal elements), but no inversion
    ties = [1, 1, 2, 3]
    assert count_inversions_standard(ties) == (sorted(ties), 0)
    assert count_inversions_mergesort(ties) == (sorted(ties), 0)

    # Case with ties (equal elements) and inversion of a unique element
    # with some equal elements
    ties2 = [2, 3, 1, 1]  # 4 inversions: two for each one.
    assert count_inversions_standard(ties2) == (ties, 4)
    assert count_inversions_mergesort(ties2) == (ties, 4)

    # Case with ties (equal elements) and inversion of equal elements
    # with other equal elements
    ties3 = [3, 2, 3, 1, 2, 1, 3]  # 1 + 3 + 2 + 4 inversions
    assert count_inversions_standard(ties3) == (sorted(ties3), 10)
    assert count_inversions_mergesort(ties3) == (sorted(ties3), 10)

    # Another test case, to be really sure it works as expected.
    ties4 = [-1, -1, 5, -1, 3, 3, 3, 4]  # 1 + 1 + 1 + 1 + 1 = 5 inversions
    assert count_inversions_standard(ties4) == (sorted(ties4), 5)
    assert count_inversions_mergesort(ties4) == (sorted(ties4), 5)

    print("All tests passed successfully!")
    print("Now timing different sorting algorithms on length-1000 lists")
    ## Timing both methods
    testlist = random.sample(range(1000), 1000)
    time_standard  = time_func_sort(count_inversions_standard, testlist)[0]
    time_divide = time_func_sort(count_inversions_mergesort, testlist)[0]

    print("The standard method took {} s".format(time_standard))
    print("The mergesort method took {} s".format(time_divide))
