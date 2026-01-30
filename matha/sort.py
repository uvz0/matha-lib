# ---------------------
# Date: 30-01-2026
# Author: AstroJr0
# ---------------------

# General O(N^2)
# these are normal sorting btw.. nothing too big

import math

def bubble_sort(array) -> list:
    """Space: O(1)
     Time: O(N^2)
     How it works: Repeatedly steps through the list, compares adjacent elements and swaps them if they are in the wrong order.
     Returns: Sorted list"""
    n = len(array)
    for i in range(n):
        swapped = False
        # Optimization: Last i elements are already in place
        for j in range(0, n - i - 1):
            if array[j] > array[j + 1]:
                array[j], array[j + 1] = array[j + 1], array[j]
                swapped = True
        if not swapped:  # Optimization: Stop if array is sorted
            break
    return array

def selection_sort(array) -> list:
    """Space: O(1)
     Time: O(N^2)
     How it works: Divides the input list into two parts: a sorted sublist and an unsorted sublist. Repeatedly selects the smallest element from the unsorted sublist.
     Returns: Sorted list"""
    size = len(array)
    for step in range(size):
        min_idx = step
        for i in range(step + 1, size):
            if array[i] < array[min_idx]:
                min_idx = i
        array[step], array[min_idx] = array[min_idx], array[step]
    return array

def insertion_sort(array) -> list:
    """Space: O(1)
     Time: O(N^2)
     How it works: Builds the final sorted array one item at a time. It assumes the first element is already sorted then compares others against it.
     Returns: Sorted list"""
    for i in range(1, len(array)):
        key = array[i]
        j = i - 1
        while j >= 0 and key < array[j]:
            array[j + 1] = array[j]
            j -= 1
        array[j + 1] = key
    return array

def gnome_sort(array) -> list:
    """Space: O(1)
     Time: O(N^2)
     How it works: Similar to insertion sort but moving an element to its proper place is accomplished by a series of swaps, similar to a Bubble Sort.
     Returns: Sorted list"""
    index = 0
    n = len(array)
    while index < n:
        if index == 0:
            index = index + 1
        if array[index] >= array[index - 1]:
            index = index + 1
        else:
            array[index], array[index - 1] = array[index - 1], array[index]
            index = index - 1
    return array

def cocktail_shaker_sort(array) -> list:
    """Space: O(1)
     Time: O(N^2)
     How it works: A variation of bubble sort that sorts in both directions on each pass through the list.
     Returns: Sorted list"""
    n = len(array)
    swapped = True
    start = 0
    end = n - 1
    while swapped:
        swapped = False
        for i in range(start, end):
            if array[i] > array[i + 1]:
                array[i], array[i + 1] = array[i + 1], array[i]
                swapped = True
        if not swapped:
            break
        swapped = False
        end = end - 1
        for i in range(end - 1, start - 1, -1):
            if array[i] > array[i + 1]:
                array[i], array[i + 1] = array[i + 1], array[i]
                swapped = True
        start = start + 1
    return array

def odd_even_sort(array) -> list:
    """Space: O(1)
     Time: O(N^2)
     How it works: A variation of bubble sort. It functions by comparing all odd/even indexed pairs of adjacent elements in the list and swapping them if wrong.
     Returns: Sorted list"""
    n = len(array)
    is_sorted = False
    while not is_sorted:
        is_sorted = True
        for i in range(1, n - 1, 2): # Odd pass
            if array[i] > array[i + 1]:
                array[i], array[i + 1] = array[i + 1], array[i]
                is_sorted = False
        for i in range(0, n - 1, 2): # Even pass
            if array[i] > array[i + 1]:
                array[i], array[i + 1] = array[i + 1], array[i]
                is_sorted = False
    return array

def comb_sort(array) -> list:
    """Space: O(1)
     Time: O(N^2) worst, O(N log N) avg
     How it works: Improves on Bubble Sort by using a gap larger than 1. The gap starts large and shrinks by a factor of 1.3.
     Returns: Sorted list"""
    n = len(array)
    gap = n
    shrink = 1.3
    sorted = False
    while not sorted:
        gap = int(gap / shrink)
        if gap <= 1:
            gap = 1
            sorted = True
        i = 0
        while i + gap < n:
            if array[i] > array[i + gap]:
                array[i], array[i + gap] = array[i + gap], array[i]
                sorted = False
            i += 1
    return array

# Efficent O(n log n)
# these are quite effeicent TBH
def merge_sort(array) -> list:
    """Space: O(N)
     Time: O(N log N)
     How it works: Recursively splits the array in half until single elements remain, then merges the sorted halves back together.
     Returns: Sorted list"""
    if len(array) > 1:
        mid = len(array) // 2
        L = array[:mid]
        R = array[mid:]
        merge_sort(L)
        merge_sort(R)
        
        i = j = k = 0
        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                array[k] = L[i]
                i += 1
            else:
                array[k] = R[j]
                j += 1
            k += 1
        while i < len(L):
            array[k] = L[i]
            i += 1
            k += 1
        while j < len(R):
            array[k] = R[j]
            j += 1
            k += 1
    return array

def quick_sort(array) -> list:
    """Space: O(log N)
     Time: O(N log N) avg
     How it works: Selects a 'pivot' element and partitions the array into elements less than and greater than the pivot. Recursively sorts partitions.
     Returns: Sorted list"""
    # Optimized with list comprehension for readability, 
    # though in-place is better for strict space optimization.
    if len(array) <= 1:
        return array
    else:
        pivot = array[len(array) // 2]
        left = [x for x in array if x < pivot]
        middle = [x for x in array if x == pivot]
        right = [x for x in array if x > pivot]
        return quick_sort(left) + middle + quick_sort(right)

def heap_sort(array) -> list:
    """Space: O(1)
     Time: O(N log N)
     How it works: Builds a max-heap from the data. Repeatedly swaps the root (max) with the last element and reduces heap size.
     Returns: Sorted list"""
    def heapify(arr, n, i):
        largest = i
        l = 2 * i + 1
        r = 2 * i + 2
        if l < n and arr[i] < arr[l]:
            largest = l
        if r < n and arr[largest] < arr[r]:
            largest = r
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            heapify(arr, n, largest)

    n = len(array)
    # Build a maxheap
    for i in range(n // 2 - 1, -1, -1):
        heapify(array, n, i)
    # Extract elements
    for i in range(n - 1, 0, -1):
        array[i], array[0] = array[0], array[i]
        heapify(array, i, 0)
    return array

def dual_pivot_quick_sort(array) -> list:
    """Space: O(log N)
     Time: O(N log N)
     How it works: An optimized QuickSort using two pivots (P1, P2) to divide the array into three parts: <P1, P1..P2, >P2.
     Returns: Sorted list"""
    def _dp_sort(arr, low, high):
        if low < high:
            if arr[low] > arr[high]:
                arr[low], arr[high] = arr[high], arr[low]
            p1, p2 = arr[low], arr[high]
            i = low + 1
            k = low + 1
            j = high - 1
            while k <= j:
                if arr[k] < p1:
                    arr[k], arr[i] = arr[i], arr[k]
                    i += 1
                elif arr[k] >= p2:
                    while arr[j] > p2 and k < j:
                        j -= 1
                    arr[k], arr[j] = arr[j], arr[k]
                    j -= 1
                    if arr[k] < p1:
                        arr[k], arr[i] = arr[i], arr[k]
                        i += 1
                k += 1
            i -= 1
            j += 1
            arr[low], arr[i] = arr[i], arr[low]
            arr[high], arr[j] = arr[j], arr[high]
            _dp_sort(arr, low, i - 1)
            _dp_sort(arr, i + 1, j - 1)
            _dp_sort(arr, j + 1, high)
            
    _dp_sort(array, 0, len(array)-1)
    return array


# Damn.. we have Hybrids here too
def intro_sort(array) -> list:
    """Space: O(log N)
     Time: O(N log N)
     How it works: Starts with QuickSort. If recursion depth exceeds limit, switches to HeapSort. For small arrays, switches to InsertionSort.
     Returns: Sorted list"""
    # Helper: Heap Sort for Intro
    def _heap_sort(arr):
        return heap_sort(arr)
        
    # Helper: Insertion for small chunks
    def _insertion(arr):
        return insertion_sort(arr)

    def _intro_sort_helper(arr, depth_limit):
        n = len(arr)
        if n < 16:
            return _insertion(arr)
        if depth_limit == 0:
            return _heap_sort(arr)
        
        pivot = arr[n // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        
        return (_intro_sort_helper(left, depth_limit - 1) + 
                middle + 
                _intro_sort_helper(right, depth_limit - 1))

    max_depth = 2 * math.log2(len(array)) if len(array) > 0 else 0
    return _intro_sort_helper(array, max_depth)

def tim_sort(array) -> list:
    """Space: O(N)
     Time: O(N log N)
     How it works: ('Tie' usually refers to Timsort). Divides array into small 'runs', sorts them via Insertion Sort, and merges them using Merge Sort logic.
     Returns: Sorted list"""
    min_run = 32
    n = len(array)

    # Sort individual subarrays of size min_run
    for start in range(0, n, min_run):
        end = min(start + min_run, n)
        # Python slice sorting is essentially Timsort (native), 
        # but here is the manual logic:
        sub_arr = array[start:end]
        # Insertion sort logic inline for speed
        for i in range(1, len(sub_arr)):
            j = i
            while j > 0 and sub_arr[j] < sub_arr[j - 1]:
                sub_arr[j], sub_arr[j - 1] = sub_arr[j - 1], sub_arr[j]
                j -= 1
        array[start:end] = sub_arr

    # Merge the runs
    size = min_run
    while size < n:
        for left in range(0, n, 2 * size):
            mid = min(n, left + size)
            right = min(n, left + 2 * size)
            if mid < right:
                # Merge logic
                merged = []
                l, r = left, mid
                while l < mid and r < right:
                    if array[l] < array[r]:
                        merged.append(array[l])
                        l += 1
                    else:
                        merged.append(array[r])
                        r += 1
                merged.extend(array[l:mid])
                merged.extend(array[r:right])
                array[left:right] = merged
        size *= 2
    return array

# We have distributions too O(N + K)
def counting_sort(array) -> list:
    """Space: O(K) where K is the range of inputs
     Time: O(N + K)
     How it works: Counts occurrences of each unique element. Then calculates positions of each element in the output sequence. Best for integers with small range.
     Returns: Sorted list"""
    if not array: return []
    max_val = max(array)
    min_val = min(array) # Handle negative numbers
    range_of_elements = max_val - min_val + 1
    
    count_arr = [0] * range_of_elements
    output_arr = [0] * len(array)
    
    for i in range(len(array)):
        count_arr[array[i] - min_val] += 1
        
    for i in range(1, len(count_arr)):
        count_arr[i] += count_arr[i - 1]
        
    for i in range(len(array) - 1, -1, -1):
        output_arr[count_arr[array[i] - min_val] - 1] = array[i]
        count_arr[array[i] - min_val] -= 1
        
    return output_arr

def bucket_sort(array) -> list:
    """Space: O(N + K)
     Time: O(N + K) avg
     How it works: Distributes elements into buckets. Each bucket is then sorted individually (often using insertion sort).
     Returns: Sorted list"""
    if len(array) == 0: return []
    bucket_count = len(array)
    max_val = max(array)
    min_val = min(array)
    buckets = [[] for _ in range(bucket_count)]
    
    # Range of each bucket
    r = (max_val - min_val) / bucket_count
    
    for num in array:
        diff = (num - min_val) / r if r != 0 else 0
        idx = int(diff) - 1 if int(diff) == bucket_count else int(diff)
        buckets[idx].append(num)
        
    sorted_array = []
    for bucket in buckets:
        # Use simple sort for buckets
        sorted_array.extend(sorted(bucket)) 
    return sorted_array

def radix_sort(array) -> list:
    """Space: O(N + K)
     Time: O(NK)
     How it works: Sorts numbers digit by digit from least significant to most significant using a stable sort (like Counting Sort).
     Returns: Sorted list"""
    def counting_sort_for_radix(arr, exp):
        n = len(arr)
        output = [0] * n
        count = [0] * 10
        for i in range(n):
            index = arr[i] // exp
            count[index % 10] += 1
        for i in range(1, 10):
            count[i] += count[i - 1]
        for i in range(n - 1, -1, -1):
            index = arr[i] // exp
            output[count[index % 10] - 1] = arr[i]
            count[index % 10] -= 1
        for i in range(n):
            arr[i] = output[i]

    if not array: return []
    max_val = max(array)
    exp = 1
    while max_val // exp > 0:
        counting_sort_for_radix(array, exp)
        exp *= 10
    return array

#Some joke ones :)
import threading
import time

def sleep_sort(array) -> list:
    """Space: O(N) (Threads)
     Time: O(max(input))
     How it works: Creates a thread for each element. The thread sleeps for a duration equal to the element's value, then appends it to a list.
     Returns: Sorted list (Non-deterministic accuracy)"""
    sorted_list = []
    threads = []
    
    # Normalize delay to speed it up slightly while maintaining order
    # Note: 0.01 multiplier makes it faster but risks inaccuracy due to OS scheduler
    def sleep_and_append(val):
        time.sleep(val * 0.01) 
        sorted_list.append(val)

    for i in array:
        t = threading.Thread(target=sleep_and_append, args=(i,))
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
        
    return sorted_list
