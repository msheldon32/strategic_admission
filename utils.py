def argmin(in_list):
    min_val = float("inf")
    min_idx = 0

    for i, x in enumerate(in_list):
        if x < min_val:
            min_val = x
            min_idx = i

    return min_idx
