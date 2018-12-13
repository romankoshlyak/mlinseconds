# Task is to write solution without using key
key = [4, 8 ,13 ,16, 20, 24, 31, 45]
def get_ind(in_array, key):
    index = 0
    for ind, key_val in enumerate(key):
        index |= in_array[key_val] << ind
    return index
def read_array(f):
    return [int(x) for x in next(f).split()]

input = open("input.txt", "r") 
res = open("res.txt", "w") 
solMap = {}
n, m = read_array(input)
for i in range(n):
    arr = read_array(input)
    in_array = arr[:m]
    in_val = arr[m]
    ind = get_ind(in_array, key)
    solMap[get_ind(in_array, key)] = in_val

n, m = read_array(input)
for i in range(n):
    in_array = read_array(input)
    res.write("{}\n".format(solMap[get_ind(in_array, key)]))
input.close() 
res.close() 
