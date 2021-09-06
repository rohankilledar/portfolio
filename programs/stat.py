import statistics
num_list = []

while True:
    new_num = int(input())
    if new_num == 0:
        break
    else:
        num_list.append(new_num)


print("input: "+ str(num_list))
print("sum: "+ str(sum(num_list)))
print("mean: "+ str(statistics.mean(num_list)))
