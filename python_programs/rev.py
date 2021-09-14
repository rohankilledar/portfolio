num_list = []
while True:
    new_num = int(input())
    if new_num < 0:
        break
    else:
        num_list.append(new_num)

num_list.reverse()
print(num_list)