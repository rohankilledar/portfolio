string_list = []
while True:
    new_string = input()
    if new_string == "":
        break
    else:
        string_list.append(new_string)



string_list.sort()
print(string_list)

if len(string_list) != len(set(string_list)):
    print("there are duplicates")