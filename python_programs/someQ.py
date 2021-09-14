def f(aList):
    for outer in range(len(aList)):
        for inner in range (len(aList) -1):
            if x(aList, inner, inner+1):
                y(aList, inner, inner+1)
    return aList
def x(aList, i, j):
    return aList[i] > aList[j]
def y(aList, i, j):
    print("hello")
    aList[i],aList[j] = aList[j],aList[i]
aList = []
while True:
    tmp = int(input("Enter a number"))
    if tmp == 0 :
        break
    aList.append(tmp)
   

print("Calling f on ", aList)
print("giving", f(aList))

