def jumpingOnClouds(c):
    # Write your code here
    cur = 0
    jump_count = 0
    last_index = len(c)-1
    while (cur != last_index):
        jump = False
        #print("current_index="+ str(cur))
        #print("jump_count=" + str(jump_count))
        #print(c)
        if cur+2 <=last_index:
            if c[cur+2] == 0:
                cur = cur + 2
                jump_count+=1
                #print("+2")
                jump = True
                
        if cur+1 <= last_index and jump ==False:
            if c[cur+1] == 0:
                cur = cur + 1
                jump_count+=1
               # print("+1")
                
    return jump_count

    
input = "0 0 1 0 0 1 0"
testcase4 = '0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 1 0 1 0 0 0 1 0 0 1 0 0 0 1 0 1 0 0 0 0 0 0 0 0 1 0 0 1 0 1 0 0'
testcase4arr = [int(s) for s in testcase4.split(' ')]
inputarr = [int(s) for s in input.split(' ')]

print(jumpingOnClouds(testcase4arr))