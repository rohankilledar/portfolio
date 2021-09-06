import re
from collections import defaultdict

def solution(A,D):
    monthly_charge = 5
    minimum_trans_amount = 100
    minimum_trans_count = 3
    month_list = []
    data_dict = defaultdict(list)

    for date in D:
        month = re.search('-(.+?)-', date)
        if month:
            #month_list.append(int(month.group(1)))
            month_list.append(month.group(1))
    #print(month_list)
    for indx, month in enumerate(month_list):
        data_dict[month].append(A[indx])
        
    print(*data_dict.items())

    month_counter = 12
    for key in data_dict:
        neg_trans = []
        trans_count = 0
        for n in data_dict[key]:
            if n<0:
                neg_trans.append(n)
                trans_count +=1
        #trans_count = sum(n<0 for n in data_dict[key])
        if trans_count >=minimum_trans_count and sum(neg_trans)<=(-minimum_trans_amount):
            month_counter-=1
    
    total_balance = sum(A)
    charges = monthly_charge*month_counter
    return total_balance-charges

    #zip_iter = zip(A,month_list)
    #input_dict = dict(zip_iter)

    

A = [100,100,100,-10]
D = ["2020-12-31","2020-12-22","2020-12-03","2020-12-29"]
print(solution(A,D))

A = [180,-50,-25,-25]
D = ["2020-01-01","2020-01-01","2020-01-01","2020-01-31"]
print(solution(A,D))

A = [1,-1,0,-105,1]
D = ["2020-12-31","2020-04-04","2020-04-04","2020-04-04","2020-07-12"]
print(solution(A,D))

A = [100,100,-10,-20,-30]
D = ["2020-01-01","2020-02-01","2020-02-11","2020-02-05","2020-02-08"]
print(solution(A,D))