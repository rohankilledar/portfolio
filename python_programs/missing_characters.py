#!/bin/python3

#
# Complete the 'missingCharacters' function below.
#
# The function is expected to return a STRING.
# The function accepts STRING s as parameter.
#
import string
def missingCharacters(s):
    # Write your code here
    s = s.lower()
    numbers = [str(n) for n in range(0,10)]
    alphabets = [a for a in string.ascii_lowercase]
    
    result = numbers + alphabets
    inputstring = [st for st in s]
    for ins in inputstring:
        if ins in result:
            result.remove(ins)
    
    return ''.join(result)
    

if __name__ == '__main__':
    s = "8hypotheticall024y6wxz"
    print(missingCharacters(s))