from typing import List
# Write any import statements here

def getMaxAdditionalDinersCount(N: int, K: int, M: int, S: List[int]) -> int:
  # Write your code here
  seat_vacant = []
  for seat in S:
    seat_vacant.append(seat)
    seat_vacant.extend(range(seat-K,seat))
    seat_vacant.extend(range(seat,seat+K+1))
    
  seat_occupied = list(set(seat_vacant))
  print(seat_occupied)
    
  all_seat = [x for x in range(1,N+1)]
  print(all_seat)
  
  empty_seat = [x for x in all_seat if x not in seat_occupied]
  print(empty_seat)
  
  seatable = [x for x in empty_seat if x-K not in empty_seat]
  
  for x in empty_seat:
    for y in range(x+1,x+K+1):
      if y in empty_seat:
        empty_seat.remove(y)

  print(empty_seat)
  
  ans = len(empty_seat )
    
  return ans

