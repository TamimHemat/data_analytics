from math import inf

the_list = [3,2,4,9]

def desc_list(l):
    
    maximum = sum(abs(x) for x in the_list)
    
    to_return = []
    
    change = True
    
    while l:
        next_num = l.pop()
        
        if next_num < maximum:
            to_return.append(next_num)
            print(to_return)
            
            if change:
                maximum = next_num
                change = False
        
        else:
            to_return.insert(0, next_num)
            print(to_return)
            maximum = next_num
    
    return to_return

print(desc_list(the_list))