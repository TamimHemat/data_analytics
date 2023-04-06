#Tamim Hemat
#A01278451
#January 17-th 2023

import numpy as np

#Part1: Standard Data Structures

#1. Create 4 tuples using the following commands and print them
tuple1 = 4, 5, 6
nested_tup = (4, 5, 6), (7, 8)
tuple3 = tuple([4, 0, 2])
tuple4 = tuple('string')

print(tuple1)
print(nested_tup)
print(tuple3)
print(tuple4)


#2. Print the first element in the nested tuple above(nested_tup)
print(nested_tup[0])


#3. Use the following to build a dictionary d1 = {'a' : 'some value', 'b' : [1, 2, 3, 4]},
#then get the element with key ‘a’ in the dictionary.
d1 = {'a' : 'some value', 'b' : [1, 2, 3, 4]}
print(d1['a'])


#4. Add the following element to the dictionary d1[5] = Another value
d1[5] = 'Another value'


#5. Print the dictionary then delete the added element in step 4 using del d1[5], then
#print the dictionary again.
print(d1)
del d1[5]
print(d1)


#6. Update the dictionary like this d1.update({'b' : 'foo', 'c' : 12}), then print d1.
d1.update({'b' : 'foo', 'c' : 12})
print(d1)


#7. Type the following code and print the by_letter dictionary. What’s the output? Comment on that.
words = ['apple', 'bat', 'bar', 'atom', 'book']
by_letter = {}
for word in words:
    letter = word[0]
    if letter not in by_letter:
        by_letter[letter] = [word]
    else:
        by_letter[letter].append(word)
        
print(by_letter)

#The output is a dictionary with the first letter of each word as the key
#and the words that start with that letter as values in a list.
#The output is:
#{'a': ['apple', 'atom'], 'b': ['bat', 'bar', 'book']}


#8. Type the following: a = set([2, 2, 2, 1, 3, 3]) - What’s the output of printing set a?
a = set([2, 2, 2, 1, 3, 3])
print(a)

#The output is {1, 2, 3} because sets only contain unique values.


#9. Type the following, and comment on the output:
a = {1, 2, 3, 4, 5}
b = {3, 4, 5, 6, 7, 8}
c = a.union(b)
print(c)
d = a.intersection(b)
print (d)
a.add(6)
print (a)
a.remove(6)
print (a)
a.pop()
print(a)

#The output is:
#{1, 2, 3, 4, 5, 6, 7, 8}
#{3, 4, 5}
#{1, 2, 3, 4, 5, 6}
#{1, 2, 3, 4, 5}
#{2, 3, 4, 5}


#Part 2: NumPy


#1. Type the following code, what are the results? Comment on the results
np.array([1,1,2,2,3])
print (type(np.array([1,1,2,2,3])))
x = np.array([1,1,2,2,3, 5])
print (x)
print (x.shape)
y = x.reshape((2,3))
print(y)
print (y.shape)

#The output is:
#<class 'numpy.ndarray'>
#[1 1 2 2 3 5]
#(6,)
# [[1 1 2]
#  [2 3 5]]
#(2, 3)


#2. Type the following code, and comment on the results:
a = np.arange(1, 20, 2)
print (a)
b = np.array([])
print (b)
c = np.zeros(6)
print (c)
d = np.ones(6)
print (d)
e = np.array(range(1,6))
print (e)
np.linspace(0, 5, 50)

#The output is:
#[ 1  3  5  7  9 11 13 15 17 19]
#[]
#[0. 0. 0. 0. 0. 0.]
#[1. 1. 1. 1. 1. 1.]
#[1 2 3 4 5]


#3. Type the following code, and comment on the results:
y = np.array([6, 10, 3, 4, 5, 6, 7])
print (y.argmin())
i = np.where(y>4)
print (i)
y.sort()
x = y.copy()
print (x)

#The output is:
#2
#(array([0, 1, 4, 5, 6], dtype=int64),)
#[ 3  4  5  6  6  7 10]


#4. Type the following code, and comment on the results:
x = np.random.random_sample((3,3))
y = np.random.random_sample((3,3))
print(np.dot(x, y))
print(np.cross(x, y))
print(np.linalg.det(x))
print(np.linalg.inv(x))

#The np.random.random_sample((3,3)) function creates a 3x3 matrix with random values between 0 and 1.
#The np.dot(x, y) function multiplies the two matrices together.
#The np.cross(x, y) function returns the cross product of the two matrices.
#The np.linalg.det(x) function returns the determinant of the matrix.
#The np.linalg.inv(x) function returns the inverse of the matrix.


#5. Use the following code in order to solve the equations, print the solution:
# 5x + 2y = 3
# 6x - 5y = 4
X = np.array([[5, 2], [6, -5]])
Y = np.array([3, 4])
solution = np.linalg.solve(X, Y)
print(solution)


#6. Type the following code and comment on the results:
data = np.random.randn(2, 3)
print(data)
print(data * 10)
print(data + data)


#The data variable is a 2x3 matrix with random values between -1 and 1.
#The data * 10 multiplies each value in the matrix by 10.
#The data + data adds each value in the matrix to itself.


#7. Type the following code and comment on the results:
arr = np.arange(10)
print(arr)
print(arr[5])
print(arr[5:8])

#The output is:
#[0 1 2 3 4 5 6 7 8 9]
#5
#[5 6 7]


#8. Type the following code, and comment on the results:
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4)
print(names)
print('*******************')
print(data)
print('*******************')
print(names == 'Bob')
print('*******************')
print(data[names == 'Bob'])
print('*******************')
print(data[names == 'Bob', 2:])
print('*******************')
print(data[names == 'Bob', 3])
print('*******************')
mask = (names == 'Bob') | (names == 'Will')
print('*******************')
print(mask)
print('*******************')
print(data[mask])


#First the names are printed.
#Then the data is printed. The data variable is a 7x4 matrix with random values between -1 and 1.
#Then the names == 'Bob' function is used to return a boolean array with True for each element
#that is 'Bob' and False for each element that is not 'Bob'.
#Then the data[names == 'Bob'] function is used to return the rows of the data matrix that are
#equal to 'Bob'.
#Then the data[names == 'Bob', 2:] function is used to return the rows of the data matrix that are
#equal to 'Bob' and the columns 2 and 3.
#Then the data[names == 'Bob', 3] function is used to return the rows of the data matrix that are
#equal to 'Bob' and the column 3.
#Then the mask variable is created by using the (names == 'Bob') | (names == 'Will') function.
#This function returns a boolean array with True for each element that is 'Bob' or 'Will' and False
#for each element that is not 'Bob' or 'Will'.
#Then the data[mask] function is used to return the rows of the data matrix that are equal to 'Bob'
#or 'Will'.


#9. Given the following matrix:
M1 = np.array([[2, 4, 6, 8, 10],
[3, 6, 9, -12, -15],
[4, 8, 12, 16, -20],
[5, -10, 15, -20, 25]])

#- Print the 1st and 2nd row, and for columns, we want the first, second, and third column.
#- Print all rows and third columns.
#- Print the first row and all columns.
#- Print the first three rows and first 2 columns.

#First and second row, first, second, and third column:
print(M1[0:2, 0:3])

#All rows and third column:
print(M1[:, 2])

#Forst row and all columns:
print(M1[0, :])

#First three rows and first two columns:
print(M1[0:3, 0:2])


#10. Create a numpy array and name it weight_lb (give acceptable weight values for 10 humans)
weight_lb = np.array([150, 200, 175, 210, 180, 185, 195, 190, 205, 215])

#Multiply by 0.453592 to go from pounds to kilograms and store the resulting numpy array as np_weight_kg
np_weight_kg = weight_lb * 0.453592

#Create np_hight_m numpy array and store acceptable height values for 10 humans
np_height_m = np.array([1.75, 1.85, 1.80, 1.90, 1.85, 1.80, 1.90, 1.85, 1.90, 1.95])

#Calculate the BMI of each human using the following equation:
#BMI = weight(kg) / height(m) ** 2

#Save the resulting numpy array as bmi
bmi = np_weight_kg / np_height_m ** 2

#print out bmi
print(bmi)

#Create a boolean numpy array: the element of the array should be True if the 
#corresponding baseball player’s BMI is below 21. You can use the < operator for this.
#Name the array light and print it.
light = bmi < 21

print(light)

#Print out a numpy array with the BMIs of all baseball players whose BMI is below 21. 
#Use light inside square brackets to do a selection on the bmi array.
print(bmi[light])

#Print the mean of of np_height_m
print(np.mean(np_height_m))

#Print the median of of np_height_m
print(np.median(np_height_m))