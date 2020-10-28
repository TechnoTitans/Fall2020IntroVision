import random
import numpy as np

x = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 90, 9]])

print(x[0][2])

# if you don't specify axis, then it takes avg of all values
print(np.average(x))

# Axis 0, is along columns
print(np.average(x, 0))

# Axis 1, is within rows (within the second dimension)
print(np.average(x, 1))
print(type(x))

exit()


def chicken():
    print("dfi")
    return 154


print(f"Random Number [0, 1): {random.random()}")
print(random.randint(0, 100))

# A list, not an array
x = [3, 4, 53, 23, 4.3, "dhf", [3, 5]]

# for each loop
for i in x:
    print(i)

print("====================")

# "Regular" for
for i in range(len(x)):
    # Put an f in front of your string, and then put any variables within curly brackets
    print(f"i: {i}")
    print(x[i])

y = 0

while y < 10 or False:
    print(y)
    y += 1

# && and
# || or

x = 15
# x = "sting"

print(chicken())
