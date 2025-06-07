rows = 5

# 1. Lower Triangle
print("Lower Triangle:")
for i in range(1, rows + 1):
    for j in range(i):
        print("*", end=" ")
    print()

# 2. Upper Triangle
print("\nUpper Triangle:")
for i in range(rows, 0, -1):
    for j in range(i):
        print("*", end=" ")
    print()

# 3. Pyramid
print("\nPyramid:")
for i in range(1, rows + 1):
    for j in range(rows - i):
        print(" ", end="") 
    for k in range(i):
        print("* ", end="")  
    print()
