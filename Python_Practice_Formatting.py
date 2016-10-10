# import some stuff:
import math as math

# Declaring different types of variables #
# NOTE: unlike JAVA, do not need to declare their type!
myName = "Michael"      # A string
firstLetter = 'M'       # A character
yearInSchool = 3        # An integer
approximatePi = 3.14    # A floating - point value

# Basic print statement #
print("Hello world!")
print()

# Basic for loop to print #
for i in range (0,5):       # It appears that i is NOT inclusive of 0
    print("Hello, world!")
print()

# Format a floating point number #
value = math.pi                                         # Need to import the math library
print("The value of pi is:","{:.2f}".format(value))    # Formatting is kind of funky compared to JAVA

# User input and indexing a matrix #
# start by creating an empty array:
scientistNamesArray = []
# Tell the user the object of the game... if you can call it that...
print("How many scientists can you name?")
print("If you can't think of another, enter \"0\"")     # Use \" if adding a quotes inside a string.
# Initialize the scientistName:
scientistName = str()
# Create a while loop to run until they cannot name another:
while scientistName is not "0":
    # Ask the user for a scientist
    scientistName = input('Name a scientist: ')
    # Add that scientist to the array if the name is valid:
    if scientistName is not "0":
        scientistNamesArray.append(scientistName)
# Print out the names in the scientist name array:
print("Wow, you sure know a lot of scientists! These are the ones you named:")
for i in range (0,scientistNamesArray.__len__()):
    print(scientistNamesArray[i])
print("Try to learn even more!")


