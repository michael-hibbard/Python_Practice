# Michael Hibbard

# Have retroactively decided to practice classes and functions in this file.

# Create a student class. Do not need to initialize yet.
class Student:

    _numStudents = 0

    # Initialize the class. "Self" is the python version of "this"
    def __init__(self,studentName, studentAge, yearInSchool):
        self._studentName = studentName
        self._studentAge = studentAge
        self._yearInSchool = yearInSchool
        Student._numStudents += 1

    def displayNumStudents(self):
        print("The total number of students is:", Student._numStudents)

    # "Syntactic Sugar" ~ @Property for getter and setter for the student name
    @property
    def studentName(self):
        return self._studentName

    @studentName.setter
    def studentName(self,newName):
        self._studentName = newName

    # Repeat the process for the student age and the student's year in school
    @property
    def studentAge(self):
        return self._studentAge

    @studentAge.setter
    def studentAge(self,newAge):
        self._studentAge = newAge

    @property
    def yearInSchool(self):
        return self._yearInSchool

    @yearInSchool.setter
    def yearInSchool(self,newYearInSchool):
        self._yearInSchool = newYearInSchool

# define the main method:
def main():

    # Gather user input to create a new student object:
    name = input("Please enter a person's name: ")
    age = input("Please enter a person's age: ")
    yearInSchool = input("Please enter year in school: ")

    # Initialize the new student object:
    newStudent = Student(name, age, yearInSchool)

    # Print the information on the student:
    print(newStudent.studentName)

# Python automatically sets name to main. Basically, is it being run directly by python, or is it being imported?
if __name__ == "__main__":
    main()


