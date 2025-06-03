# for texts where it is not plain text but follows a specific format, like python C Java codes or markdown (which is a markup language) text

# these are split by particular more filters, like for python codes the separators would be "class", "def", ... "\n" " " ""

# markdown text has specific separators which tries to break it by headings (first #, then ##, ###, ####, and more)


# python code split

from langchain.text_splitter import RecursiveCharacterTextSplitter,Language

text = """
class Student:
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade  # Grade is a float (like 8.5 or 9.2)

    def get_details(self):
        return self.name"

    def is_passing(self):
        return self.grade >= 6.0


# Example usage
student1 = Student("Aarav", 20, 8.2)
print(student1.get_details())

if student1.is_passing():
    print("The student is passing.")
else:
    print("The student is not passing.")
"""

# Initialize the splitter
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=300, 
    chunk_overlap=0,
)

# Perform the split
chunks = splitter.split_text(text)

print(len(chunks))
print('-------------------------')
print(chunks)
print('-------------------------')
print(chunks[0])
print('-------------------------')

# >>> 
# 2
# -------------------------
# ['class Student:\n    def __init__(self, name, age, grade):\n        self.name = name\n        self.age = age\n        self.grade = grade  # Grade is a float (like 8.5 or 9.2)\n\n    def get_details(self):\n        return self.name"\n\n    def is_passing(self):\n        return self.grade >= 6.0', '# Example usage\nstudent1 = Student("Aarav", 20, 8.2)\nprint(student1.get_details())\n\nif student1.is_passing():\n    print("The student is passing.")\nelse:\n    print("The student is not passing.")']
# -------------------------
# class Student:
#     def __init__(self, name, age, grade):
#         self.name = name
#         self.age = age
#         self.grade = grade  # Grade is a float (like 8.5 or 9.2)

#     def get_details(self):
#         return self.name"

#     def is_passing(self):
#         return self.grade >= 6.0
# -------------------------



# markdown text split

from langchain.text_splitter import RecursiveCharacterTextSplitter,Language

text = """
# Project Name: Smart Student Tracker

A simple Python-based project to manage and track student data, including their grades, age, and academic status.


## Features

- Add new students with relevant info
- View student details
- Check if a student is passing
- Easily extendable class-based design


## 🛠 Tech Stack

- Python 3.10+
- No external dependencies


## Getting Started

1. Clone the repo  
   ```bash
   git clone https://github.com/your-username/student-tracker.git

"""

# Initialize the splitter
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=200,
    chunk_overlap=0,
)

# Perform the split
chunks = splitter.split_text(text)

print(len(chunks))
print('-------------------------')
print(chunks)
print('-------------------------')
print(chunks[0])
print('-------------------------')


# >>>
# 3
# -------------------------
# ['# Project Name: Smart Student Tracker\n\nA simple Python-based project to manage and track student data, including their grades, age, and academic status.', '## Features\n\n- Add new students with relevant info\n- View student details\n- Check if a student is passing\n- Easily extendable class-based design', '## 🛠 Tech Stack\n\n- Python 3.10+\n- No external dependencies\n\n\n## Getting Started\n\n1. Clone the repo  \n   ```bash\n   git clone https://github.com/your-username/student-tracker.git']
# -------------------------
# # Project Name: Smart Student Tracker

# A simple Python-based project to manage and track student data, including their grades, age, and academic status.
# -------------------------