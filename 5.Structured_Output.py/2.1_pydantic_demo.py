from pydantic import BaseModel, Field, EmailStr
from typing import Optional

# Basic Example
class Student(BaseModel):
    name: str

new_student = {'name': 'Suryam Vivek Gupta'}
student = Student(**new_student)
print(student)


# Note how it gives error now, unlike TypedDict

# class Student(BaseModel):
#     name: str

# wrong_student = {'name': 32} # gives error, Input should be a valid string [type=string_type, input_value=32, input_type=int]
# student = Student(**wrong_student)
# print(student)


# how to set default values

class Student(BaseModel):
    name: str = 'Suryam'

new_student = {}
student = Student(**new_student)
print(student)


# Handling Optional Fields

class Student(BaseModel):
    name: str = 'Suryam'
    age: Optional[int] = None  # or put a number to use as default age if not provided

new_student = {}
student = Student(**new_student)
print(student)

new_student_age = {'age': 24}
student = Student(**new_student_age)
print(student)


# Coerce, i.e. Pydantic can handle implicit typecasting

class Student(BaseModel):
    name: str = 'Suryam'
    age: Optional[int] = None  # or put a number to use as default age if not provided

new_student_age = {'age': '24'} # age was defined as int, but we gave a string, so it will be converted to int
student = Student(**new_student_age)
print(student)


# EmailStr - to handle email validation

class Student(BaseModel):
    name: str = 'Suryam'
    age: Optional[int] = None 
    email: EmailStr

new_student_age = {'age': 24, 'email': 'abc@gmail.com'} 
student = Student(**new_student_age)
print(student) 

# If not in proper email format, it will give an error, value is not a valid email address: An email address must have an @-sign. [type=value_error, input_value='abc', input_type=str]
# new_student_age = {'age': 24, 'email': 'abc'} 
# student = Student(**new_student_age)
# print(student) 



# Field - to create default values, put contraints, write descriptions, regex, etc.

class Student(BaseModel):
    name: str = 'Suryam'
    age: Optional[int] = None 
    email: EmailStr
    cgpa: float = Field(gt=0, le=10, default=8, description="decmial val representing cgpa of student") # constraints: gt = greater than, le = less than or equal to (similarly we have ge and lt too), if nothing given then default will be 8, description also given

new_student_age = {'age': 24, 'email': 'abc@gmail.com', 'cgpa': 9.5} 
student = Student(**new_student_age)
print(student) 

# converting this pydantic object to dict
print(student.model_dump())

# converting this pydantic object to json
print(student.model_dump_json()) 

