from typing import TypedDict

class Person(TypedDict):
    name: str # just data representation, not a data validation tool, so even if we give a int here it wont give an error
    age: int 



new_person: Person = {'name':'Suryam', 'age':24}
print(new_person)


wrong_person: Person = {'name':100, 'age':24}
print(wrong_person)