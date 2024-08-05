conditional = True
x = 1 if conditional else 0
print(x)

print("ENUMERATE AND ZIP")
names = ['ben', 'lam', 'haos', 'hoz']
names2 = ['lam', 'bach', 'van', 'and']

for index, name in enumerate(names, start=1):
    print(index, name)

for name, last_name in zip(names, names2):
    print(f'{name}, and {last_name}')

for index, (name2, last_name2) in enumerate(zip(names, names2)):
    print(index, name2, last_name2)


print("UNPACKING")
a, b, *c, d = (1,2,3,4,5,6,7)
print("a:", a)
print("b:", b)
print("c:", c)
print("d:", d)

a, *_ = (1,2,3,4,5)
print("a:", a)

print("ADDING ATTRIBUTES")
class Person():
    pass

person = Person()
person_info = {"first_name" : "Ben", 'last_name':'Lam'}
for key, value in person_info.items():
    setattr(person, key, value)

print(person.first_name)
print(person.last_name)

print("COMPREHENSIONS")
dict_comp = {i: i*i for i in range(10)}
list_comp = [x*x for x in range(10)]
print(dict_comp)
print(list_comp)


print("LOGGING")
import logging

logging.basicConfig(level=logging.INFO, filename="log.log", filemode='w',
                    format="%(asctime)s - %(levelname)s - %(message)s")
x = 1
logging.debug("debug")
logging.info(f"the value of x is {x}")
logging.warning("warning")
logging.error("error")
logging.critical("critical")