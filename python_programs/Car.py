class Car:
    def __init__(self, model, time):
        self.model = model
        self.time = time
    
    def __str__(self):
        return '{self.model} 0 to 100 km per hour in {self.time}s'.format(self=self)

    def __lt__(self,other):
        if self.time < other.time:
            return True
        else:
            return False
    
    def __gt__(self,other):
        if self.time > other.time:
            return True
        else:
            return False
    
    def __eq__(self,other):
        if self.time == other.time:
            return True
        else:
            return False

class carEngine(Car):
    def __init__(self,name,time,eType):
        super(carEngine,self).__init__(name,time)
        self.eType = eType

    def __str__(self):
        return super(carEngine,self).__str__() + (' Engine Type is ' + self.eType)

#the code below is just to check if its correct
new_car = Car('Roadster',3)
new_car2 = Car('Mustang GT', 5)

new_car3 = carEngine('Roadster',3,'v3')
print(new_car3)

print(new_car < new_car2)
print(new_car > new_car2)
print(new_car == new_car2)

print(new_car.model)
print(new_car.time)
print(new_car)

