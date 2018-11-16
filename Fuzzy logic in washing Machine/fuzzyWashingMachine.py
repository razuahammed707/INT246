import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# variables to hold values of washing time, clothes weight and stream of water
Weight =  ctrl.Antecedent(np.arange(0,11,1),'Weight')
Time = ctrl.Consequent(np.arange(0,101,1),'Time')
water = ctrl.Antecedent(np.arange(0,81,1),'water')

#dividing the variables values into 3 judging criteria
Weight['small'] = fuzz.trimf(Weight.universe,[0,2.5,5])
Weight['medium'] = fuzz.trimf(Weight.universe,[2.5,5,7.5])
Weight['Large'] = fuzz.trimf(Weight.universe,[5,7.5,10])

Time['small'] = fuzz.trimf(Time.universe,[0,25,50])
Time['medium'] = fuzz.trimf(Time.universe,[25,50,75])
Time['Large'] = fuzz.trimf(Time.universe,[50,75,100])
water['small'] = fuzz.trimf(water.universe,[0,20,40])
water['medium'] = fuzz.trimf(water.universe,[20,40,60])
water['Large'] = fuzz.trimf(water.universe,[40,60,80])

# creating rules 
rule1  = ctrl.Rule(Weight['small'] | water['small'],Time['medium'])
rule2  = ctrl.Rule(Weight['small'] | water['medium'], Time['small'])
rule3  = ctrl.Rule(Weight['small'] | water['Large'], Time['small'])
rule4  = ctrl.Rule(Weight['medium'] | water['small'], Time['Large'])
rule5  = ctrl.Rule(Weight['medium'] |  water['medium'], Time['medium'])
rule6  = ctrl.Rule(Weight['medium'] | water['Large'], Time['small'])
rule7  = ctrl.Rule(Weight['Large'] | water['small'], Time['Large'])
rule8  = ctrl.Rule(Weight['Large'] | water['medium'], Time['Large'])
rule9  = ctrl.Rule(Weight['Large'] | water['Large'], Time['Large'])

#using rules to create the function that will make decision using the rules
washing_Time = ctrl.ControlSystem([rule1,rule2,rule3,rule4,rule5,rule6,rule7,rule8,rule9])
time = ctrl.ControlSystemSimulation(washing_Time)

#taking inputs
time.input['Weight'] = float(input("Enter the weight of clothes"))
time.input['water']=float(input("Enter the speed of water"))
time.compute() #performing operation on the basis of rules & defuzification
print(time.output['Time']) #output
Time.view(sim=time)

  
z = []
x = []
y = []
laundry_wei = np.arange(0.0,8.5,0.5)
dirt_lvl = np.arange(1.0,10.5,0.5)
for i in laundry_wei:
    for j in dirt_lvl:
        x.append(i)
        y.append(j)
        time.input['Weight'] = i
        time.input['water']=j
        time.compute() #performing operation on the basis of rules & defuzification
        z.append(time.output['Time']) #output
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='r', marker='o')
print("gcnfg")
ax.set_xlabel('laundry weight (in kg)')
ax.set_ylabel(' speed of water')
ax.set_zlabel('time (in sec)')

plt.show()