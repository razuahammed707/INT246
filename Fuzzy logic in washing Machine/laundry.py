import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class washing_machine:

  laundry_weight = ctrl.Antecedent(np.arange(0.0,8.5,0.5), 'laundry_weight')
  dirt_level = ctrl.Antecedent(np.arange(1.0,10.5,0.5), 'dirt_level')
  powder_required = ctrl.Consequent(np.arange(0,61, 1), 'powder_required')
  
  laundry_weight.automf(3)
  dirt_level.automf(3)
  
  powder_required['verylow'] = fuzz.trimf(powder_required.universe, [0, 8, 20])
  powder_required['low'] = fuzz.trimf(powder_required.universe, [8, 12, 20])
  powder_required['medium'] = fuzz.trimf(powder_required.universe, [12, 20, 40])
  powder_required['high'] = fuzz.trimf(powder_required.universe, [20, 40, 60])
  powder_required['veryhigh'] = fuzz.trimf(powder_required.universe, [40, 60, 60])

  laundry_weight.view()
  dirt_level.view()
  powder_required.view()

  rule1 = ctrl.Rule(laundry_weight['poor'] | dirt_level['poor'], powder_required['verylow'])
  rule2 = ctrl.Rule(laundry_weight['average'] | dirt_level['poor'], powder_required['low'])
  rule3 = ctrl.Rule(laundry_weight['good'] | dirt_level['poor'], powder_required['medium'])
  rule4 = ctrl.Rule(laundry_weight['poor'] | dirt_level['average'], powder_required['medium'])
  rule5 = ctrl.Rule(laundry_weight['average'] | dirt_level['average'], powder_required['medium'])
  rule6 = ctrl.Rule(laundry_weight['good'] | dirt_level['average'], powder_required['high'])
  rule7 = ctrl.Rule(laundry_weight['poor'] | dirt_level['good'], powder_required['high'])
  rule8 = ctrl.Rule(laundry_weight['average'] | dirt_level['good'], powder_required['high'])
  rule9 = ctrl.Rule(laundry_weight['good'] | dirt_level['good'], powder_required['veryhigh'])

  #rule1.view()

  washing_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
  washing = ctrl.ControlSystemSimulation(washing_ctrl)

def fuzzify_laundry(fuzz_laundry_weight,fuzz_dirt_level):
  washing_machine.washing.input['laundry_weight'] = fuzz_laundry_weight
  washing_machine.washing.input['dirt_level'] = fuzz_dirt_level
  washing_machine.washing.compute()
#  washing_machine.powder_required.view(sim = washing_machine.washing)
  return washing_machine.washing.output['powder_required']

def compute_washing_parameters(total_laundry_weight,level_of_dirt):
  
  if total_laundry_weight < 0.0 or total_laundry_weight > 8.0:
    raise Exception("Invalid value for laundry weight: %lf" % total_laundry_weight)

  if level_of_dirt < 1.0 or level_of_dirt > 10.0:
    raise Exception("Invalid value for dirt level: %lf" % level_of_dirt)
  type_fuzz = fuzzify_laundry(total_laundry_weight,level_of_dirt)
  
  print(type_fuzz)

def plot_graph():
  laundry_wei = np.arange(0.0,8.5,0.5)
  dirt_lvl = np.arange(1.0,10.5,0.5)
  x = []
  y = []
  z = []
  for i in laundry_wei:
    for j in dirt_lvl:
      x.append(i)
      y.append(j)
      z.append(fuzzify_laundry(i,j))
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  
  ax.scatter(x, y, z, c='r', marker='o')

  ax.set_xlabel('laundry weight (in kg)')
  ax.set_ylabel('dirt level')
  ax.set_zlabel('powder required')

  plt.show()

if __name__=="__main__":

  laundry_weight = float(input("laundry weight : "))
  dirt_level = float(input("dirt level : "))
  washing_parameters = compute_washing_parameters(laundry_weight,dirt_level)
  plot_graph()