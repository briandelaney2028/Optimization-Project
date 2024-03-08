import numpy as np
import pandas as pd

data = dict()

with open("Data_00-10.csv", 'r') as f:
    for line in f.readlines():
        l = line.strip().split(',')
        contents = list(filter(lambda x: x != '', l))
        if l[0].strip() == 'SIMULATION':
            for sim in contents[2:]:
                data[int(sim)] = dict()
        elif l[0].strip() == 'DENSITY' or l[0].strip() == 'INCLINE FACTOR'\
            or l[0].strip() == 'FLUID-WALL INTERACTION' or l[0].strip() == 'WALL SPEED':
            for i, (key, d) in enumerate(data.items()):
                d[contents[0].strip()] = float(contents[i+2])
        elif l[2].strip() != 'X':
            for i, (key, d) in enumerate(data.items()):
                if 'X' not in d.keys():
                    d['X'] = [float(l[2*i+4])]
                    d['Vx'] = [float(l[2*i+5])]
                else:
                    try:
                        d['X'].append(float(l[2*i+4]))
                        d['Vx'].append(float(l[2*i+5]))
                    except:
                        continue
# print(data)
temp = []
for key, d in data.items():
    density = d['DENSITY']
    incline = d['INCLINE FACTOR']
    interaction = d['FLUID-WALL INTERACTION']
    speed = d['WALL SPEED']
    for x, v in zip(d['X'], d['Vx'][::-1]):
        temp.append([x, density, incline, interaction, speed, v])

data = pd.DataFrame(temp, columns=['X', 'DENSITY', 'INCLINE FACTOR', 'FLUID-WALL INTERACTION', 'WALL SPEED', 'Vx'])
print(data)
data.to_csv("prepped_data.csv", index=False)
        
