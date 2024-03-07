import numpy as np
import pandas as pd

data = dict()

with open("formatting.csv", 'r') as f:
    for line in f.readlines():
        l = line.strip().split(',')
        contents = list(filter(lambda x: x != '', l))
        if contents[0].strip() == 'SIMULATION':
            for sim in contents[1:]:
                data[int(sim)] = dict()
        elif contents[0].strip() == 'DENSITY':
            for i, (key, d) in enumerate(data.items()):
                d[contents[0].strip()] = [contents[2*i + 1]]
                d[contents[0].strip()].append(contents[2*i + 2])
        elif contents[0].strip() == 'INCLINE FACTOR':
            for i, (key, d) in enumerate(data.items()):
                d[contents[0].strip()] = [contents[2*i + 1]]
                d[contents[0].strip()].append(contents[2*i + 2])
        elif contents[0].strip() == 'FLUID-WALL INTERACTION':
            for i, (key, d) in enumerate(data.items()):
                d[contents[0].strip()] = [contents[2*i + 1]]
                d[contents[0].strip()].append(contents[2*i + 2])
        elif contents[0].strip() == 'WALL SPEED':
            for i, (key, d) in enumerate(data.items()):
                d[contents[0].strip()] = [contents[2*i + 1]]
                d[contents[0].strip()].append(contents[2*i + 2])
        elif contents[0].strip() != 'X':
            for i, (key, d) in enumerate(data.items()):
                if 'X' not in d.keys():
                    d['X'] = [contents[2*i]]
                    d['Vx'] = [contents[2*i+1]]
                else:
                    d['X'].append(contents[2*i])
                    d['Vx'].append(contents[2*i+1])
print(data)
temp = []
for key, d in data.items():
    density = d['DENSITY'][0]
    incline = d['INCLINE FACTOR'][0]
    interaction = d['FLUID-WALL INTERACTION'][0]
    speed = d['WALL SPEED'][0]
    for x, v in zip(d['X'], d['Vx'][::-1]):
        temp.append([x, density, incline, interaction, speed, v])

data = pd.DataFrame(temp, columns=['X', 'DENSITY', 'INCLINE FACTOR', 'FLUID-WALL INTERACTION', 'WALL SPEED', 'Vx'])
print(data)
data.to_csv("prepped_data.csv", index=False)
        
