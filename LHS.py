from scipy.stats import qmc
import pandas as pd

sampler = qmc.LatinHypercube(d=4, strength=2, seed=10)
sample = sampler.random(n=49)

labels = ["Density", "Incline Factor", "Fluid-Wall Interaction", "Wall Speed"]

lower = [0.75, 0.133, 0.9, 0.5]
upper = [0.85, 0.5, 1.0, 1.0]

scaled = qmc.scale(sample, lower, upper)
df = pd.DataFrame(scaled, columns=labels)
df.to_csv("LHS_Sim_Parameters.csv", index_label="Sim #")

from seaborn import pairplot
import matplotlib.pyplot as plt
pairplot(df)
plt.show()