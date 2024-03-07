import pickle
import feyn


# lead Symbolic Regressor
symReg = feyn.Model.load('symReg.json')

# load guassian process regressor
with open('gp.pkl', 'rb') as f:
    gp = pickle.load(f)