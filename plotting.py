from feyn import Model
from generate_data import generate_data
import matplotlib.pyplot as plt


df = generate_data('test2.csv')
df_test = df.drop(['y'], axis=1)

model = Model.load('test2_model.json')

predictions = model.predict(df_test)

# display model with weights
sympy_model = model.sympify(symbolic_lr=True, include_weights=True)
print(sympy_model.as_expr())

plt.plot(df_test.iloc[0:10, 0], predictions[0:10], 'r--', label='Group 1 SymReg')
plt.plot(df_test.iloc[0:10, 0], df.iloc[0:10, 1], 'r-', label='Group 1 Poly')
plt.plot(df_test.iloc[10:20, 0], predictions[10:20], 'b--', label='Group 3 SymReg')
plt.plot(df_test.iloc[10:20, 0], df.iloc[10:20, 1], 'b-', label='Group 3 Poly')
plt.plot(df_test.iloc[20:30, 0], predictions[20:30], 'g--', label='Group 2 SymReg')
plt.plot(df_test.iloc[20:30, 0], df.iloc[20:30, 1], 'g-', label='Group 2 Poly')
plt.legend(loc='best')

plt.show()