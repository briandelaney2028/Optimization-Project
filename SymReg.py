import feyn
from generate_data import generate_data

def custom_filter(model):
    l = ''.join(model.fnames)
    return 'gaussian' not in l and 'tanh' not in l

class SymReg(object):
    def __init__(self, epochs=10, complexity=10):
        self.n_epochs = epochs
        self.complexity = complexity

        self.ql = feyn.QLattice(random_seed=1)
        self.models = []
        self.best = None
    
    def get_params(self, deep=False):
        return {'epochs':self.n_epochs, 'complexity':self.complexity}
    
    def fit(self, X, y, **kwargs):
        self.n_epochs = kwargs['epochs']
        self.complexity = kwargs['complexity']

        X['y'] = y

        for epoch in range(1, self.n_epochs+1):
            # generate new models
            new_sample = self.ql.sample_models(
                input_names=X,
                output_name='y',
                kind='regression',
                max_complexity=self.complexity
            )
            self.models += new_sample

            # filter models to remove models including gaussian functions
            self.models = list(filter(custom_filter, self.models))

            # fit models to training data
            self.models = feyn.fit_models(
                models=self.models, 
                data=X,
                loss_function='squared_error',
                criterion='bic', # Bayesion Information Criterion, could be aic Akaike " ", bic penalises complex models more than aic
                threads=8
                )
    
            # prune poorly performing models
            self.models = feyn.prune_models(self.models)

            # increase diversity of models
            # models = feyn.get_diverse_models(
            #     models=models,
            #     n=10
            # )
    
            # display progress
            if len(self.models) > 0:
                feyn.show_model(self.models[0], feyn.tools.get_progress_label(epoch, self.n_epochs), update_display=False)
    
            # update probability density function to improve structures
            self.ql.update(self.models)

   
        # select best model
        self.best = self.models[0]

    def predict(self, X):
        return self.best.predict(X)

    def save_model(self, fname):
        self.best.save(fname)

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    # x_plot = np.linspace(0, 300, 301)
    # y_plot = np.zeros_like(x_plot)
    symReg = feyn.Model.load('symReg.json')
    # for i in range(x_plot.size):
    #     df = pd.DataFrame(np.array([[x_plot[i], 0.8, 0.196, 1, 1]]), columns=['X', 'DENSITY', 'INCLINE FACTOR', 'FLUID-WALL INTERACTION', 'WALL SPPED'])
    #     y_plot[i] = symReg.predict(df)

    # fig, ax = plt.subplots()
    # ax.plot(x_plot, y_plot, label='Symbolic Regression')
    # plt.show()

    sympy_model = symReg.sympify(symbolic_lr=True, include_weights=True)
    print(sympy_model.as_expr())
    symReg.savefig('symReg.svg')