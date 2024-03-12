import feyn
import pandas as pd

# filter out models with Gaussian distributions or hyperbolic tangent
def custom_filter(model):
    l = ''.join(model.fnames)
    return 'gaussian' not in l and 'tanh' not in l

# class for Symbolic Regression
class SymReg(object):
    _estimator_type = 'regressor'
    def __init__(self, epochs=10, complexity=10):
        self.n_epochs = epochs
        self.complexity = complexity

        # establish QLattice
        self.ql = feyn.QLattice(random_seed=1)
        self.models = []
        self.best = None
    
    def get_params(self, deep=False):
        return {'epochs':self.n_epochs, 'complexity':self.complexity}
    
    # fit function
    def fit(self, X, y, **kwargs):
        if 'epochs' in kwargs.keys():
            self.n_epochs = kwargs['epochs']
        if 'complexity' in kwargs.keys():
            self.complexity = kwargs['complexity']
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=['X', 'DENSITY', 'INCLINE FACTOR', 'FLUID-WALL INTERACTION', 'WALL SPEED'])
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

            # display progress
            if len(self.models) > 0:
                feyn.show_model(self.models[0], feyn.tools.get_progress_label(epoch, self.n_epochs), update_display=False)
    
            # update probability density function to improve structures
            self.ql.update(self.models)

        # select best model
        self.best = self.models[0]

    def predict(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=['X', 'DENSITY', 'INCLINE FACTOR', 'FLUID-WALL INTERACTION', 'WALL SPEED'])
        return self.best.predict(X)

    def save_model(self, fname):
        self.best.save(fname)

if __name__ == "__main__":
    # generate tree structure and equation of model
    symReg = feyn.Model.load('symReg.json')
    sympy_model = symReg.sympify(symbolic_lr=True, include_weights=True)
    print(sympy_model.as_expr())
    symReg.savefig('symReg.svg')