import random
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from lmfit import Parameters, Minimizer

# Possible optimizer methods
methods = [
    'leastsq', # Levenberg-Marquardt
    'least_squares', # Least-Squares minimization, using Trust Region Reflective method (default)
    'differential_evolution', # Differential evolution
    'brute', # Brute force method
    'basinhopping', # Basinhopping
    'ampgo', # Adaptive Memory Programming for Global Optimization
    'nelder', # Nelder-Mead
    'lbfgsb', # L-BFGS-B
    'powell', # Powell
    'cg', # Conjugate-Gradient
    'newton', # Newton-CG
    'cobyla', # Cobyla
    'bfgs', # BFGS
    'tnc', # Truncated Newton
    'trust-ncg', # Newton-CG trust-region
    'trust-exact', # Nearly exact trust-region
    'trust-krylov', # Newton GLTR trust-region
    'trust-constr', # Trust-region for constrained optimization
    'dogleg', # Dog-leg trust-region
    'slsqp', # Sequential Linear Squares Programming
    'emcee', # Maximum likelihood via Monte-Carlo Markov Chain
    'shgo', # Simplicial Homology Global Optimization
    'dual_annealing' # Dual Annealing optimization
]

# Default equations:

def linear(t, m, b):
    # Linear model where m is slope, b is bias
    output = m*t + b
    return output

def mass_dashpot(t, y, k, b):
    # Models the equation: y'' + by' + ky = 0
    # Where k is spring constant, b is friction coefficient
    y_val = y[0]
    dydt_val = y[1]
    output = np.array([dydt_val, -k*y_val - b*dydt_val])
    return output

def newtonian(t, gamma, eta, R_0, coeff=0.7217):
    # Linear analytical solution for a Newtonian fluid (not an ODE)
    # Where parameters are defined as...
    #   t       Time
    #   gamma   Surface tension
    #   eta     Shear viscosity
    #   R_0     Initial radius
    R = R_0 - coeff*(gamma/eta)*t
    output = np.array([ R ])
    return output

def elasto_capillary(t, G, gamma, lamb, R_0):
    # Analytical solution to Hookean dumbbell model (not an ODE)
    # Where parameters are defined as...
    #   t       Time
    #   G       Linear elastic modulus
    #   gamma   Surface tension
    #   lamb    Relaxation time
    #   R_0     Initial radius
    R = R_0*((G*R_0/gamma)**(1/3))*math.exp(-t/(3*lamb))
    output = np.array([ R ])
    return output

def oldroyd_b(t, y, G, gamma, eta_S, lamb):
    # Function for Oldroyd-B model
    # Where parameters are defined as...
    #   t       Time
    #   y       [ Radius, Sigma_zz, Sigma_rr ]
    #   G       Linear elastic modulus
    #   gamma   Surface tension
    #   eta_S   Shear viscosity
    #   lamb    Relaxation time
    R = y[0]
    sigma_zz = y[1]
    sigma_rr = y[2]
    R_dot = (R*(sigma_zz - sigma_rr) - gamma)/(6*eta_S)
    sigma_zz_dot = -(4*R_dot/R)*sigma_zz - sigma_zz/lamb - 4*G*R_dot/R
    sigma_rr_dot = (2*R_dot/R)*sigma_rr - sigma_rr/lamb + 2*G*R_dot/R
    output = np.array([R_dot, sigma_zz_dot, sigma_rr_dot])
    return output

def inelastic_rate_thickening(t, y, k_2, gamma, eta):
    # Models inelastic rate thickening differential equation
    # Where parameters are defined as...
    #   t       Time
    #   y       [ Radius ]
    #   k_2     Coefficient
    #   gamma   Surface tension
    #   eta     Shear viscosity
    #   R_0     Initial radius
    R = y[0]
    a = 12*k_2
    b = -6*eta*R
    c = -gamma
    R_dot = (-b + (b**2 - 4*a*c)**(1/2)) / (2*a)
    if R_dot > 0:
        R_dot = (-b - (b**2 - 4*a*c)**(1/2)) / (2*a)
    output = np.array([ R_dot ])
    return output

def FENEP(t, y, b, lamb, G, sigma, shear_S, R_0):
    # Function for FENE-P consitutive model for dilute polymer solutions
    # Where parameters are defined as...
    #   t           Time
    #   y           [ Radius, A_zz, A_rr ]
    #   b           FENE factor
    #   lamb        Relaxation time
    #   G           Linear elastic modulus
    #   sigma       Surface tension
    #   shear_S     Solvent viscosity
    #   R_0         Initial radius
    R = y[0]
    Azz = y[1]
    Arr = y[2]
    Z = 1/(1-(Azz+2*Arr)/b)
    epsdot_ast = (2*sigma/(R*R_0/250) - G*Z*(Azz - Arr))/(3*shear_S)*lamb # Dimensionless strain rate 
    output = np.array([-epsdot_ast*R, 2*epsdot_ast*Azz - (Z*Azz-1), -epsdot_ast*Arr  - (Z*Arr-1)])
    output /= 2 # Convert to radius
    return output


def rolie_poly(t, y, L, tau_d, tau_s, G, gm, b=1, dl=-0.5):
    # Function representing Tube model for entangled polymer solutions
    # Where parameters are defined as...
    #   t           Time
    #   y           [ Radius, Srr, Szz, lmd ]
    #   L           Maximum extensibility
    #   tau_d       Disengagement time
    #   tau_s       Relaxation time
    #   G           Linear elastic modulus
    #   gm          Gamma
    #   b           Constant
    #   dl          Empirical exponent
    if (y[2] > L and L > 0):
        dy = np.array([0, 0, 0, 0])
        return dy

    else:
        # [R, Srr, Szz, lmd]
        R = y[0]
        Srr = y[1]
        Szz = y[2]
        lmd = y[3]

        # Front factor of f'/f=Cf*lmd_dot, where f is the finite extensibility
        if L==-1:
            f = 1
            Cf = 0
        else:
            f = ((3*(L**2) - (lmd**2))/(3*(L**2) - 1))*(((L**2) - 1)/((L**2) - (lmd**2)))
            Cf = 4*lmd*(L**2)/(3*(L**2)-(lmd**2))/((L**2)-(lmd**2))

        # Front factor in epsilon_dot: epsilon_dot = sr_Srr*Srr_dot + sr_Szz*Szz_dot + sr_lmd*lmd_dot 
        sr_Srr = -2/(Szz-Srr)
        sr_Szz =  2/(Szz-Srr)
        sr_lmd =  4/lmd + 2*Cf

        # Three equations: Srr_dot, Szz_dot, lmd_dot
        relaxTerm = 1/(lmd**2)*(1/tau_d + 2*b*f*(lmd-1)/tau_s * (lmd**(dl-1))) # constant of disengagement term, check Larson
        
        # Front factor in (k:S): k:S = sr_Srr*Srr_dot + sr_Szz*Szz_dot + sr_lmd*lmd_dot 
        kS_Srr = (Szz-Srr)*sr_Srr
        kS_Szz = (Szz-Srr)*sr_Szz
        kS_lmd = (Szz-Srr)*sr_lmd

        iA = np.linalg.inv([[1+Srr*(sr_Srr + 2*kS_Srr),   Srr*(   sr_Szz + 2*kS_Szz),   Srr*(   sr_lmd + 2*kS_lmd)],
                            [  Szz*(-2*sr_Srr + 2*kS_Srr), 1+Szz*(-2*sr_Szz + 2*kS_Szz),   Szz*(-2*sr_lmd + 2*kS_lmd)], 
                            [ -lmd*(kS_Srr),                -lmd*(kS_Szz),               1-lmd*(kS_lmd)]])
        iB = [[-relaxTerm*(Srr-1/3)],
                [-relaxTerm*(Szz-1/3)],
                [-f/tau_s*(lmd-1)-relaxTerm*((lmd**2)-1)/(2*lmd)*(lmd**2)]]
        dy = np.matmul(iA, iB).flatten()

        if L == -1:
            f = 1
            Cf = 0
        else:
            f = ((3*(L**2) - (lmd**2))/(3*(L**2) - 1))*(((L**2) - 1)/((L**2) - (lmd**2)))
            Cf = 4*lmd*(L**2)/(3*(L**2)-(lmd**2))/((L**2)-(lmd**2))

        dzz_rr = Szz-Srr
        R = gm/(3*G*f*(lmd**2)*dzz_rr)

        # Calculation of strain rate and viscosity

        # Front factor in epsilon_dot: epsilon_dot = sr_Srr*Srr_dot + sr_Szz*Szz_dot + sr_lmd*lmd_dot 
        sr_Srr = -2/(Szz-Srr)
        sr_Szz = 2/(Szz-Srr)
        sr_lmd = 4/lmd + 2*Cf

        epsilon_dot = sr_Srr*Srr + sr_Szz*Szz + sr_lmd*lmd
        dR = epsilon_dot*R/(-2)
        dR = np.array([dR])
        dy = np.concatenate((dR, dy))
        
    return dy

# Dictionary to store default differential equations
equation_functions = {
    'best': None,
    'linear': linear,
    'mass_dashpot': mass_dashpot,
    'newtonian': newtonian,
    'elasto_capillary': elasto_capillary,
    'oldroyd_b': oldroyd_b,
    'inelastic_rate_thickening': inelastic_rate_thickening,
    'FENEP': FENEP,
    'rolie_poly': rolie_poly
}

equation_function_args = {
    'best': [],
    'linear': ['m', 'b'],
    'mass_dashpot': ['k', 'b'],
    'newtonian': ['gamma', 'eta', 'R_0'],
    'elasto_capillary': ['G', 'gamma', 'lamb', 'R_0'],
    'oldroyd_b': ['G', 'gamma', 'eta_S', 'lamb'],
    'inelastic_rate_thickening': ['k_2', 'gamma', 'eta'],
    'FENEP': ['b', 'lamb', 'G', 'sigma', 'shear_S', 'R_0'],
    'rolie_poly': ['L', 'tau_d', 'tau_s', 'G', 'gm']
}


def solve_ode(equation_function, equation_args, time, params, init_cond, ode=True):
    # Solve ODE for given initial conditions and parameters
    # Where parameters is a dictionary mapping name to value or an lmfit Parameters object
    assert (ode == True), 'Only differential equations can be solved.'
    if equation_function == None:
        return None

    # Make sure types are correct
    if type(params) == dict:
        param_dict = params
        params = Parameters()
        for key, value in param_dict.items():
            params.add(key, value = value)

    # Add parameter values as args
    args = []
    for key in equation_args:
        args.append(params[key])

    # Solve and return
    tspan = (0, time.max() - time.min())
    solution = solve_ivp(equation_function, tspan, init_cond, t_eval=(time - time.min()), args=args, method='BDF')
    return solution.t, solution.y


def error(equation_function, equation_args, time, radius, params, init_cond=None, radius_index=0, ode=True, log_error=False):
    # Return error for a solution, in real value
    if ode == True:
        assert (type(init_cond) == np.array or type(init_cond) == np.ndarray or init_cond != None), 'Must provide initial condition for ODE.'

        # Solve initial value problem if equation is ODE
        tfit, yfit = solve_ode(equation_function, equation_args, time, params, init_cond, ode=ode)
        if yfit[radius_index, :].shape != radius.shape:
            e = (10**12)*np.ones(radius.shape)
        else:
            if log_error == True:
                e = np.log10(yfit[radius_index, :]) - np.log10(radius)
            else:
                e = yfit[radius_index, :] - radius

    else:
        # Compile input parameters
        input_param = []
        params_dict = dict(params.valuesdict())
        for arg in equation_args:
            value = params_dict[arg]
            input_param.append(value)

        # Calculate radius for all time based on equation
        yfit = np.zeros(time.shape)
        for i in range(len(time)):
            input_var = [time[i]] + input_param
            yfit[i] = equation_function(*input_var)

        # Evaluate difference as error
        if log_error == True:
            e = np.log10(yfit) - np.log10(radius)
        else:
            e = yfit - radius

    return e


def objective(equation_function, equation_args, time, radius, params, init_cond=None, radius_index=0, ode=True):
    # Return the objective of a solution to be minimized, in absolute value
    return np.abs(error(equation_function, equation_args, time, radius, params, init_cond=init_cond, radius_index=radius_index, ode=ode))


def objective_over_parameter(equation_function, parameter_ranges, equation_args, parameter_key, range_sections, time, radius, params, init_cond=None, parameter_range=None, radius_index=0, ode=True):
    # Get a list of objectives to see how performance changes over range of parameter values
    parameter_values = []
    objective_values = []
    params_dict = dict(params.valuesdict())
    
    # If parameter range not provided, use the class attributes
    if parameter_range == None:
        parameter_range = parameter_ranges[parameter_key]
    min_val = parameter_range[0]
    max_val = parameter_range[1]
    step_size = (max_val - min_val) / range_sections

    # Create a list of all possible parameter values in increasing order (where length is nuber of stations)
    parameter_values = (min_val + step_size*np.arange(range_sections)).tolist()

    # Create a list of parameter sets to supply to objective function in pooling
    param_sets = []
    for param_value in parameter_values:
        # Copy params and add the currently being checked parameter at current value
        params_dict = dict(params.valuesdict())
        guess_params = Parameters()
        for key, value in params_dict.items():
            if key != parameter_key:
                guess_params.add(key, value = value, vary = False)
        guess_params.add(parameter_key, value = param_value, vary = False)
        param_sets.append(guess_params)

    # Get mean objective for each parameter set
    for param_set in param_sets:
        obj = objective(equation_function, equation_args, time, radius, param_set, init_cond=init_cond, radius_index=radius_index, ode=ode)
        obj = np.mean(obj)
        objective_values.append(obj)

    return np.array(parameter_values), np.array(objective_values)


class Equation():
    '''
    Class to represent an ODE which can be fit to CABER data

    All values should be in base SI units (like meters, seconds, etc) (not centimeters, milliseconds etc)

    Inputs:
        equation_function           Function resembling the ones above which takes time, y, and parameters to return dy (Required, Function)
        equation_args               List of parameters which must be provided to equation_function in order (Required, List)
        parameter_ranges            Dictionary which maps parameter keys in equation_args to value ranges allowed for each parameter (Required, Dict)
        method                      Method to use to minimize object of ODE over data (String, Default: 'least_squares')
        ode                         Boolean describing whether or not equation function is a differential equation (Bool, Default: True)
        radius_index                Index where radius value is located in y, the return from equation function. Assumed to be 0, radius at the top (Int, Default: 0)
        log_error                   Boolean denoting whether or not error function should take log difference (Bool, Default: False)
        verbose                     Flag whether or not to print statements (Bool, Default: True)
    '''

    def __init__(self, equation_function, parameter_ranges, equation_args=None, method='cg', ode=True, radius_index=0, log_error=False, verbose=True):
        assert (type(parameter_ranges) == dict and type(parameter_ranges[list(parameter_ranges.keys())[0]]) == tuple), 'Provided set of parameter ranges has invalid type.'
        assert (equation_args == None or type(equation_args) == list or type(equation_args) == tuple), 'Equation arguments must be a list.'
        assert (type(method) == str), 'Optimization method must be a string.'
        assert (method in methods), 'Provided method is invalid, please choose another.'
        assert (type(ode) == bool), 'Whether or not function is an ODE must be indicated with a boolean type.'
        assert (type(radius_index) == int), 'Radius index must be an integer.'
        assert (type(verbose) == bool), 'Verbose flag must be boolean.'

        # Make sure equation function is in defaults dictionary or is a custom function
        if type(equation_function) == str:
            assert (equation_function in list(equation_functions.keys())), 'Provided equation function does not exist.'
            self.equation_function = equation_functions[equation_function]
            self.custom_function = False
            radius_index = 0
            equation_args = equation_function_args[equation_function]

            # Identify if function is not an differential equation
            if equation_function == 'linear' or equation_function == 'newtonian' or equation_function == 'elasto_capillary':
                ode = False
            else:
                ode = True

        elif callable(equation_function) == True:
            assert (equation_args != None), 'Please provide equation args as keys in order.'
            self.equation_function = equation_function
            self.custom_function = True

        else:
            assert ValueError('Equation function type not recognized')

        # Make sure args each have ranges
        for arg in equation_args:
            assert (arg in list(parameter_ranges.keys())), 'Please provide a range for all parameters / equation arguments.'

        # Save attributes
        self.equation_args = equation_args
        self.parameter_ranges = parameter_ranges
        self.method = method
        self.ode = ode
        self.radius_index = radius_index
        self.log_error = log_error
        self.verbose = verbose
        self.radius = None
        self.time = None
        self.init_cond = None
        self.parameters = None
        self.result = None


    def solve_ode(self, time, params, init_cond):
        # Solve ODE for given initial conditions and parameters
        # Where parameters is a dictionary mapping name to value or an lmfit Parameters object
        return solve_ode(self.equation_function, self.equation_args, time, params, init_cond, ode=self.ode)


    def error(self, time, radius, params, init_cond=None):
        # Return error for a solution, in real value
        return error(self.equation_function, self.equation_args, time, radius, params, init_cond=init_cond, radius_index=self.radius_index, ode=self.ode, log_error=self.log_error)


    def objective(self, time, radius, params, init_cond=None):
        # Return the objective of a solution to be minimized, in absolute value
        return objective(self.equation_function, self.equation_args, time, radius, params, init_cond=init_cond, radius_index=self.radius_index, ode=self.ode)


    def objective_over_parameter(self, parameter_key, range_sections, time, radius, params, init_cond=None, parameter_range=None):
        # Get a list of objectives to see how performance changes over range of parameter values
        return objective_over_parameter(self.equation_function, self.parameter_ranges, self.equation_args, parameter_key, range_sections, time, radius, params, init_cond=init_cond, parameter_range=parameter_range, radius_index=self.radius_index, ode=self.ode)


    def fit(self, time, radius, init_cond=None, parameter_guesses={}, vary_parameter={}, range_sections=25, max_guess_loops=None, vary_init_cond=True, vary_init_cond_pct=0.1):
        assert (type(time) == np.array or type(time) == np.ndarray), 'Time must be an array.'
        assert (type(radius) == np.array or type(radius) == np.ndarray), 'Radius must be an array.'
        assert (type(parameter_guesses) == dict), 'Parameter guesses must be in a dictionary.'
        assert (type(vary_parameter) == dict), 'Whether or not to vary parameters must be provided in a dictionary.'
        assert (type(range_sections) == int), 'Number of sections to break range into must be an integer.'
        assert (type(max_guess_loops) == int or max_guess_loops == None), 'Max number of guess algorithm loops must be an integer.'
        assert (type(vary_init_cond) == bool), 'Whether or not to vary initial condition must be a boolean.'
        assert (vary_init_cond_pct >= 0 and vary_init_cond_pct < 1), 'Initial condition variance must be a percentage.'

        # Fit given differential equation to radius data by choosing parameters
        # With inputs...
        #   t                       Time array (Required, np.array)
        #   radius                  Radius data array (Required, np.array)
        #   init_cond               Initial condition array for equation function. Required if equation is differential (Optional, np.array)
        #   parameter_guesses       Dictionary mapping parameter name (same as from ranges) to initial guess. If guess not provided for parameter, will run an algorithm to find one (Optional, Dict)
        #   vary_parameter          Dictionary mapping parameter name to bool. If bool is true, allow the parameter to change. If not provided for parameter, we assume vary to be true (Optional, Dict)
        #   range_sections          Amount of sections to break up parameter range into when looking for guess. Larger value means finer guess value (Optional, Int)
        #   max_guess_loops         Number of times to loop guessing algorithm. Defaults to the number of parameters (Optional, Int)
        #   vary_init_cond          Whether or not to vary the init condition (Optional, Bool)
        #   vary_init_cond_pct      How much to allow the initial condition to vary by in terms of percent of the initial value (Optionl, Float between 0 and 1)

        # Make sure initial condition is provided for ODE
        if self.ode == True:
            assert (type(init_cond) == np.array or type(init_cond) == np.ndarray or init_cond != None), 'Must provide initial condition.'
        elif init_cond != None:
            if self.verbose: print('Initial condition not required and will not be used.')

        self.radius = radius
        self.time = time
        self.init_cond = init_cond

        # Make sure parameter guess and variability matches
        for key, value in vary_parameter.items():
            if value == False:
                assert (key in list(parameter_guesses.keys())), 'Parameter must be able to vary if no guess is supplied. Supply a guess for parameter ' + key + ' or set vary to True.'
            elif type(value) != bool:
                raise ValueError('Vary parameter values must all be boolean.')

        # Add parameters without guess to guesses dictionary with initial value
        parameters_without_guess = []
        for key, value in self.parameter_ranges.items():
            if key not in list(parameter_guesses.keys()):
                parameters_without_guess.append(key)
                parameter_guesses[key] = (value[0] + value[1]) / 2 # Set to midpoint for now

        # Set the default max number of guess loops to length of parameter values
        if max_guess_loops == None:
            max_guess_loops = len(list(self.parameter_ranges.keys()))

        # Iterate loops and get refined guess values for parameters
        last_min_obj = -float('inf')
        min_obj = float('inf')
        if self.verbose: print('Compiling parameter guesses...')
        for _ in range(max_guess_loops):

            # Break if the objective hasn't changed in an entire run
            if min_obj == last_min_obj:
                break
            last_min_obj = min_obj

            # Shuffle parameters every run for better chance of global min
            random.shuffle(parameters_without_guess)
            for parameter in parameters_without_guess:

                # Make parameters
                params = Parameters()
                for key, value in parameter_guesses.items():
                    params.add(key, value = value, vary = False)

                # Get minimum objective for all parameter values in range
                parameter_values, objective_values = self.objective_over_parameter(parameter, range_sections, time, radius, params, init_cond=init_cond)

                # Use minimum objective to choose parameter guess
                index = np.argmin(objective_values)
                min_obj = objective_values[index]
                guess = parameter_values[index]
                parameter_guesses[parameter] = guess

                if self.verbose: print(parameter + ':', str(guess)[:7], '\t\tobjective:', str(min_obj)[:7])
        
        if self.verbose: print('Finished creating guesses.')
        if self.verbose: print(parameter_guesses)

        # Make final parameters object
        params = Parameters()
        for parameter, guess in parameter_guesses.items():
            min_val = self.parameter_ranges[parameter][0]
            max_val = self.parameter_ranges[parameter][1]
            
            # If whether or not to vary parameter explicitly provided, default to vary being True
            if parameter not in list(vary_parameter.keys()):
                vary = True            
            else:
                vary = vary_parameter[parameter]

            # Add parameter
            params.add(parameter, value = guess, min = min_val, max = max_val, vary = vary)

        # Add init conditions to parameters
        if self.ode == True and vary_init_cond == True and vary_init_cond_pct != 0:
            varying_init_cond = []
            for i in range(len(init_cond)):
                key = 'init_cond' + str(i)
                value = init_cond[i]

                # Add parameter allowing init condition to vary between value plus or minus variance percentage
                min_val = (1 - vary_init_cond_pct)*value
                min_val = (1 + vary_init_cond_pct)*value
                params.add(key, value = value, min = min_val, max = max_val, vary = True)
                varying_init_cond.append(params[key])

            init_cond = varying_init_cond

        if self.equation_function == None:
            # Choose the best model using BIC
            raise NotImplementedError()

        # Objective wrapper
        def objective_wrapper(parameters):
            return self.objective(time, radius, parameters, init_cond=init_cond)

        # Fit
        if self.verbose: print('Fitting equation to data...')
        self.parameters = params
        minner = Minimizer(objective_wrapper, params)
        result = minner.minimize(method=self.method)
        self.parameters = result.params
        self.result = result
        if self.verbose: print('Finished fitting.')

        # Return fit statistics
        return result


    def get_radius(self, time, params=None, init_cond=None):
        # Return the radius at given time(s) for fit
        if params == None: params = self.params
        assert (params != None), 'Please provide parameters.'

        # Make sure types are correct
        if type(params) != dict:
            param_dict = dict(params.valuesdict())
        else:
            param_dict = params

        # Convert to array
        if (type(time) == float or type(time) == int):
            time = np.array(time)

        if self.ode == True:
            # Make sure have initial condition
            if init_cond == None:
                init_cond = self.init_cond
            assert (type(init_cond) == np.ndarray or type(init_cond) == np.array or type(init_cond) == list), 'Please provide initial condition.'

            # Reconstruct params w halted value
            params = Parameters()
            for key, value in param_dict.items():
                params.add(key, value = value, vary = False)

            # Solve with parameters
            tfit, yfit = self.solve_ode(time, params, init_cond)
            radius = yfit[self.radius_index, :]

        else:
            # Add parameter values as args
            args = []
            for key in self.equation_args:
                args.append(param_dict[key])

            # Get radius for all time
            radius = np.zeros(time.shape)
            for i in range(len(time)):
                radius[i] = self.equation_function(time[i], *args)

        return radius


    def plot_error(self, time, radius, init_cond, params):
        # Graph fitting error over time
        error_array = self.error(time, radius, init_cond, params)
        plt.plot(time, error_array)
        plt.xlabel('Time')
        plt.ylabel('Objective')
        plt.show()


    def plot_objective_over_parameter(self, parameter_key, range_sections, time, radius, params, init_cond=None, parameter_range=None):
        # Graph objective with respect to one parameter varying over a range
        parameter_values, objective_values = self.objective_over_parameter(parameter_key, range_sections, time, radius, params, init_cond=init_cond, parameter_range=parameter_range)
        plt.plot(parameter_values, objective_values)
        plt.xlabel('Parameter Value')
        plt.ylabel('Objective')
        plt.show()


    def plot_fit(self, time=None, radius=None, params=None, init_cond=None, log=True):
        # Make sure time and radius are provided or part of the object (after fit)
        if not (type(time) == np.ndarray or type(time) == np.array or type(time) == list): time = self.time
        assert (type(time) == np.ndarray or type(time) == np.array or type(time) == list), 'Please provide time array.'
        if not (type(radius) == np.ndarray or type(radius) == np.array or type(radius) == list): radius = self.radius
        assert (type(radius) == np.ndarray or type(radius) == np.array or type(radius) == list), 'Please provide radius array.'

        # Get fit data
        fit = self.get_radius(time, params=params, init_cond=init_cond)

        # Plot both on the same plot
        if log == True:
            plt.semilogy(time, radius, label='Data')
            plt.semilogy(time, fit, label='Fit')
        else:
            plt.plot(time, radius, label='Data')
            plt.plot(time, fit, label='Fit')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Radius (m)')
        plt.show()