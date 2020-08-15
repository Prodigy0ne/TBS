#pip install rayoptics
#%matplotlib inline
import timeit
import pdb
import numpy as np
from rayoptics.environment import *
from rayoptics.optical.model_constants import Intfc, Gap, Indx, Tfrm, Zdir # Used to access the image semi-diameter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize
from scipy.optimize import minimize, Bounds
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
opm = OpticalModel() # create a new OpticalModel
sm  = opm.seq_model
osp = opm.optical_spec

osp.pupil = PupilSpec(osp, key=['object', 'pupil'], value=25)
osp.field_of_view = FieldSpec(osp, key=['object', 'height'], flds=[0.5, 0.0,-.5])  # Additional fields to account for beam width
osp.spectral_region = WvlSpec([(587.0, 1.0)])

opm.radius_mode = True

sm.gaps[0].thi=35

sm.add_surface([35.0, 6.0,  'N-BK7', 'Schott'])
sm.add_surface([-35.0, 6])
sm.set_stop() # Additional lenses to be added here

def time_func():
    opm.update_model()
print("time",timeit.Timer(time_func).timeit(number=100))
#pdb.set_trace()                  
#input=np.zeros((8,3),dtype=float)
output=np.zeros((33,30),dtype=float)

num_first_variables = 27
num_second_variables = 8
num_third_variables = 31

first_variable_store = np.zeros((num_first_variables,1),dtype=float)
second_variable_store = np.zeros((num_second_variables,1),dtype=float)
third_variable_store =  np.zeros((num_third_variables,1),dtype=float)

complete_matrix = np.zeros((num_first_variables*num_second_variables*num_third_variables,4),dtype=float)

first_variable=[3.50,260.0]
second_variable=[2.0,10.0]
third_variable=[2.0,250]

################################################################################
# A) Try to use a minimization algorithm in the Scipy package
# Create a function that can be minised
def return_value(x):
    value1=x[0]
    value2=x[1]
    value3=x[2]
    opm = OpticalModel() # create a new OpticalModel
    sm  = opm.seq_model
    osp = opm.optical_spec

    osp.pupil = PupilSpec(osp, key=['object', 'pupil'], value=25)
    osp.field_of_view = FieldSpec(osp, key=['object', 'height'], flds=[0.5, 0.0,-.5])  # Additional fields to account for beam width
    osp.spectral_region = WvlSpec([(587.0, 1.0)])

    opm.radius_mode = True

    sm.gaps[0].thi=35

    sm.add_surface([value1, value2, 'N-BK7', 'Schott'])
    sm.add_surface([-1.0*value1, value3])
    sm.set_stop()
    opm.update_model()
    #pdb.set_trace()
    for i, sg in enumerate(sm.path()):
        s = sm.list_surface_and_gap(sg[Intfc], gp=sg[Gap])
        sd = s[-1]

    return sd

bounds = Bounds([3.5,2.0,2.0], [260.0,10.0,250.0])
x0=[1.0,1.0,1.0]
res = minimize(return_value, x0, method='L-BFGS-B', bounds=bounds, tol=1e-9)

layout_plt = plt.figure(FigureClass=InteractiveLayout, opt_model=opm,
                        do_draw_rays=True, do_paraxial_layout=False).plot()

print("Value of SD predicted by minimisation algorithm = ", res.fun)
print("Value of first_variable minimisation algorithm = ", res.x[0])
print("Value of second_variable minimisation algorithm = ", res.x[1])
print("Value of third_variable minimisation algorithm = ", res.x[2])
################################################################################

################################################################################
# B) Build up a matrix of values and find the minimum using a fairly crude approach

step=0
for value1 in np.arange(first_variable[0],first_variable[1],9.5):
    #step2=0
    for value2 in np.arange(second_variable[0],second_variable[1],1.0):
        #step3=0
        for value3 in np.arange(third_variable[0],third_variable[1],8.0):

            opm = OpticalModel() # create a new OpticalModel
            sm  = opm.seq_model
            osp = opm.optical_spec

            osp.pupil = PupilSpec(osp, key=['object', 'pupil'], value=25)
            osp.field_of_view = FieldSpec(osp, key=['object', 'height'], flds=[0.5, 0.0,-.5])  # Additional fields to account for beam width
            osp.spectral_region = WvlSpec([(587.0, 1.0)])

            opm.radius_mode = True

            sm.gaps[0].thi=35

            sm.add_surface([value1, value2, 'N-BK7', 'Schott'])
            sm.add_surface([-1.0*value1, value3])
            sm.set_stop()
            opm.update_model()
            #pdb.set_trace()
            for i, sg in enumerate(sm.path()):
                s = sm.list_surface_and_gap(sg[Intfc], gp=sg[Gap])
                sd = s[-1]
            #print(sd) # This is the image semi-diameter
            #first_variable_store[step1,0]=value1
            #second_variable_store[step2,0]=value2/value3
            complete_matrix[step,0]=value1
            complete_matrix[step,1]=value2
            complete_matrix[step,2]=value3
            complete_matrix[step,3]=sd

            step+=1


# find where minimum value occurs
minimum = np.amin(complete_matrix[:,3])
result = np.where(complete_matrix[:,3] == np.amin(complete_matrix[:,3]))
value1_min = complete_matrix[result[0][0],0]
value2_min = complete_matrix[result[0][0],1]
value3_min = complete_matrix[result[0][0],2]

layout_plt = plt.figure(FigureClass=InteractiveLayout, opt_model=opm,
                        do_draw_rays=True, do_paraxial_layout=False).plot()

print("Value of SD predicted by basic search = ", minimum)
print("Value of first_variable basic search = ", value1_min)
print("Value of second_variable basic search = ", value2_min)
print("Value of third_variable basic search = ", value3_min)
################################################################################

################################################################################
# C) Fit a machine learning regressor to the previous matrix and see how it compares

model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=6),random_state=0)

param_grid = [{'n_estimators': [10, 100, 500],'learning_rate':[0.1,0.001,0.00001]}]
search = GridSearchCV(model, param_grid, cv=5)#Create the input and output


scalerx = preprocessing.StandardScaler().fit(complete_matrix[:,0:3])
scalery = preprocessing.StandardScaler().fit(complete_matrix[:,3].reshape(-1, 1))
X=scalerx.transform(complete_matrix[:,0:3])
y=scalery.transform(complete_matrix[:,3].reshape(-1, 1))
search.fit(X, y)
# report performance
best_output = scalery.inverse_transform(search.best_estimator_.predict(X))
print("best_output =", best_output)
#Now produce a scatter plot of predicted versus actual value from matrix generated
#earlier
plt.scatter(np.log10(complete_matrix[:,3]), np.log10(best_output))
plt.xlabel("log(Original Model)")
plt.ylabel("log(Machine Learning prediction)")
plt.show()
pdb.set_trace()
################################################################################
