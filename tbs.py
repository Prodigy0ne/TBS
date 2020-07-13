#pip install rayptics
#%matplotlib inline
import timeit
import pdb
import numpy as np
from rayoptics.environment import *
from rayoptics.optical.model_constants import Intfc, Gap, Indx, Tfrm, Zdir # Used to access the image semi-diameter


opm = OpticalModel() # create a new OpticalModel
sm  = opm.seq_model
osp = opm.optical_spec

osp.pupil = PupilSpec(osp, key=['object', 'pupil'], value=25)
osp.field_of_view = FieldSpec(osp, key=['object', 'height'], flds=[0.5, 0.0,-.5])  # Additional fields to account for beam width
osp.spectral_region = WvlSpec([(587.0, 1.0)]) 

opm.radius_mode = True

sm.gaps[0].thi=35 

sm.add_surface([35.0, 6.0, 'N-BK7', 'Schott']) 
sm.add_surface([-35.0, 6])
sm.set_stop() # Additional lenses to be added here

def time_func():
    opm.update_model()
print("time",timeit.Timer(time_func).timeit(number=100))
#pdb.set_trace() 
input=np.zeros((8,3),dtype=float)
output=np.zeros((8,1),dtype=float)
first_variable=[35.0,200.0]
second_variable=[4.0,10.0]
step1=0
for value1 in first_variable:
    for value2 in second_variable:
        opm = OpticalModel() # create a new OpticalModel
        sm  = opm.seq_model
        osp = opm.optical_spec

        osp.pupil = PupilSpec(osp, key=['object', 'pupil'], value=25)
        osp.field_of_view = FieldSpec(osp, key=['object', 'height'], flds=[0.5, 0.0,-.5])  # Additional fields to account for beam width
        osp.spectral_region = WvlSpec([(587.0, 1.0)]) 

        opm.radius_mode = True

        sm.gaps[0].thi=35 

        sm.add_surface([value1, value2, 'N-BK7', 'Schott']) 
        sm.add_surface([-1.0*value1, 75])
        sm.set_stop()
        opm.update_model()
        pdb.set_trace()
        for i, sg in enumerate(sm.path()):
            s = sm.list_surface_and_gap(sg[Intfc], gp=sg[Gap])
            sd = s[-1]
        print(sd) # This is the image semi-diameter  
        input[step1,0]=value1
        input[step1,1]=value2
        input[step1,2]=value1*-1.0
        output[step1,0]=sd
        step1+=1

print(input)
print(output)