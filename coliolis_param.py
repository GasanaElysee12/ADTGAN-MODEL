import numpy as np 

def Coliolis_parameter(lat):
    omega=7.29e-5
    
    f0=(2.*omega*np.sin(np.radians(lat)))
    
    

    return f0