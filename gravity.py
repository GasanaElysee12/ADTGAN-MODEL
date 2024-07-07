import numpy as np 



def Gravity(lat):
    

  return 9.78031846*(1.0000+ 0.005324*(np.sin(np.radians(lat)))**2 - 58e-7*(np.sin(2.*np.radians(lat)))**2)

# print(Gravity(LAT))np.radians()