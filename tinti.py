import numpy as np

def i1(t,Tr):
    p1 = np.pi*(4*t - Tr)*Tr + 8*t**1.5*np.sqrt(-t + Tr)
    p2 = 4*Tr*np.sqrt(t*(-t + Tr)) 
    p3 = - 2*(4*t - Tr)*Tr*np.arctan((-2*t + Tr)/(2.*np.sqrt(t*(-t + Tr))))
    return (p1+p2+p3)/16.0

def i2(t,Tr,Ts):
    p1 = (Tr*np.sqrt(t*(-t + Tr)) + 2*np.sqrt(t**3*(-t + Tr)) - t*Tr*np.sqrt((-t + Tr + Ts)/(t - Ts)) + 
          Tr*Ts*np.sqrt((-t + Tr + Ts)/(t - Ts)) - 2*t*np.sqrt((t - Ts)*(-t + Tr + Ts)) - 
          2*Ts*np.sqrt((t - Ts)*(-t + Tr + Ts)) + (4*t - Tr)*Tr*np.arcsin(np.sqrt(t/Tr)) + 
          Tr*(-4*t + Tr)*np.arcsin(np.sqrt((t - Ts)/Tr)))/4. 
          
    p2 = (-((2*t + Tr - 6*Ts)*np.sqrt((t - Ts)*(-t + Tr + Ts))) + 
          Tr*(-4*t + Tr + 8*Ts)*np.arcsin(np.sqrt((t - Ts)/Tr)))/4.
    return p1+p2

def i3(t,Tr,Ts):
    p1 = (Tr*np.sqrt(t*(-t + Tr)) + 2*np.sqrt(t**3*(-t + Tr)) - t*Tr*np.sqrt((-t + Tr + Ts)/(t - Ts)) + 
          Tr*Ts*np.sqrt((-t + Tr + Ts)/(t - Ts)) - 2*t*np.sqrt((t - Ts)*(-t + Tr + Ts)) - 
          2*Ts*np.sqrt((t - Ts)*(-t + Tr + Ts)) + (4*t - Tr)*Tr*np.arcsin(np.sqrt(t/Tr)) + 
          Tr*(-4*t + Tr)*np.arcsin(np.sqrt((t - Ts)/Tr)))/4. 
          
    p2 = (2*(-2*t*np.sqrt((t - Ts)*(-t + Tr + Ts)) - Tr*np.sqrt((t - Ts)*(-t + Tr + Ts)) + 
        6*Ts*np.sqrt((t - Ts)*(-t + Tr + Ts)) + 2*t*np.sqrt((t - 2*Ts)*(-t + Tr + 2*Ts)) + 
        Tr*np.sqrt((t - 2*Ts)*(-t + Tr + 2*Ts)) - 4*Ts*np.sqrt((t - 2*Ts)*(-t + Tr + 2*Ts))) - 
        Tr*(-4*t + Tr + 8*Ts)*np.arctan((-2*t + Tr + 2*Ts)/(2.*np.sqrt((t - Ts)*(-t + Tr + Ts)))) + 
        Tr*(-4*t + Tr + 8*Ts)*np.arctan((-2*t + Tr + 4*Ts)/(2.*np.sqrt((t - 2*Ts)*(-t + Tr + 2*Ts)))))/8.
    return p1+p2
          

def i4(t,Tr,Ts):
    p1 = t*Tr*np.arccos(np.sqrt((t - Ts)/Tr)) 
    
    p2 = (-(np.sqrt((t - Ts)*(-t + Tr + Ts))*(2*t + Tr + 2*Ts)) - 
        Tr**2*np.arccos(1.0/(np.sqrt(Tr/(t - Ts)))))/4. 
                        
    p3 = (2*(-2*t*np.sqrt((t - Ts)*(-t + Tr + Ts)) - Tr*np.sqrt((t - Ts)*(-t + Tr + Ts)) + 
        6*Ts*np.sqrt((t - Ts)*(-t + Tr + Ts)) + 2*t*np.sqrt((t - 2*Ts)*(-t + Tr + 2*Ts)) + 
        Tr*np.sqrt((t - 2*Ts)*(-t + Tr + 2*Ts)) - 4*Ts*np.sqrt((t - 2*Ts)*(-t + Tr + 2*Ts))) - 
        Tr*(-4*t + Tr + 8*Ts)*np.arctan((-2*t + Tr + 2*Ts)/(2.*np.sqrt((t - Ts)*(-t + Tr + Ts)))) + 
        Tr*(-4*t + Tr + 8*Ts)*np.arctan((-2*t + Tr + 4*Ts)/(2.*np.sqrt((t - 2*Ts)*(-t + Tr + 2*Ts)))))/8.
    return p1+p2+p3

def i5(t,Tr,Ts):
    p1 = (4*(2*t + Tr - 4*Ts)*np.sqrt(-((t - 2*Ts)*(t - Tr - 2*Ts))) + np.pi*Tr*(-4*t + Tr + 8*Ts) + 
    2*Tr*(-4*t + Tr + 8*Ts)*np.arctan((-2*t + Tr + 4*Ts)/(2.*np.sqrt((t - 2*Ts)*(-t + Tr + 2*Ts)))))/16.
    return p1

def si3(t, Tr, Ts):
    p1 = t*Tr*np.arccos(np.sqrt((t - Ts)/Tr)) + (-(np.sqrt((t - Ts)*(-t + Tr + Ts))*(2*t + Tr + 2*Ts)) - 
      Tr**2*np.arccos(1.0/np.sqrt(Tr/(t - Ts))))/4.

    p2 = (-4*(2*t + Tr - 6*Ts)*np.sqrt((t - Ts)*(-t + Tr + Ts)) + np.pi*Tr*(-4*t + Tr + 8*Ts) - 
      2*Tr*(-4*t + Tr + 8*Ts)*np.arctan((-2*t + Tr + 2*Ts)/(2.*np.sqrt((t - Ts)*(-t + Tr + Ts)))))/16.

    return p1+p2

def tinti(t, Ts, Tr, t0):
    n = len(t)
    dt = t[1]-t[0]
    
    # normalizing constant
    k = 2 / (np.pi * Tr * Ts**2)
    
    stf = np.zeros(n)
    
    if Tr > 2*Ts:
        stf[(t >= 0) & (t <= Ts)] = i1(t[(t >= 0) & (t <= Ts)], Tr)
        stf[(t > Ts) & (t < 2*Ts)] = i2(t[(t > Ts) & (t < 2*Ts)], Tr, Ts)
        stf[(t >= 2*Ts) & (t < Tr)] = i3(t[(t >= 2*Ts) & (t < Tr)], Tr, Ts)
        stf[(t >= Tr) & (t < Tr+Ts)] = i4(t[(t >= Tr) & (t < Tr+Ts)], Tr, Ts)
        stf[(t >= Tr+Ts) & (t < Tr + 2*Ts)] = i5(t[(t >= Tr+Ts) & (t < Tr + 2*Ts)], Tr, Ts)

    elif Tr > Ts and Tr <= 2*Ts:
        stf[(t >= 0) & (t <= Ts)] = i1(t[(t >= 0) & (t <= Ts)], Tr)
        stf[(t > Ts) & (t < Tr)] = i2(t[(t > Ts) & (t < Tr)], Tr, Ts)
        stf[(t >= Tr) & (t < 2*Ts)] = si3(t[(t >= Tr) & (t < 2*Ts)], Tr, Ts)
        stf[(t >= 2*Ts) & (t < Ts+Tr)] = i4(t[(t >= 2*Ts) & (t < Ts+Tr)], Tr, Ts)
        stf[(t >= Ts+Tr) & (t<2*Ts+Tr)] = i5(t[(t >= Ts+Tr) & (t<2*Ts+Tr)], Tr, Ts)

    else:
        print "Invalid parameters assigned. (tr=%f, ts=%f)" % (Tr, Ts)
    
    d = int(np.ceil((t0)/dt))
    stf = np.roll(stf, d)
    return k*stf


if __name__ == "__main__":
    # testing goes here.
    from pylab import *

    dt = 0.002
    tt = 5.0

    t = arange(0, tt, dt)

    ts = 0.1
    tr = 1.0
    t0 = 0.5

    stf = tinti(t, ts, tr, t0)

    figure()
    plot(t, stf)


    ts = 0.6
    tr = 1.0
    t0 = 0.5

    stf = tinti(t, ts, tr, t0)

    plot(t, stf)

    show()
