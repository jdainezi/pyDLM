import numpy, pdb, sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize

# cf = clamped-free, ff = free-free
def omega_nm_cf(nmodesx,nmodesy,D,L1,L2,rho,nu):
    omega_nm = numpy.zeros((nmodesx,nmodesy))
    Gx = numpy.zeros(nmodesx)
    Gy = numpy.zeros(nmodesy)
    Hx = numpy.zeros(nmodesx)
    Hy = numpy.zeros(nmodesy)
    Jx = numpy.zeros(nmodesx)
    Jy = numpy.zeros(nmodesy)
    for i in range(nmodesx):
        if i==0:
            Gx[i]=0.597
            Hx[i]=-0.087
            Jx[i]=0.471
        elif i==1:
            Gx[i]=1.494
            Hx[i]=1.347
            Jx[i]=3.284
        else: 
            Gx[i]=i+1-0.5 #+1 pq i deveria comecar em 1
            Hx[i]=(i+1-0.5)**2*(1-2/((i+1-0.5)*numpy.pi))
            Jx[i]=(i+1-0.5)**2*(1+2/((i+1-0.5)*numpy.pi))
    for j in range(nmodesy):
        if j==1:
            Jy[j]=12/numpy.pi**2
        elif j==2:
            Gy[j]=1.506
            Hy[j]=1.248
            Jy[j]=5.017
        elif j>=3: 
            Gy[j]=j-0.5 
            Hy[j]=(j-0.5)**2*(1-2/((j-0.5)*numpy.pi))
            Jy[j]=(j-0.5)**2*(1+6/((j-0.5)*numpy.pi))
    for i in range(nmodesx):
        for j in range(nmodesy):
            omega_nm[i,j] = numpy.sqrt(numpy.pi**4*D/(L1**4*rho)*(Gx[i]**4+Gy[j]**4*(L1/L2)**4+2*(L1/L2)**2*(nu*Hx[i]*Hy[j]+(1-nu)*Jx[i]*Jy[j])))
    return omega_nm

def omega_nm_ff(nmodesx,nmodesy,D,L1,L2,rho,nu):
    omega_nm = numpy.zeros((nmodesx,nmodesy))
    Gx = numpy.zeros(nmodesx)
    Gy = numpy.zeros(nmodesy)
    Hx = numpy.zeros(nmodesx)
    Hy = numpy.zeros(nmodesy)
    Jx = numpy.zeros(nmodesx)
    Jy = numpy.zeros(nmodesy)
    for i in range(nmodesx):
        if i==1:
            Jx[i]=12/numpy.pi**2
        elif i==2:
            Gx[i]=1.506
            Hx[i]=1.248
            Jx[i]=5.017
        elif i>=3: 
            Gx[i]=i-0.5 
            Hx[i]=(i-0.5)**2*(1-2/((i-0.5)*numpy.pi))
            Jx[i]=(i-0.5)**2*(1+6/((i-0.5)*numpy.pi))
    for j in range(nmodesy):
        if j==1:
            Jy[j]=12/numpy.pi**2
        elif j==2:
            Gy[j]=1.506
            Hy[j]=1.248
            Jy[j]=5.017
        elif j>=3: 
            Gy[j]=j-0.5 
            Hy[j]=(j-0.5)**2*(1-2/((j-0.5)*numpy.pi))
            Jy[j]=(j-0.5)**2*(1+6/((j-0.5)*numpy.pi))
    for i in range(nmodesx):
        for j in range(nmodesy):
            omega_nm[i,j] = numpy.sqrt(numpy.pi**4*D/(L1**4*rho)*(Gx[i]**4+Gy[j]**4*(L1/L2)**4+2*(L1/L2)**2*(nu*Hx[i]*Hy[j]+(1-nu)*Jx[i]*Jy[j])))
    return omega_nm

def calc_gama1(gama1):
    return numpy.tan(gama1/2) + numpy.tanh(gama1/2)

def calc_gama2(gama2):
    return numpy.tan(gama2/2) - numpy.tanh(gama2/2)

def calc_gama3(gama3):
    return numpy.cos(gama3)*numpy.cosh(gama3)+1

def calc_X(x,L1,gama3,mode):
    X = numpy.cos(gama3[mode]*x/L1)\
    -numpy.cosh(gama3[mode]*x/L1)\
    +((numpy.sin(gama3[mode])-numpy.sinh(gama3[mode]))\
    /(numpy.cos(gama3[mode])+numpy.cosh(gama3[mode])))\
    *(numpy.sin(gama3[mode]*x/L1)-numpy.sinh(gama3[mode]*x/L1))
    return X

def calc_Y(y,L2,gama1,gama2,mode):
    if mode == 0:
        Y[:,:] = 1
    elif mode == 1:
        Y[:,:] = 1-2*y/L2
    elif mode%2 == 0:
        Y[:,:] = numpy.cos(gama1[mode]*(y/L2-0.5)) - numpy.sin(gama1[mode]/2)/numpy.sinh(gama1[mode]/2)*numpy.cosh(gama1[mode]*(y/L2-0.5))
    elif mode%2 != 0:
        Y[:,:] = numpy.sin(gama2[mode]*(y/L2-0.5)) - numpy.sin(gama2[mode]/2)/numpy.sinh(gama2[mode]/2)*numpy.sinh(gama2[mode]*(y/L2-0.5))
    return Y

def naca_t(x):
    '''Return the thickness distribution of a naca0012 foil'''
    t = 0.12/0.2*(0.2969*x**0.5-0.126*x-0.3516*x**2+0.2843*x**3-0.1036*x**4)
    return t


def love_kirchhoff_cf(Lx,Ly,Lz,E,rho,nu,nx,ny,modesx,modesy):
    '''nx number o points in x, modesx number of modes in x'''
    #The plate is clamped in x=0 here, make sure to invert x and y in the input if it is clamped in y=0
    tp=Lz
    L1 = Lx
    L2 = Ly
    
    D=tp**3*E/(12*(1-nu**2))
    
    xsteps=nx
    ysteps=ny
    x=((numpy.ones((xsteps,ysteps))).T*numpy.linspace(L1*1.0/xsteps,L1,xsteps)).T
    y=numpy.ones((xsteps,ysteps))*numpy.linspace(0,L2,ysteps)
    
    X=numpy.zeros((xsteps,ysteps,modesx))
    Y=numpy.zeros((xsteps,ysteps,modesy))
    gama1 = numpy.zeros((modesy))
    gama2 = numpy.zeros((modesy))
    gama3 = numpy.zeros((modesx))
    
    for i in range(modesy):
        if i<= 5:
            gama1[i] = scipy.optimize.ridder(calc_gama1,(2*i-1)*numpy.pi+1.0e-12,(2*i+1)*numpy.pi)
            gama2[i] = scipy.optimize.ridder(calc_gama2,(2*i-1)*numpy.pi+1.0e-12,(2*i+1)*numpy.pi)
        else:
            gama1[i] = ((i*4+3)*numpy.pi)/2
            gama2[i] = ((i*4+1)*numpy.pi)/2
            
    for i in range(modesx):
        gama3[i] = scipy.optimize.ridder(calc_gama3,(i)*numpy.pi,(i+1)*numpy.pi)
    omega = omega_nm_cf(modesx,modesy,D,L1,L2,rho,nu)
    dx = L1/xsteps
    dy = L2/ysteps
    
    for i in range(modesx):
        X[:,:,i] = numpy.cos(gama3[i]*x/L1)\
        -numpy.cosh(gama3[i]*x/L1)\
        +((numpy.sin(gama3[i])-numpy.sinh(gama3[i]))\
        /(numpy.cos(gama3[i])+numpy.cosh(gama3[i])))\
        *(numpy.sin(gama3[i]*x/L1)-numpy.sinh(gama3[i]*x/L1))
    for j in range(modesy):
        if j == 0:
            Y[:,:,j] = 1
        elif j == 1:
            Y[:,:,j] = 1-2*y/L2
        elif j%2 == 0:
            Y[:,:,j] = numpy.cos(gama1[j]*(y/L2-0.5)) - numpy.sin(gama1[j]/2)/numpy.sinh(gama1[j]/2)*numpy.cosh(gama1[j]*(y/L2-0.5))
        elif j%2 != 0:
            Y[:,:,j] = numpy.sin(gama2[j]*(y/L2-0.5)) - numpy.sin(gama2[j]/2)/numpy.sinh(gama2[j]/2)*numpy.sinh(gama2[j]*(y/L2-0.5))
    
    phi = numpy.zeros((nx*ny,modesx*modesy))
    for i in range(modesx):
        for j in range(modesy):
            phi[:,i*modesy+j] = (X[:,:,i]*Y[:,:,j]).flatten()

    omega = omega.flatten() 
    return phi, omega

def love_kirchhoff_ff(Lx,Ly,Lz,E,rho,nu,nx,ny,modesx,modesy):
    '''nx number o points in x, modesx number of modes in x'''
    #The plate is clamped in x=0 here, make sure to invert x and y in the input if it is clamped in y=0
    tp=Lz
    L1 = Lx
    L2 = Ly
    
    D=tp**3*E/(12*(1-nu**2))
    
    xsteps=nx
    ysteps=ny
    x=((numpy.ones((xsteps,ysteps))).T*numpy.linspace(L1*1.0/xsteps,L1,xsteps)).T
    y=numpy.ones((xsteps,ysteps))*numpy.linspace(0,L2,ysteps)
    
    X=numpy.zeros((xsteps,ysteps,modesx))
    Y=numpy.zeros((xsteps,ysteps,modesy))
    gama1 = numpy.zeros((max([modesy,modesx])))
    gama2 = numpy.zeros((max([modesy,modesx])))
    gama3 = numpy.zeros((modesx))
    
    for i in range(max([modesy,modesx])):
        if i<= 5:
            gama1[i] = scipy.optimize.ridder(calc_gama1,(2*i-1)*numpy.pi+1.0e-12,(2*i+1)*numpy.pi)
            gama2[i] = scipy.optimize.ridder(calc_gama2,(2*i-1)*numpy.pi+1.0e-12,(2*i+1)*numpy.pi)
        else:
            gama1[i] = ((i*4+3)*numpy.pi)/2
            gama2[i] = ((i*4+1)*numpy.pi)/2
            
    for i in range(modesx):
        gama3[i] = scipy.optimize.ridder(calc_gama3,(i)*numpy.pi,(i+1)*numpy.pi)
    omega = omega_nm_ff(modesx,modesy,D,L1,L2,rho,nu)
    dx = L1/xsteps
    dy = L2/ysteps
    
    for i in range(modesx):
        if i == 0:
            X[:,:,i] = 1
        elif i == 1:
            X[:,:,i] = 1-2*x/L1
        elif i%2 == 0:
            X[:,:,i] = numpy.cos(gama1[i]*(x/L1-0.5)) - numpy.sin(gama1[i]/2)/numpy.sinh(gama1[i]/2)*numpy.cosh(gama1[i]*(x/L1-0.5))
        elif i%2 != 0:
            X[:,:,i] = numpy.sin(gama2[i]*(x/L1-0.5)) - numpy.sin(gama2[i]/2)/numpy.sinh(gama2[i]/2)*numpy.sinh(gama2[i]*(x/L1-0.5))
    for j in range(modesy):
        if j == 0:
            Y[:,:,j] = 1
        elif j == 1:
            Y[:,:,j] = 1-2*y/L2
        elif j%2 == 0:
            Y[:,:,j] = numpy.cos(gama1[j]*(y/L2-0.5)) - numpy.sin(gama1[j]/2)/numpy.sinh(gama1[j]/2)*numpy.cosh(gama1[j]*(y/L2-0.5))
        elif j%2 != 0:
            Y[:,:,j] = numpy.sin(gama2[j]*(y/L2-0.5)) - numpy.sin(gama2[j]/2)/numpy.sinh(gama2[j]/2)*numpy.sinh(gama2[j]*(y/L2-0.5))
    
    phi = numpy.zeros((nx*ny,modesx*modesy))
    for i in range(modesx):
        for j in range(modesy):
            phi[:,i*modesy+j] = (X[:,:,i]*Y[:,:,j]).flatten()

    omega = omega.flatten() 
    return phi, omega

def euler_modes_cf(L,EI,mi,x,nmodes):
    phi = []
    omega_n = []
    beta = numpy.array([1.875,4.6941,7.8547,10.9955])/L
    for i in range(nmodes):
        try:
            betak = beta[i]
        except:
            betak = (2*(i+1)-1)*numpy.pi/(2*L)
        sigmak = (numpy.sin(betak*L)-numpy.sinh(betak*L))/(numpy.cos(betak*L)+numpy.cosh(betak*L))
        if betak == 0:
            phi.append(numpy.ones_like(x))
        elif i<10:
            phi.append(numpy.cos(betak*x)-numpy.cosh(betak*x)+sigmak*(numpy.sin(betak*x)-numpy.sinh(betak*x)))
        else:
            phi.append(numpy.cos(betak*x)-numpy.sin(betak*x)-numpy.exp(-betak*x)-numpy.exp(betak*x-betak*L)*(numpy.sin(betak*L)))
        omega_n.append(betak**2*numpy.sqrt(EI/mi))
    phi = numpy.array(phi).T
    omega_n = numpy.array(omega_n)
    
    return phi, omega_n

def euler_modes_ff(L,EI,mi,x,nmodes):
    phi = []
    omega_n = []
    beta = numpy.array([0,1.50562,2.49975,3.50001,4.5])/L*numpy.pi
    for i in range(nmodes):
        try:
            betak = beta[i]
        except:
            betak = (i+0.5)*numpy.pi/L
        if betak == 0:
            phi.append(numpy.ones_like(x))
        elif i<10:
            sigmak = (numpy.sin(betak*L)-numpy.sinh(betak*L))/(numpy.cos(betak*L)-numpy.cosh(betak*L))
            phi.append(numpy.sin(betak*x)-numpy.sinh(betak*x)-sigmak*(numpy.cos(betak*x)-numpy.cosh(betak*x)))
        else:
            sigmak = (numpy.sin(betak*L)-numpy.sinh(betak*L))/(numpy.cos(betak*L)-numpy.cosh(betak*L))
            phi.append(numpy.exp(-betak*x)+numpy.cos(betak*x)-numpy.sin(betak*x)*(1+(numpy.exp(-betak*L)-numpy.cos(betak*L)+numpy.sin(betak*L))/(numpy.sinh(betak*L)-numpy.sin(betak*L)))-(numpy.exp(-betak*L)-numpy.cos(betak*L)+numpy.sin(betak*L))*numpy.exp(betak*(x-L)))
        omega_n.append(betak**2*numpy.sqrt(EI/mi))
    phi = numpy.array(phi).T
    omega_n = numpy.array(omega_n)
    
    return phi, omega_n
