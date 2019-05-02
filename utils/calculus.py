'''Code to perform 1st, 2nd, 3rt or 4th derivatives of a function
 using 4th order finite difference scheme in the centre and 2nd 
order in the borders, integration and Laplacian'''
import numpy, scipy.interpolate

def dy_dx(y,dx):
    dy = numpy.zeros_like(y)
    for i in range(len(y)):
        if i in [0,1]:
            dy[i]=(-1.5*y[i]+2*y[i+1]-0.5*y[i+2])/dx
        elif i in [len(y)-1,len(y)-2]:
            dy[i]=-(-1.5*y[i]+2*y[i-1]-0.5*y[i-2])/dx
        else:
            dy[i]=(1.0/12*y[i-2]-2.0/3*y[i-1]+2.0/3*y[i+1]-1.0/12*y[i+2])/dx
    return dy

def d2y_dx2(y,dx):
    d2y = numpy.zeros_like(y)
    for i in range(len(y)):
        if i in [0,1]:
            d2y[i]=(2*y[i]-5*y[i+1]+4*y[i+2]-1*y[i+3])/(dx)**2
        elif i in [len(y)-1,len(y)-2]:
            d2y[i]=(2*y[i]-5*y[i-1]+4*y[i-2]-1*y[i-3])/(dx)**2
        else:
            d2y[i]=(-1.0/12*y[i-2]+4.0/3*y[i-1]-5.0/2*y[i]+4.0/3*y[i+1]-1.0/12*y[i+2])/(dx)**2
    return d2y

def d3y_dx3(y,dx):
    d3y = numpy.zeros_like(y)
    for i in range(len(y)):
        if i in [0,1,2]:
            d3y[i]=(-2.5*y[i]+9*y[i+1]-12*y[i+2]+7*y[i+3]-1.5*y[i+4])/(dx)**3
        elif i in [len(y)-1,len(y)-2,len(y)-3]:
            d3y[i]=(-2.5*y[i]+9*y[i-1]-12*y[i-2]+7*y[i-3]-1.5*y[i-4])/(dx)**3
        else:
            d3y[i]=(1.0/8*y[i-3]-1.0*y[i-2]+13.0/8*y[i-1]-13.0/8*y[i+1]+1.0*y[i+2]-1.0/8*y[i+3])/(dx)**3
    return d3y

def d4y_dx4(y,dx):
    d4y = numpy.zeros_like(y)
    for i in range(len(y)):
        if i in [0,1,2]:
            d4y[i]=(3*y[i]-14*y[i+1]+26*y[i+2]-24*y[i+3]+11*y[i+4]-2*y[i+5])/(dx)**4
        elif i in [len(y)-1,len(y)-2,len(y)-3]:
            d4y[i]=(3*y[i]-14*y[i-1]+26*y[i-2]-24*y[i-3]+11*y[i-4]-2*y[i-5])/(dx)**4
        else:
            d4y[i]=(-1.0/6*y[i-3]+2.0*y[i-2]-13.0/2*y[i-1]+28.0/3*y[i]-13.0/2*y[i+1]+2.0*y[i+2]-1.0/6*y[i+3])/(dx)**4
    return d4y

def laplacian_2d(xy,dx,dy):
    '''xy eh o vetor 2d dos valores da funcao'''
    lap = numpy.zeros_like(xy)
    for line in range(xy.shape[0]):
        lap[line,:] += d2y_dx2(xy[line,:],dy)
    for col in range(xy.shape[1]):
        lap[:,col] += d2y_dx2(xy[:,col],dx)
    return lap

def integrate_1d(fx,h):
    '''fx eh o vetor dos valores das funcoes, h eh o espacamento do intervalo. Usa o metodo 3/8 de Simpson'''
    Nx = len(fx)
    x = numpy.linspace(0,1,Nx)
    xnew = numpy.linspace(0,1,3*Nx+1)
    fun = scipy.interpolate.UnivariateSpline(x,fx)
    fx = fun(xnew)
    h *= Nx/(Nx*3+1.0)
    integ = 0.0
    for i in range(3*Nx+1):
        if i in [0,3*Nx]:
            integ += 3*h/8*(fx[i])
        elif i%3 == 0:
            integ += 3*h/8*(2*fx[i])
        else:
            integ += 3*h/8*(3*fx[i])
    return integ

def integrate_2d(fxy,hx,hy):
    '''fxy eh a matriz dos valores das funcoes, hx e hy sao os espacamentos do intervalo. Usa o metodo 3/8 de Simpson. Dominio deve ser retangular'''
    Nx = fxy.shape[0]
    Ny = fxy.shape[1]
    x = numpy.linspace(0,1,Nx)
    y = numpy.linspace(0,1,Ny)
    xnew = numpy.linspace(0,1,3*Nx+1)
    ynew = numpy.linspace(0,1,3*Ny+1)
    fun = scipy.interpolate.interp2d(x,y,fxy, kind='cubic')
    fxy = fun(xnew,ynew).T
    hx *= Nx/(Nx*3+1.0)
    hy *= Ny/(Ny*3+1.0)
    integ2d = numpy.zeros(3*Ny+1)
    for i in range(3*Nx+1):
        for j in range(3*Ny+1):
            if i in [0,3*Nx]:
                integ2d[j] += 3*hx/8*(fxy[i,j])
            elif i%3 == 0:
                integ2d[j] += 3*hx/8*(2*fxy[i,j])
            else:
                integ2d[j] += 3*hx/8*(3*fxy[i,j])
    integ = 0.0
    for i in range(3*Ny+1):
        if i in [0,3*Ny]:
            integ += 3*hy/8*(integ2d[i])
        elif i%3 == 0:
            integ += 3*hy/8*(2*integ2d[i])
        else:
            integ += 3*hy/8*(3*integ2d[i])
    return integ

