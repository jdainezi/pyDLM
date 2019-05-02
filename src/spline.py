import numpy, pdb
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy.optimize

def IPS(xstr,ystr,xa,ya,xpc,ypc,area,dof_node,opt):

    #INFINITE PLATE SPLINE (IPS) METHOD
    # opt = 1 => wa3 solution. GG matrix calculated by the meshes displacements
    # opt = 2 => waN solution. GG matrix calculated by the meshes angular displacements
    
    pan_ya  = xa.shape[1]
    pan_xa  = xa.shape[0]
    pan_ys  = xstr.shape[1]
    pan_xs  = xstr.shape[0]
    n_pan   = pan_xa*pan_ya             # all aerodynamic panels
    n_nodes = (pan_xs)*(pan_ys) # Number of nodes, including the clamped ones.
    ## Variable changes: transforming matrices into vectors!
    
    xstr1 = numpy.zeros((n_nodes)) 
    ystr1 = numpy.zeros((n_nodes,1))
    xstr1 = xstr.flatten()
    ystr1[:,0] = ystr.flatten()   
 
    xa1  = numpy.zeros((n_pan)) 
    ya1  = numpy.zeros((n_pan))
    xpc1 = numpy.zeros((n_pan)) 
    ypc1 = numpy.zeros((n_pan))

    xa1 = xa.flatten()
    ya1 = ya.flatten()
    xpc1 = xpc.flatten()
    ypc1 = ypc.flatten()   
 
    # TEST
    # xa2  = xa
    # ya2  = ya
    # xa12 = xa1
    # ya12 = ya1
    # 
    # clear xa ya xa1 ya1
    # 
    # load('Discretized_variables_original','xdl','ydl','xbox','ybox','xpc','ypc','S','xe','ye','xa','ya','xa1','ya1')
    # 
    # max(max(xstr1 - ye)) # OK!
    # max(max(ystr1 - xe)) # OK!
    # max(max(xa12 - ya))  # OK!
    # max(max(ya12 - xa))  # OK!
    # max(max(xpc1 - ya1)) # OK!
    # max(max(ypc1 - xa1)) # OK!
    # 
    # xa1 = xa12 ya1 = ya12 xa = xa2 ya = ya2
    # clear xa2 ya2 xa12 ya12
    # END
    
    ## M Matrix for force calculation using displacement
    
    # The matrix MM calculates the relation between suppose forces applied at
    # determined point of the structutres knowing the displacement at the same
    # determied points
    
    MM_disp = numpy.zeros((n_nodes + 3,n_nodes + 3))
    rr      = numpy.zeros((n_nodes + 3,n_nodes + 3))
    for ii in range(0,n_nodes):
        for jj in range(0,n_nodes):
    
            MM_disp[0,jj + 3] = 1
            MM_disp[2,jj + 3] = ystr1[jj]
            MM_disp[1,jj + 3] = xstr1[jj]
            MM_disp[jj + 3,0] = 1
            MM_disp[jj + 3,2] = ystr1[jj]
            MM_disp[jj + 3,1] = xstr1[jj]
            rr[ii + 3,jj + 3] = (xstr1[jj] - xstr1[ii])**2 + (ystr1[jj] - ystr1[ii])**2
    
            if rr[ii + 3,jj + 3] == 0:
                MM_disp[ii + 3,jj + 3] = 0
            else:
                MM_disp[ii + 3,jj + 3] = rr[ii + 3,jj + 3]*numpy.log(rr[ii + 3,jj + 3])
    
    ## MM_rot matrix for 'angular force' calculation using chord rotation
    
    # The matrix MM_rot calculates the relation between suppose 'angular forces' 
    # applied at determined point of the structutres knowing the rotation at 
    # the same determined points
    
    MM_rot = numpy.zeros((n_nodes + 1,n_nodes + 1))
    rn     = numpy.zeros((n_nodes + 1,n_nodes + 1))
    xn     = numpy.zeros((n_nodes + 1,n_nodes + 1))
    for ii in range(0,n_nodes):
        for jj in range(0,n_nodes):
    
            MM_rot[0,ii + 1] = 1
            MM_rot[ii + 1,0] = 1
    
            rn[ii + 1,jj + 1] = (ystr1[ii] - ystr1[jj])**2 + (xstr1[ii] - xstr1[jj])**2
            xn[ii + 1,jj + 1] = (ystr1[ii] - ystr1[jj])
    
            if rn[ii + 1,jj + 1] == 0:
    #             MM_rot(ii + 1,jj + 1) = 0 # Pay attetion at this added noise
                MM_rot[ii + 1,jj + 1] = 1 # Equal to Wander's original program!
            else:
                MM_rot[ii + 1,jj + 1] = 2*(1 + numpy.log(rn[ii + 1,jj + 1]))*(xn[ii + 1,jj + 1])
    
    ## T1 Matrix for relate force (structural mesh) to displacement (aerodynamic mesh)
    
    # This matrix relates the force described in the structural mesh to one
    # displacement on the aerodynamic mesh of the doublet force localization
    
    T1 = numpy.zeros((n_pan,n_nodes + 3))
    r1 = numpy.zeros((n_pan,n_nodes + 3))
    
    for ii in range(0,n_pan):
    
        T1[ii,0] = 1
        T1[ii,2] = ya1[ii]
        T1[ii,1] = xa1[ii]
    
        for jj in range(0,n_nodes):
    
            r1[ii,jj + 3] = (ya1[ii] - ystr1[jj])**2 + (xa1[ii] - xstr1[jj])**2
    
            if r1[ii,jj + 3] == 0:
                T1[ii,jj + 3] = 0
            else:
                T1[ii,jj + 3] = r1[ii,jj+ 3]*numpy.log(r1[ii,jj + 3])
    
    SS = T1.T*area
    
    ## D1 Matrix for relate moment (structural mesh) to rotation (aerodynamic mesh)
    
    # This matrix relates the moment described in the structural mesh to one
    #  rotation at the aerodynamic mesh on the control point localization
    
    D1 = numpy.zeros((n_pan,n_nodes + 1))
    r1 = numpy.zeros((n_pan,n_nodes + 1))
    x1 = numpy.zeros_like(r1)
    for ii in range(n_pan):
    
        D1[ii,0] = 1 
    
        for jj in range(n_nodes):
    
            r1[ii,jj + 1] = (xpc1[ii] - xstr1[jj])**2 + (ypc1[ii] - ystr1[jj])**2
            x1[ii,jj + 1] = (xpc1[ii] - xstr1[jj]) 
    
            if r1[ii,jj + 1] == 0:
                D1[ii,jj + 1] = 0
            else:
                D1[ii,jj + 1] = 2*(1 + numpy.log(r1[ii,jj + 1]))*(x1[ii,jj + 1]) 
    D1 = numpy.hstack([numpy.zeros((n_pan,1)),D1[:,0].reshape(n_pan,1),numpy.zeros((n_pan,1)),D1[:,1:]]) # Resizing D1 in order to have the same size of D2. Resizing: adding first and third column (zeros).
    
    ## D2 Matrix for relate force (structural mesh) to displacement (aerodynamic mesh)
    
    # This matrix relates the force described in the structural mesh to one
    # displacement at the aerodynamic mesh on the control velocity point localization
    
    D2 = numpy.zeros((n_pan,n_nodes + 3))
    r2 = numpy.zeros((n_pan,n_nodes + 3))
    for ii in range(n_pan):
    
        D2[ii,0] = 1
        D2[ii,2] = ypc1[ii]
        D2[ii,1] = xpc1[ii]
    
        for jj in range(n_nodes):
    
            r2[ii,jj + 3] = (ypc1[ii] - ystr1[jj])**2 + (xpc1[ii] - xstr1[jj])**2
    
            if r2[ii,jj + 3] == 0:
                D2[ii,jj + 3] = 0
            else:
                D2[ii,jj + 3] = r2[ii,jj + 3]*numpy.log(r2[ii,jj + 3])
    
    ## Spline matrix GG
    
    if opt == 1:
        # GG matrix calculated by the meshes displacements
        try:
            GG1 = numpy.linalg.inv(MM_disp)
        except:
            raise ValueError('Check if your structural grids are aligned. If they do not define a plane, try opt=2')
       #ind = numpy.hstack([range(0,3),numpy.arange(3,GG1.shape[0],(pan_xs)) #clamped nodes?])
        ind = range(0,3)
        GG1 = numpy.delete(GG1,ind,1)
        
        # Resizing GG matrix.
       #GG = numpy.zeros((GG1.shape[0],(pan_ys)*(pan_xs-1 #clamped?)*dof_node))
        GG = numpy.zeros((GG1.shape[0],(pan_ys)*(pan_xs)*dof_node))
        if dof_node >= 3:
            for ii in range(GG1.shape[1]):
                GG[:,2+dof_node*(ii)] = GG1[:,ii]
        else:
            GG=GG1
        
        
        # TESTE: FUNCIONA PARA 1 CASO GERAL! matre_teste_100.mat
    #     GG1 = inv(MM_disp) GG1(:,1:3) = [] # PRA TODOS OS GDL, SO TIRA AS 3 PRIMEIRAS COLUNAS
    #     
    #     # Resizing GG matrix.
    #     GG = zeros(size(GG1,1),(pan_xs + 1)*(pan_ys + 1)*dof_node) ## GG PRA TODOS OS GDL
    #     for ii = 1:size(GG1,2)
    #         GG(:,1 + dof_node*(ii - 1)) = GG1(:,ii) # GG SO PROS GDLs VERTICAIS
    #     end
    #     load ('C:\Users\user\Desktop\Mestrado\2011\NOVO\Nova_Versao\Structural Data\matre_teste_100.mat','dof_free')
    #     
    #     GG = GG(:,dof_free) # SO OS GDLs LIVRES
        # END
        
    elif opt == 2:
        # GG matrix calculated by the meshes angular displacements
        GG1 = numpy.linalg.inv(MM_rot)
       #ind = numpy.hstack([0,range(1,GG1.shape[0],(pan_ys))#clamped?])
        ind = 0
        GG1 = numpy.delete(GG1,ind,1)
        GG1 = numpy.vstack([numpy.zeros((1,GG1.shape[1])),GG1[0,:].reshape(1,GG1.shape[1]),numpy.zeros((1,GG1.shape[1])),GG1[1:,:]])
        
        # Resizing GG matrix.
       #GG = numpy.zeros((GG1.shape[0],(pan_xs)*(pan_ys-1 ##clamped?)*dof_node))
        GG = numpy.zeros((GG1.shape[0],(pan_xs)*(pan_ys)*dof_node))
        for ii in range(GG1.shape[1]):
            GG[:,2 + dof_node*(ii)] = GG1[:,ii]

    return GG,SS,D1,D2 

def beam(xstr,ystr,xa,ya,xpc,ypc,area,EI,dof_node,aoa=0):
    sweep = numpy.arctan((xstr[0,-1]-xstr[0,0])/(ystr[0,-1]-ystr[0,0]))
    Ts = numpy.eye(3)
    Ts[0:2,0:2] = numpy.array([numpy.cos(sweep),numpy.sin(sweep),-numpy.sin(sweep),numpy.cos(sweep)]).reshape(2,2)
    nstruct = len(xstr.T)
    npx = xa.shape[0]
    npy = xa.shape[1]
    xa = xa.flatten()
    ya = ya.flatten()
    xpc = xpc.flatten()
    ypc = ypc.flatten()
    struct_coord = numpy.vstack([xstr,ystr,numpy.zeros_like(ystr)])
    ac_coord = numpy.vstack([xa,ya,numpy.zeros_like(ya)])
    pc_coord = numpy.vstack([xpc,ypc,numpy.zeros_like(ypc)])

    #transforming points to local beam coordinates
    struct_coord_prime = numpy.dot(Ts,struct_coord)
    ac_coord_prime = numpy.dot(Ts,ac_coord)
    pc_coord_prime = numpy.dot(Ts,pc_coord)
    xa_prime = ac_coord_prime[0,:].reshape(1,npx*npy)
    ya_prime = ac_coord_prime[1,:].reshape(1,npx*npy)
    xpc_prime = pc_coord_prime[0,:].reshape(1,npx*npy)
    ypc_prime = pc_coord_prime[1,:].reshape(1,npx*npy)
    xstr_prime = struct_coord_prime[0,:].reshape(1,nstruct)
    ystr_prime = struct_coord_prime[1,:].reshape(1,nstruct)

    if dof_node == 1:
        R1 = numpy.hstack([numpy.ones((nstruct,1)),ystr_prime.T])
        A11 = numpy.zeros((nstruct,nstruct))
        eta = numpy.zeros((nstruct,nstruct))
        for i in range(nstruct):
            for j in range(nstruct):
                eta[i,j] = ystr_prime[0,j]-ystr_prime[0,i]
                if j<i:
                    A11[i,j] = eta[i,j]**3/(12*EI)
                else:
                    A11[i,j] = -eta[i,j]**3/(12*EI)
        C = numpy.vstack([numpy.hstack([numpy.zeros((2,2)),R1.T]),numpy.hstack([R1,A11])])
        D2 = numpy.zeros((npx*npy,nstruct+2))
        SS = numpy.zeros((nstruct+2,npx*npy))
        SS[0:2,:] = numpy.vstack([numpy.ones((1,npx*npy)),ypc_prime])
        D2[:,0:2] = numpy.hstack([numpy.ones((npx*npy,1)),ypc_prime.T])
        for i in range(npx*npy):
            for j in range(nstruct):
                D2[i,2+j] = numpy.absolute(ypc_prime[0,i]-ystr_prime[0,j])**3/(12*EI)
                SS[2+j,i] = numpy.absolute(ya_prime[0,i]-ystr_prime[0,j])**3/(12*EI)
        GG = numpy.delete(numpy.linalg.inv(C),[0,1],1)
        SS = numpy.dot(SS,numpy.eye(npx*npy)*area)
       #D1 = numpy.dot(numpy.linalg.pinv(GG),numpy.ones(nstruct)*aoa)
        D1 = numpy.zeros((npx*npy,nstruct+2))
    
    return GG,SS,D1,D2

def RBA(xstr,ystr,xa,ya,xpc,ypc,area):
    #Rigid-body attachment Method, works for 6 dof only
    nx = xstr.shape[0]
    ny = xstr.shape[1]
    npx = xa.shape[0]
    npy = xa.shape[1]
    xstr = xstr.flatten()
    ystr = ystr.flatten()
    xa = xa.flatten()
    ya = ya.flatten()
    xpc = xpc.flatten()
    ypc = ypc.flatten()
    D1 = numpy.zeros((npx*npy,6*nx*ny))
    D2 = numpy.zeros((npx*npy,6*nx*ny))
    SS = numpy.zeros((6*nx*ny,npx*npy))
    GG = numpy.eye(6*nx*ny)
    D1[:,4::6] = 1
    for k in range(npx*npy):
        for i in range(nx*ny):
            D2[k,i*6:(i+1)*6] = numpy.array([0,0,1,ypc[k]-ystr[i],-(xpc[k]-xstr[i]),0])
            SS[i*6:(i+1)*6,k] = numpy.array([0,0,1,ya[k]-ystr[i],-(xa[k]-xstr[i]),0])
    SS = numpy.dot(SS,numpy.eye(npx*npy)*area)

    return GG,SS,D1,D2

def linear_GAF(GAF,k):
    #real and imaginary coeff of y = ax + b
    ar = numpy.zeros((GAF.shape[0],GAF.shape[1]))
    br = numpy.zeros((GAF.shape[0],GAF.shape[1]))
    ai = numpy.zeros((GAF.shape[0],GAF.shape[1]))
    bi = numpy.zeros((GAF.shape[0],GAF.shape[1]))
    for i in range(GAF.shape[0]):
        for j in range(GAF.shape[1]):
            #imag part
            fitfunc = lambda params, x: params[0] * x
            errfunc = lambda p, x, y: fitfunc(p, x) - y #create error function for least squares fit

            init_a = 0.5    #find initial value for a (gradient)
          # init_b = 0.0
          # init_p = numpy.array((init_a,init_b)) 
            init_p = numpy.array((init_a)) 

            #calculate best fitting parameters (i.e. m and b) using the error function
            p1, success = scipy.optimize.leastsq(errfunc, init_p.copy(), args = (k, GAF[i,j,:].imag))
            ai[i,j] = p1[0]
            bi[i,j] = 0#p1[1]
            #real part
            fitfunc = lambda params, x: params[0] * x + params[1]
            errfunc = lambda p, x, y: fitfunc(p, x) - y #create error function for least squares fit

            init_a = 0.5    #find initial value for a (gradient)
            init_b = 0.0
            init_p = numpy.array((init_a,init_b)) 

            #calculate best fitting parameters (i.e. m and b) using the error function
            p1, success = scipy.optimize.leastsq(errfunc, init_p.copy(), args = (k, GAF[i,j,:].real))
            ar[i,j] = p1[0]
            br[i,j] = p1[1]
            
    return ar, br, ai, bi

def interp_GAF(GAF,k,newk):
    f = interp1d(k,GAF) 
    newGAF = f(newk)
    return newGAF
