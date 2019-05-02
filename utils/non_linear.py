import numpy, math, pdb, copy
from calculus import *
from dynamics import *
import scipy.optimize

def magnet_force(zi,mu0,Br,A1,A2,a1,a2,r1,r2,t):
    if type(zi) == numpy.ndarray:
        FL = numpy.zeros_like(zi)
        FR = numpy.zeros_like(zi)
        for i in range(len(zi)):
            z = zi[i][0]
            if (a2-z)>r2:
                FL[i] = Br**2*A2**2*(t+r2)**2*((a2-z)**(-2) +(a2-z+2*t)**(-2) -2*(a2-z+t)**(-2))/(numpy.pi*mu0*t**2)
            else:
                FL[i] = Br**2*A2**2*(t+r2)**2*((r2)**(-2) +(r2+2*t)**(-2) -2*(r2+t)**(-2))/(numpy.pi*mu0*t**2)
            if (a1+z)>r1:
                FR[i] = Br**2*A1**2*(t+r1)**2*((a1+z)**(-2) +(a1+z+2*t)**(-2) -2*(a1+z+t)**(-2))/(numpy.pi*mu0*t**2)
            else:
                FR[i] = Br**2*A1**2*(t+r1)**2*((r1)**(-2) +(r1+2*t)**(-2) -2*(r1+t)**(-2))/(numpy.pi*mu0*t**2)
    else:
        z = zi
        if (a2-z)>r2:
            FL = Br**2*A2**2*(t+r2)**2*((a2-z)**(-2) +(a2-z+2*t)**(-2) -2*(a2-z+t)**(-2))/(numpy.pi*mu0*t**2)
        else:
            FL = Br**2*A2**2*(t+r2)**2*((r2)**(-2) +(r2+2*t)**(-2) -2*(r2+t)**(-2))/(numpy.pi*mu0*t**2)
        if (a1+z)>r1:
            FR = Br**2*A1**2*(t+r1)**2*((a1+z)**(-2) +(a1+z+2*t)**(-2) -2*(a1+z+t)**(-2))/(numpy.pi*mu0*t**2)
        else:
            FR = Br**2*A1**2*(t+r1)**2*((r1)**(-2) +(r1+2*t)**(-2) -2*(r1+t)**(-2))/(numpy.pi*mu0*t**2)
            
    Ft = FL - FR
    return Ft

def magnet_force_derivative(zi,mu0,Br,A1,A2,a1,a2,r1,r2,t):
    if type(zi) == numpy.ndarray:
        FL = numpy.zeros_like(zi)
        FR = numpy.zeros_like(zi)
        for i in range(len(zi)):
            z = zi[i][0]
            if (a2-z)>r2:
                FL[i] = Br**2*A2**2*(t+r2)**2*(2*(a2-z)**(-3) +2*(a2-z+2*t)**(-3) -4*(a2-z+t)**(-3))/(numpy.pi*mu0*t**2)
            else:
                FL[i] = 0.01 
            if (a1+z)>r1:
                FR[i] = Br**2*A1**2*(t+r1)**2*(-2*(a1+z)**(-3) -2*(a1+z+2*t)**(-3) +4*(a1+z+t)**(-3))/(numpy.pi*mu0*t**2)
            else:
                FR[i] = 0.01
    else:
        z = zi
        if (a2-z)>r2:
            FL = Br**2*A2**2*(t+r2)**2*(2*(a2-z)**(-3) +2*(a2-z+2*t)**(-3) -4*(a2-z+t)**(-3))/(numpy.pi*mu0*t**2)
        else:
            FL = 0.01 
        if (a1+z)>r1:
            FR = Br**2*A1**2*(t+r1)**2*(-2*(a1+z)**(-3) -2*(a1+z+2*t)**(-3) +4*(a1+z+t)**(-3))/(numpy.pi*mu0*t**2)
        else:
            FR = 0.01
    
    Ft = FL - FR
    return Ft

#def x(A0,An,Bn,t,D,nabs,a1,a2,c,s):
#    f = A0 + An.dot(c(t)) + Bn.dot(s(t))
#    for i in range(nabs):
#        if f[D/2-nabs+i] < -a1:
#            f[D/2-nabs+i] = -a1
#            f[-nabs+i] = -f[-nabs+i]
#        if f[D/2-nabs+i] > a2:
#            f[D/2-nabs+i] = a2
#            f[-nabs+i] = -f[-nabs+i]
#    return f

def harmonic_balance(f,dfdx,T,n,a0,A0,B0,tol,err_type,nabs=0,a1=0,a2=0):

    #(M,C,K,freqlen,NH):

    #f: function handle for dynamical system, f = f(t,x) = f(t+T,x)
    #dfdx: function handle for gradient of f, [dfdx]_ij = dfi/dxj
    #T: period of orbit desired, x(t) = x(t+T)
    #n: Sampling Factor, n>2 to satisfy Nyquist Criterion
    #a0: initial constant vector
    #A0: initial cosine coefficent matrix
    #B0: initial sine coefficient matrix
    
    #D is dimension of state space
    #NH is number of harmonics in truncated Fourier series
   #D = 2*M.shape[0]
    D = A0.shape[0]
    NH = A0.shape[1]
    err_msg = 0
    
    #initialize solution vectors/matrices
    a = a0
    A = A0
    B = B0
    
    #FFT things
    nextpow2 = lambda n: int(math.log(n,2)) + math.ceil(math.log(n,2)-int(math.log(n,2)))
    NFFT = int(2**nextpow2(n*NH))
    dt = T/NFFT
    ts = numpy.arange(0,T,dt)
    fs = numpy.zeros((D,NFFT))
    gs = numpy.zeros((D,D,NFFT))
    
    #column vector of harmonic numbers m = [1...M]
    m = numpy.arange(1,NH+1).reshape(NH,1)

    #multi harmonic cosine and sine functions
    c = lambda t: numpy.cos(2*math.pi*m*t/T)
    s = lambda t: numpy.sin(2*math.pi*m*t/T)
    
    #approximate solution function handle
    x = lambda A0,An,Bn,t: A0 + An.dot(c(t)) + Bn.dot(s(t))

    #Initialize Jacobians
    Jaa = numpy.zeros((D,D))
    JaA = numpy.zeros((D,NH,D))
    JaB = numpy.zeros((D,NH,D))
    JAa = numpy.zeros((D,NH,D))
    JAA = numpy.zeros((D,NH,NH,D))
    JAB = numpy.zeros((D,NH,NH,D))
    JBa = numpy.zeros((D,NH,D))
    JBA = numpy.zeros((D,NH,NH,D))
    JBB = numpy.zeros((D,NH,NH,D))
    
    MaxIter = 100
    for kk in range(1,MaxIter+1):
        if kk==MaxIter:
            err_msg = 1
        
        #Time Samples
        for mfft in range(NFFT):
           #fs[:,mfft] = f(ts[mfft],x(a,A,B,ts[mfft],D,nabs,a1,a2,c,s)).reshape(D) #right hand side of ss equation
           #gs[:,:,mfft] = dfdx(ts[mfft],x(a,A,B,ts[mfft],D,nabs,a1,a2,c,s)) #state space matrix
            fs[:,mfft] = f(ts[mfft],x(a,A,B,ts[mfft])).reshape(D) #right hand side of ss equation
            gs[:,:,mfft] = dfdx(ts[mfft],x(a,A,B,ts[mfft])) #state space matrix
        F = numpy.fft.fft(fs,NFFT,1)/NFFT;
        G = numpy.fft.fft(gs,NFFT,2)/NFFT;
        
        #Residuals
        ra =  F[:,0].real
        RA =  F[:,1:NH+1].real - B.dot(numpy.diag(math.pi*m[:,0])/T)
        RB = -F[:,1:NH+1].imag + A.dot(numpy.diag(math.pi*m[:,0])/T)
        
        #Jacobians
        Jaa[:,:] = G[:,:,0].real
        for mm in m[:,0]:
            nn = mm;
            JAa[:,mm-1,:] =  G[:,:,mm].real
            JBa[:,mm-1,:] = -G[:,:,mm].imag
            JaA[:,nn-1,:] =  G[:,:,nn].real
            JaB[:,nn-1,:] = -G[:,:,nn].imag

        for mm in m[:,0]:
            for nn in m[:,0]:
                mu = mm+nn;
                nu = mm-nn;
                if nu>0:
                    JAA[:,mm-1,nn-1,:] =  (G[:,:,mu].real+G[:,:,nu].real)/2
                    JAB[:,mm-1,nn-1,:] = (-G[:,:,mu].imag+G[:,:,nu].imag)/2 - (math.pi*mm/T)*numpy.eye(D)*(mm==nn)
                    JBA[:,mm-1,nn-1,:] = (-G[:,:,mu].imag-G[:,:,nu].imag)/2 + (math.pi*mm/T)*numpy.eye(D)*(mm==nn)
                    JBB[:,mm-1,nn-1,:] = (-G[:,:,mu].real+G[:,:,nu].real)/2
                else:
                    JAA[:,mm-1,nn-1,:] =  (G[:,:,mu].real+G[:,:,-nu].real)/2
                    JAB[:,mm-1,nn-1,:] = (-G[:,:,mu].imag-G[:,:,-nu].imag)/2 - (math.pi*mm/T)*numpy.eye(D)*(mm==nn)
                    JBA[:,mm-1,nn-1,:] = (-G[:,:,mu].imag+G[:,:,-nu].imag)/2 + (math.pi*mm/T)*numpy.eye(D)*(mm==nn)
                    JBB[:,mm-1,nn-1,:] = (-G[:,:,mu].real+G[:,:,-nu].real)/2
        
        ##
        #initialize Newton's method
        u = numpy.zeros((D*(2*NH+1),1)) #vector of unknowns
        v = numpy.zeros((D*(2*NH+1),1)) #vector of residuals
        W = numpy.zeros((D*(2*NH+1),D*(2*NH+1)))  #Jacobian matrix
        
        #Pack up unknowns and residuals
        for ii in range(D):
            pa = ii
            u[pa] = a[ii]
            v[pa] = ra[ii]
            for mm in m[:,0]:
                pA = ii+D*mm
                pB = ii+D*(NH+mm)
                u[pA] = A[ii,mm-1]
                u[pB] = B[ii,mm-1]
                v[pA] = RA[ii,mm-1]
                v[pB] = RB[ii,mm-1]
        
        
        for ii in range(D):
            pa = ii
            for jj in range(D):
                qa = jj
                W[pa,qa] = Jaa[ii,jj]
        
        for mm in m[:,0]:
            for ii in range(D):
                pA = ii+D*mm
                pB = ii+D*(NH+mm)
                for jj in range(D):
                    qa = jj
                    W[pA,qa] = JAa[ii,mm-1,jj]
                    W[pB,qa] = JBa[ii,mm-1,jj]
        
        for nn in m[:,0]:
            for jj in range(D):
                qA = jj+D*nn
                qB = jj+D*(NH+nn)
                for ii in range(D):
                    pa = ii
                    W[pa,qA] = JaA[ii,nn-1,jj]
                    W[pa,qB] = JaB[ii,nn-1,jj]
        
        for mm in m[:,0]:
            for ii in range(D):
                pA = ii+D*mm
                pB = ii+D*(NH+mm)
                for nn in m[:,0]:
                    for jj in range(D):
                        qA = jj+D*nn
                        qB = jj+D*(NH+nn)
                        W[pA,qA] = JAA[ii,mm-1,nn-1,jj]
                        W[pA,qB] = JAB[ii,mm-1,nn-1,jj]
                        W[pB,qA] = JBA[ii,mm-1,nn-1,jj]
                        W[pB,qB] = JBB[ii,mm-1,nn-1,jj]
        
        ##
        #Do Newton iteration
        try:
            du = -numpy.linalg.pinv(W).dot(v) #Pseudo-Inverse
        except:
            err_msg = 1
            break
           #pdb.set_trace()
    #     du = -W\v;       #Gausssian Elimination
        uplus = u + du
        
        #Unpack unknowns
        for ii in range(D):
            pa = ii
            a[ii] = uplus[pa]
            for mm in m[:,0]:
                pA = ii+D*mm
                pB = ii+D*(NH+mm)
                A[ii,mm-1] = uplus[pA]
                B[ii,mm-1] = uplus[pB]
        
        #calculate  error
        if err_type == 'rel':
            err_vec = numpy.zeros_like(du)
            for ii in range(len(u)):
                if u[ii,0] == 0:
                    err_vec[ii,0] = du[ii,0]
                else:
                    err_vec[ii,0] = du[ii,0]/u[ii,0]
            err = numpy.max(numpy.absolute(err_vec))
        elif err_type == 'abs':
            err = numpy.max(numpy.absolute(du))
        if err<tol:
            break
    
    #output initial condition
   #x0 = x(a,A,B,0,D,nabs,a1,a2,c,s);
    x0 = x(a,A,B,0);

    return a,A,B,x0,err_msg

def area_func(K3,freqlen,phi,ndof,Mhh,ybabs,omegas,A,Mbar_inv,Knl,nmodes,nabs,ref_gridid,amplitude,Mbar,freq):
    err_type = 'abs'
    tol = 1e-10
    s = 3
    M = 4
    D = A.shape[0]
    a_aero_abs = numpy.zeros((D,1))
    A_aero_abs = numpy.zeros((D,M))
    B_aero_abs = numpy.zeros((D,M))
    Mag_aero_abs = numpy.zeros((M+1,2*freqlen))
    phi_pinv = numpy.zeros((phi.shape[1],phi.shape[0]))
    if ndof == 1:
        phi_pinv = numpy.linalg.pinv(phi)
    else:
        for dof in range(ndof):
           phi_pinv[dof::ndof] = numpy.linalg.pinv(phi[dof::ndof])
    ybhh = numpy.dot(Mhh,numpy.dot(phi_pinv,ybabs))
    time_resp = numpy.zeros((int(2*freqlen),int(2*freqlen)))
    rms = numpy.zeros(int(2*freqlen))

    for i in range(int(2*freqlen)):
        T = 2*numpy.pi/omegas[i]
        time = numpy.linspace(0,T,int(2*freqlen))
        fabs = lambda t,x: numpy.dot(A,x) + numpy.vstack([numpy.zeros((Mhh.shape[0],1)),-numpy.dot(Mbar_inv,omegas[i]**2*amplitude*numpy.sin(omegas[i]*t)*ybhh[:,0]).reshape(Mhh.shape[0],1)]) - numpy.vstack([numpy.zeros((Mhh.shape[0],1)),numpy.dot(Mbar_inv,numpy.dot(phi.T,K3*numpy.dot(Knl,numpy.dot(phi,x[nmodes+nabs:2*(nmodes+nabs)]))**3))])
        dfdxabs = lambda t,x: A - numpy.vstack([numpy.zeros((Mhh.shape[0],A.shape[1])),numpy.hstack([numpy.zeros((nmodes+nabs,nmodes+nabs)),numpy.dot(Mbar_inv,numpy.dot(phi.T,numpy.dot(3*K3*Knl*numpy.dot(Knl,numpy.dot(phi,x[nmodes+nabs:2*(nmodes+nabs)]))**2,phi)))])])
        
        anew,Anew,Bnew,x0,err = harmonic_balance(fabs,dfdxabs,T,s,a_aero_abs,A_aero_abs,B_aero_abs,tol,err_type,nabs)
        if err == 1:
            err_type = 'rel'
            anew,Anew,Bnew,x0,err = harmonic_balance(fabs,dfdxabs,T,s,a_aero_abs,A_aero_abs,B_aero_abs,tol,err_type,nabs)
            err_type = 'abs'
            if err == 1:
                a_aero_abs = numpy.zeros((D,1))
                A_aero_abs = numpy.zeros((D,M))
                B_aero_abs = numpy.zeros((D,M))
                anew,Anew,Bnew,x0,err = harmonic_balance(fabs,dfdxabs,T,s,a_aero_abs,A_aero_abs,B_aero_abs,tol,err_type,nabs)
        a_aero_abs = anew
        A_aero_abs = Anew
        B_aero_abs = Bnew
        
        Mag_aero_abs[0,i] = numpy.absolute(numpy.dot(phi[ref_gridid,:],a_aero_abs[0:Mbar.shape[0]]))
        time_resp[:,i] = numpy.dot(phi[ref_gridid,:],a_aero_abs[0:Mbar.shape[0]])
        for ii in range(M):
            Mag_aero_abs[ii+1,i] = numpy.dot(phi[ref_gridid,:],numpy.sqrt(A_aero_abs[0:Mbar.shape[0],ii]**2+B_aero_abs[0:Mbar.shape[0],ii]**2))
            time_resp[:,i] += numpy.dot(phi[ref_gridid,:],A_aero_abs[0:Mbar.shape[0],ii])*numpy.sin((ii+1)*omegas[i]*time)+numpy.dot(phi[ref_gridid,:],B_aero_abs[0:Mbar.shape[0],ii])*numpy.cos((ii+1)*omegas[i]*time)
        rms[i] = numpy.sqrt(numpy.mean(time_resp[:,i]**2))
    area = integrate_1d(numpy.log10(rms),freq[1])
    return area

def optimize_K3(struct_mesh_dict,Mtot,Mhh_old,Bhh_old,Khh_old,k,rho,semi_chord,M_old,B_old,K_old,omega_n,amplitude,freqlen,sgrids,ref_grid,phi_old,mi,K1,K30,ndof,nmodes,damp,ext_shape,x_rot):
    ref_gridid = numpy.argwhere(struct_mesh_dict['full_set']['grid_map']==ref_grid)[0][0] +1
    nabs = 0
    grid_ids = []
    ms = []
    for i,gset in enumerate(sgrids):
        nabs += len(gset)
        grid_ids.extend(gset)
    for i,gset in enumerate(sgrids):
        mset = numpy.ones(len(gset))*Mtot*float(mi[i])/len(gset)
        ms.extend(mset)
    ks = numpy.zeros(nabs)
    bs = 0.1*2*numpy.array(ms)*100

    K,B,M = insert_shock_absorbers(struct_mesh_dict,grid_ids,ks,bs,ms,M_old,B_old,K_old,ndof)
    Knl,Bnl,Mnl = insert_shock_absorbers(struct_mesh_dict,grid_ids,numpy.ones(nabs),numpy.zeros(nabs),numpy.zeros(nabs),M_old*0,B_old*0,K_old*0,ndof)
    phi = numpy.vstack([numpy.hstack([phi_old,numpy.zeros((phi_old.shape[0],nabs))]),numpy.hstack([numpy.zeros((nabs,phi_old.shape[1])),numpy.eye(nabs)])])
    Mhh = numpy.dot(numpy.dot(phi.T,M),phi)
    Khh = numpy.dot(numpy.dot(phi.T,K),phi)
    Bhh = numpy.dot(numpy.dot(phi.T,B),phi)
    ybabs = numpy.zeros((phi.shape[0],1))
    Mbar_inv = numpy.linalg.pinv(Mhh)
    Mbar = Mhh

    if ext_shape == 'bending':
        if ndof == 1:
            ybabs += 1.0
        else:
            ybabs[2::ndof] = 1.0
    elif ext_shape == 'torsion':
        x_abs = struct_mesh_dict['full_set']['x']
        for i,grid in enumerate(grid_ids):
            x_abs = numpy.hstack([x_abs,struct_mesh_dict['full_set']['x'][grid]])
        if ndof <= 3:
            ybabs += 0.1*(x_rot - x_abs).reshape(phi.shape[0],1)
        else:
            ybabs[2::ndof] += 0.1*(x_rot - x_abs).reshape(phi.shape[0],1)
            ybabs[4::ndof] += 0.1
    
    freqlen = 100    
    freq = numpy.hstack([numpy.linspace(0.1,1.1*omega_n[3]/2/numpy.pi,freqlen),numpy.linspace(1.1*omega_n[3]/2/numpy.pi,0.1,freqlen)])
    omegas = freq*numpy.pi*2
    
    A = numpy.vstack([numpy.hstack([numpy.zeros_like(Mhh),numpy.eye(Mhh.shape[0])]),\
        numpy.hstack([numpy.dot(-numpy.linalg.inv(Mhh),Khh),numpy.dot(-numpy.linalg.inv(Mhh),Bhh)])]).real
    res = scipy.optimize.minimize(area_func,K30,args=(freqlen,phi,ndof,Mhh,ybabs,omegas,A,Mbar_inv,Knl,nmodes,nabs,ref_gridid,amplitude,Mbar,freq))
    K3 = res.x[0]
    print "Optimized K3 value {} N/m3".format(K3)
    return K3


if __name__ == '__main__': 
    N=500
    W = numpy.hstack([numpy.linspace(0.8,1.2,N/2),numpy.linspace(1.2,0.8,N/2)])
    
    mu = 0.05
    muorig = 0.01
    Fs = lambda x: 0.1*x**3 + 0.01*x**2 +x
    Gs = lambda x: 0.3*x**2 +0.02*x + 1
    e = 0.001
    
    tol = 1e-10
    
    D = 2
    M = 6
    s = 3

    import matplotlib.pyplot as plt
    plt.figure()

    err_type = 'abs'
    Mag0 = numpy.zeros((M,N))
    rms = numpy.zeros(N)
    time_resp0 = numpy.zeros((N,N))
    a = numpy.zeros((D,1))
    A = numpy.zeros((D,M))
    B = numpy.zeros((D,M))
    for n in range(N):
       #print n
        T = 2*math.pi/W[n]
        time = numpy.linspace(0,T,N)
       #f = lambda t,x: numpy.array([[x[1,0]],[-mu*x[1,0] - Fs(x[0,0]) + e*math.cos(W[n]*t)]])
       #dfdx = lambda t,x: numpy.array([[0, 1],[ -Gs(x[0,0]), -mu]])

        f = lambda t,x: numpy.array([[x[1,0]],[-muorig*x[1,0] -x[0,0] + e*math.cos(W[n]*t)]])
        dfdx = lambda t,x: numpy.array([[0, 1],[-1, -muorig]])
        
        anew,Anew,Bnew,x0,err = harmonic_balance(f,dfdx,T,s,a,A,B,tol,err_type)
        if err == 1:
            err_type = 'rel'
            anew,Anew,Bnew,x0,err = harmonic_balance(f,dfdx,T,s,a,A,B,tol,err_type)
            err_type = 'abs'
            if err == 1:
                a = numpy.zeros((D,1))
                A = numpy.zeros((D,M))
                B = numpy.zeros((D,M))
                anew,Anew,Bnew,x0,err = harmonic_balance(f,dfdx,T,s,a,A,B,tol,err_type)
        a = anew
        A = Anew
        B = Bnew
        
        Mag0[0,n] = numpy.absolute(a[0])
        time_resp0[:,n] = a[0]
        for ii in range(M-1):
            Mag0[ii+1,n] = math.sqrt(A[0,ii]**2+B[0,ii]**2)
            time_resp0[:,n] += A[0,ii]*numpy.sin((ii+1)*W[n]*time)+B[0,ii]*numpy.cos((ii+1)*W[n]*time)
        rms[n] = numpy.sqrt(numpy.mean(time_resp0[:,n]**2))
    plt.semilogy(W,rms)
    err_type = 'abs'
    Mag0 = numpy.zeros((M,N))
    rms = numpy.zeros(N)
    time_resp0 = numpy.zeros((N,N))
    a = numpy.zeros((D,1))
    A = numpy.zeros((D,M))
    B = numpy.zeros((D,M))
    for n in range(N):
       #print n
        T = 2*math.pi/W[n]
        time = numpy.linspace(0,T,N)
       #f = lambda t,x: numpy.array([[x[1,0]],[-mu*x[1,0] - Fs(x[0,0]) + e*math.cos(W[n]*t)]])
       #dfdx = lambda t,x: numpy.array([[0, 1],[ -Gs(x[0,0]), -mu]])

        f = lambda t,x: numpy.array([[x[1,0]],[-muorig*x[1,0]/1.1 -x[0,0]/1.1 + e*math.cos(W[n]*t)/1.1]])
        dfdx = lambda t,x: numpy.array([[0, 1],[-1/1.1, -muorig/1.1]])
        
        anew,Anew,Bnew,x0,err = harmonic_balance(f,dfdx,T,s,a,A,B,tol,err_type)
        if err == 1:
            err_type = 'rel'
            anew,Anew,Bnew,x0,err = harmonic_balance(f,dfdx,T,s,a,A,B,tol,err_type)
            err_type = 'abs'
            if err == 1:
                a = numpy.zeros((D,1))
                A = numpy.zeros((D,M))
                B = numpy.zeros((D,M))
                anew,Anew,Bnew,x0,err = harmonic_balance(f,dfdx,T,s,a,A,B,tol,err_type)
        a = anew
        A = Anew
        B = Bnew
        
        Mag0[0,n] = numpy.absolute(a[0])
        time_resp0[:,n] = a[0]
        for ii in range(M-1):
            Mag0[ii+1,n] = math.sqrt(A[0,ii]**2+B[0,ii]**2)
            time_resp0[:,n] += A[0,ii]*numpy.sin((ii+1)*W[n]*time)+B[0,ii]*numpy.cos((ii+1)*W[n]*time)
        rms[n] = numpy.sqrt(numpy.mean(time_resp0[:,n]**2))
    plt.semilogy(W,rms)
    legend = ['original','original+mass']

    Br = 1.2
    mu0 = 1.257e-06
    a1_vec = [0.1,0.15,0.175,0.2,0.25,0.3,0.5]
    a2_vec = [0.1,0.15,0.175,0.2,0.25,0.3,0.5]
   #a1_vec = [0.7]#,0.2,0.3,0.5,0.7,1,1.2]
   #a2_vec = [0.7]#,0.2,0.3,0.5,0.7,1,1.2]
    r1 = 0.006
    r2 = 0.006
    A1 = math.pi*r1**2
    A2 = math.pi*r2**2
    tk = 0.0015
    K3_vec = [0.1,0.5,1,2,3,4,5]
    time_resp = numpy.zeros((N,N,len(a1_vec)))
    
    for jj in range(len(a1_vec)):
       #K3 = K3_vec[jj]
        K1 = 0.1
        D = 4
        M = 6
        err_type = 'abs'
        a1 = a1_vec[jj]
        a2 = a2_vec[jj]
        Mag = numpy.zeros((M+1,N))
        rms = numpy.zeros(N)
        
        #initial Fourier series coefficient guesses
        a = numpy.zeros((D,1))
        A = numpy.zeros((D,M))
        B = numpy.zeros((D,M))

        for n in range(N):
           #print n
           #if n == N/2:
           #    a = numpy.zeros((D,1))
           #    A = numpy.zeros((D,M))
           #    B = numpy.zeros((D,M))
                
            T = 2*math.pi/W[n]
            time = numpy.linspace(0,T,N)
           #f = lambda t,x: numpy.array([[x[1,0]],[-mu*x[1,0] - Fs(x[0,0]) + e*math.cos(W[n]*t)]])
           #dfdx = lambda t,x: numpy.array([[0, 1],[ -Gs(x[0,0]), -mu]])

           # Magnet
            f = lambda t,x: numpy.array([[x[2,0]],[x[3,0]],[-muorig*x[2,0] - magnet_force(x[0,0]-x[1,0],mu0,Br,A1,A2,a1,a2,r1,r2,tk) -x[0,0] - mu*(x[2,0]-x[3,0]) + e*math.cos(W[n]*t)],[magnet_force(x[0,0]-x[1,0],mu0,Br,A1,A2,a1,a2,r1,r2,tk)/0.1 - mu*(x[3,0]-x[2,0])/0.1 + e*math.cos(W[n]*t)]])
            dfdx = lambda t,x: numpy.array([[0,0, 1,0],[0,0, 0,1],[ -magnet_force_derivative(x[0,0]-x[1,0],mu0,Br,A1,A2,a1,a2,r1,r2,tk) -1,magnet_force_derivative(x[0,0]-x[1,0],mu0,Br,A1,A2,a1,a2,r1,r2,tk), -mu-muorig,mu],[magnet_force_derivative(x[0,0]-x[1,0],mu0,Br,A1,A2,a1,a2,r1,r2,tk)/0.1,-magnet_force_derivative(x[0,0]-x[1,0],mu0,Br,A1,A2,a1,a2,r1,r2,tk)/0.1,mu/0.1,-mu/0.1]])
            ## cubic
           #f = lambda t,x: numpy.array([[x[2,0]],[x[3,0]],[-muorig*x[2,0] - K3*(x[0,0]-x[1,0])**3 -x[0,0] - mu*(x[2,0]-x[3,0]) + e*math.cos(W[n]*t)],[K3*(x[0,0]-x[1,0])**3/0.1 - mu*(x[3,0]-x[2,0])/0.1]])
           #dfdx = lambda t,x: numpy.array([[0,0, 1,0],[0,0, 0,1],[ -3*K3*(x[0,0]-x[1,0])**2 -1,3*K3*(x[0,0]-x[1,0])**2, -mu-muorig,mu],[3*K3*(x[0,0]-x[1,0])**2/0.1,-3*K3*(x[0,0]-x[1,0])**2/0.1,mu/0.1,-mu/0.1]])
            ### linear
           #f = lambda t,x: numpy.array([[x[2,0]],[x[3,0]],[-muorig*x[2,0] -K1*(x[0,0]-x[1,0]) -x[0,0] - mu*(x[2,0]-x[3,0]) + e*math.cos(W[n]*t)],[K1*(x[0,0]-x[1,0])/0.1 - mu*(x[3,0]-x[2,0])/0.1]])
           #dfdx = lambda t,x: numpy.array([[0,0, 1,0],[0,0, 0,1],[-K1-1, K1, -mu-muorig,mu],[K1/0.1,-K1/0.1,mu/0.1,-mu/0.1]])
            
            
            anew,Anew,Bnew,x0,err = harmonic_balance(f,dfdx,T,s,a,A,B,tol,err_type)
            if err == 1:
                err_type = 'rel'
                anew,Anew,Bnew,x0,err = harmonic_balance(f,dfdx,T,s,a,A,B,tol,err_type)
                err_type = 'abs'
                if err == 1:
                    a = numpy.zeros((D,1))
                    A = numpy.zeros((D,M))
                    B = numpy.zeros((D,M))
                    anew,Anew,Bnew,x0,err = harmonic_balance(f,dfdx,T,s,a,A,B,tol,err_type)
            a = anew
            A = Anew
            B = Bnew
            
            Mag[0,n] = numpy.absolute(a[0])
            time_resp[:,n,jj] = a[0]
            for ii in range(M):
                Mag[ii+1,n] = math.sqrt(A[0,ii]**2+B[0,ii]**2)
                time_resp[:,n,jj] += A[0,ii]*numpy.sin((ii+1)*W[n]*time)+B[0,ii]*numpy.cos((ii+1)*W[n]*time)
            rms[n] = numpy.sqrt(numpy.mean(time_resp[:,n,jj]**2))
        plt.semilogy(W,rms)
        legend.append('with magnetic absorber d={}m'.format(a1))
       #legend.append('K3={}'.format(K3))
       #legend = ['original','harmonic 0','harmonic 1','harmonic 2','harmonic 3','harmonic 4','harmonic 5','harmonic 6']
    plt.xlabel('\Omega')
    plt.ylabel('Harmonic Magnitude RMS')
    plt.legend(legend,loc='best')
    plt.xlim([min(W),max(W)])
    plt.show()
    pdb.set_trace()
