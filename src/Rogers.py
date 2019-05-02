import numpy, pdb, sys
from kernel import kernel
from calculus import dy_dx

def rogers(GAF,k2):

    # This sub-routine carries out the approximation for the unsteady 
    # aerodynamic load by rational functions, varing the reduced frequency in a
    # few known values.
  
   #check_k = numpy.zeros((GAF.shape[0],GAF.shape[1]),dtype=int)
   #for ii in range(GAF.shape[0]):
   #    for jj in range(GAF.shape[1]):
   #        dGAF_dk = dy_dx(GAF[ii,jj,:],1)
   #        i = 3
   #        while i < len(k2): 
   #            if numpy.sign(dGAF_dk.real[i]*dGAF_dk.real[i-1]) == -1 or numpy.sign(dGAF_dk.imag[i]*dGAF_dk.imag[i-1]) == -1:
   #                check_k[ii,jj] = i-1
   #                break
   #            else:
   #                check_k[ii,jj] = len(k2) - 1
   #            i += 1
   #kmaxind = (numpy.max(check_k)+numpy.min(check_k))/2
   #k2orig = k2[:]
   #k2 = k2[0:kmaxind]
            
    y2 = numpy.zeros(4)
    for i in range(1,len(y2)+1):
        y2[i-1] = 1.7*numpy.max(k2)*(i/(4.0+1))**2

    n_k2 = len(k2)
    n_y2 = len(y2)
    
    A1 = numpy.zeros((2,3 + n_y2,n_k2))
    A2 = numpy.zeros((2,3,n_k2))
    for kk in range(n_k2):
    
        k1 = k2[kk]
        A1[0:2,0:3,kk]=numpy.array([1,0,-k1**2,0,k1,0]).reshape(2,3)
        A2[0:2,0:3,kk]=numpy.array([1,0,-k1**2,0,k1,0]).reshape(2,3)
    
        for jj in range(n_y2):
    
            A1[0:2,3 + jj,kk] = numpy.array([k1**2/(k1**2 + y2[jj]**2),k1*y2[jj]/(k1**2 - y2[jj]**2)])
    
    # Rogers approximation for the laminar flow
    C1 = numpy.zeros((A1.shape[1],A1.shape[1],n_k2))
    D1 = numpy.zeros((A1.shape[1],1         ,n_k2))
    C2 = numpy.zeros((A2.shape[1],A2.shape[1],n_k2))
    D2 = numpy.zeros((A2.shape[1],1         ,n_k2))
    PP_GAF = numpy.zeros((GAF.shape[0],GAF.shape[1],3 + n_y2)) 
    PP_GAF2 = numpy.zeros_like(PP_GAF)
    for ii in range(GAF.shape[0]):
        for jj in range(GAF.shape[1]):

           #if ii == jj:
            
                CN1 = numpy.zeros((3 + n_y2,3 + n_y2))
                DN1 = numpy.zeros((3 + n_y2,1))
    
                B1  = numpy.zeros((2,n_k2))
                for kk in range(n_k2):
    
                    B1[0,kk] = GAF[ii,jj,kk].real # Real coefficients of the aerodynamic matrix
                    B1[1,kk] = GAF[ii,jj,kk].imag # Imaginary coefficients of the aerodynamic matrix
    
                    C1[:,:,kk] = numpy.dot(A1[:,:,kk].T,A1[:,:,kk])
                    D1[:,:,kk] = numpy.dot(A1[:,:,kk].T,B1[:,kk]).reshape(D1.shape[0],D1.shape[1])
    
                    CN1 = CN1 + C1[:,:,kk]
                    DN1 = DN1 + D1[:,:,kk]
                    
                PP_GAF[ii,jj,:] = numpy.linalg.solve(CN1,DN1).reshape(PP_GAF.shape[2])
           #else:
           #    CN2 = numpy.zeros((3,3))
           #    DN2 = numpy.zeros((3,1))
    
           #    B2  = numpy.zeros((2,n_k2))
           #    for kk in range(n_k2):
    
           #        B2[0,kk] = GAF[ii,jj,kk].real # Real coefficients of the aerodynamic matrix
           #        B2[1,kk] = GAF[ii,jj,kk].imag # Imaginary coefficients of the aerodynamic matrix
    
           #        C2[:,:,kk] = numpy.dot(A2[:,:,kk].T,A2[:,:,kk])
           #        D2[:,:,kk] = numpy.dot(A2[:,:,kk].T,B2[:,kk]).reshape(D2.shape[0],D2.shape[1])
    
           #        CN2 = CN2 + C2[:,:,kk]
           #        DN2 = DN2 + D2[:,:,kk]
           #        
           #    PP_GAF[ii,jj,:] = numpy.vstack([numpy.linalg.solve(CN2,DN2),numpy.zeros((n_y2,1))]).reshape(PP_GAF.shape[2])
   #import matplotlib.pyplot as plt
   #for ii in range(GAF.shape[0]):
   #    for jj in range(GAF.shape[1]):
   #       #if ii == jj:
   #        interpGAF = PP_GAF[ii,jj,0] +  1j*k2*PP_GAF[ii,jj,1] + (1j*k2)**2*PP_GAF[ii,jj,2] + PP_GAF[ii,jj,3]*1j*k2/(1j*k2-y2[0]) + PP_GAF[ii,jj,4]*1j*k2/(1j*k2-y2[1]) + PP_GAF[ii,jj,5]*1j*k2/(1j*k2-y2[2]) + PP_GAF[ii,jj,6]*1j*k2/(1j*k2-y2[3])
   #       #else:
   #       #    interpGAF = PP_GAF[ii,jj,0] +  1j*k2*PP_GAF[ii,jj,1] + (1j*k2)**2*PP_GAF[ii,jj,2]
   #        plt.figure()
   #        plt.plot(k2orig,GAF[ii,jj,:].real)
   #        plt.plot(k2orig,GAF[ii,jj,:].imag)
   #        plt.plot(k2,interpGAF.real,'*')
   #        plt.plot(k2,interpGAF.imag,'o')
   #        plt.savefig('/home/jh/Desktop/GAF{}.png'.format(str(ii)+'_'+str(jj)))
   #        plt.close()
    
    return y2,PP_GAF

