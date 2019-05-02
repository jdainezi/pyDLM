from kernel import *
from Rogers import *
from spline import *
from dynamics import *
from graphics_tools import *
from calculus import *
from non_linear import *
import scipy.optimize, scipy.linalg, scipy.io, scipy.integrate

def static_calculation(AIC,struct_mesh_dict,aero_mesh_dict,GG_SS,D1_GG,ngrids,npan,ndof,x_rot,aoa,M,K,rho,Sref,v,path,plot_path,result_dict):

    #static calculation
    #Applying the angle of attack
    x_aoa = -(struct_mesh_dict['full_set']['x']-x_rot)*(1-numpy.cos(aoa*numpy.pi/180))
    y_aoa = numpy.zeros_like(x_aoa)
    z_aoa = -(struct_mesh_dict['full_set']['x']-x_rot)*(numpy.sin(aoa*numpy.pi/180))
    shape_aoa = numpy.zeros(x_aoa.shape[0]*ndof)
    if ndof >= 3:
        shape_aoa[0::ndof] = x_aoa
        shape_aoa[1::ndof] = y_aoa
        shape_aoa[2::ndof] = z_aoa
    elif ndof==1:
        shape_aoa = z_aoa

    #weight
    W = numpy.zeros_like(shape_aoa)
    if ndof >= 3:
        W[2::ndof] = -numpy.diag(M[2::ndof,2::ndof])*9.81
    elif ndof==1:
        W = -numpy.diag(M)*9.81

    #deformation
    err = 1
    err_list = numpy.ones_like(shape_aoa)
    h_old = numpy.zeros_like(shape_aoa)
    it = 1
    while err>0.01 and it < 20:
        print 'Coupling aero forces and structure. Iteraction {}, error {}'.format(str(it),str(err))
        Fstruct = -numpy.dot(0.5*rho*v**2*numpy.dot(numpy.dot(GG_SS,numpy.linalg.pinv(AIC)),D1_GG),shape_aoa+h_old) #aero force on struct grids
        h = numpy.dot(numpy.linalg.pinv(K),Fstruct+W)
        for i in range(len(h)):
            if h_old[i] != 0:
                err_list[i] = numpy.absolute((h[i]-h_old[i])/h_old[i])
            else:
                err_list[i] = numpy.absolute(h[i]-h_old[i])
        err = numpy.max(err_list)
        h_old = h
       #pdb.set_trace()
        it += 1

    #preparing results dictionary:
    result_dict['structural_nodes'] = {}
    result_dict['aero_nodes'] = {}
    result_dict['cp'] = {}
    result_dict['cl'] = {}
    result_dict['cl_strip'] = {} 
    result_dict['displacement'] = {}
    result_dict['force'] = {}

    GG_D2 = GG_SS/aero_mesh_dict['full_set']['area']
    h_aero = numpy.dot(GG_D2.T,h).real
    dtheta_aero = numpy.dot(D1_GG,h).real
    Faero = -0.5*rho*v**2*numpy.dot(numpy.linalg.pinv(AIC)*aero_mesh_dict['full_set']['area'],dtheta_aero+numpy.ones_like(dtheta_aero)*aoa*numpy.pi/180).real
    cp = (Faero/aero_mesh_dict['full_set']['area'])/(0.5*rho*v**2)
    cl = Faero/(0.5*rho*v**2*Sref)
    if ndof == 1:
        Fstruct = Fstruct
    elif ndof >= 3:
        Fstruct = Fstruct[2::ndof]
    
    result_dict['structural_nodes']['full_set'] = {}
    result_dict['structural_nodes']['full_set']['x'] = struct_mesh_dict['full_set']['x']
    result_dict['structural_nodes']['full_set']['y'] = struct_mesh_dict['full_set']['y']
    result_dict['structural_nodes']['full_set']['z'] = struct_mesh_dict['full_set']['z']
    result_dict['structural_nodes']['full_set']['grid_map'] = struct_mesh_dict['full_set']['grid_map']
    result_dict['aero_nodes']['full_set'] = {}
    result_dict['aero_nodes']['full_set']['xac'] = aero_mesh_dict['full_set']['ac_x'] 
    result_dict['aero_nodes']['full_set']['yac'] = aero_mesh_dict['full_set']['ac_y']
    result_dict['aero_nodes']['full_set']['zac'] = aero_mesh_dict['full_set']['ac_z']
    result_dict['aero_nodes']['full_set']['area'] = aero_mesh_dict['full_set']['area']
    result_dict['cp']['full_set'] = cp
    result_dict['cl']['full_set'] = numpy.sum(cl)
    result_dict['displacement']['full_set'] = {}
    result_dict['displacement']['full_set']['struct'] = h
    result_dict['displacement']['full_set']['aero'] = h_aero
    result_dict['force']['full_set'] = {}
    result_dict['force']['full_set']['struct'] = Fstruct
    result_dict['force']['full_set']['aero'] = Faero
    #plotting:

    maxlen = numpy.max([numpy.max(struct_mesh_dict['full_set']['x'])-numpy.min(struct_mesh_dict['full_set']['x']),numpy.max(struct_mesh_dict['full_set']['y'])-numpy.min(struct_mesh_dict['full_set']['y']),numpy.max(struct_mesh_dict['full_set']['z'])-numpy.min(struct_mesh_dict['full_set']['z'])])
    #displacement
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    nclamped = 0
    for i,component in enumerate(struct_mesh_dict.keys()):
        if component != 'full_set':
            nx = struct_mesh_dict[component]['x'].shape[0] 
            ny = struct_mesh_dict[component]['x'].shape[1]
            structind1 = struct_mesh_dict[struct_mesh_dict.keys()[i]]['global_index']-nclamped
            try:
                nclamped += struct_mesh_dict[struct_mesh_dict.keys()[i]]['nclamped']
            except:
                nclamped += 0
            structind2 = struct_mesh_dict[struct_mesh_dict.keys()[i+1]]['global_index']-nclamped
            result_dict['displacement'][component] = {}
            if ndof == 1:
                disp = h[structind1:structind2]
            elif ndof >= 3:
                disp = h[2+structind1*ndof:structind2*ndof:ndof]
            result_dict['displacement'][component]['struct'] = disp.reshape(nx,ny)
            if component in aero_mesh_dict.keys():
                for j,aerocomp in enumerate(aero_mesh_dict.keys()):
                    if aerocomp == component:
                        break
                npx = aero_mesh_dict[component]['area'].shape[0] 
                npy = aero_mesh_dict[component]['area'].shape[1]
                aeroind1 = aero_mesh_dict[aero_mesh_dict.keys()[j]]['global_index']
                aeroind2 = aero_mesh_dict[aero_mesh_dict.keys()[j+1]]['global_index']
                result_dict['displacement'][component]['aero'] = h_aero[aeroind1:aeroind2].reshape(npx,npy)
            ax.plot_wireframe(struct_mesh_dict[component]['x'],struct_mesh_dict[component]['y'],struct_mesh_dict[component]['z'])
            ax.plot_wireframe(struct_mesh_dict[component]['x'],struct_mesh_dict[component]['y'],struct_mesh_dict[component]['z']+disp.reshape(nx,ny),color='red')
            ax.legend(['orig','deformed'])
            ax.set_xlim([numpy.min(struct_mesh_dict['full_set']['x']),numpy.min(struct_mesh_dict['full_set']['x'])+maxlen])
            ax.set_ylim([numpy.min(struct_mesh_dict['full_set']['y']),numpy.min(struct_mesh_dict['full_set']['y'])+maxlen])
    plt.savefig(plot_path+'/Static_displacement.png')
    plt.close()
    #force
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    nclamped = 0
    for i,component in enumerate(struct_mesh_dict.keys()):
        if component != 'full_set':
            result_dict['force'][component] = {}
            nx = struct_mesh_dict[component]['x'].shape[0] 
            ny = struct_mesh_dict[component]['x'].shape[1]
            structind1 = struct_mesh_dict[struct_mesh_dict.keys()[i]]['global_index']-nclamped
            try:
                nclamped += struct_mesh_dict[struct_mesh_dict.keys()[i]]['nclamped']
            except:
                nclamped += 0
            structind2 = struct_mesh_dict[struct_mesh_dict.keys()[i+1]]['global_index']-nclamped
            force = Fstruct[structind1:structind2]
            result_dict['force'][component]['struct'] = force.reshape(nx,ny)
            if component in aero_mesh_dict.keys():
                for j,aerocomp in enumerate(aero_mesh_dict.keys()):
                    if aerocomp == component:
                        break
                npx = aero_mesh_dict[component]['area'].shape[0] 
                npy = aero_mesh_dict[component]['area'].shape[1]
                aeroind1 = aero_mesh_dict[aero_mesh_dict.keys()[j]]['global_index']
                aeroind2 = aero_mesh_dict[aero_mesh_dict.keys()[j+1]]['global_index']
                result_dict['force'][component]['aero'] = Faero[aeroind1:aeroind2].reshape(npx,npy)
                cl_strip = numpy.sum(Faero[aeroind1:aeroind2].reshape(npx,npy),axis=0)/(0.5*rho*v**2*numpy.sum(aero_mesh_dict[component]['area'],axis=0))
                result_dict['cl_strip'][component] = cl_strip
                ax.plot_wireframe(struct_mesh_dict[component]['x'],struct_mesh_dict[component]['y'],struct_mesh_dict[component]['z'])
                ax.plot_wireframe(aero_mesh_dict[component]['ac_x'],aero_mesh_dict[component]['ac_y'],aero_mesh_dict[component]['ac_z']+Faero[aeroind1:aeroind2].reshape(npx,npy),color='red')
                ax.set_xlim([numpy.min(struct_mesh_dict['full_set']['x']),numpy.min(struct_mesh_dict['full_set']['x'])+maxlen])
                ax.set_ylim([numpy.min(struct_mesh_dict['full_set']['y']),numpy.min(struct_mesh_dict['full_set']['y'])+maxlen])
    plt.savefig(plot_path+'/Static_aero_force.png')
    plt.close()
    #cp
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    nclamped = 0
    for i,component in enumerate(struct_mesh_dict.keys()):
        if component != 'full_set':
            if component in aero_mesh_dict.keys():
                nx = struct_mesh_dict[component]['x'].shape[0] 
                ny = struct_mesh_dict[component]['x'].shape[1]
                npx = aero_mesh_dict[component]['area'].shape[0] 
                npy = aero_mesh_dict[component]['area'].shape[1]
                structind1 = struct_mesh_dict[struct_mesh_dict.keys()[i]]['global_index']-nclamped
                try:
                    nclamped += struct_mesh_dict[struct_mesh_dict.keys()[i]]['nclamped']
                except:
                    nclamped += 0
                structind2 = struct_mesh_dict[struct_mesh_dict.keys()[i+1]]['global_index']-nclamped
                for j,aerocomp in enumerate(aero_mesh_dict.keys()):
                    if aerocomp == component:
                        break
                aeroind1 = aero_mesh_dict[aero_mesh_dict.keys()[j]]['global_index']
                aeroind2 = aero_mesh_dict[aero_mesh_dict.keys()[j+1]]['global_index']
                result_dict['cp'][component] = cp[aeroind1:aeroind2].reshape(npx,npy)
                ax.plot_wireframe(aero_mesh_dict[component]['ac_x'],aero_mesh_dict[component]['ac_y'],aero_mesh_dict[component]['ac_z'])
                ax.plot_wireframe(aero_mesh_dict[component]['ac_x'],aero_mesh_dict[component]['ac_y'],cp[aeroind1:aeroind2].reshape(npx,npy),color='green')
                ax.set_xlim([numpy.min(struct_mesh_dict['full_set']['x']),numpy.min(struct_mesh_dict['full_set']['x'])+maxlen])
                ax.set_ylim([numpy.min(struct_mesh_dict['full_set']['y']),numpy.min(struct_mesh_dict['full_set']['y'])+maxlen])
                ax.set_title('Cp')
           #    ax.set_zlim([-maxlen/2,maxlen/2])
    plt.savefig(plot_path+'/Cp.png')
    plt.close()
    #cl
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    nclamped = 0
    for i,component in enumerate(struct_mesh_dict.keys()):
        if component != 'full_set':
            if component in aero_mesh_dict.keys():
                nx = struct_mesh_dict[component]['x'].shape[0] 
                ny = struct_mesh_dict[component]['x'].shape[1]
                npx = aero_mesh_dict[component]['area'].shape[0] 
                npy = aero_mesh_dict[component]['area'].shape[1]
                structind1 = struct_mesh_dict[struct_mesh_dict.keys()[i]]['global_index']-nclamped
                try:
                    nclamped += struct_mesh_dict[struct_mesh_dict.keys()[i]]['nclamped']
                except:
                    nclamped += 0
                structind2 = struct_mesh_dict[struct_mesh_dict.keys()[i+1]]['global_index']-nclamped
                for j,aerocomp in enumerate(aero_mesh_dict.keys()):
                    if aerocomp == component:
                        break
                aeroind1 = aero_mesh_dict[aero_mesh_dict.keys()[j]]['global_index']
                aeroind2 = aero_mesh_dict[aero_mesh_dict.keys()[j+1]]['global_index']
                result_dict['cl'][component] = numpy.sum(cl[aeroind1:aeroind2])
                ax.plot_wireframe(aero_mesh_dict[component]['ac_x'],aero_mesh_dict[component]['ac_y'],aero_mesh_dict[component]['ac_z'])
                ax.plot_wireframe(aero_mesh_dict[component]['ac_x'],aero_mesh_dict[component]['ac_y'],cl[aeroind1:aeroind2].reshape(npx,npy),color='green')
                ax.set_xlim([numpy.min(struct_mesh_dict['full_set']['x']),numpy.min(struct_mesh_dict['full_set']['x'])+maxlen])
                ax.set_ylim([numpy.min(struct_mesh_dict['full_set']['y']),numpy.min(struct_mesh_dict['full_set']['y'])+maxlen])
                ax.set_title('Cl')
           #    ax.set_zlim([-maxlen/2,maxlen/2])
    plt.savefig(plot_path+'/Cl.png')
    plt.close()
    #cl strip:
    plt.figure()
    legend = []
    for i,component in enumerate(aero_mesh_dict.keys()):
        print component, 'total Cl =', result_dict['cl'][component]
        if component != 'full_set':
            legend.append(component)
            plt.plot(aero_mesh_dict[component]['ac_y'][0,:],result_dict['cl_strip'][component])
    plt.legend(legend,loc='best')
    plt.xlabel('y[m]')
    plt.ylabel('Cl')
    plt.title('Cl spanwise')
    plt.savefig(plot_path+'/Cl_spanwise.png')
    plt.close()
    
    return result_dict

def solve_PK(k,i,mode,GAF,ks,Mhh,Khh,Chh,rho,semi_chord,vel):
    if k <= ks[-1]:
        newGAF = interp_GAF(GAF,ks,k)
    else:
        newGAF = (GAF[:,:,-1]-GAF[:,:,-2])*(k-ks[-1])/(ks[-1]-ks[-2]) + GAF[:,:,-1]
    if k > 0:
        A = numpy.vstack([numpy.hstack([numpy.zeros_like(Mhh),numpy.eye(Mhh.shape[0])]),\
            numpy.hstack([numpy.dot(-numpy.linalg.inv(Mhh),Khh-0.5*rho*vel**2*newGAF.real),numpy.dot(-numpy.linalg.inv(Mhh),Chh-0.5*rho*semi_chord*vel*newGAF.imag/k)])])
    else:
        A = numpy.vstack([numpy.hstack([numpy.zeros_like(Mhh),numpy.eye(Mhh.shape[0])]),\
            numpy.hstack([numpy.dot(-numpy.linalg.inv(Mhh),Khh-0.5*rho*vel**2*newGAF.real),numpy.dot(-numpy.linalg.inv(Mhh),Chh)])])
    val, vec = numpy.linalg.eig(A)
    ind_sort = numpy.argsort(val.imag)
    val = val[ind_sort][Mhh.shape[0]:2*Mhh.shape[0]]
    new_omega = val[mode].imag
    
    omega = new_omega
    knew = omega*semi_chord/vel
    if k==0:
        err = numpy.absolute(knew - k)/k
    else:
        err = numpy.absolute(knew - k)

    return err

def mp_flutter(pk_input):
    Mhh = pk_input[0]
    Chh = pk_input[1]
    Khh = pk_input[2]
    GAF = pk_input[3]
    V = pk_input[4]
    ks = pk_input[5]
    semi_chord = pk_input[6]
    rho = pk_input[7]
    omega_n = pk_input[8]
    mode = pk_input[9]
    mi = pk_input[10]
    abs_damp = pk_input[11]
    omegat = pk_input[12]
    g = numpy.zeros(len(V))
    k = numpy.ones(len(V))*omega_n[mode]*semi_chord/V[0]
    pk_modes = numpy.zeros((len(V),Mhh.shape[0]),dtype=complex)
    print 'Solving eigenvalue problem for mode {}'.format(mode+1)
    err = 0
    for i,vel in enumerate(V):
        err1 = 0.05
        errmin = 1
        err_list = []
        newk_list = []
        k[i] = k[i-1]*V[i-1]/vel
        omegas = ks*vel/semi_chord
        omega = k[i]*vel/semi_chord
        if i<0:
            if k[i] <= 0:
                k[i] = 1e-32
            if err > 0.08:
                k[i] = k[i-1]*V[i-1]/vel
                g[i] = g[i-1] 
                pk_modes[i,:] = pk_modes[i-1,:]
                if numpy.absolute(g[i]/(k[i]*vel/semi_chord))>0.5:
                    k[i] = 0
                else:
                    continue
            newk = -1
            if k[i,mode] > 3:
                interval0 = k[i]/50.0
                interval_p = k[i]/50.0
                interval_m = k[i]/50.0
                fact = 0.5
            else:
                interval0 = k[i]/100.0
                interval_p = k[i]/100.0
                interval_m = k[i]/100.0
                fact = 1
            it = 0
            while errmin>0.01:
                newk, err, ierr, numfunc = scipy.optimize.fminbound(solve_PK,k[i]-interval_m,k[i]+interval_p,args=(i,mode,GAF,ks,Mhh,Khh,Chh,rho,semi_chord,vel),full_output=1)
                interval_p += interval0
                interval_m += interval0
                if interval_m > k[i]:
                    interval_m = k[i]
                it += 1
                err_list.append(err)
                newk_list.append(newk)
                errmin = err
                if it == 50*fact:
                    fact *= 2
                    interval0 /= 1.5
                    interval_p = k[i]/100.0/fact
                    interval_m = k[i]/100.0/fact
                    if interval_m > k[i]:
                        interval_m = k[i]
                    newk = newk_list[numpy.argmin(err_list)]
                    errmin = numpy.min(err_list)
                    if fact == 8:
                        print 'Error could not be less than 0.01 at v={}m/s'.format(vel),'Minimun error found eps={}'.format(errmin)
                        break
           #if newk < 0.05:
           #    newk = 1e-16
            k[i] = newk
            if k[i] <= ks[-1]:
                newGAF = interp_GAF(GAF,ks,k[i])
            else:
                newGAF = (GAF[:,:,-1]-GAF[:,:,-2])*(k[i]-ks[-1])/(ks[-1]-ks[-2]) + GAF[:,:,-1]
            if k[i] > 0:
                A = numpy.vstack([numpy.hstack([numpy.zeros_like(Mhh),numpy.eye(Mhh.shape[0])]),\
                    numpy.hstack([numpy.dot(-numpy.linalg.inv(Mhh),Khh-0.5*rho*vel**2*newGAF.real),numpy.dot(-numpy.linalg.inv(Mhh),Chh-0.5*rho*semi_chord*vel*newGAF.imag/k[i])])])
            else:
                A = numpy.vstack([numpy.hstack([numpy.zeros_like(Mhh),numpy.eye(Mhh.shape[0])]),\
                    numpy.hstack([numpy.dot(-numpy.linalg.inv(Mhh),Khh-0.5*rho*vel**2*newGAF.real),numpy.dot(-numpy.linalg.inv(Mhh),Chh-0.5*rho*semi_chord*vel*interp_GAF(GAF,ks,1e-32).imag/1e-32)])])
            val, vec = numpy.linalg.eig(A)
            ind_sort = numpy.argsort(val.imag)
            val = val[ind_sort][Mhh.shape[0]:2*Mhh.shape[0]]
            if k[i]<=0:
                val[mode] = -numpy.absolute(val[mode])
            vec = vec[0:Mhh.shape[0],ind_sort][:,Mhh.shape[0]:2*Mhh.shape[0]]
            new_omega = val[mode].imag
            g[i] = val[mode].real#*semi_chord/vel
           #if newk == 1e-16:
           #    g[i,mode] = g[i-1,mode]
            pk_modes[i,:] = vec[:,mode]
        else:
            it = 0
            while err1 >= 0.001 and it<100:
                if k[i] <= 0:
                    k[i] = 0
                if k[i] <= ks[-1]:
                    newGAF = interp_GAF(GAF,ks,k[i])
                else:
                    newGAF = (GAF[:,:,-1]-GAF[:,:,-2])*(k[i]-ks[-1])/(ks[-1]-ks[-2]) + GAF[:,:,-1]
                omega_abs = k[i]*vel/semi_chord
                abs_term = 0
                for ii in range(len(mi)):
                    if omega_abs > 0.0:
                        abs_term += mi[ii]*omegat[ii]**2/(omegat[ii]**2 +2*1j*abs_damp[ii]*omegat[ii]*omega_abs -omega_abs**2)
                Mhh_abs = Mhh*(1+abs_term)
                if k[i] > 0:
                    A = numpy.vstack([numpy.hstack([numpy.zeros_like(Mhh),numpy.eye(Mhh.shape[0])]),\
                        numpy.hstack([numpy.dot(-numpy.linalg.inv(Mhh_abs),Khh-0.5*rho*vel**2*newGAF.real),numpy.dot(-numpy.linalg.inv(Mhh_abs),Chh-0.5*rho*semi_chord*vel*newGAF.imag/k[i])])])
                else:
                    A = numpy.vstack([numpy.hstack([numpy.zeros_like(Mhh),numpy.eye(Mhh.shape[0])]),\
                        numpy.hstack([numpy.dot(-numpy.linalg.inv(Mhh_abs),Khh-0.5*rho*vel**2*newGAF.real),numpy.dot(-numpy.linalg.inv(Mhh_abs),Chh-0.5*rho*semi_chord*vel*interp_GAF(GAF,ks,1e-32).imag/1e-32)])])
                val, vec = numpy.linalg.eig(A)
                ind_sort = numpy.argsort(val.imag)
                #to fix problem of different real parts when k = 0
                if k[i]==0.0:
                    #correlation between the zero freq mode and the others to find the other overdamped solution:
                    corr = numpy.zeros(2*Mhh.shape[0])
                    for ii in range(2*Mhh.shape[0]):
                        corr[ii] = scalar_product(vec[0:Mhh.shape[0],ind_sort][:,Mhh.shape[0]+mode],vec[0:Mhh.shape[0],ind_sort[ii]])
                    ind2 = numpy.argsort(corr)
                    val[ind_sort[Mhh.shape[0]+mode]] = 0*numpy.max([val[ind_sort[ind2[-1]]],val[ind_sort[ind2[-2]]]])
                val = val[ind_sort][Mhh.shape[0]:2*Mhh.shape[0]]
                vec = vec[0:Mhh.shape[0],ind_sort][:,Mhh.shape[0]:2*Mhh.shape[0]]
                new_omega = val[mode].imag
                g[i] = val[mode].real#*semi_chord/vel
                pk_modes[i,:] = vec[:,mode]
            
                if omega == 0:
                    err1 = numpy.max(numpy.absolute((new_omega-omega)))
                else:
                    err1 = numpy.max(numpy.absolute((new_omega-omega)/omega))
                omega = new_omega
                k[i] = omega*semi_chord/vel
                it += 1
    return {'g':g, 'k':k, 'pk_modes':pk_modes}

def mp_corr(input_corr):
    '''Calculates the eigenvectors correlation from pk-method with multiple processing'''
    Mhh = input_corr[0]
    pk_modes = input_corr[1]
    j = input_corr[2]
    V  = input_corr[3]
    correlation = numpy.zeros((Mhh.shape[0],len(V)))
    for i,vel in enumerate(V):
        if i != 0:
            for kk in range(Mhh.shape[0]):
                correlation[kk,i] = scalar_product(pk_modes[i,j,:],pk_modes[i-1,kk,:])
        else:
            correlation[j,i] = 1
    return correlation 


def flutter_calculation(Mhh,Chh,Khh,GAF,V,ks,semi_chord,rho,omega_n,ncores,miabs=[0],abs_damp=[0],omegat=[0]):
    #P-K method
    g = numpy.zeros((len(V),Mhh.shape[0]))
    k = numpy.ones((len(V),Mhh.shape[0]))*omega_n*semi_chord/V[0]
    pk_modes = numpy.zeros((len(V),Mhh.shape[0],Mhh.shape[1]),dtype=complex)
    pool = mp.Pool(processes=ncores)
    pk_input = []
    for mode in range(Mhh.shape[0]):
        pk_input.append([Mhh,Chh,Khh,GAF,V,ks,semi_chord,rho,omega_n,mode,miabs,abs_damp,omegat])
    pk_dict = pool.map(mp_flutter,pk_input)
    pool.close()
    for mode in range(Mhh.shape[0]):
        g[:,mode] = pk_dict[mode]['g']
        k[:,mode] = pk_dict[mode]['k']
        pk_modes[:,mode,:] = pk_dict[mode]['pk_modes']
                        
    #ordering the results based on the eigenvectors
    corr_input = []
    for mode in range(Mhh.shape[0]):
        corr_input.append([Mhh,pk_modes,mode,V])
    pool = mp.Pool(processes=ncores)
    corr_list = pool.map(mp_corr,corr_input)
    pool.close()
    correlation = numpy.zeros((Mhh.shape[0],Mhh.shape[1],len(V)))
    for mode in range(Mhh.shape[0]):
        correlation[mode,:,:] = corr_list[mode]
    
   #indsortold = numpy.arange(Mhh.shape[0])
   #for i,vel in enumerate(V):
   #    if i != 0:
   #        indsort = []
   #        for j in range(Mhh.shape[0]):
   #            indsort.append(numpy.argmax(correlation[:,j,i]))
   #        indsort = numpy.array(indsort)
   #        indsort = indsortold[indsort]
   #        check =[]
   #        for j in range(Mhh.shape[0]):
   #            if indsort[j] in check:
   #                ind1 = []
   #                ind1.append(j)
   #                ind1.extend(numpy.argwhere(check==indsort[j])[0])
   #                indsort[ind1] = indsortold[ind1]
   #            check.append(indsort[j])
   #        k[i,:] = k[i,indsort]
   #        g[i,:] = g[i,indsort]
   #        pk_modes[i,:,:] = pk_modes[i,indsort,:]
   #        indsortold = indsort
   #    else:
   #        continue
    #no parallel calc          
    indsortold = numpy.arange(Mhh.shape[0])
    for i,vel in enumerate(V):
        correlation = numpy.zeros_like(Mhh)
        if i != 0:
            for j in range(Mhh.shape[0]):
                for kk in range(Mhh.shape[0]):
                    correlation[j,kk] = scalar_product(pk_modes[i,j,:],pk_modes[i-1,kk,:])
            indsort = []
            for j in range(Mhh.shape[0]):
                indsort.append(numpy.argmax(correlation[:,j]))
            indsort = numpy.array(indsort)
            check =[]
            for j in range(Mhh.shape[0]):
                if indsort[j] in check:
                    ind1 = []
                    ind1.append(j)
                    ind1.extend(numpy.argwhere(check==indsort[j])[0])
                    indsort[ind1] = indsortold[ind1]
                check.append(indsort[j])
            k[i,:] = k[i,indsort]
            g[i,:] = g[i,indsort]
            pk_modes[i,:,:] = pk_modes[i,indsort,:]
            indsortold = indsort
        else:
            continue

    return g, k, pk_modes

def get_v_flutter(g,k_flutter,pk_modes,V,nmodes,ks,Mhh,Chh,Khh,GAF,semi_chord,rho):
   
    v_flutter = V[-1]
    flutter_mode0 = pk_modes[0,0] 
    for i in range(1,len(V)):
        for j in range(nmodes):
            if g[i,j] > 1E-5 and v_flutter == V[-1]:
                flutter_mode0 = pk_modes[i,j,:]
                v_flutter = (V[i-1]-V[i])*g[i-1,j]/(g[i,j]-g[i-1,j])+V[i-1]
                k_flutter = (k_flutter[i-1,j]-k_flutter[i,j])*g[i-1,j]/(g[i,j]-g[i-1,j])+k_flutter[i-1,j]
                if g[i-1,j]>=0:
                    v_flutter = V[i-1]
                print 'Flutter speed found: {} m/s at mode {}, frequency {}Hz'.format(v_flutter,j+1,k_flutter*v_flutter/semi_chord/(2*numpy.pi))
    if v_flutter == V[-1]:
        print 'Flutter speed not found'.format(v_flutter,j+1)
        k_flutter = k_flutter[i,j]
        
    vel = v_flutter
    k = numpy.zeros(nmodes)
    g = numpy.zeros(nmodes)
    pk_modes = numpy.zeros((nmodes,nmodes),dtype=complex)
    ar,br,ai,bi = linear_GAF(GAF,ks) #linear interpolation coefs (real and imag)
    for mode in range(nmodes):
        err = 1
        errold = 1
        k[mode] = k_flutter
        omegas = ks*vel/semi_chord
        omega = k[mode]*vel/semi_chord
        i=0
        newk = -1
        interval0 = k[mode]/100.0
        interval_p = k[mode]/100.0
        interval_m = k[mode]/100.0
        it = 0
       #while err>0.01 and it<30:
       #    newk, err, ierr, numfunc = scipy.optimize.fminbound(solve_PK,k[mode]-interval_m,k[mode]+interval_p,args=(i,mode,GAF,ks,Mhh,Khh,Chh,rho,semi_chord,vel),full_output=1)
       #    interval_p += interval0
       #    interval_m += interval0
       #    if interval_m > k[mode]:
       #        interval_m = k[mode]
       #    it += 1
       #    if errold < err:
       #        break 
       #    errold = err*1.0
       #k[mode] = newk
       #if k[mode] <= ks[-1]:
       #    newGAF = interp_GAF(GAF,ks,k[mode])
       #else:
       #    newGAF = (GAF[:,:,-1]-GAF[:,:,-2])*(k[mode]-ks[-1])/(ks[-1]-ks[-2]) + GAF[:,:,-1]
       #A = numpy.vstack([numpy.hstack([numpy.zeros_like(Mhh),numpy.eye(Mhh.shape[0])]),\
       #    numpy.hstack([numpy.dot(-numpy.linalg.inv(Mhh),Khh-0.5*rho*vel**2*newGAF.real),numpy.dot(-numpy.linalg.inv(Mhh),Chh-0.5*rho*semi_chord*vel*newGAF.imag/k[mode])])])
       #val, vec = numpy.linalg.eig(A)
       #ind_sort = numpy.argsort(val.imag)
       #if k[mode]==0:
       #    #correlation between the zero freq mode and the others to find the other overdamped solution:
       #    corr = numpy.zeros(2*Mhh.shape[0])
       #    for ii in range(2*Mhh.shape[0]):
       #        corr[ii] = scalar_product(vec[0:Mhh.shape[0],ind_sort][:,Mhh.shape[0]+mode],vec[0:Mhh.shape[0],ind_sort])
       #    ind2 = numpy.argsort(corr[ii])
       #    val[ind_sort[Mhh.shape[0]+mode]] = numpy.max([val[ind_sort[ind2[-1]]],val[ind_sort[ind2[-2]]]]) 
       #val = val[ind_sort][Mhh.shape[0]:2*Mhh.shape[0]]
       #vec = vec[0:Mhh.shape[0],ind_sort][:,Mhh.shape[0]:2*Mhh.shape[0]]
       #new_omega = val[mode].imag
       #pk_modes[mode,:] = vec[:,mode]
        while err >= 0.01 and it<50:
            if k[mode] <= ks[-1]:
                if k[mode] <= ks[0]:
                    k[mode] = ks[0]
                newGAF = interp_GAF(GAF,ks,k[mode])
            else:
                newGAF = (GAF[:,:,-1]-GAF[:,:,-2])*(k[mode]-ks[-1])/(ks[-1]-ks[-2]) + GAF[:,:,-1]
            if k[mode] == 0:
                A = numpy.vstack([numpy.hstack([numpy.zeros_like(Mhh),numpy.eye(Mhh.shape[0])]),\
                    numpy.hstack([numpy.dot(-numpy.linalg.inv(Mhh),Khh-0.5*rho*vel**2*newGAF.real),numpy.dot(-numpy.linalg.inv(Mhh),Chh)])])
            else:
                A = numpy.vstack([numpy.hstack([numpy.zeros_like(Mhh),numpy.eye(Mhh.shape[0])]),\
                    numpy.hstack([numpy.dot(-numpy.linalg.inv(Mhh),Khh-0.5*rho*vel**2*newGAF.real),numpy.dot(-numpy.linalg.inv(Mhh),Chh-0.5*rho*semi_chord*vel*newGAF.imag/k[mode])])])
            val, vec = numpy.linalg.eig(A)
            ind_sort = numpy.argsort(val.imag)
            if k[i]==0:
                #correlation between the zero freq mode and the others to find the other overdamped solution:
                corr = numpy.zeros(2*Mhh.shape[0])
                for ii in range(2*Mhh.shape[0]):
                    corr[ii] = scalar_product(vec[0:Mhh.shape[0],ind_sort][:,Mhh.shape[0]+mode],vec[0:Mhh.shape[0],ind_sort[ii]])
                ind2 = numpy.argsort(corr)
                val[ind_sort[Mhh.shape[0]+mode]] = numpy.max([val[ind_sort[ind2[-1]]],val[ind_sort[ind2[-2]]]])
            val = val[ind_sort][Mhh.shape[0]:2*Mhh.shape[0]]
            vec = vec[0:Mhh.shape[0],ind_sort][:,Mhh.shape[0]:2*Mhh.shape[0]]
            new_omega = val[mode].imag
            g[mode] = val[mode].real*semi_chord/vel
            pk_modes[mode,:] = vec[:,mode]
        
            if omega == 0:
                err = numpy.max(numpy.absolute((new_omega-omega)))
            else:
                err = numpy.max(numpy.absolute((new_omega-omega)/omega))
            omega = new_omega
            k[mode] = omega*semi_chord/vel
            it += 1
    correlation = numpy.zeros(nmodes)
    for i in range(nmodes):
        correlation[i] = scalar_product(flutter_mode0,pk_modes[i,:])
    indflutter = numpy.argmax(correlation)
    k_flutter = k[indflutter]
    flutter_mode = pk_modes[indflutter,:]

    return v_flutter, flutter_mode, k_flutter

def scalar_product(X,Y):
    #calculates the correlation between 2 eigenvectors
    S1=0
    S2=0
    S3=0
    S4=0
    for i in range(len(X)):
        S1 += X[i].real*Y[i].real+X[i].imag*Y[i].imag
        S2 += X[i].real*Y[i].imag-X[i].imag*Y[i].real
        S3 += numpy.absolute(X[i])**2
        S4 += numpy.absolute(Y[i])**2
    if S3 == 0 and S4 == 0:
        XY = 1
    elif S3 == 0 or S4 == 0:
        XY = 0
    else:
        XY = numpy.sqrt(S1**2+S2**2)/numpy.sqrt(S3*S4)
    return XY

def rel_K_B(K_old,B_old,grids_wallid,ndof,struct_mesh_dict):
    Krel = numpy.zeros_like(K_old)
    Brel = numpy.zeros_like(B_old)
    grids_wallid -= 1
    for i,grid in enumerate(grids_wallid):
        prevy=struct_mesh_dict['full_set']['y'][grid-1]
        acty=struct_mesh_dict['full_set']['y'][grid]
        nexty=struct_mesh_dict['full_set']['y'][grid+1]
        try:
            prevx=struct_mesh_dict['full_set']['x'][grids_wallid[i-1]]
            actx=struct_mesh_dict['full_set']['x'][grid]
            try:
                nextx=struct_mesh_dict['full_set']['x'][grids_wallid[i+1]]
            except:
                nextx=struct_mesh_dict['full_set']['x'][grids_wallid[0]]
        except:
            prevx=struct_mesh_dict['full_set']['x'][grid]
            actx=struct_mesh_dict['full_set']['x'][grid]
            nextx=struct_mesh_dict['full_set']['x'][grid]
        if numpy.absolute(acty-nexty)<=numpy.absolute(acty-prevy):
            if ndof == 1:
                Krel[grid,grid] = K_old[grid,grid] + K_old[grid,grid+1]
                Brel[grid,grid] = B_old[grid,grid] + B_old[grid,grid+1]
                if numpy.absolute(actx-nextx)<numpy.absolute(actx-prevx):
                    Krel[grid,grid] += K_old[grid,grids_wallid[i+1]]
                    Brel[grid,grid] += B_old[grid,grids_wallid[i+1]]
                elif numpy.absolute(actx-nextx)>numpy.absolute(actx-prevx):
                    Krel[grid,grid] += K_old[grid,grids_wallid[i-1]]
                    Brel[grid,grid] += B_old[grid,grids_wallid[i-1]]
                elif numpy.absolute(actx-nextx)-numpy.absolute(actx-prevx)<0.01*numpy.min([numpy.absolute(actx-nextx),numpy.absolute(actx-prevx)]) and prevx+nextx != 0:
                    Krel[grid,grid] += K_old[grid,grids_wallid[i-1]] + K_old[grid,grids_wallid[i+1]]
                    Brel[grid,grid] += B_old[grid,grids_wallid[i-1]] + B_old[grid,grids_wallid[i+1]]
            else:
                Krel[grid*ndof+2,grid*ndof+2] = K_old[grid*ndof+2,grid*ndof+2] + K_old[(grid)*ndof+2,(grid+1)*ndof+2]
                Brel[grid*ndof+2,grid*ndof+2] = B_old[grid*ndof+2,grid*ndof+2] + B_old[(grid)*ndof+2,(grid+1)*ndof+2]
                if numpy.absolute(actx-nextx)<numpy.absolute(actx-prevx):
                    Krel[grid*ndof+2,grid*ndof+2] += K_old[grid*ndof+2,grids_wallid[i+1]*ndof+2]
                    Brel[grid*ndof+2,grid*ndof+2] += B_old[grid*ndof+2,grids_wallid[i+1]*ndof+2]
                elif numpy.absolute(actx-nextx)>numpy.absolute(actx-prevx):
                    Krel[grid*ndof+2,grid*ndof+2] += K_old[grid*ndof+2,grids_wallid[i-1]*ndof+2]
                    Brel[grid*ndof+2,grid*ndof+2] += B_old[grid*ndof+2,grids_wallid[i-1]*ndof+2]
                elif numpy.absolute(actx-nextx)-numpy.absolute(actx-prevx)<0.01*numpy.min([numpy.absolute(actx-nextx),numpy.absolute(actx-prevx)]) and prevx+nextx != 0:
                    Krel[grid*ndof+2,grid*ndof+2] += K_old[grid*ndof+2,grids_wallid[i-1]*ndof+2] + K_old[grid*ndof+2,grids_wallid[i+1]*ndof+2]
                    Brel[grid*ndof+2,grid*ndof+2] += B_old[grid*ndof+2,grids_wallid[i-1]*ndof+2] + B_old[grid*ndof+2,grids_wallid[i+1]*ndof+2]
        else:
            if ndof == 1:
                Krel[grid,grid] = K_old[grid,grid] + K_old[grid,grid-1]
                Brel[grid,grid] = B_old[grid,grid] + B_old[grid,grid-1]
                if numpy.absolute(actx-nextx)<numpy.absolute(actx-prevx):
                    Krel[grid,grid] += K_old[grid,grids_wallid[i+1]]
                    Brel[grid,grid] += B_old[grid,grids_wallid[i+1]]
                elif numpy.absolute(actx-nextx)>numpy.absolute(actx-prevx):
                    Krel[grid,grid] += K_old[grid,grids_wallid[i-1]]
                    Brel[grid,grid] += B_old[grid,grids_wallid[i-1]]
                elif numpy.absolute(actx-nextx)-numpy.absolute(actx-prevx)<0.01*numpy.min([numpy.absolute(actx-nextx),numpy.absolute(actx-prevx)]) and prevx+nextx != 0:
                    Krel[grid,grid] += K_old[grid,grids_wallid[i-1]] + K_old[grid,grids_wallid[i+1]]
                    Brel[grid,grid] += B_old[grid,grids_wallid[i-1]] + B_old[grid,grids_wallid[i+1]]
            else:
                Krel[grid*ndof+2,grid*ndof+2] = K_old[grid*ndof+2,grid*ndof+2] + K_old[(grid)*ndof+2,(grid-1)*ndof+2]
                Brel[grid*ndof+2,grid*ndof+2] = B_old[grid*ndof+2,grid*ndof+2] + B_old[(grid)*ndof+2,(grid-1)*ndof+2]
                if numpy.absolute(actx-nextx)<numpy.absolute(actx-prevx):
                    Krel[grid*ndof+2,grid*ndof+2] += K_old[grid*ndof+2,grids_wallid[i+1]*ndof+2]
                    Brel[grid*ndof+2,grid*ndof+2] += B_old[grid*ndof+2,grids_wallid[i+1]*ndof+2]
                elif numpy.absolute(actx-nextx)>numpy.absolute(actx-prevx):
                    Krel[grid*ndof+2,grid*ndof+2] += K_old[grid*ndof+2,grids_wallid[i-1]*ndof+2]
                    Brel[grid*ndof+2,grid*ndof+2] += B_old[grid*ndof+2,grids_wallid[i-1]*ndof+2]
                elif numpy.absolute(actx-nextx)-numpy.absolute(actx-prevx)<0.01*numpy.min([numpy.absolute(actx-nextx),numpy.absolute(actx-prevx)]) and prevx+nextx != 0:
                    Krel[grid*ndof+2,grid*ndof+2] += K_old[grid*ndof+2,grids_wallid[i-1]*ndof+2] + K_old[grid*ndof+2,grids_wallid[i+1]*ndof+2]
                    Brel[grid*ndof+2,grid*ndof+2] += B_old[grid*ndof+2,grids_wallid[i-1]*ndof+2] + B_old[grid*ndof+2,grids_wallid[i+1]*ndof+2]
    return Krel,Brel

def frf_analysis(struct_mesh_dict,GAF,Mtot,Mhh_old,Bhh_old,Khh_old,k,v,rho,semi_chord,M_old,B_old,K_old,omega_n,freqlen,wall_nodes,sgrids,infabs,ref_grid,phi_old,mi,abs_damping,tmode,ndof,nmodes,damp,ext_shape,x_rot,plot_path):
    nabs = len(sgrids)
    nclamped = 0
    grid_ids = sgrids
    zeta_old = numpy.diag(Bhh_old)/(2*numpy.sqrt(numpy.diag(Khh_old)*numpy.diag(Mhh_old)))
   #grids_wall = wall_nodes
   #grids_wallid = numpy.zeros(len(wall_nodes))
   #for i in range(len(wall_nodes)):
   #    grids_wallid[i] = numpy.argwhere(struct_mesh_dict['full_set']['grid_map']==grids_wall[i])[0][0] + 1

    ref_gridid = numpy.argwhere(struct_mesh_dict['full_set']['grid_map']==ref_grid)[0][0] +1
    yb = numpy.zeros((phi_old.shape[0],1))
    if ext_shape == 'bending':
        if ndof == 1:
            yb += 1.0
        else:
            yb[2::ndof] = 1.0
    elif ext_shape == 'torsion':
        if ndof <= 3:
            yb += 0.1*(x_rot - struct_mesh_dict['full_set']['x']).reshape(phi_old.shape[0]/ndof,1)
        else:
            yb[2::ndof] += 0.1*(x_rot - struct_mesh_dict['full_set']['x']).reshape(phi_old.shape[0]/ndof,1)
            yb[4::ndof] += 0.1
    
    miabs = numpy.zeros(len(mi)) 
    if mi != 0 and infabs != '1':
        nabs = 0
        grid_ids = []
        grid_idst = []
        grid_idsf = []
        nabst = 0
        nabsf = 0
        for i,gset in enumerate(sgrids):
            nabs += len(gset)
            grid_ids.extend(gset)
            if 't' in mi[i]:
                nabst += len(gset)
                grid_idst.extend(gset)
            else:
                nabsf += len(gset)
                grid_idsf.extend(gset)
        ks = []
        ms = []
        bs = []
        for i,gset in enumerate(sgrids):
           #pdb.set_trace()
            if 't' in mi[i]:
                infabs = '0'
                miabs[i] = float(mi[i].replace('t',''))
                if type(tmode[i]) != type('Hz'):
                    omegat  = omega_n[tmode[i]]*2/(1+numpy.sqrt(1+miabs[i]))
                else:
                    omegat  = 2*numpy.pi*float(tmode[i].replace('Hz',''))*2/(1+numpy.sqrt(1+miabs[i]))
            else:
                continue
            mset = numpy.ones(len(gset))*Mtot*miabs[i]/len(gset)
            ms.extend(mset)
            ks.extend((omegat)**2*mset)
            bs.extend(abs_damping[i]*2*numpy.sqrt(mset*((omegat)**2*mset)))

        K,B,M = insert_tshock_absorbers(struct_mesh_dict,grid_idst,ks,bs,ms,M_old,B_old,K_old,ndof)
        ks = []
        ms = []
        bs = []
        for i,gset in enumerate(sgrids):
           #pdb.set_trace()
            if 't' not in mi[i]:
                miabs[i] = float(mi[i])
                if type(tmode[i]) != type('Hz'):
                    omegat  = omega_n[tmode[i]]*2/(1+numpy.sqrt(1+miabs[i]))
                else:
                    omegat  = 2*numpy.pi*float(tmode[i].replace('Hz',''))*2/(1+numpy.sqrt(1+miabs[i]))
            else:
                continue
            mset = numpy.ones(len(gset))*Mtot*miabs[i]/len(gset)
            ms.extend(mset)
            ks.extend((omegat)**2*mset)
            bs.extend(abs_damping[i]*2*numpy.sqrt(mset*((omegat)**2*mset)))

        K,B,M = insert_shock_absorbers(struct_mesh_dict,grid_idsf,ks,bs,ms,M,B,K,ndof)
       
       #if all(numpy.diag(M)[3::ndof])<=0 and all(numpy.diag(M)[4::ndof])<=0 and all(numpy.diag(M)[5::ndof])<=0:
       #    Mmax = numpy.max(M)
       #    M /= Mmax
       #    for i in range(M_old.shape[0]/ndof):
       #        M[i*ndof+3,i*ndof+3] += 0.0001
       #        M[i*ndof+4,i*ndof+4] += 0.0001
       #        M[i*ndof+5,i*ndof+5] += 0.0001
       #    K /= Mmax
       #    B /= Mmax
       #pdb.set_trace()
        A = numpy.vstack([numpy.hstack([numpy.zeros_like(M),numpy.eye(M.shape[0])]),\
            numpy.hstack([numpy.dot(-numpy.linalg.inv(M),K),numpy.dot(-numpy.linalg.inv(M),B)])])
        if ndof == 6:
            val,vec = numpy.linalg.eig(A)
        else:
            val,vec = numpy.linalg.eig(A)
        ind_sort = numpy.argsort(val.imag)[M.shape[0]:]
        ind_sort_full = numpy.argsort(val.imag)
        omega_new = []
        zeta = []
        nmodesabs = 0
        while val.imag[ind_sort][nmodesabs] < omega_n[-1]*1.1:
            if val.imag[ind_sort][nmodesabs] > 0:
                x = -val.real[ind_sort][nmodesabs]/val.imag[ind_sort][nmodesabs]
                zeta.append(numpy.absolute(x)/numpy.sqrt(1+x**2))
            else:
                x = ((-val.real[ind_sort_full][M.shape[0]-nmodesabs]-val.real[ind_sort_full][M.shape[0]+nmodesabs])/2)/((val.real[ind_sort_full][M.shape[0]-nmodesabs]-val.real[ind_sort_full][M.shape[0]+nmodesabs])/2)
                zeta.append(numpy.absolute(x)/numpy.sqrt(-1+x**2))
            omega_new.append(numpy.absolute(val.real[ind_sort][nmodesabs]/zeta[-1]))
            nmodesabs += 1
        omega_new = numpy.array(omega_new)
        zeta = numpy.array(zeta)
        phi = vec[0:M.shape[0],ind_sort][:,0:nmodesabs]
        if ndof != 6:
            for i in range(phi.shape[1]):
                phi[:,i] /= max(numpy.absolute(phi[0:M_old.shape[0],i]))#*numpy.cos(numpy.angle(phi[0,i]))
        else:
            for i in range(phi.shape[1]):
                phi[:,i] /= max(numpy.absolute(phi[2:M_old.shape[0]:ndof,i]))#*numpy.cos(numpy.angle(phi[0,i]))
        ybabs = numpy.zeros((phi.shape[0],1))
        if ext_shape == 'bending':
            if ndof == 1:
                ybabs += 1.0
            else:
                ybabs[2:phi_old.shape[0]:ndof] = 1.0
                ybabs[phi_old.shape[0]:] = 1.0
        elif ext_shape == 'torsion':
            x_abs = struct_mesh_dict['full_set']['x']
            for grid in grid_ids:
               #grid = numpy.argwhere(struct_mesh_dict['full_set']['grid_map']==grid)[0][0] +1
                x_abs = numpy.hstack([x_abs,struct_mesh_dict['full_set']['x'][grid]])
            if ndof <= 3:
                ybabs[0:phi_old.shape[0]] += 0.1*(x_rot - x_abs)[0:phi_old.shape[0]].reshape(phi_old.shape[0],1)
                ybabs[phi_old.shape[0]:-nabsf] += 0.1
                ybabs[-nabsf:] += 0.1*(x_rot - x_abs)[-nabsf:].reshape(nabsf,1)
            else:
                ybabs[2:phi_old.shape[0]:ndof] += 0.1*(x_rot - x_abs).reshape(phi.shape[0],1)[2:phi_old.shape[0]:ndof]
                ybabs[4:phi_old.shape[0]:ndof] += 0.1
                ybabs[phi_old.shape[0]:-nabsf] += 0.1
                ybabs[-nabsf:] += 0.1*(x_rot - x_abs[-nabsf:]).reshape(nabsf,1)
        ybhh = numpy.zeros_like(numpy.dot(numpy.linalg.pinv(phi),ybabs))
        if ndof == 1:
            if nabst == 0:
                ybhh = numpy.dot(numpy.linalg.pinv(phi),ybabs)
            else:
                ybhh += numpy.dot(numpy.linalg.pinv(phi[0:phi_old.shape[0]]),ybabs[0:phi_old.shape[0]])
                ybhh += numpy.dot(numpy.linalg.pinv(phi[phi_old.shape[0]:-nabsf]),ybabs[phi_old.shape[0]:-nabsf])
                ybhh += numpy.dot(numpy.linalg.pinv(phi[-nabsf:]),ybabs[-nabsf:])
        else:
            for dof in [2]:
               ybhh  += numpy.dot(numpy.linalg.pinv(phi[dof::ndof,:]),ybabs[dof::ndof])
       #for i in range(nmodesabs):
       #    if abs(ybhh[i,0])<0.1:
       #        phi[0:M_old.shape[0],i] *= 0.001

       #pdb.set_trace()
       #phi = phi.real

#       bs = numpy.zeros_like(ms)
#       K,B,M = insert_shock_absorbers(struct_mesh_dict,grid_ids,ks,bs,ms,M_old,B_old,K_old,ndof)
#       val,vec = numpy.linalg.eig(numpy.dot(numpy.linalg.inv(M),K))
#       ind_sort = numpy.argsort(val)
#       omega_new = []
#       nmodesabs = 0
#       while numpy.sqrt(val[ind_sort])[nmodesabs] < omega_n[-1]*1.1:
#           omega_new.append(numpy.sqrt(val[ind_sort])[nmodesabs])
#           nmodesabs += 1
#       phi = vec[:,ind_sort][:,0:nmodesabs]
#       for i in range(phi.shape[1]):
#           phi[:,i] /= max(numpy.absolute(phi[:,i]))
       #################################
       #nmodesabs = 0
       #count = 0
       #phinew2 = []
       #omega_new2 = []
       #egvec = numpy.zeros_like(phi[:,0])
       #egval = 0
       #for i in range(phi.shape[1]):
       #    if max(numpy.absolute(phi[:-nabs,i])) > 0.1*max(numpy.absolute(phi[-nabs:,i])):
       #        phinew2.append(phi[:,i])
       #        omega_new2.append(omega_new[i])
       #        nmodesabs += 1
       #    elif max(numpy.absolute(phi[:-nabs,i])) <= 0.1*max(numpy.absolute(phi[-nabs:,i])): 
       #        count += 1
       #        egvec += phi[:,i]
       #        egval += omega_new[i]
       #if count >= 1: 
       #    nmodesabs += 1
       #    phinew2.append(egvec/count)
       #    omega_new2.append(egval/count)
       #phi = numpy.array(phinew2).T
       #omega_new = numpy.array(omega_new2)

      ##nmodesabs = 0
      ##count = 0
      ##phinew2 = []
      ##omega_new2 = []
      ##for i in range(phi.shape[1]):
      ##    if max(numpy.absolute(phi[:-nabs,i])) > 0.1*max(numpy.absolute(phi[-nabs:,i])):
      ##        phinew2.append(phi[:,i])
      ##        omega_new2.append(omega_new[i])
      ##        nmodesabs += 1
      ##    elif max(numpy.absolute(phi[:-nabs,i])) <= 0.1*max(numpy.absolute(phi[-nabs:,i])) and count == 0: 
      ##        count = 1
      ##        nmodesabs += 1
      ##        egvec = numpy.zeros_like(phi[:,i])
      ##        egvec[-nabs:] = 1
      ##        phinew2.append(egvec)
      ##        omega_new2.append(omega_new[i])
      ##phi = numpy.array(phinew2).T
      ##omega_new = numpy.array(omega_new2)
       #omega_new = numpy.array(omega_new)
       #phi = phi[:,[0,1,2,3,4,5,6,7,126,127,128,129,130,131]]
       #nmodesabs = 14
       #omega_new = omega_new[[0,1,2,3,4,5,6,7,126,127,128,129,130,131]]
        ################################
        Mhh = numpy.dot(numpy.dot(phi.T,M),phi)
        Khh = numpy.dot(numpy.dot(phi.T,K),phi)
        Bhh = numpy.dot(numpy.dot(phi.T,B),phi)
#       if damp[0] == 'flat':
#            Bhh = numpy.diag(2*numpy.sqrt(numpy.diag(Khh)*numpy.diag(Mhh)))*float(damp[1])
#            B = numpy.dot(numpy.dot(numpy.linalg.pinv(phi.T),Bhh),numpy.linalg.pinv(phi))
#       elif damp[0] == 'proportional':
#            Bhh = Mhh*float(damp[1]) + Khh*float(damp[2])
#            B = numpy.dot(numpy.dot(numpy.linalg.pinv(phi.T),Bhh),numpy.linalg.pinv(phi))
#       A = numpy.vstack([numpy.hstack([numpy.zeros_like(Mhh),numpy.eye(Mhh.shape[0])]),\
#           numpy.hstack([numpy.dot(-numpy.linalg.inv(Mhh),Khh),numpy.dot(-numpy.linalg.inv(Mhh),Bhh)])])
#       val, vec = numpy.linalg.eig(A)
#       ind_sort = numpy.argsort(val.imag)
#       val = val[ind_sort][Mhh.shape[0]:]
#       zeta = -val.real/val.imag
#       omega_new = val.imag
    elif mi != 0 and infabs == '1':
        for i,gset in enumerate(sgrids):
           #pdb.set_trace()
            if 't' in mi[i]:
                infabs = '0'
                miabs[i] = float(mi[i].replace('t',''))
            else:
                miabs[i] = float(mi[i])
        
    freq = numpy.linspace(0.1,1.1*omega_n[-1]/2/numpy.pi,freqlen)
        
    omegas = freq*numpy.pi*2
    frf_orig = transmissibility(struct_mesh_dict,omegas,grid_ids,ref_gridid,ndof,nmodes,phi_old,omega_n,mi,Mtot,tmode,zeta_old,abs_damping,yb)
   #frf_aero_orig = aero_transmissibility(struct_mesh_dict,GAF,k,v,rho,phi_old,semi_chord,Mhh_old,Bhh_old,Khh_old,omega_n,omegas,grid_ids,ref_gridid,ndof,nmodes,mi,Mtot,tmode,yb,nabs=0)
    frf_aero_orig = aero_transmissibility(GAF,rho,Mhh_old,Khh_old,Bhh_old,k,semi_chord,v,struct_mesh_dict,omegas,grid_ids,ref_gridid,ndof,nmodes,phi_old,omega_n,mi,Mtot,tmode,zeta_old,abs_damping,yb)
   #frf_orig = FRF(struct_mesh_dict,omegas,ndof,grids_wallid,ref_gridid,phi_old,Mhh_old,M_old,Khh_old,K_old,Bhh_old,B_old,n_abs=0)
   #frf_aero_orig = FRF_aero(GAF,v,rho,semi_chord,k,struct_mesh_dict,omegas,ndof,grids_wallid,ref_gridid,phi_old,Mhh_old,M_old,Khh_old,K_old,Bhh_old,B_old,n_abs=0)
    if mi != 0:
       #frf = FRF(struct_mesh_dict,omegas,ndof,grids_wallid,ref_gridid,phi,Mhh,M,Khh,K,Bhh,B,nabs)
        if infabs == '1':
            frf = transmissibility(struct_mesh_dict,omegas,grid_ids,ref_gridid,ndof,nmodes,phi_old,omega_n,miabs,Mtot,tmode,zeta_old,abs_damping,yb,nabs=nabs)
        else:
            frf = transmissibility(struct_mesh_dict,omegas,grid_ids,ref_gridid,ndof,nmodesabs,phi,omega_new,miabs,Mtot,tmode,zeta,abs_damping,ybabs,nabs=0,nabst=nabst,nabsf=nabsf)
       #frf_aero = FRF_aero(GAFabs,v,rho,semi_chord,k,struct_mesh_dict,omegas,ndof,grids_wallid,ref_gridid,phi,Mhh,M,Khh,K,Bhh,B,nabs)
        if infabs == '1':
            frf_aero = aero_transmissibility(GAF,rho,Mhh_old,Khh_old,Bhh_old,k,semi_chord,v,struct_mesh_dict,omegas,grid_ids,ref_gridid,ndof,nmodes,phi_old,omega_n,miabs,Mtot,tmode,zeta_old,abs_damping,yb,nabs=nabs)
        else:
            # transforming GAF matrix to match system with abs:
            GAFabs = numpy.zeros((nmodesabs,nmodesabs,GAF.shape[2]),dtype=complex)
            for i in range(len(k)):
                AICabs = numpy.vstack([numpy.hstack([numpy.dot(numpy.dot(numpy.linalg.pinv(phi_old).T,GAF[:,:,i]),numpy.linalg.pinv(phi_old)),numpy.zeros((phi_old.shape[0],nabs))]),numpy.hstack([numpy.zeros((nabs,phi_old.shape[0])),numpy.zeros((nabs,nabs))])])
                GAFabs[:,:,i] = numpy.dot(numpy.dot(phi.T,AICabs),phi) 
            frf_aero = aero_transmissibility_old(struct_mesh_dict,GAFabs,k,v,rho,phi,semi_chord,Mhh,Bhh,Khh,omega_new,omegas,grid_ids,ref_gridid,ndof,nmodesabs,miabs,Mtot,tmode,ybabs,abs_damping,nabs=0,nabst=nabst,nabsf=nabsf)
   #frf = FRF(omegas,ndof,grids_wall,ref_grid,M,K,B,n_abs=len(grid_ids))[:,:,0]
   #frf = numpy.array(frf)
   #frf = FRF_nl(omegas,ndof,grids_wall,ref_grid,M,K,K3,B,n_abs=len(grid_ids))
    frf_results = {}
    frf_results['freq'] = freq
    frf_results['frf_orig'] = frf_orig
    frf_results['frf_aero_orig'] = frf_aero_orig
    if mi != 0:
        frf_results['frf_abs'] = frf
        frf_results['frf_aero_abs'] = frf_aero

    return frf_results

def state_space_time(t,x,nmodes,aero_modes,M,phi,Mhh,Mbar_inv,Kbar,Cbar,Eh,D,qinf,vel,R,semi_chord,accel,yb,dt,ndof,miabs,abs_damp,omegat):
    err = 1000
    omegas_old = 0
    it = 0
    while err > 1 and it<20:
        abs_term = 0
        for ii in range(len(miabs)):
            if omegas_old > 0:
                abs_term += miabs[ii]*omegat[ii]**2/(omegat[ii]**2 +2*1j*abs_damp[ii]*omegat[ii]*omegas_old -omegas_old**2)
        Mbar_inv_abs = Mbar_inv/(1+abs_term)
        Mhh_abs = Mhh*(1+abs_term)
        A = numpy.hstack([numpy.vstack([numpy.zeros((nmodes,nmodes)),-numpy.dot(Mbar_inv_abs,Kbar),numpy.zeros((aero_modes,nmodes))]),numpy.vstack([numpy.eye(nmodes),-numpy.dot(Mbar_inv_abs,Cbar),Eh]),numpy.vstack([numpy.zeros((nmodes,aero_modes)),numpy.dot(qinf*Mbar_inv_abs,D),vel*R/semi_chord])])
        b = numpy.zeros((A.shape[0],1),dtype=complex)
       #ybhh = numpy.dot(phi.T,numpy.dot(M,yb))
        ybhh = numpy.dot(Mhh_abs,numpy.dot(numpy.linalg.pinv(phi),yb))
        if ndof != 1:
            ybhh *= 0
            for dof in range(ndof):
                ybhh += numpy.dot(Mhh_abs,numpy.dot(numpy.linalg.pinv(phi[dof::ndof,:]),yb[dof::ndof]))
        i = int(t/dt)
        b[nmodes:2*nmodes,0] = -numpy.dot(Mbar_inv_abs,accel[i]*ybhh[:,0])
       #b[nmodes:2*nmodes,0] = -amplitude*ybhh[:,0]*(2*numpy.pi*freq)**2*numpy.sin(2*numpy.pi*freq*t)*numpy.dot(phi.T,numpy.diag(M))
       # To compute complex values
        A = numpy.vstack([numpy.hstack([A,numpy.zeros_like(A)]),numpy.hstack([numpy.zeros_like(A),A])])
        b = numpy.vstack([b.real,b.imag])
        x_dot = numpy.dot(A,x.reshape(len(x),1)) + b
        if all(miabs) == 0:
            break
        omegas = (numpy.angle(x_dot[0,0]+1j*x_dot[A.shape[0]/2,0])-numpy.angle(x[0]+1j*x[A.shape[0]/2]))/dt
        err = numpy.max(numpy.absolute(omegas-omegas_old))
        omegas_old = omegas
        it += 1
    return x_dot.real.reshape(len(x))

#def jacobian(t,x,nmodes,aero_modes,M,phi,Mhh,Mbar_inv,Kbar,Cbar,Eh,D,qinf,vel,R,semi_chord,freq,amplitude,yb):
#    A = numpy.hstack([numpy.vstack([numpy.zeros((nmodes,nmodes)),-numpy.dot(Mbar_inv,Kbar),numpy.zeros((aero_modes,nmodes))]),numpy.vstack([numpy.eye(nmodes),-numpy.dot(Mbar_inv,Cbar),Eh]),numpy.vstack([numpy.zeros((nmodes,aero_modes)),numpy.dot(qinf*Mbar_inv,D),vel*R/semi_chord])])
#    A = numpy.vstack([numpy.hstack([A,numpy.zeros_like(A)]),numpy.hstack([numpy.zeros_like(A),A])])
#    return A 

def state_space_time_gust(t,x,nmodes,aero_modes,M,phi,Mhh,Mbar_inv,Kbar,Cbar,E,D,qinf,vel,R,semi_chord,accel,yb,wg,wgdot,Lgust,PPg,dt,ndof,miabs,abs_damp,omegat):
    err = 1000
    omegas_old = 0 
    it = 0
    while err > 1 and it<20:
        abs_term = 0
        for ii in range(len(miabs)):
            if omegas_old > 0:
                abs_term += miabs[ii]*omegat[ii]**2/(omegat[ii]**2 +2*1j*abs_damp[ii]*omegat[ii]*omegas_old -omegas_old**2)
        Mbar_inv_abs = Mbar_inv/(1+abs_term)
        Mhh_abs = Mhh*(1+abs_term)
        A = numpy.hstack([numpy.vstack([numpy.zeros((nmodes,nmodes)),-numpy.dot(Mbar_inv_abs,Kbar),numpy.zeros((2*aero_modes,nmodes))]),numpy.vstack([numpy.eye(nmodes),-numpy.dot(Mbar_inv_abs,Cbar),E[:,0:nmodes]]),numpy.vstack([numpy.zeros((nmodes,2*aero_modes)),numpy.dot(qinf*Mbar_inv_abs,D),vel*R/semi_chord])])
        b = numpy.zeros((A.shape[0],1),dtype=complex)
   #    ybhh = numpy.dot(phi.T,numpy.dot(M,yb))
        ybhh = numpy.dot(Mhh_abs,numpy.dot(numpy.linalg.pinv(phi),yb))
        if ndof != 1:
            ybhh *= 0
            for dof in range(ndof):
                ybhh += numpy.dot(Mhh_abs,numpy.dot(numpy.linalg.pinv(phi[dof::ndof,:]),yb[dof::ndof]))
        i = int(t/dt)
        b[nmodes:2*nmodes,0] = -numpy.dot(Mbar_inv_abs,accel[i]*ybhh[:,0])
        #b[nmodes:2*nmodes,0] = -amplitude*ybhh[:,0]*(2*numpy.pi*freq)**2*numpy.sin(2*numpy.pi*freq*t)*numpy.dot(phi.T,numpy.diag(M))
        # To compute complex values
        A = numpy.vstack([numpy.hstack([A,numpy.zeros_like(A)]),numpy.hstack([numpy.zeros_like(A),A])])
        b = numpy.vstack([b.real,b.imag])
        wgbar = numpy.hstack([wg[:,i]/vel,wgdot[:,i]/vel])
        Bg = numpy.hstack([numpy.vstack([numpy.zeros((nmodes,wg.shape[0])),qinf*numpy.dot(Mbar_inv,PPg[:,:,0]),numpy.zeros((2*aero_modes,wg.shape[0]))]),numpy.vstack([numpy.zeros((nmodes,wg.shape[0])),qinf*Lgust/vel*numpy.dot(Mbar_inv,PPg[:,:,1]),E[:,nmodes:]])])
        x_dot = numpy.dot(A,x.reshape(len(x),1)) + b + numpy.hstack([numpy.dot(Bg,wgbar),numpy.dot(Bg,wgbar)]).reshape(A.shape[0],1)
        if all(miabs) == 0:
            break
        omegas = (numpy.angle(x_dot[0,0]+1j*x_dot[A.shape[0]/2,0])-numpy.angle(x[0]+1j*x[A.shape[0]/2]))/dt
        err = numpy.max(numpy.absolute(omegas-omegas_old))
        omegas_old = omegas
        it += 1
    return x_dot.real.reshape(len(x))

def time_solver(plot_path,PP_GAF,Ri,M,phi,Mhh,Khh,Chh,vel,rho,nmodes,semi_chord,tmax,dt,amplitude,freq,yb,ndof,miabs=[0],abs_damping=[0],omegat=[0]):
    aero_modes = nmodes*(len(Ri))
    qinf = 0.5*rho*vel**2
    
    if vel > 0:
        Mbar = Mhh - qinf*semi_chord**2/vel**2*PP_GAF[:,:,2]
    else:
        Mbar = Mhh#- 0.5*rho*semi_chord**2*PP_GAF[:,:,2]
    Mbar_inv = numpy.linalg.inv(Mbar)
    Kbar = Khh - qinf*PP_GAF[:,:,0]
    if vel > 0:
        Cbar = Chh - qinf*semi_chord/vel*PP_GAF[:,:,1]
    else:
        Cbar = Chh
    R = []
    for i,r in enumerate(Ri):
        if i == 0:
            D = numpy.eye(nmodes)
            Eh = PP_GAF[:,:,3+i]
        else:
            D = numpy.hstack([D,numpy.eye(nmodes)])
            Eh = numpy.vstack([Eh,PP_GAF[:,:,3+i]])
        R.extend(-numpy.ones(nmodes)*r)
    R = numpy.diag(R)
    if freq == 'white_noise':
        accel = white_noise_accel(amplitude,numpy.arange(0,tmax+2*dt,dt),dt)
        plot_fft(plot_path,'Base_fft',numpy.arange(0,tmax+2*dt,dt),accel)
    else:
        accel = harmonic_motion(amplitude,freq,numpy.arange(0,tmax+2*dt,dt))
    x0 = numpy.zeros(2*(2*nmodes+aero_modes),dtype=complex)
    i=0
    r = scipy.integrate.ode(state_space_time).set_integrator('zvode')
    r.set_initial_value(x0,0).set_f_params(nmodes,aero_modes,M,phi,Mhh,Mbar_inv,Kbar,Cbar,Eh,D,qinf,vel,R,semi_chord,accel,yb,dt,ndof,miabs,abs_damping,omegat)#.set_jac_params(nmodes,aero_modes,M,phi,Mhh,Mbar_inv,Kbar,Cbar,Eh,D,qinf,vel,R,semi_chord,freq,amplitude,yb)
    time = [0]
    resp = [x0]
    while r.t < tmax and i < tmax/dt+1:
        print 'Solving ODE at t={}s'.format(r.t+dt)
        r.integrate(r.t+dt)
        time.append(r.t)
        resp.append(r.y[0:2*nmodes+aero_modes]+1j*r.y[2*nmodes+aero_modes:])
        i += 1
    return time, resp

def time_solver_gust(plot_path,PP_GAF,PPg,Ri,Rg,M,phi,Mhh,Khh,Chh,vel,rho,nmodes,semi_chord,tmax,dt,amplitude,freq,yb,gust_sp,x0gust,Vgust,Lgust,x,ndof,miabs=[0],abs_damping=[0],omegat=[0]):
    aero_modes = nmodes*(len(Ri))
    qinf = 0.5*rho*vel**2
    
    if vel > 0:
        Mbar = Mhh - qinf*semi_chord**2/vel**2*PP_GAF[:,:,2]
    else:
        Mbar = Mhh#- 0.5*rho*semi_chord**2*PP_GAF[:,:,2]
    Mbar_inv = numpy.linalg.inv(Mbar)
    Kbar = Khh - qinf*PP_GAF[:,:,0]
    if vel > 0:
        Cbar = Chh - qinf*semi_chord/vel*PP_GAF[:,:,1]
    else:
        Cbar = Chh
    R = []
    for i,r in enumerate(Ri):
        if i == 0:
            Dh = numpy.eye(nmodes)
            Eh = PP_GAF[:,:,3+i]
        else:
            Dh = numpy.hstack([Dh,numpy.eye(nmodes)])
            Eh = numpy.vstack([Eh,PP_GAF[:,:,3+i]])
        R.extend(-numpy.ones(nmodes)*r)
    for i,r in enumerate(Rg):
        if i == 0:
            Eg = PPg[:,:,3+i]
            Dg = numpy.eye(nmodes)
        else:
            Eg = numpy.vstack([Eg,PPg[:,:,3+i]])
            Dg = numpy.hstack([Dg,numpy.eye(nmodes)])
        R.extend(-numpy.ones(nmodes)*r)
    R = numpy.diag(R)
    E = numpy.vstack([numpy.hstack([Eh,numpy.zeros((Eh.shape[0],Eg.shape[1]))]),numpy.hstack([numpy.zeros((Eg.shape[0],Eh.shape[1])),Eg])])
    D = numpy.hstack([Dh,Dg])
    if freq == 'white_noise':
        accel = white_noise_accel(amplitude,numpy.arange(0,tmax+2*dt,dt),dt)
        plot_fft(plot_path,'Base_fft',numpy.arange(0,tmax+2*dt,dt),accel)
    else:
        accel = harmonic_motion(amplitude,freq,numpy.arange(0,tmax+2*dt,dt))
    x0 = numpy.zeros(2*(2*nmodes+2*aero_modes),dtype=complex)
    wg = numpy.zeros((len(x),len(numpy.arange(0,tmax+2*dt,dt))))
    wgdot = numpy.zeros_like(wg)
    if gust_sp == '1-cos':
        wg = gust_shape(vel,Vgust,Lgust,x0gust,x,numpy.arange(0,tmax+2*dt,dt))
        wgdot = numpy.zeros_like(wg)
        for i in range(len(x)):
            wgdot[i,:] = dy_dx(wg[i,:],dt)
    elif gust_sp == 'white_noise':
        wngust = white_noise_gust(white_noise(1,numpy.arange(0,tmax+2*dt,dt),dt),dt,Lgust,vel,Vgust)
        accel += wngust
        plot_fft(plot_path,'Base_fft',numpy.arange(0,tmax+2*dt,dt),accel)
    i=0
    r = scipy.integrate.ode(state_space_time_gust).set_integrator('zvode')
    r.set_initial_value(x0,0).set_f_params(nmodes,aero_modes,M,phi,Mhh,Mbar_inv,Kbar,Cbar,E,D,qinf,vel,R,semi_chord,accel,yb,wg,wgdot,Lgust,PPg,dt,ndof,miabs,abs_damping,omegat)#.set_jac_params(nmodes,aero_modes,M,phi,Mhh,Mbar_inv,Kbar,Cbar,Eh,D,qinf,vel,R,semi_chord,freq,amplitude,yb)
    time = [0]
    resp = [x0]
    while r.t < tmax and i < tmax/dt+1:
        print 'Solving ODE at t={}s'.format(r.t+dt)
        r.integrate(r.t+dt)
        time.append(r.t)
        resp.append(r.y[0:2*nmodes+2*aero_modes]+1j*r.y[2*nmodes+2*aero_modes:])
        i += 1
    return time, resp

def state_space_time_nl(t,x,nmodes,aero_modes,M,phi,Mhh,Mbar_inv,Kbar,Cbar,Eh,D,qinf,vel,R,semi_chord,accel,yb,dt,ndof,miabs,abs_damp,omegat,Knl,K3,nabs):
    err = 1000
    omegas_old = 0
    it = 0
    while err > 1 and it<20:
        Mbar_inv_abs = Mbar_inv
        Mhh_abs = Mhh
        A = numpy.hstack([numpy.vstack([numpy.zeros((nmodes,nmodes)),-numpy.dot(Mbar_inv_abs,Kbar),numpy.zeros((aero_modes,nmodes))]),numpy.vstack([numpy.eye(nmodes),-numpy.dot(Mbar_inv_abs,Cbar),Eh]),numpy.vstack([numpy.zeros((nmodes,aero_modes)),numpy.dot(qinf*Mbar_inv_abs,D),vel*R/semi_chord])])
        b = numpy.zeros((A.shape[0],1),dtype=complex)
       #ybhh = numpy.dot(phi.T,numpy.dot(M,yb))
        ybhh = numpy.dot(Mhh_abs,numpy.dot(numpy.linalg.pinv(phi),yb))
        if ndof != 1:
            ybhh *= 0
            for dof in range(ndof):
                ybhh += numpy.dot(Mhh_abs,numpy.dot(numpy.linalg.pinv(phi[dof::ndof,:]),yb[dof::ndof]))
        i = int(t/dt)
        b[nmodes:2*nmodes,0] = -numpy.dot(Mbar_inv_abs,accel[i]*ybhh[:,0])
       #b[nmodes:2*nmodes,0] = -amplitude*ybhh[:,0]*(2*numpy.pi*freq)**2*numpy.sin(2*numpy.pi*freq*t)*numpy.dot(phi.T,numpy.diag(M))
       # To compute complex values
        A = numpy.vstack([numpy.hstack([A,numpy.zeros_like(A)]),numpy.hstack([numpy.zeros_like(A),A])])
        b = numpy.vstack([b.real,b.imag]) + numpy.vstack([-numpy.vstack([numpy.zeros((Mhh.shape[0],1)),numpy.dot(Mbar_inv,numpy.dot(phi.T,K3*numpy.dot(Knl,numpy.dot(phi,(x[nmodes:2*(nmodes)]+1j*x[A.shape[0]/2+nmodes:A.shape[0]/2+2*(nmodes)]).reshape(nmodes,1)))**3)).real,numpy.zeros((aero_modes,1))]),-numpy.vstack([numpy.zeros((Mhh.shape[0],1)),numpy.dot(Mbar_inv,numpy.dot(phi.T,K3*numpy.dot(Knl,numpy.dot(phi,(x[nmodes:2*(nmodes)]+1j*x[A.shape[0]/2+nmodes:A.shape[0]/2+2*(nmodes)]).reshape(nmodes,1)))**3)).imag,numpy.zeros((aero_modes,1))])])
        x_dot = numpy.dot(A,x.reshape(len(x),1)) + b
        if all(miabs) == 0:
            break
        omegas = (numpy.angle(x_dot[0,0]+1j*x_dot[A.shape[0]/2,0])-numpy.angle(x[0]+1j*x[A.shape[0]/2]))/dt
        err = numpy.max(numpy.absolute(omegas-omegas_old))
        omegas_old = omegas
        it += 1
    return x_dot.real.reshape(len(x))

def state_space_time_gust_nl(t,x,nmodes,aero_modes,M,phi,Mhh,Mbar_inv,Kbar,Cbar,E,D,qinf,vel,R,semi_chord,accel,yb,wg,wgdot,Lgust,PPg,dt,ndof,miabs,abs_damp,omegat,Knl,K3,nabs):
    err = 1000
    omegas_old = 0 
    it = 0
    while err > 1 and it<20:
        Mbar_inv_abs = Mbar_inv
        Mhh_abs = Mhh
        A = numpy.hstack([numpy.vstack([numpy.zeros((nmodes,nmodes)),-numpy.dot(Mbar_inv_abs,Kbar),numpy.zeros((2*aero_modes,nmodes))]),numpy.vstack([numpy.eye(nmodes),-numpy.dot(Mbar_inv_abs,Cbar),E[:,0:nmodes]]),numpy.vstack([numpy.zeros((nmodes,2*aero_modes)),numpy.dot(qinf*Mbar_inv_abs,D),vel*R/semi_chord])])
        b = numpy.zeros((A.shape[0],1),dtype=complex)
   #    ybhh = numpy.dot(phi.T,numpy.dot(M,yb))
        ybhh = numpy.dot(Mhh_abs,numpy.dot(numpy.linalg.pinv(phi),yb))
        if ndof != 1:
            ybhh *= 0
            for dof in range(ndof):
                ybhh += numpy.dot(Mhh_abs,numpy.dot(numpy.linalg.pinv(phi[dof::ndof,:]),yb[dof::ndof]))
        i = int(t/dt)
        b[nmodes:2*nmodes,0] = -numpy.dot(Mbar_inv_abs,accel[i]*ybhh[:,0])
        #b[nmodes:2*nmodes,0] = -amplitude*ybhh[:,0]*(2*numpy.pi*freq)**2*numpy.sin(2*numpy.pi*freq*t)*numpy.dot(phi.T,numpy.diag(M))
        # To compute complex values
        A = numpy.vstack([numpy.hstack([A,numpy.zeros_like(A)]),numpy.hstack([numpy.zeros_like(A),A])])
        b = numpy.vstack([b.real,b.imag]) + numpy.vstack([-numpy.vstack([numpy.zeros((Mhh.shape[0],1)),numpy.dot(Mbar_inv,numpy.dot(phi.T,K3*numpy.dot(Knl,numpy.dot(phi,(x[nmodes:2*(nmodes)]+1j*x[A.shape[0]/2+nmodes:A.shape[0]/2+2*(nmodes)]).reshape(nmodes,1)))**3)).real,numpy.zeros((2*aero_modes,1))]),-numpy.vstack([numpy.zeros((Mhh.shape[0],1)),numpy.dot(Mbar_inv,numpy.dot(phi.T,K3*numpy.dot(Knl,numpy.dot(phi,(x[nmodes:2*(nmodes)]+1j*x[A.shape[0]/2+nmodes:A.shape[0]/2+2*(nmodes)]).reshape(nmodes,1)))**3)).imag,numpy.zeros((2*aero_modes,1))])])
        wgbar = numpy.hstack([wg[:,i]/vel,wgdot[:,i]/vel])
        Bg = numpy.hstack([numpy.vstack([numpy.zeros((nmodes,wg.shape[0])),qinf*numpy.dot(Mbar_inv,PPg[:,:,0]),numpy.zeros((2*aero_modes,wg.shape[0]))]),numpy.vstack([numpy.zeros((nmodes,wg.shape[0])),qinf*Lgust/vel*numpy.dot(Mbar_inv,PPg[:,:,1]),E[:,nmodes:]])])
        x_dot = numpy.dot(A,x.reshape(len(x),1)) + b + numpy.hstack([numpy.dot(Bg,wgbar),numpy.dot(Bg,wgbar)]).reshape(A.shape[0],1)
        if all(miabs) == 0:
            break
        omegas = (numpy.angle(x_dot[0,0]+1j*x_dot[A.shape[0]/2,0])-numpy.angle(x[0]+1j*x[A.shape[0]/2]))/dt
        err = numpy.max(numpy.absolute(omegas-omegas_old))
        omegas_old = omegas
        it += 1
    return x_dot.real.reshape(len(x))

def time_solver_nl(plot_path,PP_GAF,Ri,M,phi,Mhh,Khh,Chh,vel,rho,nmodes,semi_chord,tmax,dt,amplitude,freq,yb,ndof,Knl,K3,miabs=[0],abs_damping=[0],omegat=[0]):
    aero_modes = nmodes*(len(Ri))
    qinf = 0.5*rho*vel**2
    
    if vel > 0:
        Mbar = Mhh - qinf*semi_chord**2/vel**2*PP_GAF[:,:,2]
    else:
        Mbar = Mhh#- 0.5*rho*semi_chord**2*PP_GAF[:,:,2]
    Mbar_inv = numpy.linalg.inv(Mbar)
    Kbar = Khh - qinf*PP_GAF[:,:,0]
    if vel > 0:
        Cbar = Chh - qinf*semi_chord/vel*PP_GAF[:,:,1]
    else:
        Cbar = Chh
    R = []
    for i,r in enumerate(Ri):
        if i == 0:
            D = numpy.eye(nmodes)
            Eh = PP_GAF[:,:,3+i]
        else:
            D = numpy.hstack([D,numpy.eye(nmodes)])
            Eh = numpy.vstack([Eh,PP_GAF[:,:,3+i]])
        R.extend(-numpy.ones(nmodes)*r)
    R = numpy.diag(R)
    if freq == 'white_noise':
        accel = white_noise_accel(amplitude,numpy.arange(0,tmax+2*dt,dt),dt)
        plot_fft(plot_path,'Base_fft',numpy.arange(0,tmax+2*dt,dt),accel)
    else:
        accel = harmonic_motion(amplitude,freq,numpy.arange(0,tmax+2*dt,dt))
    x0 = numpy.zeros(2*(2*nmodes+aero_modes),dtype=complex)
    i=0
    r = scipy.integrate.ode(state_space_time_nl).set_integrator('lsoda')
    r.set_initial_value(x0,0).set_f_params(nmodes,aero_modes,M,phi,Mhh,Mbar_inv,Kbar,Cbar,Eh,D,qinf,vel,R,semi_chord,accel,yb,dt,ndof,miabs,abs_damping,miabs*0,Knl,K3,len(miabs))#.set_jac_params(nmodes,aero_modes,M,phi,Mhh,Mbar_inv,Kbar,Cbar,Eh,D,qinf,vel,R,semi_chord,freq,amplitude,yb)
    time = [0]
    resp = [x0]
    while r.t < tmax and i < tmax/dt+1:
        print 'Solving ODE at t={}s'.format(r.t+dt)
        r.integrate(r.t+dt)
        time.append(r.t)
        resp.append(r.y[0:2*nmodes+aero_modes]+1j*r.y[2*nmodes+aero_modes:])
        i += 1
    return time, resp

def time_solver_gust_nl(plot_path,PP_GAF,PPg,Ri,Rg,M,phi,Mhh,Khh,Chh,vel,rho,nmodes,semi_chord,tmax,dt,amplitude,freq,yb,gust_sp,x0gust,Vgust,Lgust,x,ndof,Knl,K3,miabs=[0],abs_damping=[0],omegat=[0]):
    aero_modes = nmodes*(len(Ri))
    qinf = 0.5*rho*vel**2
    
    if vel > 0:
        Mbar = Mhh - qinf*semi_chord**2/vel**2*PP_GAF[:,:,2]
    else:
        Mbar = Mhh#- 0.5*rho*semi_chord**2*PP_GAF[:,:,2]
    Mbar_inv = numpy.linalg.inv(Mbar)
    Kbar = Khh - qinf*PP_GAF[:,:,0]
    if vel > 0:
        Cbar = Chh - qinf*semi_chord/vel*PP_GAF[:,:,1]
    else:
        Cbar = Chh
    R = []
    for i,r in enumerate(Ri):
        if i == 0:
            Dh = numpy.eye(nmodes)
            Eh = PP_GAF[:,:,3+i]
        else:
            Dh = numpy.hstack([Dh,numpy.eye(nmodes)])
            Eh = numpy.vstack([Eh,PP_GAF[:,:,3+i]])
        R.extend(-numpy.ones(nmodes)*r)
    for i,r in enumerate(Rg):
        if i == 0:
            Eg = PPg[:,:,3+i]
            Dg = numpy.eye(nmodes)
        else:
            Eg = numpy.vstack([Eg,PPg[:,:,3+i]])
            Dg = numpy.hstack([Dg,numpy.eye(nmodes)])
        R.extend(-numpy.ones(nmodes)*r)
    R = numpy.diag(R)
    E = numpy.vstack([numpy.hstack([Eh,numpy.zeros((Eh.shape[0],Eg.shape[1]))]),numpy.hstack([numpy.zeros((Eg.shape[0],Eh.shape[1])),Eg])])
    D = numpy.hstack([Dh,Dg])
    if freq == 'white_noise':
        accel = white_noise_accel(amplitude,numpy.arange(0,tmax+2*dt,dt),dt)
        plot_fft(plot_path,'Base_fft',numpy.arange(0,tmax+2*dt,dt),accel)
    else:
        accel = harmonic_motion(amplitude,freq,numpy.arange(0,tmax+2*dt,dt))
    x0 = numpy.zeros(2*(2*nmodes+2*aero_modes),dtype=complex)
    wg = numpy.zeros((len(x),len(numpy.arange(0,tmax+2*dt,dt))))
    wgdot = numpy.zeros_like(wg)
    if gust_sp == '1-cos':
        wg = gust_shape(vel,Vgust,Lgust,x0gust,x,numpy.arange(0,tmax+2*dt,dt))
        wgdot = numpy.zeros_like(wg)
        for i in range(len(x)):
            wgdot[i,:] = dy_dx(wg[i,:],dt)
    elif gust_sp == 'white_noise':
        wngust = white_noise_gust(white_noise(1,numpy.arange(0,tmax+2*dt,dt),dt),dt,Lgust,vel,Vgust)
        accel += wngust
        plot_fft(plot_path,'Base_fft',numpy.arange(0,tmax+2*dt,dt),accel)
    i=0
    r = scipy.integrate.ode(state_space_time_gust_nl).set_integrator('lsoda')
    r.set_initial_value(x0,0).set_f_params(nmodes,aero_modes,M,phi,Mhh,Mbar_inv,Kbar,Cbar,E,D,qinf,vel,R,semi_chord,accel,yb,wg,wgdot,Lgust,PPg,dt,ndof,miabs,abs_damping,miabs*0,Knl,K3,len(miabs))#.set_jac_params(nmodes,aero_modes,M,phi,Mhh,Mbar_inv,Kbar,Cbar,Eh,D,qinf,vel,R,semi_chord,freq,amplitude,yb)
    time = [0]
    resp = [x0]
    while r.t < tmax and i < tmax/dt+1:
        print 'Solving ODE at t={}s'.format(r.t+dt)
        r.integrate(r.t+dt)
        time.append(r.t)
        resp.append(r.y[0:2*nmodes+2*aero_modes]+1j*r.y[2*nmodes+2*aero_modes:])
        i += 1
    return time, resp

def wave_propagation(h,phinew,time,dt,freq_t,struct_mesh_dict,ndof,nabs,grid_ids):
    phase0 = numpy.angle(h)
    phase = numpy.angle(h)
    for i in range(phinew.shape[0]):
        for j,t in enumerate(time):
            dphi = phase[i,j] - phase[i,j-1]
            if dphi > numpy.pi and t != time[0]:
                phase[i,j:] -= 2*numpy.pi
            if dphi < -numpy.pi and t != time[0]:
                phase[i,j:] += 2*numpy.pi
    omega_resp_full = dy_dx(phase.T,dt).T
    omega_resp = {}
    phase_resp = {}
    wave_vec = {}
    group_v = {}
    phase_grid = {}
    abs_dict = {}
    nclamped = 0
    T = 1/freq_t
    start_t = 0
    for i,t in enumerate(time):
        if t>= 4*T:
            start_t = i
            break
    for i,component in enumerate(struct_mesh_dict.keys()):
        if component != 'full_set':
            nx = struct_mesh_dict[component]['x'].shape[0]
            ny = struct_mesh_dict[component]['x'].shape[1]
            structind1 = struct_mesh_dict[struct_mesh_dict.keys()[i]]['global_index']-nclamped
            try:
                nclamped += struct_mesh_dict[struct_mesh_dict.keys()[i]]['nclamped']
            except:
                nclamped += 0
            structind2 = struct_mesh_dict[struct_mesh_dict.keys()[i+1]]['global_index']-nclamped
            phase_grid[component] = numpy.zeros((nx*ndof,ny*ndof,len(time)))
            omega_resp[component] = numpy.zeros((nx*ndof,ny*ndof,len(time)))
            phase_resp[component] = numpy.zeros((nx*ndof,ny*ndof,len(time)))
            wave_vec[component] = numpy.zeros((nx*ndof,ny*ndof,ndof,2,len(time)))
            group_v[component] = numpy.zeros((nx*ndof,ny*ndof,ndof,2,len(time)))
            dx = struct_mesh_dict[component]['dx']
            dy = struct_mesh_dict[component]['dy']
            for t in range(len(time)):
                phase_grid[component][:,:,t] = phase[structind1*ndof:structind2*ndof,t].reshape(nx*ndof,ny*ndof)
                omega_resp[component][:,:,t] = omega_resp_full[structind1*ndof:structind2*ndof,t].reshape(nx*ndof,ny*ndof)
                phase_resp[component][:,:,t] = phase0[structind1*ndof:structind2*ndof,t].reshape(nx*ndof,ny*ndof)
                for dof in range(ndof):
                    for ix in range(nx):
                        wave_vec[component][ix,:,dof,1,t] = -dy_dx(phase_grid[component][ix*ndof + dof,dof::ndof,t],dy)
                    for iy in range(ny):
                        wave_vec[component][:,iy,dof,0,t] = -dy_dx(phase_grid[component][dof::ndof,iy*ndof + dof,t],dx)
            for dof in range(ndof):
                for ix in range(nx):
                    for iy in range(ny):
                       #pdb.set_trace()
                       #arg0 = numpy.argsort(wave_vec[component][ix,iy,dof,0,start_t:])
                       #arg1 = numpy.argsort(wave_vec[component][ix,iy,dof,1,start_t:])
                       #f0 = scipy.interpolate.interp1d(wave_vec[component][ix,iy,dof,0,start_t:][arg0],omega_resp[component][ix*ndof+dof,iy*ndof+dof,start_t:][arg0])
                       #f1 = scipy.interpolate.interp1d(wave_vec[component][ix,iy,dof,1,start_t:][arg1],omega_resp[component][ix*ndof+dof,iy*ndof+dof,start_t:][arg1])
                       #k0min = min(wave_vec[component][ix,iy,dof,0,start_t:])
                       #k0max = max(wave_vec[component][ix,iy,dof,0,start_t:])
                       #k0range = numpy.linspace(k0min,k0max,1000)
                       #dk0 = k0range[1]-k0range[0]
                       #domega0 = dy_dx(f0(k0range),dk0)
                       #k1min = min(wave_vec[component][ix,iy,dof,1,start_t:])
                       #k1max = max(wave_vec[component][ix,iy,dof,1,start_t:])
                       #k1range = numpy.linspace(k1min,k1max,1000)
                       #dk1 = k1range[1]-k1range[0]
                       #domega1 = dy_dx(f1(k1range),dk1)
                       #Vg0 = scipy.interpolate.interp1d(k0range,domega0)
                       #Vg1 = scipy.interpolate.interp1d(k1range,domega1)
                       #group_v[component][ix,iy,dof,:,:] = [Vg0(wave_vec[component][ix,iy,dof,0,start_t:]),Vg1(wave_vec[component][ix,iy,dof,1,start_t:])]
                        group_v[component][ix,iy,dof,:,:] = [dy_dx(omega_resp[component][ix*ndof+dof,iy*ndof+dof,:],dt)/dy_dx(wave_vec[component][ix,iy,dof,0,:],dt),dy_dx(omega_resp[component][ix*ndof+dof,iy*ndof+dof,:],dt)/dy_dx(wave_vec[component][ix,iy,dof,1,:],dt)]
            
            if nabs != 0:
                abs_dict['abs_'+component] = {}
                abs_gridx_list = []
                abs_gridy_list = []
                for ix in range(nx):
                    xlist = []
                    ylist = []
                    for iy in range(ny):
                        for gid in grid_ids:
                            if gid == struct_mesh_dict[component]['grid_map'][ix,iy]:
                                xlist.append(struct_mesh_dict[component]['x'][ix,iy])
                                ylist.append(struct_mesh_dict[component]['y'][ix,iy])
                    abs_gridx_list.append(xlist)
                    abs_gridy_list.append(ylist)
                abs_dict['abs_'+component]['x'] = numpy.array(abs_gridx_list)
                abs_dict['abs_'+component]['y'] = numpy.array(abs_gridy_list)
                abs_dict['abs_'+component]['dx'] = abs_gridx_list[1][0]
                abs_dict['abs_'+component]['dy'] = abs_gridy_list[0][1]
                abs_dict['abs_'+component]['nx'] = numpy.array(abs_gridx_list).shape[0]
                abs_dict['abs_'+component]['ny'] = numpy.array(abs_gridx_list).shape[1]
                nx = abs_dict['abs_'+component]['nx']
                ny = abs_dict['abs_'+component]['ny']
                phase_grid['abs_'+component] = numpy.zeros((nx,ny,len(time)))
                omega_resp['abs_'+component] = numpy.zeros((nx,ny,len(time)))
                phase_resp['abs_'+component] = numpy.zeros((nx,ny,len(time)))
                wave_vec['abs_'+component] = numpy.zeros((nx,ny,2,len(time)))
                group_v['abs_'+component] = numpy.zeros((nx,ny,2,len(time)))
                dx = abs_dict['abs_'+component]['dx']
                dy = abs_dict['abs_'+component]['dy']
                for t in range(len(time)):
                    phase_grid['abs_'+component][:,:,t] = phase[-nabs:,t].reshape(nx,ny)
                    omega_resp['abs_'+component][:,:,t] = omega_resp_full[-nabs:,t].reshape(nx,ny)
                    phase_resp['abs_'+component][:,:,t] = phase0[-nabs:,t].reshape(nx,ny)
                    for ix in range(nx):
                        wave_vec['abs_'+component][ix,:,1,t] = -dy_dx(phase_grid['abs_'+component][ix,:,t],dy)
                    for iy in range(ny):
                        wave_vec['abs_'+component][:,iy,0,t] = -dy_dx(phase_grid['abs_'+component][:,iy,t],dx)
                for ix in range(nx):
                    for iy in range(ny):
                       #arg0 = numpy.argsort(wave_vec['abs_'+component][ix,iy,0,:])
                       #arg1 = numpy.argsort(wave_vec['abs_'+component][ix,iy,1,:])
                       #f0 = scipy.interpolate.interp1d(wave_vec['abs_'+component][ix,iy,0,arg0],omega_resp['abs_'+component][ix,iy,arg0])
                       #f1 = scipy.interpolate.interp1d(wave_vec['abs_'+component][ix,iy,1,arg1],omega_resp['abs_'+component][ix,iy,arg1])
                       #k0min = min(wave_vec['abs_'+component][ix,iy,0,arg0])
                       #k0max = max(wave_vec['abs_'+component][ix,iy,0,arg0])
                       #k0range = numpy.linspace(k0min,k0max,1000)
                       #dk0 = k0range[1]-k0range[0]
                       #domega0 = dy_dx(f0(k0range),dk0)
                       #k1min = min(wave_vec['abs_'+component][ix,iy,1,arg0])
                       #k1max = max(wave_vec['abs_'+component][ix,iy,1,arg0])
                       #k1range = numpy.linspace(k1min,k1max,1000)
                       #dk1 = k1range[1]-k1range[0]
                       #domega1 = dy_dx(f1(k1range),dk1)
                       #Vg0 = scipy.interpolate.interp1d(k0range,domega0)
                       #Vg1 = scipy.interpolate.interp1d(k1range,domega1)
                       #group_v['abs_'+component][ix,iy,:,:] = [Vg0(wave_vec['abs_'+component][ix,iy,0,:]),Vg1(wave_vec['abs_'+component][ix,iy,1,:])]
                        group_v['abs_'+component][ix,iy,:,:] = [dy_dx(omega_resp['abs_'+component][ix,iy,:],dt)/dy_dx(wave_vec['abs_'+component][ix,iy,0,:],dt),dy_dx(omega_resp['abs_'+component][ix,iy,:],dt)/dy_dx(wave_vec['abs_'+component][ix,iy,1,:],dt)]
                    
    return abs_dict,phase_resp,omega_resp,group_v,wave_vec,start_t

def non_linear_analysis(struct_mesh_dict,Ri,PP_GAF,Mtot,Mhh_old,Bhh_old,Khh_old,k,v,rho,semi_chord,M_old,B_old,K_old,omega_n,amplitude,freqlen,sgrids,ref_grid,phi_old,miabs,K1,K3,ndof,nmodes,abs_damping,ext_shape,x_rot,plot_path):
    nabs = len(sgrids)
    nclamped = 0
    grid_ids = sgrids
    zeta_old = numpy.diag(Bhh_old)/(2*numpy.sqrt(numpy.diag(Khh_old)*numpy.diag(Mhh_old)))

    ref_gridid = numpy.argwhere(struct_mesh_dict['full_set']['grid_map']==ref_grid)[0][0] +1
    yb = numpy.zeros((phi_old.shape[0],1))
    if ext_shape == 'bending':
        if ndof == 1:
            yb += 1.0
        else:
            yb[2::ndof] = 1.0
    elif ext_shape == 'torsion':
        if ndof <= 3:
            yb += 0.1*(x_rot - struct_mesh_dict['full_set']['x']).reshape(phi_old.shape[0]/ndof,1)
        else:
            yb[2::ndof] += 0.1*(x_rot - struct_mesh_dict['full_set']['x']).reshape(phi_old.shape[0]/ndof,1)
            yb[4::ndof] += 0.1
    phi_pinv_old = numpy.zeros((phi_old.shape[1],phi_old.shape[0]))
    if ndof == 1:
        phi_pinv_old = numpy.linalg.pinv(phi_old)
    else:
        for dof in range(ndof):
           phi_pinv_old[dof::ndof] = numpy.linalg.pinv(phi_old[dof::ndof])
    ybhh = numpy.dot(Mhh_old,numpy.dot(phi_pinv_old,yb))
    
    mi = numpy.zeros(len(miabs))
    if miabs != 0:
        nabs = 0
        grid_ids = []
        grid_idst = []
        grid_idsf = []
        nabst = 0
        nabsf = 0
        for i,gset in enumerate(sgrids):
            nabs += len(gset)
            grid_ids.extend(gset)
            if 't' in miabs[i]:
                nabst += len(gset)
                grid_idst.extend(gset)
            else:
                nabsf += len(gset)
                grid_idsf.extend(gset)
        ks = []
        ms = []
        bs = []
        for i,gset in enumerate(sgrids):
           #pdb.set_trace()
            if 't' in miabs[i]:
                mi[i] = float(miabs[i].replace('t',''))
            else:
                continue
            mset = numpy.ones(len(gset))*Mtot*mi[i]/len(gset)
            ms.extend(mset)
            ks.extend(K1*numpy.ones(len(gset)))
            bs.extend(abs_damping[i]*2*numpy.sqrt(mset*((100*2*numpy.pi)**2*mset)))

        K,B,M = insert_tshock_absorbers(struct_mesh_dict,grid_idst,ks,bs,ms,M_old,B_old,K_old,ndof)
        Knl,Bnl,Mnl = insert_tshock_absorbers(struct_mesh_dict,grid_idst,numpy.ones(nabst),numpy.zeros(nabst),numpy.zeros(nabst),M_old*0,B_old*0,K_old*0,ndof)
        ks = []
        ms = []
        bs = []
        for i,gset in enumerate(sgrids):
           #pdb.set_trace()
            if 't' not in miabs[i]:
                mi[i] = float(miabs[i])
            else:
                continue
            mset = numpy.ones(len(gset))*Mtot*mi[i]/len(gset)
            ms.extend(mset)
            ks.extend(K1*numpy.ones(len(gset)))
            bs.extend(abs_damping[i]*2*numpy.sqrt(mset*((100*2*numpy.pi)**2*mset)))

        K,B,M = insert_shock_absorbers(struct_mesh_dict,grid_idsf,ks,bs,ms,M,B,K,ndof)
        Knl,Bnl,Mnl = insert_shock_absorbers(struct_mesh_dict,grid_idsf,numpy.ones(nabsf),numpy.zeros(nabsf),numpy.zeros(nabsf),Mnl,Bnl,Knl,ndof)

        phi = numpy.vstack([numpy.hstack([phi_old,numpy.zeros((phi_old.shape[0],nabs))]),numpy.hstack([numpy.zeros((nabs,phi_old.shape[1])),numpy.eye(nabs)])])
        Mhh = numpy.dot(numpy.dot(phi.T,M),phi)
        Khh = numpy.dot(numpy.dot(phi.T,K),phi)
        Bhh = numpy.dot(numpy.dot(phi.T,B),phi)
        ybabs = numpy.zeros((phi.shape[0],1))
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
        
    freq = numpy.hstack([numpy.linspace(0.1,1.1*omega_n[3]/2/numpy.pi,freqlen),numpy.linspace(1.1*omega_n[3]/2/numpy.pi,0.1,freqlen)])
    omegas = freq*numpy.pi*2

    Aorig = numpy.vstack([numpy.hstack([numpy.zeros_like(Mhh_old),numpy.eye(Mhh_old.shape[0])]),\
            numpy.hstack([numpy.dot(-numpy.linalg.inv(Mhh_old),Khh_old),numpy.dot(-numpy.linalg.inv(Mhh_old),Bhh_old)])]) 

    err_type = 'abs'
    tol = 1e-10
    s = 3
    M = 1
    aero_modes = nmodes*(len(Ri))
    qinf = 0.5*rho*v**2
    
    if v > 0:
        Mbar = Mhh_old - qinf*semi_chord**2/v**2*PP_GAF[:,:,2]
    else:
        Mbar = Mhh_old#- 0.5*rho*semi_chord**2*PP_GAF[:,:,2]
    Mbar_inv = numpy.linalg.inv(Mbar)
    Kbar = Khh_old - qinf*PP_GAF[:,:,0]
    if v > 0:
        Cbar = Bhh_old - qinf*semi_chord/v*PP_GAF[:,:,1]
    else:
        Cbar = Bhh_old
    R = []
    for i,r in enumerate(Ri):
        if i == 0:
            D = numpy.eye(nmodes)
            E = PP_GAF[:,:,3+i]
        else:
            D = numpy.hstack([D,numpy.eye(nmodes)])
            E = numpy.vstack([E,PP_GAF[:,:,3+i]])
        R.extend(-numpy.ones(nmodes)*r)
    R = numpy.diag(R)
    Aaero = numpy.hstack([numpy.vstack([numpy.zeros((nmodes,nmodes)),-numpy.dot(Mbar_inv,Kbar),numpy.zeros((aero_modes,nmodes))]),numpy.vstack([numpy.eye(nmodes),-numpy.dot(Mbar_inv,Cbar),E[:,0:nmodes]]),numpy.vstack([numpy.zeros((nmodes,aero_modes)),numpy.dot(qinf*Mbar_inv,D),v*R/semi_chord])])
    D = Aaero.shape[0]
    a_aero = numpy.zeros((D,1))
    A_aero = numpy.zeros((D,M))
    B_aero = numpy.zeros((D,M))
    Mag_aero = numpy.zeros((M+1,2*freqlen))
    time_resp_orig = numpy.zeros((int(2*freqlen),int(2*freqlen)))
    rms_orig = numpy.zeros(int(2*freqlen))
    for i in range(int(2*freqlen)):
        T = 2*numpy.pi/omegas[i]
        time = numpy.linspace(0,T,int(2*freqlen))
        forig = lambda t,x: numpy.dot(Aaero,x) + numpy.vstack([numpy.zeros((Mhh_old.shape[0],1)),-numpy.dot(Mbar_inv,omegas[i]**2*amplitude*numpy.sin(omegas[i]*t)*ybhh[:,0]).reshape(Mhh_old.shape[0],1),numpy.zeros((aero_modes,1))])
        dfdxorig = lambda t,x: Aaero
        
        anew,Anew,Bnew,x0,err = harmonic_balance(forig,dfdxorig,T,s,a_aero,A_aero,B_aero,tol,err_type)
        if err == 1:
            err_type = 'rel'
            anew,Anew,Bnew,x0,err = harmonic_balance(forig,dfdxorig,T,s,a_aero,A_aero,B_aero,tol,err_type)
            err_type = 'abs'
            if err == 1:
                a_aero = numpy.zeros((D,1))
                A_aero = numpy.zeros((D,M))
                B_aero = numpy.zeros((D,M))
                anew,Anew,Bnew,x0,err = harmonic_balance(forig,dfdxorig,T,s,a_aero,A_aero,B_aero,tol,err_type)
        a_aero = anew
        A_aero = Anew
        B_aero = Bnew
        
        Mag_aero[0,i] = numpy.absolute(numpy.dot(phi_old[ref_gridid,:],a_aero[0:Mbar.shape[0]]))
        time_resp_orig[:,i] = numpy.dot(phi_old[ref_gridid,:],a_aero[0:Mbar.shape[0]])
        for ii in range(M):
            Mag_aero[ii+1,i] = numpy.dot(phi_old[ref_gridid,:],numpy.sqrt(A_aero[0:Mbar.shape[0],ii]**2+B_aero[0:Mbar.shape[0],ii]**2))
            time_resp_orig[:,i] += numpy.dot(phi_old[ref_gridid,:],A_aero[0:Mbar.shape[0],ii])*numpy.sin((ii+1)*omegas[i]*time)+numpy.dot(phi_old[ref_gridid,:],B_aero[0:Mbar.shape[0],ii])*numpy.cos((ii+1)*omegas[i]*time)
        rms_orig[i] = numpy.sqrt(numpy.mean(time_resp_orig[:,i]**2))
   #plt.figure()
   #plt.semilogy(freq,Mag_aero.T+1)
   #plt.show()

    if mi != [0]:
        M = 4
        aero_modes = (nmodes+nabs)*(len(Ri))
        qinf = 0.5*rho*v**2
        
        if v > 0:
            Mbar = Mhh - qinf*semi_chord**2/v**2*numpy.hstack([numpy.vstack([PP_GAF[:,:,2],numpy.zeros((nabs,nmodes))]),numpy.zeros((nmodes+nabs,nabs))])
        else:
            Mbar = Mhh
        Mbar_inv = numpy.linalg.inv(Mbar)
        Kbar = Khh - qinf*numpy.hstack([numpy.vstack([PP_GAF[:,:,0],numpy.zeros((nabs,nmodes))]),numpy.zeros((nmodes+nabs,nabs))])
        if v > 0:
            Cbar = Bhh - qinf*semi_chord/v*numpy.hstack([numpy.vstack([PP_GAF[:,:,1],numpy.zeros((nabs,nmodes))]),numpy.zeros((nmodes+nabs,nabs))])
        else:
            Cbar = Bhh
        R = []
        for i,r in enumerate(Ri):
            if i == 0:
                D = numpy.eye(nmodes+nabs)
                E = numpy.hstack([numpy.vstack([PP_GAF[:,:,3+i],numpy.zeros((nabs,nmodes))]),numpy.zeros((nmodes+nabs,nabs))])
            else:
                D = numpy.hstack([D,numpy.eye(nmodes+nabs)])
                E = numpy.vstack([E,numpy.hstack([numpy.vstack([PP_GAF[:,:,3+i],numpy.zeros((nabs,nmodes))]),numpy.zeros((nmodes+nabs,nabs))])])
            R.extend(-numpy.ones(nmodes+nabs)*r)
        R = numpy.diag(R)
        if v==0:
            A = numpy.vstack([numpy.hstack([numpy.zeros_like(Mhh),numpy.eye(Mhh.shape[0])]),\
                numpy.hstack([numpy.dot(-numpy.linalg.inv(Mhh),Khh),numpy.dot(-numpy.linalg.inv(Mhh),Bhh)])]).real
        else:
            A = numpy.hstack([numpy.vstack([numpy.zeros((nmodes+nabs,nmodes+nabs)),-numpy.dot(Mbar_inv,Kbar),numpy.zeros((aero_modes,nmodes+nabs))]),numpy.vstack([numpy.eye(nmodes+nabs),-numpy.dot(Mbar_inv,Cbar),E[:,0:nmodes+nabs]]),numpy.vstack([numpy.zeros((nmodes+nabs,aero_modes)),numpy.dot(qinf*Mbar_inv,D),v*R/semi_chord])]).real
        
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
            print i
            T = 2*numpy.pi/omegas[i]
            time = numpy.linspace(0,T,int(2*freqlen))
#           if v==0:
#               fabs = lambda t,x: numpy.dot(A,x) + numpy.vstack([numpy.zeros((Mhh.shape[0],1)),-numpy.dot(Mbar_inv,omegas[i]**2*amplitude*numpy.sin(omegas[i]*t)*ybhh[:,0]).reshape(Mhh.shape[0],1)]) - numpy.vstack([numpy.zeros((Mhh.shape[0],1)),numpy.dot(Mbar_inv,numpy.dot(phi_pinv,magnet_force(numpy.dot(Knl,numpy.dot(phi,x[nmodes+nabs:2*(nmodes+nabs)])),mu0,Br,A1,A2,a1,a2,r1,r2,tk)))])
#               dfdxabs = lambda t,x: A - numpy.vstack([numpy.zeros((Mhh.shape[0],A.shape[1])),numpy.hstack([numpy.zeros((nmodes+nabs,nmodes+nabs)),numpy.dot(Mbar_inv,numpy.dot(phi.T,numpy.dot(numpy.dot(Knl,magnet_force_derivative(Knl*numpy.dot(phi,x[nmodes+nabs:2*(nmodes+nabs)]),mu0,Br,A1,A2,a1,a2,r1,r2,tk)),phi)))])])
#           else:
#               fabs = lambda t,x: numpy.dot(A,x) + numpy.vstack([numpy.zeros((Mhh.shape[0],1)),-numpy.dot(Mbar_inv,omegas[i]**2*amplitude*numpy.sin(omegas[i]*t)*ybhh[:,0]).reshape(Mhh.shape[0],1),numpy.zeros((aero_modes,1))]) - numpy.vstack([numpy.zeros((Mhh.shape[0],1)),numpy.dot(Mbar_inv,numpy.dot(phi_pinv,magnet_force(numpy.dot(Knl,numpy.dot(phi,x[nmodes+nabs:2*(nmodes+nabs)])),mu0,Br,A1,A2,a1,a2,r1,r2,tk))),numpy.zeros((aero_modes,1))])
#               dfdxabs = lambda t,x: A - numpy.vstack([numpy.zeros((Mhh.shape[0],A.shape[1])),numpy.hstack([numpy.zeros((nmodes+nabs,nmodes+nabs)),numpy.dot(Mbar_inv,numpy.dot(phi.T,numpy.dot(numpy.dot(Knl,magnet_force_derivative(Knl*numpy.dot(phi,x[nmodes+nabs:2*(nmodes+nabs)]),mu0,Br,A1,A2,a1,a2,r1,r2,tk)),phi))),numpy.zeros((nmodes+nabs,aero_modes))]),numpy.zeros((aero_modes,A.shape[1]))])
            if v==0:
                fabs = lambda t,x: numpy.dot(A,x) + numpy.vstack([numpy.zeros((Mhh.shape[0],1)),-numpy.dot(Mbar_inv,omegas[i]**2*amplitude*numpy.sin(omegas[i]*t)*ybhh[:,0]).reshape(Mhh.shape[0],1)]) - numpy.vstack([numpy.zeros((Mhh.shape[0],1)),numpy.dot(Mbar_inv,numpy.dot(phi.T,K3*numpy.dot(Knl,numpy.dot(phi,x[nmodes+nabs:2*(nmodes+nabs)]))**3))])
                dfdxabs = lambda t,x: A - numpy.vstack([numpy.zeros((Mhh.shape[0],A.shape[1])),numpy.hstack([numpy.zeros((nmodes+nabs,nmodes+nabs)),numpy.dot(Mbar_inv,numpy.dot(phi.T,numpy.dot(3*K3*Knl*numpy.dot(Knl,numpy.dot(phi,x[nmodes+nabs:2*(nmodes+nabs)]))**2,phi)))])])
            else:
                fabs = lambda t,x: numpy.dot(A,x) + numpy.vstack([numpy.zeros((Mhh.shape[0],1)),-numpy.dot(Mbar_inv,omegas[i]**2*amplitude*numpy.sin(omegas[i]*t)*ybhh[:,0]).reshape(Mhh.shape[0],1),numpy.zeros((aero_modes,1))]) - numpy.vstack([numpy.zeros((Mhh.shape[0],1)),numpy.dot(Mbar_inv,numpy.dot(phi.T,K3*numpy.dot(Knl,numpy.dot(phi,x[nmodes+nabs:2*(nmodes+nabs)]))**3)),numpy.zeros((aero_modes,1))])
                dfdxabs = lambda t,x: A - numpy.vstack([numpy.zeros((Mhh.shape[0],A.shape[1])),numpy.hstack([numpy.zeros((nmodes+nabs,nmodes+nabs)),numpy.dot(Mbar_inv,numpy.dot(phi.T,numpy.dot(Knl*3*K3*numpy.dot(Knl,numpy.dot(phi,x[nmodes+nabs:2*(nmodes+nabs)]))**2,phi))),numpy.zeros((nmodes+nabs,aero_modes))]),numpy.zeros((aero_modes,A.shape[1]))])
           #fabs = lambda t,x: numpy.dot(A,x) + numpy.vstack([numpy.zeros((Mhh.shape[0],1)),-numpy.dot(Mbar_inv,omegas[i]**2*amplitude*numpy.sin(omegas[i]*t)*ybhh[:,0]).reshape(Mhh.shape[0],1),numpy.zeros((aero_modes,1))]) - numpy.vstack([numpy.zeros((Mhh.shape[0],1)),numpy.dot(Mbar_inv,numpy.dot(numpy.linalg.pinv(phi),magnet_force(numpy.dot(Knl,numpy.dot(phi,x[nmodes+nabs:2*(nmodes+nabs)])),mu0,Br,A1,A2,a1,a2,r1,r2,tk))),numpy.zeros((aero_modes,1))])
           #dfdxabs = lambda t,x: A - numpy.vstack([numpy.zeros((Mhh.shape[0],A.shape[1])),numpy.hstack([numpy.zeros((nmodes+nabs,nmodes+nabs)),numpy.dot(Mbar_inv,numpy.dot(phi.T,numpy.dot(numpy.dot(Knl,magnet_force_derivative(Knl*numpy.dot(phi,x[nmodes+nabs:2*(nmodes+nabs)]),mu0,Br,A1,A2,a1,a2,r1,r2,tk)),phi))),numpy.zeros((nmodes+nabs,aero_modes))]),numpy.zeros((aero_modes,A.shape[1]))])
               #fabs = lambda t,x: numpy.dot(A,x) + numpy.vstack([numpy.zeros((Mhh.shape[0],1)),-numpy.dot(Mbar_inv,omegas[i]**2*amplitude*numpy.sin(omegas[i]*t)*ybhh[:,0]).reshape(Mhh.shape[0],1),numpy.zeros((aero_modes,1))]) - numpy.vstack([numpy.zeros((Mhh.shape[0],1)),numpy.dot(Mbar_inv,numpy.dot(numpy.linalg.pinv(phi),100*numpy.dot(Knl,numpy.dot(phi,x[nmodes+nabs:2*(nmodes+nabs)]))**3)),numpy.zeros((aero_modes,1))])
               #dfdxabs = lambda t,x: A - numpy.vstack([numpy.zeros((Mhh.shape[0],A.shape[1])),numpy.hstack([numpy.zeros((nmodes+nabs,nmodes+nabs)),numpy.dot(Mbar_inv,numpy.dot(phi.T,numpy.dot(numpy.dot(Knl,(300*Knl*numpy.dot(phi,x[nmodes+nabs:2*(nmodes+nabs)]))**2),phi))),numpy.zeros((nmodes+nabs,aero_modes))]),numpy.zeros((aero_modes,A.shape[1]))])
           #fabs = lambda t,x: numpy.dot(A,x) + numpy.vstack([numpy.zeros((Mhh.shape[0],1)),-numpy.dot(Mbar_inv,omegas[i]**2*amplitude*numpy.sin(omegas[i]*t)*ybhh[:,0]).reshape(Mhh.shape[0],1)]) - numpy.vstack([numpy.zeros((Mhh.shape[0],1)),numpy.dot(Mbar_inv,numpy.dot(numpy.linalg.pinv(phi),100*numpy.dot(Knl,numpy.dot(phi,x[nmodes+nabs:2*(nmodes+nabs)]))**3))])
           #dfdxabs = lambda t,x: A - numpy.vstack([numpy.zeros((Mhh.shape[0],A.shape[1])),numpy.hstack([numpy.zeros((nmodes+nabs,nmodes+nabs)),numpy.dot(Mbar_inv,numpy.dot(phi.T,numpy.dot(numpy.dot(Knl,(300*Knl*numpy.dot(phi,x[nmodes+nabs:2*(nmodes+nabs)]))**2),phi)))])])
            
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

    return [freq,rms_orig,rms]
