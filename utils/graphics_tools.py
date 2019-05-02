#from pygame.locals import *
#import pygame.draw
#import pygame.time
import multiprocessing as mp
import numpy
import scipy.interpolate, scipy.io
import matplotlib.pyplot as plt
import os, pdb
from math import sin, cos
ORIGINX = 0
ORIGINY = 0

def plot_geom(mesh_dict,plot_path):
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    maxx = 0
    maxy = 0
    maxz = 0
    minx = 0
    miny = 0
    minz = 0
    legend = []
    rgb=[(0,0,0),(0,0,1),(0,1,0),(1,0,0),(1,1,0),(1,0,1),(0,1,1),(1,1,1),(0.5,0,0),(0,0.5,0),(0,0,0.5),(0.25,0.5,0.75),(0.5,0.25,0.75),(0.75,0.5,0.25),(0.25,0.25,0.75),(0.5,0.5,0.5)]
    for i,component in enumerate(mesh_dict):
        if component != 'full_set':
           #rgb = (numpy.random.random(),numpy.random.random(),numpy.random.random())
            ax.plot_wireframe(mesh_dict[component]['x'],mesh_dict[component]['y'],mesh_dict[component]['z'],color=rgb[i])
            legend.append(component)
        if numpy.max(mesh_dict[component]['x'])>maxx:
            maxx = numpy.max(mesh_dict[component]['x'])
        if numpy.max(mesh_dict[component]['y'])>maxy:
            maxy = numpy.max(mesh_dict[component]['y'])
        if numpy.max(mesh_dict[component]['z'])>maxz:
            maxz = numpy.max(mesh_dict[component]['z'])
        if numpy.min(mesh_dict[component]['x'])<minx:
            minx = numpy.min(mesh_dict[component]['x'])
        if numpy.min(mesh_dict[component]['y'])<miny:
            miny = numpy.min(mesh_dict[component]['y'])
        if numpy.min(mesh_dict[component]['z'])<minz:
            minz = numpy.min(mesh_dict[component]['z'])
    max_dimension = numpy.max([maxx-minx,maxy-miny,maxz-minz])
    ax.set_xlim([minx,minx+max_dimension])
    ax.set_ylim([miny,miny+max_dimension])
    ax.set_zlim([minz,minz+max_dimension])
    ax.legend(legend)
    plt.savefig(plot_path+'/geometry.png')
    plt.close()
    return


def draw_3dline(surface, color, a, b, maxx):
    """Convert 3D coordinates to 2D and draw line."""
    if a[1]>=0:
        ax, ay = -(a[1]-(maxx-a[0])*a[1]/(5*maxx+a[0]))+ORIGINX, -(a[2]+0.2*(maxx-a[0]))+ORIGINY
    elif a[1]<0:
        ax, ay = -(a[1]-(maxx-a[0])*a[1]/(5*maxx+a[0]))+ORIGINX, -(a[2]+0.2*(maxx-a[0]))+ORIGINY
    if b[1]>=0:
        bx, by = -(b[1]-(maxx-b[0])*b[1]/(5*maxx+b[0]))+ORIGINX, -(b[2]+0.2*(maxx-b[0]))+ORIGINY
    if b[1]<0:
        bx, by = -(b[1]-(maxx-b[0])*b[1]/(5*maxx+b[0]))+ORIGINX, -(b[2]+0.2*(maxx-b[0]))+ORIGINY
    pygame.draw.line(surface, color, (ax, ay), (bx, by))

def draw_panel(surface, color, panel, maxx):
    """Draw 3D cube."""
    a, b, c, d = panel
    draw_3dline(surface, color, a, b, maxx)
    draw_3dline(surface, color, b, c, maxx)
    draw_3dline(surface, color, c, d, maxx)
    draw_3dline(surface, color, d, a, maxx)

def rotate_3dpoint(p, angle, axis):
    """Rotate a 3D point around given axis."""
    ret = [0, 0, 0]
    cosang = cos(angle)
    sinang = sin(angle)
    ret[0] += (cosang+(1-cosang)*axis[0]*axis[0])*p[0]
    ret[0] += ((1-cosang)*axis[0]*axis[1]-axis[2]*sinang)*p[1]
    ret[0] += ((1-cosang)*axis[0]*axis[2]+axis[1]*sinang)*p[2]
    ret[1] += ((1-cosang)*axis[0]*axis[1]+axis[2]*sinang)*p[0]
    ret[1] += (cosang+(1-cosang)*axis[1]*axis[1])*p[1]
    ret[1] += ((1-cosang)*axis[1]*axis[2]-axis[0]*sinang)*p[2]
    ret[2] += ((1-cosang)*axis[0]*axis[2]-axis[1]*sinang)*p[0]
    ret[2] += ((1-cosang)*axis[1]*axis[2]+axis[0]*sinang)*p[1]
    ret[2] += (cosang+(1-cosang)*axis[2]*axis[2])*p[2]
    return ret

def rotate_object(obj, angle, axis):
    """Rotate an object around given axis."""
    for i in range(len(obj)):
        obj[i] = rotate_3dpoint(obj[i], angle, axis)

def window_plot(mesh_dict):
    global ORIGINX, ORIGINY
    pygame.init()
    screen = pygame.display.set_mode((800,600))
    # Move origin to center of screen
    ORIGINX = screen.get_width()/2
    ORIGINY = screen.get_height()*3/4
    scale = screen.get_width()*3/8/numpy.max(mesh_dict['full_set']['y'])
    for component in mesh_dict:
        if component != 'full_set':
            for i in range(mesh_dict[component]['x'].shape[0]-1):
                for j in range(mesh_dict[component]['x'].shape[1]-1):
                    panel = [(mesh_dict[component]['x'][i,j]*scale,mesh_dict[component]['y'][i,j]*scale,mesh_dict[component]['z'][i,j]*scale),(mesh_dict[component]['x'][i+1,j]*scale,mesh_dict[component]['y'][i+1,j]*scale,mesh_dict[component]['z'][i+1,j]*scale),(mesh_dict[component]['x'][i+1,j+1]*scale,mesh_dict[component]['y'][i+1,j+1]*scale,mesh_dict[component]['z'][i+1,j+1]*scale),(mesh_dict[component]['x'][i,j+1]*scale,mesh_dict[component]['y'][i,j+1]*scale,mesh_dict[component]['z'][i,j+1]*scale)]
                    draw_panel(screen, 255, panel, scale*numpy.max(mesh_dict['full_set']['x']))

   #t = 0 
    while 1:
        pygame.display.flip()
        event = pygame.event.poll()
   #    draw_panel(screen, 0, panel, scale*numpy.max(mesh_dict['full_set']['x']))
   #    for component in mesh_dict:
   #        if component != 'full_set':
   #            for i in range(mesh_dict[component]['x'].shape[0]-1):
   #                for j in range(mesh_dict[component]['x'].shape[1]-1): 
   #                    panel = [(mesh_dict[component]['x'][i,j]*scale,mesh_dict[component]['y'][i,j]*scale,(mesh_dict[component]['z'][i,j]+5*numpy.cos(0.1*t))*scale),(mesh_dict[component]['x'][i+1,j]*scale,mesh_dict[component]['y'][i+1,j]*scale,(mesh_dict[component]['z'][i+1,j]+5*numpy.cos(0.1*t))*scale),(mesh_dict[component]['x'][i+1,j+1]*scale,mesh_dict[component]['y'][i+1,j+1]*scale,(mesh_dict[component]['z'][i+1,j+1]+5*numpy.cos(0.1*t))*scale),(mesh_dict[component]['x'][i,j+1]*scale,mesh_dict[component]['y'][i,j+1]*scale,(mesh_dict[component]['z'][i,j+1]+5*numpy.cos(0.1*t))*scale)]
   #                    draw_panel(screen, 255, panel, scale*numpy.max(mesh_dict['full_set']['x']))
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            break
   #    t+=0.05
    return

def plot_flutter(plot_path,V,g,k_flutter,semi_chord,nmodes,omega_n,nabs=0):
    '''Plot Vg and V-omega diagrams'''
    plt.figure()
    plt.plot(V,g)
    legend = []
    for i in range(nmodes):
        legend.append('Mode {}'.format(i+1))
    plt.legend(legend,loc='best',fontsize = 'x-small')
    plt.xlabel('Velocity [m/s]')
    plt.ylabel('g')
    plt.title('V-g diagram')
    plt.grid()
    if nabs == 0:
        plt.savefig(plot_path+'/V-g_diagram.png')
    else:
        plt.savefig(plot_path+'/V-g_diagram_{}_abs.png'.format(nabs))
    plt.close()
    
    plt.figure()
    plt.plot(V,k_flutter*V.reshape(len(V),1)/semi_chord)
    plt.legend(legend,loc='best',fontsize = 'x-small')
    plt.xlabel('Velocity [m/s]')
    plt.ylabel('Omega [rad/s]')
    plt.title('V-omega diagram')
    plt.grid()
    if nabs == 0:
        plt.savefig(plot_path+'/V-omega_diagram.png')
    else:
        plt.savefig(plot_path+'/V-omega_diagram_{}_abs.png'.format(nabs))
    plt.close()

    plt.figure()
    gama = numpy.zeros_like(k_flutter)
    freqnat = omega_n
    for i in range(k_flutter.shape[0]):
        for j in range(k_flutter.shape[1]):
            if k_flutter[i,j] != 0:
                x = g[i,j]/(k_flutter[i,j]*V[i]/semi_chord)
                gama[i,j] = x/numpy.sqrt(1+x**2)
                freqnat[j] = k_flutter[i,j]*V[i]/semi_chord/numpy.sqrt(1-gama[i,j]**2)
            else:
                gama[i,j] = g[i,j]/freqnat[j]
    plt.plot(V,gama)
    plt.legend(legend,loc='best',fontsize = 'x-small')
    plt.xlabel('Velocity [m/s]')
    plt.ylabel('Gama')
    plt.ylim([-1,0.25])
    plt.title('V-gama diagram')
    plt.grid()
    if nabs == 0:
        plt.savefig(plot_path+'/V-gama.png')
    else:
        plt.savefig(plot_path+'/V-gama_{}_abs.png'.format(nabs))
    plt.close()

    plt.figure()
    plt.plot(g,k_flutter)
    plt.legend(legend,loc='best',fontsize = 'x-small')
    plt.xlabel('g')
    plt.ylabel('k')
    plt.title('g-k diagram')
    plt.grid()
    if nabs == 0:
        plt.savefig(plot_path+'/g-k_diagram.png')
    else:
        plt.savefig(plot_path+'/g-k_diagram_{}_abs.png'.format(nabs))
    plt.close()

    return

class Plot_class:
    def __init__(self,path,struct_mesh_dict,aero_mesh_dict,ndof,h,tmax,nabs=0,sgrids=[]):
        self.path = path 
        self.struct_mesh_dict = struct_mesh_dict
        self.aero_mesh_dict = aero_mesh_dict
        self.h = h
        self.tmax = tmax
        self.ndof = ndof
        self.nabs = nabs
        if type(sgrids) == type({}):
            self.sgrids = sgrids
        else:   
            self.sgrids = numpy.array(sgrids)

def mp_plot(plot_list):
    '''Function to plot lots of figures using parallel processing'''
    pltclass = plot_list[0]
    j = plot_list[1]
    t = plot_list[2]
    nclamped = 0
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    for i,component in enumerate(pltclass.struct_mesh_dict.keys()):
        if component != 'full_set':
            nx = pltclass.struct_mesh_dict[component]['x'].shape[0]
            ny = pltclass.struct_mesh_dict[component]['x'].shape[1]
           #npx = pltclass.aero_mesh_dict[component]['area'].shape[0]
           #npy = pltclass.aero_mesh_dict[component]['area'].shape[1]
            structind1 = pltclass.struct_mesh_dict[pltclass.struct_mesh_dict.keys()[i]]['global_index']-nclamped
            try:
                nclamped += pltclass.struct_mesh_dict[pltclass.struct_mesh_dict.keys()[i]]['nclamped']
            except:
                nclamped += 0
            structind2 = pltclass.struct_mesh_dict[pltclass.struct_mesh_dict.keys()[i+1]]['global_index']-nclamped
           #for jj,aerocomp in enumerate(pltclass.aero_mesh_dict.keys()):
           #    if aerocomp == component:
           #        break
           #aeroind1 = pltclass.aero_mesh_dict[pltclass.aero_mesh_dict.keys()[jj]]['global_index']
           #aeroind2 = pltclass.aero_mesh_dict[pltclass.aero_mesh_dict.keys()[jj+1]]['global_index']
            if pltclass.ndof == 1:
                dispz = pltclass.h[structind1:structind2,j]
                dispx = numpy.zeros_like(dispz)
                dispy = numpy.zeros_like(dispz)
            elif pltclass.ndof >= 3:
                dispx = pltclass.h[structind1*pltclass.ndof:structind2*pltclass.ndof:pltclass.ndof,j]
                dispy = pltclass.h[1+structind1*pltclass.ndof:structind2*pltclass.ndof:pltclass.ndof,j]
                dispz = pltclass.h[2+structind1*pltclass.ndof:structind2*pltclass.ndof:pltclass.ndof,j]
            maxlen = numpy.max([numpy.max(pltclass.struct_mesh_dict['full_set']['x'])-numpy.min(pltclass.struct_mesh_dict['full_set']['x']),numpy.max(pltclass.struct_mesh_dict['full_set']['y'])-numpy.min(pltclass.struct_mesh_dict['full_set']['y']),numpy.max(pltclass.struct_mesh_dict['full_set']['z'])-numpy.min(pltclass.struct_mesh_dict['full_set']['z'])])
            ax.plot_wireframe(pltclass.struct_mesh_dict[component]['x']+0.1*maxlen*dispx.reshape(nx,ny),pltclass.struct_mesh_dict[component]['y']+0.1*maxlen*dispy.reshape(nx,ny),pltclass.struct_mesh_dict[component]['z']+0.1*maxlen*dispz.reshape(nx,ny))
            ax.set_xlim([numpy.min(pltclass.struct_mesh_dict['full_set']['x']),numpy.min(pltclass.struct_mesh_dict['full_set']['x'])+maxlen])
            ax.set_ylim([numpy.min(pltclass.struct_mesh_dict['full_set']['y']),numpy.min(pltclass.struct_mesh_dict['full_set']['y'])+maxlen])
            ax.set_zlim([-maxlen/2,maxlen/2])
            ax.set_title('t=0-{0:.3g}s'.format(pltclass.tmax.real))
    if pltclass.nabs != 0:
        ax.plot(pltclass.struct_mesh_dict['full_set']['x'][pltclass.sgrids],pltclass.struct_mesh_dict['full_set']['y'][pltclass.sgrids],0.1*maxlen+0.1*maxlen*pltclass.h[-pltclass.nabs:,j],'ro')
    plt.savefig(pltclass.path+'/{}.png'.format(j))
    plt.close()

    return

def mp_plot_time(plot_list):
    '''Function to plot lots of figures using parallel processing'''
    pltclass = plot_list[0]
    j = plot_list[1]
    t = plot_list[2]
    v_time = plot_list[3]
    freq = plot_list[4]
    frf_path = plot_list[5] + '/frf_results.mat'
    ext_shape = plot_list[6]
    nclamped = 0
    try:
        frf_dict = scipy.io.loadmat(frf_path)
        fig = plt.figure(figsize=(20,7.5))
        ax0 = fig.add_subplot(121)
       #pdb.set_trace()
        if v_time < frf_dict['V'][0][0]:
            if pltclass.nabs == 0:
                frf_funcr = scipy.interpolate.interp1d(frf_dict['freq'][0,:],frf_dict['frf_orig'][0,:].real)
                frf_funci = scipy.interpolate.interp1d(frf_dict['freq'][0,:],frf_dict['frf_orig'][0,:].imag)
            else:
                frf_funcr = scipy.interpolate.interp1d(frf_dict['freq'][0,:],frf_dict['frf_abs'][0,:].real)
                frf_funci = scipy.interpolate.interp1d(frf_dict['freq'][0,:],frf_dict['frf_abs'][0,:].imag)
            ax0.plot(frf_dict['freq'][0,:],numpy.log10(numpy.absolute(frf_funcr(frf_dict['freq'][0,:])+1j*frf_funci(frf_dict['freq'][0,:])+1)),linewidth=2)
            ax0.plot(freq,numpy.log10(numpy.absolute(frf_funcr(freq)+1j*frf_funci(freq)+1)),'ro', markersize=8)
        else:
            if pltclass.nabs == 0:
                frf_funcr = scipy.interpolate.interp2d(frf_dict['freq'][0,:],frf_dict['V'][0],frf_dict['frf_aero_orig'].real)
                frf_funci = scipy.interpolate.interp2d(frf_dict['freq'][0,:],frf_dict['V'][0],frf_dict['frf_aero_orig'].imag)
            else:                                                                             
                frf_funcr = scipy.interpolate.interp2d(frf_dict['freq'][0,:],frf_dict['V'][0],frf_dict['frf_aero_abs'].real)
                frf_funci = scipy.interpolate.interp2d(frf_dict['freq'][0,:],frf_dict['V'][0],frf_dict['frf_aero_abs'].imag)
            ax0.plot(frf_dict['freq'][0,:],numpy.log10(numpy.absolute(frf_funcr(frf_dict['freq'][0,:],v_time)+1j*frf_funci(frf_dict['freq'][0,:],v_time)+1)),linewidth=2)
            if type(freq) is not str:
                ax0.plot(freq,numpy.log10(numpy.absolute(frf_funcr(freq,v_time)+1j*frf_funci(freq,v_time)+1)),'ro', markersize=8)
        ax0.grid()
        ax0.set_xlabel('frequency [Hz]')
        ax0.set_ylabel('Magnitude [log]')
        ax0.set_title('FRF')
        ax = fig.add_subplot(122, projection='3d')
    except:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    for i,component in enumerate(pltclass.struct_mesh_dict.keys()):
        if component != 'full_set':
            nx = pltclass.struct_mesh_dict[component]['x'].shape[0]
            ny = pltclass.struct_mesh_dict[component]['x'].shape[1]
           #npx = pltclass.aero_mesh_dict[component]['area'].shape[0]
           #npy = pltclass.aero_mesh_dict[component]['area'].shape[1]
            structind1 = pltclass.struct_mesh_dict[pltclass.struct_mesh_dict.keys()[i]]['global_index']-nclamped
            try:
                nclamped += pltclass.struct_mesh_dict[pltclass.struct_mesh_dict.keys()[i]]['nclamped']
            except:
                nclamped += 0
            structind2 = pltclass.struct_mesh_dict[pltclass.struct_mesh_dict.keys()[i+1]]['global_index']-nclamped
            if pltclass.ndof == 1:
                dispz = pltclass.h[structind1:structind2,j]
                dispx = numpy.zeros_like(dispz)
                dispy = numpy.zeros_like(dispz)
            elif pltclass.ndof >= 3:
                dispx = pltclass.h[structind1*pltclass.ndof:structind2*pltclass.ndof:pltclass.ndof,j]
                dispy = pltclass.h[1+structind1*pltclass.ndof:structind2*pltclass.ndof:pltclass.ndof,j]
                dispz = pltclass.h[2+structind1*pltclass.ndof:structind2*pltclass.ndof:pltclass.ndof,j]
            maxlen = numpy.max([numpy.max(pltclass.struct_mesh_dict['full_set']['x'])-numpy.min(pltclass.struct_mesh_dict['full_set']['x']),numpy.max(pltclass.struct_mesh_dict['full_set']['y'])-numpy.min(pltclass.struct_mesh_dict['full_set']['y']),numpy.max(pltclass.struct_mesh_dict['full_set']['z'])-numpy.min(pltclass.struct_mesh_dict['full_set']['z'])])
            ax.plot_wireframe(pltclass.struct_mesh_dict[component]['x']+dispx.reshape(nx,ny),pltclass.struct_mesh_dict[component]['y']+dispy.reshape(nx,ny),pltclass.struct_mesh_dict[component]['z']+dispz.reshape(nx,ny))
            ax.set_xlim([numpy.min(pltclass.struct_mesh_dict['full_set']['x']),numpy.min(pltclass.struct_mesh_dict['full_set']['x'])+maxlen])
            ax.set_ylim([numpy.min(pltclass.struct_mesh_dict['full_set']['y']),numpy.min(pltclass.struct_mesh_dict['full_set']['y'])+maxlen])
            ax.set_zlim([-maxlen/2,maxlen/2])
            ax.set_title('V= {} m/s, f= {} Hz'.format(v_time,freq))
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
    if pltclass.nabs != 0:
        ax.plot(pltclass.struct_mesh_dict['full_set']['x'][pltclass.sgrids],pltclass.struct_mesh_dict['full_set']['y'][pltclass.sgrids],0.1+pltclass.h[-pltclass.nabs:,j],'ro')
       #if pltclass.ndof == 1:
       #    ax.plot(pltclass.struct_mesh_dict['full_set']['x'][pltclass.sgrids-1],pltclass.struct_mesh_dict['full_set']['y'][pltclass.sgrids-1],0.1+pltclass.h[pltclass.sgrids-1,j]+pltclass.h[-pltclass.nabs:,j],'ro')
       #elif pltclass.ndof >= 3:
       #    ax.plot(pltclass.struct_mesh_dict['full_set']['x'][pltclass.sgrids-1],pltclass.struct_mesh_dict['full_set']['y'][pltclass.sgrids-1],0.1+pltclass.h[(pltclass.sgrids-1)*pltclass.ndof+2,j]+pltclass.h[-pltclass.nabs:,j],'ro')
    if ext_shape == 'torsion':
        ax.view_init(0, 90)
        dx = numpy.max(pltclass.struct_mesh_dict['full_set']['x']) - numpy.min(pltclass.struct_mesh_dict['full_set']['x'])
        ax.set_xlim([numpy.min(pltclass.struct_mesh_dict['full_set']['x'])-dx/2,numpy.min(pltclass.struct_mesh_dict['full_set']['x'])+1.5*dx])
    elif ext_shape == 'bending':
        ax.view_init(0, 0) 
       #ax.view_init(0, 90)
       #dx = numpy.max(pltclass.struct_mesh_dict['full_set']['x']) - numpy.min(pltclass.struct_mesh_dict['full_set']['x'])
       #ax.set_xlim([numpy.min(pltclass.struct_mesh_dict['full_set']['x'])-dx/2,numpy.min(pltclass.struct_mesh_dict['full_set']['x'])+1.5*dx])
    plt.savefig(pltclass.path+'/{}.png'.format(j))
    plt.close()

    return

def flutter_video(struct_mesh_dict,aero_mesh_dict,plot_path,semi_chord,phi,omega_n,v_flutter, flutter_mode, k_flutter, ndof, ncores, nabs=0, sgrids = []):
    path = plot_path+'/temp'
    if not os.path.isdir(path):
        os.mkdir(path)
    fmax = numpy.max(numpy.hstack([omega_n/2/numpy.pi,k_flutter*v_flutter/semi_chord/2/numpy.pi]))
    tmax = 10
    time = numpy.linspace(0,tmax,1200)
    h = numpy.zeros((phi.shape[0],1200))
    for j,t in enumerate(time):
        for mode in range(phi.shape[1]):
            h[:,j] += (flutter_mode[mode]*phi[:,mode]*numpy.sin(t*k_flutter*v_flutter/semi_chord)).real#+ (flutter_mode[mode]*phi[:,mode]*numpy.cos(t*k_flutter*v_flutter/semi_chord)).imag
    h /= numpy.max(numpy.absolute(h))
    pltclass = Plot_class(path,struct_mesh_dict,aero_mesh_dict,ndof,h,tmax,nabs,sgrids)
    for j,t in enumerate(time):
        mp_plot([pltclass,j,t])

    os.system('avconv -r 120 -i "{}%d.png" {}'.format(path+'/',plot_path+'/')+'flutter_v_{0:.4g}.mp4'.format(v_flutter))
    os.system('avconv -r 20 -i "{}%d.png" {}'.format(path+'/',plot_path+'/')+'slow_flutter_v_{0:.4g}.mp4'.format(v_flutter))
    os.system('rm -r {}'.format(path))
    
    return

def divergence_video(plot_path,div_V,div_mode,struct_mesh_dict,aero_mesh_dict,ndof):
    path = plot_path+'/tempdiv'
    if not os.path.isdir(path):
        os.mkdir(path)
    tmax = 5
    time = numpy.linspace(0,tmax,300)
    h = numpy.zeros((div_mode.shape[0],300))
    for j,t in enumerate(time):
        h[:,j] = div_mode*0.5*t
    pltclass = Plot_class(path,struct_mesh_dict,aero_mesh_dict,ndof,h,tmax)
    for j,t in enumerate(time):
        mp_plot([pltclass,j,t])
    os.system('avconv -r 60 -i "{}%d.png" {}'.format(path+'/',plot_path+'/')+'divergence_v_{0:.4g}.mp4'.format(div_V))
    os.system('rm -r {}'.format(path))
    return

def plot_frf(frf_results,v,freqlen,miabs,plot_path):
    plot_path += '/frf'
    if not os.path.isdir(plot_path):
        os.mkdir(plot_path)
    for ii in range(len(v)):
        frf_orig = frf_results['frf_orig'][ii,:]
        frf_aero_orig = frf_results['frf_aero_orig'][ii,:]
        freq = frf_results['freq'][ii,:]
        plt.figure()
        plt.semilogy(freq,numpy.absolute(frf_orig+1))
        plt.semilogy(freq,numpy.absolute(frf_aero_orig+1))
        if miabs != 0:
            frf_abs = frf_results['frf_abs'][ii,:]
            frf_aero_abs = frf_results['frf_aero_abs'][ii,:]
            plt.semilogy(freq,numpy.absolute(frf_abs+1))
            plt.semilogy(freq,numpy.absolute(frf_aero_abs+1))
            legend = ['original','original + aero forces','with absorbers','with absorbers + aero forces']
        else:
            legend = ['original','original + aero forces']
        plt.legend(legend,loc='best',fontsize = 'x-small')
        plt.xlabel('Frequencia (Hz)')
        plt.ylabel('Magnitude (DB)')
        plt.grid()
        plt.savefig(plot_path+'/{}frf.png'.format(v[ii]))
        plt.close()
    if len(v) > 1:
        X = []
        for i in range(int(freqlen)):
            X.append(v)
        Xi = numpy.array(X).T
        Yi = frf_results['freq']
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(Yi, Xi, numpy.log10(numpy.absolute(frf_results['frf_aero_orig']+1)))
        ax.set_xlabel('frequency [Hz]')
        ax.set_ylabel('Velocity [m/s]')
        ax.set_zlabel('FRF')
        plt.savefig(plot_path+'/frf_aero_3D.png')
        plt.close()
        plt.figure()
        Zi = numpy.log10(numpy.absolute(frf_results['frf_aero_orig']+1))
        X = []
        Y = []
        for i in range(int(freqlen)):
            X.append(numpy.linspace(v[0],v[-1],freqlen))
            Y.append(Yi[0,:])
        X=numpy.array(X).T
        Y=numpy.array(Y)
        Z = numpy.zeros((int(freqlen),int(freqlen)))
        for i in range(int(freqlen)):
            func = scipy.interpolate.interp1d(Xi[:,i],Zi[:,i])
            Z[:,i] = func(X[:,i])
        plt.imshow(Z.T, vmin=Z.min(), vmax=Z.max(), origin='lower',extent=[X.min(), X.max(), Y.min(), Y.max()])
        plt.ylabel('frequency [Hz]')
        plt.xlabel('Velocity [m/s]')
        plt.savefig(plot_path+'/frf_aero_colormap.png')
        plt.close()
        if miabs != 0:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_wireframe(Yi, Xi, numpy.log10(numpy.absolute(frf_results['frf_aero_abs']+1)))
            ax.set_xlabel('frequency [Hz]')
            ax.set_ylabel('Velocity [m/s]')
            ax.set_zlabel('FRF')
            plt.savefig(plot_path+'/frf_aero_abs_3D.png')
            plt.close()
            plt.figure()
            Zi = numpy.log10(numpy.absolute(frf_results['frf_aero_abs']+1))
            Z = numpy.zeros((int(freqlen),int(freqlen)))
            for i in range(int(freqlen)):
                func = scipy.interpolate.interp1d(Xi[:,i],Zi[:,i])
                Z[:,i] = func(X[:,i])
            plt.imshow(Z.T, vmin=Z.min(), vmax=Z.max(), origin='lower',extent=[X.min(), X.max(), Y.min(), Y.max()])
            plt.ylabel('frequency [Hz]')
            plt.xlabel('Velocity [m/s]')
            plt.savefig(plot_path+'/frf_aero_abs_colormap.png')
            plt.close()
        return

def time_video(struct_mesh_dict,aero_mesh_dict,plot_path,ref_grid,semi_chord,phi,time,v_time, resp, freq, ndof, ncores, frf_path, ext_shape, nabs=0, sgrids = []):
    path = plot_path+'/temptime'
    if not os.path.isdir(path):
        os.mkdir(path)
    h = numpy.zeros((phi.shape[0],len(time)))
    for j,t in enumerate(time):
        for mode in range(phi.shape[1]):
            h[:,j] += (resp[j][mode]*phi[:,mode]).real
    pltclass = Plot_class(path,struct_mesh_dict,aero_mesh_dict,ndof,h,time[-1],nabs,sgrids)
    ####plot abs displacement 
    if len(sgrids) > 0:
        plt.figure()
        plt.plot(time,pltclass.h[-pltclass.nabs:,:].T)
        plt.xlabel('Time [s]')
        plt.ylabel('Displacement [m]')
        plt.savefig(plot_path+'/Abs_displacement.png')
        plt.close()

        plt.figure()
        plt.plot(time,pltclass.h[-pltclass.nabs:,:].T-pltclass.h[pltclass.sgrids,:].T)
        plt.xlabel('Time [s]')
        plt.ylabel('Displacement [m]')
        plt.savefig(plot_path+'/Abs_rel_displacement.png')
        plt.close()
    ####plot abs wavelet 
    if len(sgrids) > 0:
        for i in range(pltclass.nabs):
            plot_fft(plot_path,'Abs_{}'.format(i+1),time,pltclass.h[-pltclass.nabs+i,:])
    ####plot ref grid displacement
    ref_gridid = numpy.argwhere(struct_mesh_dict['full_set']['grid_map']==ref_grid)[0][0]
    plt.figure()
    if ndof == 1:
        plt.plot(time,h[ref_gridid,:].T)
    else:
        plt.plot(time,h[ref_gridid*ndof+2,:].T)
    plt.xlabel('Time [s]')
    plt.ylabel('Displacement [m]')
    plt.savefig(plot_path+'/Ref_grid_displacement.png')
    plt.close()
    ####plot ref grid wavelet
    if ndof == 1:
        plot_fft(plot_path,'Ref_grid',time,pltclass.h[ref_gridid,:])
    else:
        plot_fft(plot_path,'Ref_grid',time,pltclass.h[ref_gridid*ndof+2,:])

    for j,t in enumerate(time):
        mp_plot_time([pltclass,j,t,v_time,freq,frf_path,ext_shape])

    fps = len(time)/time[-1]
    os.system('avconv -r {} -i "{}%d.png" {}'.format(fps,path+'/',plot_path+'/')+'time_response_v_{}_freq_{}.mp4'.format(v_time, freq))
    os.system('avconv -r {} -i "{}%d.png" {}'.format(30,path+'/',plot_path+'/')+'slow_time_response_v_{}_freq_{}.mp4'.format(v_time, freq))
    os.system('rm -r {}'.format(path))
    
    return

def plot_laststep(struct_mesh_dict,aero_mesh_dict,plot_path,semi_chord,phi,time,v_time, resp, freq, ndof, ncores, ref_grid, nabs=0, sgrids = []):
    h = numpy.zeros(phi.shape[0])
    j = len(time) - 1
    for mode in range(phi.shape[1]):
        h += resp[j][mode]*phi[:,mode]
    if ndof == 1:
        h /= h[ref_grid-1]
    else:
        h /= h[2+ndof*(ref_grid-1)]
    pltclass = Plot_class(plot_path,struct_mesh_dict,aero_mesh_dict,ndof,h,time[-1],nabs,sgrids)
    
    nclamped = 0
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    for i,component in enumerate(pltclass.struct_mesh_dict.keys()):
        if component != 'full_set':
            nx = pltclass.struct_mesh_dict[component]['x'].shape[0]
            ny = pltclass.struct_mesh_dict[component]['x'].shape[1]
            structind1 = pltclass.struct_mesh_dict[pltclass.struct_mesh_dict.keys()[i]]['global_index']-nclamped
            try:
                nclamped += pltclass.struct_mesh_dict[pltclass.struct_mesh_dict.keys()[i]]['nclamped']
            except:
                nclamped += 0
            structind2 = pltclass.struct_mesh_dict[pltclass.struct_mesh_dict.keys()[i+1]]['global_index']-nclamped
            if pltclass.ndof == 1:
                dispz = pltclass.h[structind1:structind2]
                dispx = numpy.zeros_like(dispz)
                dispy = numpy.zeros_like(dispz)
            elif pltclass.ndof >= 3:
                dispx = pltclass.h[structind1*pltclass.ndof:structind2*pltclass.ndof:pltclass.ndof]
                dispy = pltclass.h[1+structind1*pltclass.ndof:structind2*pltclass.ndof:pltclass.ndof]
                dispz = pltclass.h[2+structind1*pltclass.ndof:structind2*pltclass.ndof:pltclass.ndof]
            maxlen = numpy.max([numpy.max(pltclass.struct_mesh_dict['full_set']['x'])-numpy.min(pltclass.struct_mesh_dict['full_set']['x']),numpy.max(pltclass.struct_mesh_dict['full_set']['y'])-numpy.min(pltclass.struct_mesh_dict['full_set']['y']),numpy.max(pltclass.struct_mesh_dict['full_set']['z'])-numpy.min(pltclass.struct_mesh_dict['full_set']['z'])])
            ax.plot_wireframe(pltclass.struct_mesh_dict[component]['x']+dispx.reshape(nx,ny),pltclass.struct_mesh_dict[component]['y']+dispy.reshape(nx,ny),pltclass.struct_mesh_dict[component]['z']+dispz.reshape(nx,ny))
            ax.set_xlim([numpy.min(pltclass.struct_mesh_dict['full_set']['x']),numpy.min(pltclass.struct_mesh_dict['full_set']['x'])+maxlen])
            ax.set_ylim([numpy.min(pltclass.struct_mesh_dict['full_set']['y']),numpy.min(pltclass.struct_mesh_dict['full_set']['y'])+maxlen])
            ax.set_zlim([-2,2])
            ax.set_title('v={} m/s,f={} Hz'.format(v_time,freq))
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
    if pltclass.nabs != 0:
        ax.plot(pltclass.struct_mesh_dict['full_set']['x'][pltclass.sgrids-1],pltclass.struct_mesh_dict['full_set']['y'][pltclass.sgrids-1],0.1+pltclass.h[-pltclass.nabs:],'ro')
   
    ax.view_init(0, 0) 
    plt.savefig(pltclass.path+'/Response_shape_v{}_f{}.png'.format(v_time,freq))
    plt.close()
    return

def plot_energy(plot_path,time_model,nabs,miabs,phinew,time,nmodesabs,Khhnew,Mhhnew,resp,ref_grid,ndof,vtime,freq_t):
    engpath = plot_path+'/energy'
    if not os.path.isdir(engpath):
        os.mkdir(engpath)
    if time_model == 'original' or miabs == 0:
        K_eng = numpy.zeros((phinew.shape[0],len(time)))
        P_eng = numpy.zeros((phinew.shape[0],len(time))) 
        G_eng = numpy.zeros((phinew.shape[0],len(time))) 
        for j,t in enumerate(time):
            for mode in range(phinew.shape[1]):
                P_eng[:,j] += Khhnew[mode,mode]*(resp[j][mode].real)**2/2*phinew[:,mode]**2
                K_eng[:,j] += Mhhnew[mode,mode]*(resp[j][mode+nmodesabs].real)**2/2*phinew[:,mode]**2
                G_eng[:,j] += 9.81*Mhhnew[mode,mode]*resp[j][mode].real*phinew[:,mode]
        T_eng = K_eng+P_eng
        plt.figure()
        plt.plot(time,K_eng[(ref_grid-1)*ndof:ref_grid*ndof,:].T,'-')
        plt.plot(time,P_eng[(ref_grid-1)*ndof:ref_grid*ndof,:].T,'-')
        plt.plot(time,T_eng[(ref_grid-1)*ndof:ref_grid*ndof,:].T,linewidth=2)
        plt.title('Energy at node {}, v={}m/s, f={}Hz'.format(ref_grid,vtime,freq_t))
        plt.grid()
        plt.legend(['Kinetic Energy','Potential Energy','Total Energy'],loc='best')
        plt.xlabel('Time [s]')
        plt.ylabel('Energy [J]')
        plt.savefig(engpath+'/energy_node{}_V{}_f{}.png'.format(ref_grid,vtime,freq_t))
        plt.close()
        plt.figure()
        plt.plot(time,numpy.sum(K_eng,axis=0),'-')
        plt.plot(time,numpy.sum(P_eng,axis=0),'-')
        plt.plot(time,numpy.sum(T_eng,axis=0),linewidth=2)
        plt.title('Energy at the plate, v={}m/s, f={}Hz'.format(vtime,freq_t))
        plt.grid()
        plt.legend(['Kinetic Energy','Potential Energy','Total Energy'],loc='best')
        plt.xlabel('Time [s]')
        plt.ylabel('Energy [J]')
        plt.savefig(engpath+'/energy_plate_V{}_f{}.png'.format(vtime,freq_t))
        plt.close()
        plt.figure()
        plt.plot(time,numpy.sum(K_eng,axis=0),'-')
        plt.plot(time,numpy.sum(P_eng,axis=0),'-')
        plt.plot(time,numpy.sum(G_eng,axis=0),'-')
        plt.plot(time,numpy.sum(T_eng+G_eng,axis=0),linewidth=2)
        plt.title('Energy at the plate, v={}m/s, f={}Hz'.format(vtime,freq_t))
        plt.grid()
        plt.legend(['Kinetic Energy','Potential Elastic Energy','Potential Gravitational Energy','Total Energy'],loc='best')
        plt.xlabel('Time [s]')
        plt.ylabel('Energy [J]')
        plt.savefig(engpath+'/energy_plate_gravity_V{}_f{}.png'.format(vtime,freq_t))
        plt.close()
    elif time_model == 'absorbers':
        K_eng = numpy.zeros((phinew.shape[0],len(time)))
        P_eng = numpy.zeros((phinew.shape[0],len(time))) 
        G_eng = numpy.zeros((phinew.shape[0],len(time))) 
        for j,t in enumerate(time):
            for mode in range(phinew.shape[1]):
                P_eng[:,j] += Khhnew[mode,mode]*(resp[j][mode].real)**2/2*phinew[:,mode]**2
                K_eng[:,j] += Mhhnew[mode,mode]*(resp[j][mode+nmodesabs].real)**2/2*phinew[:,mode]**2
                G_eng[:,j] += 9.81*Mhhnew[mode,mode]*resp[j][mode].real*phinew[:,mode]
        T_eng = K_eng+P_eng
        plt.figure()
        plt.plot(time,K_eng[(ref_grid-1)*ndof:ref_grid*ndof,:].T,'-')
        plt.plot(time,P_eng[(ref_grid-1)*ndof:ref_grid*ndof,:].T,'-')
        plt.plot(time,T_eng[(ref_grid-1)*ndof:ref_grid*ndof,:].T,linewidth=2)
        plt.title('Energy at node {}, v={}m/s, f={}Hz'.format(ref_grid,vtime,freq_t))
        plt.grid()
        plt.legend(['Kinetic Energy','Potential Energy','Total Energy'],loc='best')
        plt.xlabel('Time [s]')
        plt.ylabel('Energy [J]')
        plt.savefig(engpath+'/energy_node{}_V{}_f{}.png'.format(ref_grid,vtime,freq_t))
        plt.close()
        plt.figure()
        plt.plot(time,numpy.sum(K_eng[:-nabs,:],axis=0),'-')
        plt.plot(time,numpy.sum(P_eng[:-nabs,:],axis=0),'-')
        plt.plot(time,numpy.sum(T_eng[:-nabs,:],axis=0),linewidth=2)
        plt.title('Energy at the plate, v={}m/s, f={}Hz'.format(vtime,freq_t))
        plt.grid()
        plt.legend(['Kinetic Energy','Potential Energy','Total Energy'],loc='best')
        plt.xlabel('Time [s]')
        plt.ylabel('Energy [J]')
        plt.savefig(engpath+'/energy_plate_V{}_f{}.png'.format(vtime,freq_t))
        plt.close()
        plt.figure()
        plt.plot(time,numpy.sum(K_eng[-nabs:,:],axis=0),'-')
        plt.plot(time,numpy.sum(P_eng[-nabs:,:],axis=0),'-')
        plt.plot(time,numpy.sum(T_eng[-nabs:,:],axis=0),linewidth=2)
        plt.title('Energy at the absorbers, v={}m/s, f={}Hz'.format(vtime,freq_t))
        plt.grid()
        plt.legend(['Kinetic Energy','Potential Energy','Total Energy'],loc='best')
        plt.xlabel('Time [s]')
        plt.ylabel('Energy [J]')
        plt.savefig(engpath+'/energy_abs_V{}_f{}.png'.format(vtime,freq_t))
        plt.close()
        plt.figure()
        plt.plot(time,numpy.sum(T_eng[:-nabs,:],axis=0),linewidth=2)
        plt.plot(time,numpy.sum(T_eng[-nabs:,:],axis=0),linewidth=2)
        plt.title('Total energy comparison, v={}m/s, f={}Hz'.format(vtime,freq_t))
        plt.grid()
        plt.legend(['Plate','Absorbers'],loc='best')
        plt.xlabel('Time [s]')
        plt.ylabel('Energy [J]')
        plt.savefig(engpath+'/tot_energy_comp_V{}_f{}.png'.format(vtime,freq_t))
        plt.close()
        plt.figure()
        plt.plot(time,numpy.sum(K_eng[:-nabs,:],axis=0),linewidth=2)
        plt.plot(time,numpy.sum(K_eng[-nabs:,:],axis=0),linewidth=2)
        plt.title('Kinetic energy comparison, v={}m/s, f={}Hz'.format(vtime,freq_t))
        plt.grid()
        plt.legend(['Plate','Absorbers'],loc='best')
        plt.xlabel('Time [s]')
        plt.ylabel('Energy [J]')
        plt.savefig(engpath+'/kin_energy_comp_V{}_f{}.png'.format(vtime,freq_t))
        plt.close()
        plt.figure()
        plt.plot(time,numpy.sum(P_eng[:-nabs,:],axis=0),linewidth=2)
        plt.plot(time,numpy.sum(P_eng[-nabs:,:],axis=0),linewidth=2)
        plt.title('Potential energy comparison, v={}m/s, f={}Hz'.format(vtime,freq_t))
        plt.grid()
        plt.legend(['Plate','Absorbers'],loc='best')
        plt.xlabel('Time [s]')
        plt.ylabel('Energy [J]')
        plt.savefig(engpath+'/pot_energy_comp_V{}_f{}.png'.format(vtime,freq_t))
        plt.close() 
        plt.figure()
        plt.plot(time,numpy.sum(K_eng,axis=0),'-')
        plt.plot(time,numpy.sum(P_eng,axis=0),'-')
        plt.plot(time,numpy.sum(G_eng,axis=0),'-')
        plt.plot(time,numpy.sum(T_eng+G_eng,axis=0),linewidth=2)
        plt.title('Energy at the plate, v={}m/s, f={}Hz'.format(vtime,freq_t))
        plt.grid()
        plt.legend(['Kinetic Energy','Potential Elastic Energy','Potential Gravitational Energy','Total Energy'],loc='best')
        plt.xlabel('Time [s]')
        plt.ylabel('Energy [J]')
        plt.savefig(engpath+'/energy_plate_gravity_V{}_f{}.png'.format(vtime,freq_t))
        plt.close()
    return K_eng,P_eng,G_eng

def mp_plot_group(plot_list):
    pltclass = plot_list[0]
    j = plot_list[1]
    t = plot_list[2]
    v_time = plot_list[3]
    freq = plot_list[4]
    frf_path = plot_list[5] + '/frf_results.mat'
    ext_shape = plot_list[6]
    nclamped = 0
    try:
        frf_dict = scipy.io.loadmat(frf_path)
        fig = plt.figure(figsize=(20,15))
        ax0 = fig.add_subplot(121)
       #pdb.set_trace()
        if v_time < frf_dict['V'][0][0]:
            if pltclass.nabs == 0:
                frf_funcr = scipy.interpolate.interp1d(frf_dict['freq'][0,:],frf_dict['frf_orig'][0,:].real)
                frf_funci = scipy.interpolate.interp1d(frf_dict['freq'][0,:],frf_dict['frf_orig'][0,:].imag)
            else:
                frf_funcr = scipy.interpolate.interp1d(frf_dict['freq'][0,:],frf_dict['frf_abs'][0,:].real)
                frf_funci = scipy.interpolate.interp1d(frf_dict['freq'][0,:],frf_dict['frf_abs'][0,:].imag)
            ax0.plot(frf_dict['freq'][0,:],numpy.log10(numpy.absolute(frf_funcr(frf_dict['freq'][0,:])+1j*frf_funci(frf_dict['freq'][0,:])+1)),linewidth=2)
            ax0.plot(freq,numpy.log10(numpy.absolute(frf_funcr(freq)+1j*frf_funci(freq)+1)),'ro', markersize=8)
        else:
            if pltclass.nabs == 0:
                frf_funcr = scipy.interpolate.interp2d(frf_dict['freq'][0,:],frf_dict['V'][0],frf_dict['frf_aero_orig'].real)
                frf_funci = scipy.interpolate.interp2d(frf_dict['freq'][0,:],frf_dict['V'][0],frf_dict['frf_aero_orig'].imag)
            else:                                                                             
                frf_funcr = scipy.interpolate.interp2d(frf_dict['freq'][0,:],frf_dict['V'][0],frf_dict['frf_aero_abs'].real)
                frf_funci = scipy.interpolate.interp2d(frf_dict['freq'][0,:],frf_dict['V'][0],frf_dict['frf_aero_abs'].imag)
            ax0.plot(frf_dict['freq'][0,:],numpy.log10(numpy.absolute(frf_funcr(frf_dict['freq'][0,:],v_time)+1j*frf_funci(frf_dict['freq'][0,:],v_time)+1)),linewidth=2)
            ax0.plot(freq,numpy.log10(numpy.absolute(frf_funcr(freq,v_time)+1j*frf_funci(freq,v_time)+1)),'ro', markersize=8)
        ax0.grid()
        ax0.set_xlabel('frequency [Hz]')
        ax0.set_ylabel('Magnitude [log]')
        ax0.set_title('FRF')
        ax = fig.add_subplot(122)
    except:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    for i,component in enumerate(pltclass.h.keys()):
        if component in pltclass.struct_mesh_dict.keys():
           #ax.quiver(pltclass.struct_mesh_dict[component]['x'],pltclass.struct_mesh_dict[component]['y'],pltclass.h[component][:,:,0,0,j],pltclass.h[component][:,:,0,1,j],color='b')
            if j != 0:
                xx = ax.contour(pltclass.struct_mesh_dict[component]['x'],pltclass.struct_mesh_dict[component]['y'],pltclass.h[component][:,:,j],colors='b',levels=[0])
                ax.clabel(xx, fontsize=9, inline=1)
            ax.set_title('V= {} m/s, f= {} Hz, t= {}s'.format(v_time,freq,t))
            ax.set_xlabel('x')
            ax.set_ylabel('y')
        else:
           #ax.quiver(pltclass.sgrids[component]['x'],pltclass.sgrids[component]['y'],pltclass.h[component][:,:,0,j],pltclass.h[component][:,:,1,j],color='r')
            if j != 0:
                xx = ax.contour(pltclass.sgrids[component]['x'],pltclass.sgrids[component]['y'],pltclass.h[component][:,:,j],colors='r',levels=[0])
                ax.clabel(xx, fontsize=9, inline=1)
    plt.savefig(pltclass.path+'/{}.png'.format(j))
    plt.close()

    return

def group_v_video(struct_mesh_dict,aero_mesh_dict,plot_path,semi_chord,phi,time,v_time, group_v, freq, ndof, ncores, frf_path, ext_shape, nabs=0, abs_dict = []):
    path = plot_path+'/tempquiver'
    if not os.path.isdir(path):
        os.mkdir(path)
    pltclass = Plot_class(path,struct_mesh_dict,aero_mesh_dict,ndof,group_v,time[-1],nabs,abs_dict)
    for j,t in enumerate(time):
        mp_plot_group([pltclass,j,t,v_time,freq,frf_path,ext_shape])

    fps = len(time)/time[-1]
    os.system('avconv -r {} -i "{}%d.png" {}'.format(fps,path+'/',plot_path+'/')+'group_v_response_v_{}_freq_{}.mp4'.format(v_time, freq))
    os.system('avconv -r {} -i "{}%d.png" {}'.format(10,path+'/',plot_path+'/')+'slow_group_v_response_v_{}_freq_{}.mp4'.format(v_time, freq))
    os.system('rm -r {}'.format(path))
    
    return

def plot_fft(plot_path,name,time,h):
    yi = numpy.fft.fftfreq(len(h),time[1])
    indmax = numpy.argmax(yi)
   #Xi = numpy.zeros((len(time),indmax+1))
   #Yi = numpy.zeros((len(time),indmax+1))
   #for i in range(len(time)):
   #    if i <= indmax:
   #        Xi[:,i] = numpy.array(time)
   #Zi = numpy.zeros((len(time),indmax+1))
   #iold = 0
   #count = 0
   #for i in range(1,len(time)):
   #    if numpy.sign(h[i]*h[i-1]) == -1:
   #        count += 1
   #    if count == 2:
   #        count = 0
   #        for j in range(i-iold):
   #            if j >= numpy.argmax(numpy.fft.fftfreq(i-iold,time[1])):
   #                break
   #            Zi[iold:i,j] = numpy.absolute(numpy.fft.fft(h[iold:i])[j])
   #            Yi[iold:i,j] = numpy.fft.fftfreq(i-iold,time[1])[j]
   #        plt.figure()
   #        plt.plot(Yi[iold,0:i-iold]/2/numpy.pi,Zi[iold,0:i-iold])
   #        plt.xlabel('freq[Hz]')
   #        plt.ylabel('magnitude')
   #        plt.savefig(plot_path+'/{}_fft_t={}-{}s.png'.format(name,time[iold],time[i]))
   #        plt.close()
   #        iold = i
   #if iold != len(time):
   #    for j in range(i-iold):
   #        if j >= numpy.argmax(numpy.fft.fftfreq(len(h[iold:]),time[1])):
   #            break
   #        Zi[iold:,j] = numpy.absolute(numpy.fft.fft(h[iold:])[j])
   #        Yi[iold:,j] = numpy.fft.fftfreq(len(h[iold:]),time[1])[j]
   #Yimax = numpy.max(Yi)
   #Yimin = numpy.min(Yi)
   #X = numpy.zeros((len(time),len(time)))
   #Y = numpy.zeros((len(time),len(time)))
   #Z = numpy.zeros((len(time),len(time)))
   #for i in range(len(time)):
   #    Y[i,:] = numpy.linspace(0,Yimax,len(time))
   #    X[:,i] = Xi[:,0]
   #    for j in range(1,indmax+1):
   #        if Yi[i,j] == 0:
   #            if j > 1:
   #                f = scipy.interpolate.interp1d(Yi[i,0:j],Zi[i,0:j])
   #            break
   #    for j in range(len(time)):
   #        try:
   #            Z[i,j] = f(Y[i,j])
   #        except:
   #            Z[i,j] = 0
    plt.figure()
    plt.semilogy(yi[0:indmax+1],numpy.absolute(numpy.fft.fft(h)[0:indmax+1]))
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('log(FFT Magnitude)')
    plt.savefig(plot_path+'/{}_fft_total.eps'.format(name))
    plt.close()
   #plt.figure()
   #plt.imshow(Z, vmin=Z.min(), vmax=Z.max(), origin='lower',extent=[X.min(), X.max(), Y.min()/2/numpy.pi, Y.max()/2/numpy.pi])
   #plt.xlabel('time[s]')
   #plt.ylabel('freq[Hz]')
   #plt.savefig(plot_path+'/Wavelet_{}.png'.format(name))
   #plt.close()
    return
