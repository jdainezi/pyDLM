#imports
import argparse, numpy, pdb 
import scipy.linalg, scipy.sparse, scipy.sparse.linalg, scipy.io, sys, os
import scipy.interpolate
import matplotlib.pyplot as plt
import multiprocessing as mp
from collections import OrderedDict
from mpl_toolkits.mplot3d import Axes3D
project_dir = '/home/jh/Documents/scripts/pyDLM'
sys.path.append(project_dir+'/utils')
from graphics_tools import *
from modal import *
from kernel import *
from Rogers import *
from spline import *
from standard_atmosphere import *
from continuous_models import *
from analysis import *
from dynamics import *
from calculus import *
from non_linear import *
#from numba import vectorize

#functions
def apply_profile(project_dir,profile,nchord,chord,profile2=None,sec_span=None,sec_location=None,theta=None):
    #calculates the incremental in z due to the profile, for all chordwise elements.
    if profile=='flat_plate':
        delta_z = numpy.zeros(nchord)
        delta_x = numpy.linspace(0,chord,nchord)
        return delta_z, delta_x
    else:
        profile = numpy.loadtxt(os.path.join('..','profiles',profile))
        profile2 = numpy.loadtxt(os.path.join('..','profiles',profile2))
        profile *= chord/(numpy.max(profile[:,0]-numpy.min(profile[:,0])))
        profile2 *= chord/(numpy.max(profile2[:,0]-numpy.min(profile2[:,0])))
        delta_z = (profile[:,1]*(1-sec_location/sec_span) + profile2[:,1]*sec_location/sec_span)*numpy.cos(theta*numpy.pi/180) + (profile[:,0]*(1-sec_location/sec_span) + profile2[:,0]*sec_location/sec_span)*numpy.sin(theta*numpy.pi/180)
        delta_x = (profile[:,0]*(1-sec_location/sec_span) + profile2[:,0]*sec_location/sec_span)*numpy.cos(theta*numpy.pi/180) + (profile[:,1]*(1-sec_location/sec_span) + profile2[:,1]*sec_location/sec_span)*numpy.sin(theta*numpy.pi/180)
        return delta_z, delta_x

def generate_mesh(data_dict,project_dir):
    mesh_info = {}
    for component in data_dict:
        mesh_info[component] = {}
        mesh_info[component]['profile'] = []
        mesh_info[component]['x'] = []
        mesh_info[component]['y'] = []
        mesh_info[component]['z'] = []
        mesh_info[component]['theta'] = []
        mesh_info[component]['chord'] = []
        mesh_info[component]['nchord'] = []
        for section in data_dict[component]['sections']:
            mesh_info[component]['profile'].append(section[0])
            mesh_info[component]['x'].append(float(section[1]))
            mesh_info[component]['y'].append(float(section[2]))
            mesh_info[component]['z'].append(float(section[3]))
            mesh_info[component]['theta'].append(float(section[4]))
            mesh_info[component]['chord'].append(float(section[5]))
            mesh_info[component]['nchord'].append(int(section[6]))
    mesh_dict = {}
    for component in mesh_info:
        orientation = data_dict[component]['orientation']
        nstrips = data_dict[component]['n_strips']
        if orientation == 'xz':
            x = numpy.zeros((mesh_info[component]['nchord'][0]+1,data_dict[component]['n_strips']+1))
            y = numpy.zeros_like(x)
            z = numpy.zeros_like(x)
            info_ind = []
            for span_index in range(x.shape[1]):
                for ind, y_slice in enumerate(mesh_info[component]['y']):
                    if mesh_info[component]['y'][0]+span_index*(mesh_info[component]['y'][-1]-mesh_info[component]['y'][0])/nstrips<y_slice:
                        info_ind.append(ind-1)
                        break
                    elif span_index==nstrips:
                        info_ind.append(ind-1)
            nslice = []
            count = 1
            for i,ind in enumerate(info_ind[1:]):
                if ind != info_ind[i-2]:
                    nslice.append(count*numpy.ones(count))
                count += 1
                nslice.append(count*numpy.ones(count))
            for span_index in range(x.shape[1]):
                theta = (mesh_info[component]['theta'][info_ind[span_index]] + (mesh_info[component]['theta'][info_ind[span_index]+1]-mesh_info[component]['theta'][info_ind[span_index]])*(mesh_info[component]['y'][0]+span_index*(mesh_info[component]['y'][-1]-mesh_info[component]['y'][0])/nstrips-mesh_info[component]['y'][info_ind[span_index]])/(mesh_info[component]['y'][info_ind[span_index]+1]-mesh_info[component]['y'][info_ind[span_index]]))
                chord = (mesh_info[component]['chord'][info_ind[span_index]] + (mesh_info[component]['chord'][info_ind[span_index]+1]-mesh_info[component]['chord'][info_ind[span_index]])*(mesh_info[component]['y'][0]+span_index*(mesh_info[component]['y'][-1]-mesh_info[component]['y'][0])/nstrips-mesh_info[component]['y'][info_ind[span_index]])/(mesh_info[component]['y'][info_ind[span_index]+1]-mesh_info[component]['y'][info_ind[span_index]]))/numpy.cos(theta*numpy.pi/180)
                x[:,span_index] = (mesh_info[component]['x'][info_ind[span_index]] + (mesh_info[component]['x'][info_ind[span_index]+1]-mesh_info[component]['x'][info_ind[span_index]])*(mesh_info[component]['y'][0]+span_index*(mesh_info[component]['y'][-1]-mesh_info[component]['y'][0])/nstrips-mesh_info[component]['y'][info_ind[span_index]])/(mesh_info[component]['y'][info_ind[span_index]+1]-mesh_info[component]['y'][info_ind[span_index]]))*numpy.ones(mesh_info[component]['nchord'][info_ind[span_index]]+1)
                y[:,span_index] = (mesh_info[component]['y'][0]+span_index*(mesh_info[component]['y'][-1]-mesh_info[component]['y'][0])/nstrips)*numpy.ones(mesh_info[component]['nchord'][info_ind[span_index]]+1)
                z[:,span_index] = (mesh_info[component]['z'][info_ind[span_index]] + (mesh_info[component]['z'][info_ind[span_index]+1]-mesh_info[component]['z'][info_ind[span_index]])*(mesh_info[component]['y'][0]+span_index*(mesh_info[component]['y'][-1]-mesh_info[component]['y'][0])/nstrips-mesh_info[component]['y'][info_ind[span_index]])/(mesh_info[component]['y'][info_ind[span_index]+1]-mesh_info[component]['y'][info_ind[span_index]]))*numpy.ones(mesh_info[component]['nchord'][info_ind[span_index]]+1) + numpy.linspace(-chord/4*numpy.sin(theta*numpy.pi/180),3/4*chord*numpy.sin(theta*numpy.pi/180),mesh_info[component]['nchord'][info_ind[span_index]]+1)
                delta_z, delta_x = apply_profile(project_dir,mesh_info[component]['profile'][info_ind[span_index]],mesh_info[component]['nchord'][info_ind[span_index]]+1,chord,profile2=mesh_info[component]['profile'][info_ind[span_index]+1],sec_span=mesh_info[component]['y'][info_ind[span_index]+1]-mesh_info[component]['y'][info_ind[span_index]],sec_location=mesh_info[component]['y'][0]+span_index*(mesh_info[component]['y'][-1]-mesh_info[component]['y'][0])/nstrips-mesh_info[component]['y'][info_ind[span_index]],theta=theta)
                x[:,span_index] += delta_x
                z[:,span_index] += delta_z
        elif orientation == 'yz':
            x = numpy.zeros((data_dict[component]['n_strips']+1,mesh_info[component]['nchord'][0]+1))
            y = numpy.zeros_like(x)
            z = numpy.zeros_like(x)
            info_ind = []
            for span_index in range(x.shape[0]):
                for ind, x_slice in enumerate(mesh_info[component]['x']):
                    if mesh_info[component]['x'][0]+span_index*(mesh_info[component]['x'][-1]-mesh_info[component]['x'][0])/nstrips<x_slice:
                        info_ind.append(ind-1)
                        break
                    elif span_index==nstrips:
                        info_ind.append(ind-1)
            nslice = []
            count = 1
            for i,ind in enumerate(info_ind[1:]):
                if ind != info_ind[i-2]:
                    nslice.append(count*numpy.ones(count))
                count += 1
                nslice.append(count*numpy.ones(count))
            for span_index in range(x.shape[0]):
                theta = (mesh_info[component]['theta'][info_ind[span_index]] + (mesh_info[component]['theta'][info_ind[span_index]+1]-mesh_info[component]['theta'][info_ind[span_index]])*(mesh_info[component]['x'][0]+span_index*(mesh_info[component]['x'][-1]-mesh_info[component]['x'][0])/nstrips-mesh_info[component]['x'][info_ind[span_index]])/(mesh_info[component]['x'][info_ind[span_index]+1]-mesh_info[component]['x'][info_ind[span_index]]))
                chord = (mesh_info[component]['chord'][info_ind[span_index]] + (mesh_info[component]['chord'][info_ind[span_index]+1]-mesh_info[component]['chord'][info_ind[span_index]])*(mesh_info[component]['x'][0]+span_index*(mesh_info[component]['x'][-1]-mesh_info[component]['x'][0])/nstrips-mesh_info[component]['x'][info_ind[span_index]])/(mesh_info[component]['x'][info_ind[span_index]+1]-mesh_info[component]['x'][info_ind[span_index]]))/numpy.cos(theta*numpy.pi/180)
                y[span_index,:] = (mesh_info[component]['y'][info_ind[span_index]] + (mesh_info[component]['y'][info_ind[span_index]+1]-mesh_info[component]['y'][info_ind[span_index]])*(mesh_info[component]['x'][0]+span_index*(mesh_info[component]['x'][-1]-mesh_info[component]['x'][0])/nstrips-mesh_info[component]['x'][info_ind[span_index]])/(mesh_info[component]['x'][info_ind[span_index]+1]-mesh_info[component]['x'][info_ind[span_index]]))*numpy.ones(mesh_info[component]['nchord'][info_ind[span_index]]+1)
                x[span_index,:] = (mesh_info[component]['x'][0]+span_index*(mesh_info[component]['x'][-1]-mesh_info[component]['x'][0])/nstrips)*numpy.ones(mesh_info[component]['nchord'][info_ind[span_index]]+1)
                z[span_index,:] = (mesh_info[component]['z'][info_ind[span_index]] + (mesh_info[component]['z'][info_ind[span_index]+1]-mesh_info[component]['z'][info_ind[span_index]])*(mesh_info[component]['x'][0]+span_index*(mesh_info[component]['x'][-1]-mesh_info[component]['x'][0])/nstrips-mesh_info[component]['x'][info_ind[span_index]])/(mesh_info[component]['x'][info_ind[span_index]+1]-mesh_info[component]['x'][info_ind[span_index]]))*numpy.ones(mesh_info[component]['nchord'][info_ind[span_index]]+1) + numpy.linspace(-chord/4*numpy.sin(theta*numpy.pi/180),3/4*chord*numpy.sin(theta*numpy.pi/180),mesh_info[component]['nchord'][info_ind[span_index]]+1)
                delta_z, delta_y = apply_profile(project_dir,mesh_info[component]['profile'][info_ind[span_index]],mesh_info[component]['nchord'][info_ind[span_index]]+1,chord,profile2=mesh_info[component]['profile'][info_ind[span_index]+1],sec_span=mesh_info[component]['x'][info_ind[span_index]+1]-mesh_info[component]['x'][info_ind[span_index]],sec_location=mesh_info[component]['x'][0]+span_index*(mesh_info[component]['x'][-1]-mesh_info[component]['x'][0])/nstrips-mesh_info[component]['x'][info_ind[span_index]],theta=theta)
                y[span_index,:] += delta_y
                z[span_index,:] += delta_z
        if orientation == 'xy':
            x = numpy.zeros((mesh_info[component]['nchord'][0]+1,data_dict[component]['n_strips']+1))
            y = numpy.zeros_like(x)
            z = numpy.zeros_like(x)
            info_ind = []
            for span_index in range(x.shape[1]):
                for ind, z_slice in enumerate(mesh_info[component]['z']):
                    if mesh_info[component]['z'][0]+span_index*(mesh_info[component]['z'][-1]-mesh_info[component]['z'][0])/nstrips<z_slice:
                        info_ind.append(ind-1)
                        break
                    elif span_index==nstrips:
                        info_ind.append(ind-1)
            nslice = []
            count = 1
            for i,ind in enumerate(info_ind[1:]):
                if ind != info_ind[i-2]:
                    nslice.append(count*numpy.ones(count))
                count += 1
                nslice.append(count*numpy.ones(count))
            for span_index in range(x.shape[1]):
                theta = (mesh_info[component]['theta'][info_ind[span_index]] + (mesh_info[component]['theta'][info_ind[span_index]+1]-mesh_info[component]['theta'][info_ind[span_index]])*(mesh_info[component]['z'][0]+span_index*(mesh_info[component]['z'][-1]-mesh_info[component]['z'][0])/nstrips-mesh_info[component]['z'][info_ind[span_index]])/(mesh_info[component]['z'][info_ind[span_index]+1]-mesh_info[component]['z'][info_ind[span_index]]))
                chord = (mesh_info[component]['chord'][info_ind[span_index]] + (mesh_info[component]['chord'][info_ind[span_index]+1]-mesh_info[component]['chord'][info_ind[span_index]])*(mesh_info[component]['z'][0]+span_index*(mesh_info[component]['z'][-1]-mesh_info[component]['z'][0])/nstrips-mesh_info[component]['z'][info_ind[span_index]])/(mesh_info[component]['z'][info_ind[span_index]+1]-mesh_info[component]['z'][info_ind[span_index]]))/numpy.cos(theta*numpy.pi/180)
                x[:,span_index] = (mesh_info[component]['x'][info_ind[span_index]] + (mesh_info[component]['x'][info_ind[span_index]+1]-mesh_info[component]['x'][info_ind[span_index]])*(mesh_info[component]['z'][0]+span_index*(mesh_info[component]['z'][-1]-mesh_info[component]['z'][0])/nstrips-mesh_info[component]['z'][info_ind[span_index]])/(mesh_info[component]['z'][info_ind[span_index]+1]-mesh_info[component]['z'][info_ind[span_index]]))*numpy.ones(mesh_info[component]['nchord'][info_ind[span_index]]+1)
                z[:,span_index] = (mesh_info[component]['z'][0]+span_index*(mesh_info[component]['z'][-1]-mesh_info[component]['z'][0])/nstrips)*numpy.ones(mesh_info[component]['nchord'][info_ind[span_index]]+1)
                y[:,span_index] = (mesh_info[component]['y'][info_ind[span_index]] + (mesh_info[component]['y'][info_ind[span_index]+1]-mesh_info[component]['y'][info_ind[span_index]])*(mesh_info[component]['z'][0]+span_index*(mesh_info[component]['z'][-1]-mesh_info[component]['z'][0])/nstrips-mesh_info[component]['z'][info_ind[span_index]])/(mesh_info[component]['z'][info_ind[span_index]+1]-mesh_info[component]['z'][info_ind[span_index]]))*numpy.ones(mesh_info[component]['nchord'][info_ind[span_index]]+1) + numpy.linspace(-chord/4*numpy.sin(theta*numpy.pi/180),3/4*chord*numpy.sin(theta*numpy.pi/180),mesh_info[component]['nchord'][info_ind[span_index]]+1)
                delta_y, delta_x = apply_profile(project_dir,mesh_info[component]['profile'][info_ind[span_index]],mesh_info[component]['nchord'][info_ind[span_index]]+1,chord,profile2=mesh_info[component]['profile'][info_ind[span_index]+1],sec_span=mesh_info[component]['z'][info_ind[span_index]+1]-mesh_info[component]['z'][info_ind[span_index]],sec_location=mesh_info[component]['z'][0]+span_index*(mesh_info[component]['z'][-1]-mesh_info[component]['z'][0])/nstrips-mesh_info[component]['z'][info_ind[span_index]],theta=theta)
                x[:,span_index] += delta_x
                y[:,span_index] += delta_y
        if orientation == 'yz':
            area = numpy.zeros((data_dict[component]['n_strips'],mesh_info[component]['nchord'][0]))
        else:
            area = numpy.zeros((mesh_info[component]['nchord'][0],data_dict[component]['n_strips']))
        normals_plus = []
        normals_minus = []
        #collocation points at the centre of each panel (Katz)
        collocation_x = numpy.zeros_like(area)
        collocation_y = numpy.zeros_like(area)
        collocation_z = numpy.zeros_like(area)
        ac_x = numpy.zeros_like(area)
        ac_y = numpy.zeros_like(area)
        ac_z = numpy.zeros_like(area)
        #Doublet line points coordinates (left and right):
        xl = numpy.zeros_like(area)
        xr = numpy.zeros_like(area)
        yl = numpy.zeros_like(area)
        yr = numpy.zeros_like(area)
        for i in range(area.shape[0]):
            for j in range(area.shape[1]):
                #the area is estimated as the average of the cross products in two oposite vertices.
                #border vectors
                v12 = numpy.array([x[i+1,j],y[i+1,j],z[i+1,j]])-numpy.array([x[i,j],y[i,j],z[i,j]])
                v13 = numpy.array([x[i,j+1],y[i,j+1],z[i,j+1]])-numpy.array([x[i,j],y[i,j],z[i,j]])
                v42 = numpy.array([x[i+1,j],y[i+1,j],z[i+1,j]])-numpy.array([x[i+1,j+1],y[i+1,j+1],z[i+1,j+1]])
                v43 = numpy.array([x[i,j+1],y[i,j+1],z[i,j+1]])-numpy.array([x[i+1,j+1],y[i+1,j+1],z[i+1,j+1]])
                normal = (numpy.cross(v12,v13)+numpy.cross(v43,v42))/2.0
                area[i,j] = numpy.linalg.norm(normal)
                normals_plus.append(normal/area[i,j])
                normals_minus.append(normal*numpy.array([1,-1,1])/area[i,j])
                if x[i+1,j]>x[i,j]:
                    collocation_x[i,j] = (x[i,j]+3*x[i+1,j]+x[i,j+1]+3*x[i+1,j+1])/8.0
                    collocation_y[i,j] = (y[i,j]+3*y[i+1,j]+y[i,j+1]+3*y[i+1,j+1])/8.0
                    collocation_z[i,j] = (z[i,j]+3*z[i+1,j]+z[i,j+1]+3*z[i+1,j+1])/8.0
                    ac_x[i,j] = (3*x[i,j]+x[i+1,j]+3*x[i,j+1]+x[i+1,j+1])/8.0
                    ac_y[i,j] = (3*y[i,j]+y[i+1,j]+3*y[i,j+1]+y[i+1,j+1])/8.0
                    ac_z[i,j] = (3*z[i,j]+z[i+1,j]+3*z[i,j+1]+z[i+1,j+1])/8.0
                    xl[i,j] = (3*x[i,j]+x[i+1,j])/4.0
                    xr[i,j] = (3*x[i,j+1]+x[i+1,j+1])/4.0
                    yl[i,j] = (3*y[i,j]+y[i+1,j])/4.0
                    yr[i,j] = (3*y[i,j+1]+y[i+1,j+1])/4.0
                else:
                    collocation_x[i,j] = (3*x[i,j]+x[i+1,j]+3*x[i,j+1]+x[i+1,j+1])/8.0
                    collocation_y[i,j] = (3*y[i,j]+y[i+1,j]+3*y[i,j+1]+y[i+1,j+1])/8.0
                    collocation_z[i,j] = (3*z[i,j]+z[i+1,j]+3*z[i,j+1]+z[i+1,j+1])/8.0
                    ac_x[i,j] = (x[i,j]+3*x[i+1,j]+x[i,j+1]+3*x[i+1,j+1])/8.0
                    ac_y[i,j] = (y[i,j]+3*y[i+1,j]+y[i,j+1]+3*y[i+1,j+1])/8.0
                    ac_z[i,j] = (z[i,j]+3*z[i+1,j]+z[i,j+1]+3*z[i+1,j+1])/8.0
                    xl[i,j] = (x[i,j]+3*x[i+1,j])/4.0
                    xr[i,j] = (x[i,j+1]+3*x[i+1,j+1])/4.0
                    yl[i,j] = (y[i,j]+3*y[i+1,j])/4.0
                    yr[i,j] = (y[i,j+1]+3*y[i+1,j+1])/4.0
        if data_dict[component]['sym'] == '1':
            mesh_dict[component+'_right'] = {}
            mesh_dict[component+'_right']['x'] = x
            mesh_dict[component+'_right']['y'] = y
            mesh_dict[component+'_right']['z'] = z
            mesh_dict[component+'_right']['collocation_x'] = collocation_x
            mesh_dict[component+'_right']['collocation_y'] = collocation_y
            mesh_dict[component+'_right']['collocation_z'] = collocation_z
            mesh_dict[component+'_right']['ac_x'] = ac_x
            mesh_dict[component+'_right']['ac_y'] = ac_y
            mesh_dict[component+'_right']['ac_z'] = ac_z
            mesh_dict[component+'_right']['normals'] = normals_plus
            mesh_dict[component+'_right']['area'] = area
            mesh_dict[component+'_right']['n_elem'] = len(x.flatten())
            mesh_dict[component+'_right']['xl'] = xl
            mesh_dict[component+'_right']['xr'] = xr
            mesh_dict[component+'_right']['yl'] = yl 
            mesh_dict[component+'_right']['yr'] = yr
            mesh_dict[component+'_left'] = {}
            mesh_dict[component+'_left']['x'] = x
            mesh_dict[component+'_left']['y'] = -y
            mesh_dict[component+'_left']['z'] = z
            mesh_dict[component+'_left']['collocation_x'] = collocation_x
            mesh_dict[component+'_left']['collocation_y'] = -collocation_y
            mesh_dict[component+'_left']['collocation_z'] = collocation_z
            mesh_dict[component+'_left']['ac_x'] = ac_x
            mesh_dict[component+'_left']['ac_y'] = -ac_y
            mesh_dict[component+'_left']['ac_z'] = ac_z
            mesh_dict[component+'_left']['normals'] = normals_minus
            mesh_dict[component+'_left']['area'] = area
            mesh_dict[component+'_left']['n_elem'] = len(x.flatten())
            #The left components have a mirrored order of elements:
            mesh_dict[component+'_left']['xl'] = xr
            mesh_dict[component+'_left']['xr'] = xl
            mesh_dict[component+'_left']['yl'] = -yr 
            mesh_dict[component+'_left']['yr'] = -yl
        else:
            mesh_dict[component] = {}
            mesh_dict[component]['x'] = x
            mesh_dict[component]['y'] = y
            mesh_dict[component]['z'] = z
            mesh_dict[component]['collocation_x'] = collocation_x
            mesh_dict[component]['collocation_y'] = collocation_y
            mesh_dict[component]['collocation_z'] = collocation_z
            mesh_dict[component]['ac_x'] = ac_x
            mesh_dict[component]['ac_y'] = ac_y
            mesh_dict[component]['ac_z'] = ac_z
            mesh_dict[component]['normals'] = normals_plus
            mesh_dict[component]['area'] = area
            mesh_dict[component]['n_elem'] = len(x.flatten())
            mesh_dict[component]['xl'] = xl
            mesh_dict[component]['xr'] = xr
            mesh_dict[component]['yl'] = yl 
            mesh_dict[component]['yr'] = yr
    #create the mesh for the full set.
    cpnt_ind = 0
    fs_x = [] #full_set x
    fs_y = []
    fs_z = []
    fs_ac_x = [] 
    fs_ac_y = []
    fs_ac_z = []
    fs_xr = []
    fs_yr = []
    fs_xl = []
    fs_yl = []
    fs_collocation_x = [] 
    fs_collocation_y = []
    fs_collocation_z = []
    fs_normals = []
    fs_area = []
    for component in mesh_dict:
        mesh_dict[component]['global_index'] = cpnt_ind
        cpnt_ind += len(mesh_dict[component]['area'].flatten())
        fs_x.extend(mesh_dict[component]['x'].flatten())
        fs_y.extend(mesh_dict[component]['y'].flatten())
        fs_z.extend(mesh_dict[component]['z'].flatten())
        fs_ac_x.extend(mesh_dict[component]['ac_x'].flatten())
        fs_ac_y.extend(mesh_dict[component]['ac_y'].flatten())
        fs_ac_z.extend(mesh_dict[component]['ac_z'].flatten())
        fs_xr.extend(mesh_dict[component]['xr'].flatten())
        fs_xl.extend(mesh_dict[component]['xl'].flatten())
        fs_yr.extend(mesh_dict[component]['yr'].flatten())
        fs_yl.extend(mesh_dict[component]['yl'].flatten())
        fs_collocation_x.extend(mesh_dict[component]['collocation_x'].flatten())
        fs_collocation_y.extend(mesh_dict[component]['collocation_y'].flatten())
        fs_collocation_z.extend(mesh_dict[component]['collocation_z'].flatten())
        fs_area.extend(mesh_dict[component]['area'].flatten())
        for i in range(mesh_dict[component]['x'].shape[0]):
            for j in range(mesh_dict[component]['x'].shape[1]):
                fs_normals.append(mesh_dict[component]['normals'])
    mesh_dict['full_set'] = {}
    mesh_dict['full_set']['global_index'] = len(fs_xr)
    mesh_dict['full_set']['x'] = fs_x
    mesh_dict['full_set']['y'] = fs_y
    mesh_dict['full_set']['z'] = fs_z
    mesh_dict['full_set']['collocation_x'] = fs_collocation_x
    mesh_dict['full_set']['collocation_y'] = fs_collocation_y
    mesh_dict['full_set']['collocation_z'] = fs_collocation_z
    mesh_dict['full_set']['ac_x'] = fs_ac_x
    mesh_dict['full_set']['ac_y'] = fs_ac_y
    mesh_dict['full_set']['ac_z'] = fs_ac_z
    mesh_dict['full_set']['xr'] = fs_xr
    mesh_dict['full_set']['xl'] = fs_xl
    mesh_dict['full_set']['yr'] = fs_yr
    mesh_dict['full_set']['yl'] = fs_yl
    mesh_dict['full_set']['normals'] = fs_normals
    mesh_dict['full_set']['area'] = fs_area
    return mesh_dict

def read_aero_dat(dat):
    data_dict = {}
    with open(dat, 'r') as infile:
        for line in infile:
            data = line.split()
            if 'COMPONENT' in data:
                component = data[1]
                data_dict[data[1]] = {}
                data_dict[data[1]]['orientation'] = data[2]
                data_dict[data[1]]['n_strips'] = int(data[3])
                data_dict[data[1]]['sym'] = int(data[4])
                data_dict[data[1]]['sections'] = []
            elif 'SECTION' in data:
                data_dict[component]['sections'].append(data[1:])               
    return data_dict

def read_struct_dat(dat):
    data_dict = {}
    nj = 0 #number of joints
    data_dict['joints'] = {}
    data_dict['joints']['grids'] = [1e100]
    with open(dat, 'r') as infile:
        for line in infile:
            data = line.split()
            if data == []:
                data = ['0']
            if ':' in data[0] and '#' not in data[0]:
                if data[0] == 'grids:':
                    grids = []
                    for group in data[1:]:
                        grids.extend(numpy.arange(int(group.split('-')[0]),int(group.split('-')[1])+1))
                    data_dict[component][data[0].replace(':','')] = numpy.array(grids)
                else:
                    data_dict[component][data[0].replace(':','')] = data[1:]
            if 'COMPONENT' in data:
                component = data[1]
                data_dict[data[1]] = {}
                try:
                    data_dict[data[1]]['sym'] = data[2]
                except:
                    continue
            elif 'JOINT' in data:
                nj += 1
                data_dict['joints'][nj] = {}
                data_dict['joints'][nj]['components'] = [data[1],data[2]]
                data_dict['joints'][nj]['grids1'] = data[3].split(',')
                data_dict['joints'][nj]['grids2'] = data[4].split(',')
                data_dict['joints'][nj]['stiff'] = data[5::2]
                data_dict['joints'][nj]['dof'] = data[6::2]
    return data_dict

def read_param_dat(dat):
    data_dict = {}
    with open(dat, 'r') as infile:
        for line in infile:
            data = line.split()
            try:
                if '#' not in data[0]:
                    data_dict[data[0].replace(':','')] = data[1:]
            except:
                continue
    return data_dict

def load_fem(FEM_path,FEM_unit,ndof,struct_dict,damp):
    if damp[0] == 'proportional':
        damping = [float(damp[1]),float(damp[2])]
    else:
        damping = [0,0]
    if ndof ==1:
        if FEM_unit == 'm':
            K_full = numpy.loadtxt(FEM_path+'/K.txt') 
            M_full = numpy.loadtxt(FEM_path+'/M.txt')
        elif FEM_unit == 'mm':
            K_full = numpy.loadtxt(FEM_path+'/K.txt')*1000 #modelo estava em N/mm
            M_full = numpy.loadtxt(FEM_path+'/M.txt')*1000
        M = numpy.zeros((K_full.shape[0]/6*ndof,K_full.shape[1]/6*ndof))
        K = numpy.zeros((K_full.shape[0]/6*ndof,K_full.shape[1]/6*ndof))
        for i in range(K_full.shape[0]/6):
            for j in range(K_full.shape[1]/6):
                M[i,j] = M_full[i*6+2,j*6+2]
                K[i,j] = K_full[i*6+2,j*6+2]
        C = numpy.zeros_like(M)
    elif ndof ==3:
        if FEM_unit == 'm':
            K_full = numpy.loadtxt(FEM_path+'/K.txt') 
            M_full = numpy.loadtxt(FEM_path+'/M.txt')
        elif FEM_unit == 'mm':
            K_full = numpy.loadtxt(FEM_path+'/K.txt')*1000 #modelo estava em N/mm
            M_full = numpy.loadtxt(FEM_path+'/M.txt')*1000
        M = numpy.zeros((K_full.shape[0]/6*ndof,K_full.shape[1]/6*ndof))
        K = numpy.zeros((K_full.shape[0]/6*ndof,K_full.shape[1]/6*ndof))
        for i in range(K_full.shape[0]/6):
            for j in range(K_full.shape[1]/6):
                M[i:i+3,j:j+3] = M_full[i*6:i*6+3,j*6:i*6+3]
                K[i:i+3,j:j+3] = K_full[i*6:i*6+3,j*6:i*6+3]
        C = numpy.zeros_like(M)
    elif ndof ==6:
        if FEM_unit == 'm':
            K = numpy.loadtxt(FEM_path+'/K.txt') 
            M = numpy.loadtxt(FEM_path+'/M.txt')
        elif FEM_unit == 'mm':
            K = numpy.loadtxt(FEM_path+'/K.txt')*1000 #modelo estava em N/mm
            M = numpy.loadtxt(FEM_path+'/M.txt')*1000
        C = numpy.zeros_like(M)
    else:
        raise ValueError('Nastran model works for ndof = 1, 3 or 6 only.')
   #val, vec = scipy.linalg.eig(K,M)
   #Mhh = numpy.dot(numpy.dot(vec.T,M),vec)
   #Khh = numpy.dot(numpy.dot(vec.T,K),vec)
   #vecx = vec[0::6,:]
   #vecy = vec[1::6,:]
   #vecz = vec[2::6,:]
   #vecrx = vec[3::6,:]
   #vecry = vec[4::6,:]
   #vecrz = vec[5::6,:]
   #newvec = numpy.zeros_like(vec)
   #newvec[0::6,:] = -vecx
   #newvec[1::6,:] = vecz
   #newvec[2::6,:] = vecy
   #newvec[3::6,:] = -vecrx
   #newvec[4::6,:] = vecrz
   #newvec[5::6,:] = vecry
   #vec = newvec
   #M = numpy.dot(numpy.dot(numpy.linalg.pinv(vec).T,Mhh),numpy.linalg.pinv(vec)) 
   #K = numpy.dot(numpy.dot(numpy.linalg.pinv(vec).T,Khh),numpy.linalg.pinv(vec)) 
    
    C = damping[0]*M + damping[1]*K

    if FEM_unit == 'mm':
        xs = numpy.loadtxt(FEM_path+'/x.txt')/1000
        ys = numpy.loadtxt(FEM_path+'/y.txt')/1000
        zs = numpy.loadtxt(FEM_path+'/z.txt')/1000
    elif FEM_unit == 'm':
        xs = numpy.loadtxt(FEM_path+'/x.txt')
        ys = numpy.loadtxt(FEM_path+'/y.txt')
        zs = numpy.loadtxt(FEM_path+'/z.txt')
    grid_map = numpy.loadtxt(FEM_path+'/grid_map.txt')

    struct_mesh_dict = {}
    global_id = 0
    for component in struct_dict.keys():
        if component != 'joints':
            struct_mesh_dict[component] = {}
            grids = struct_dict[component]['grids']-1 #nastran counts from 1 and python from 0
            struct_mesh_dict[component]['global_index'] = global_id
            struct_mesh_dict[component]['spline'] = struct_dict[component]['spline'][0]
            nx = int(struct_dict[component]['nx'][0])
            ny = int(struct_dict[component]['ny'][0])
            struct_mesh_dict[component]['grid_map'] = grid_map[global_id:global_id+nx*ny].reshape(nx,ny)
            struct_mesh_dict[component]['x'] = xs[global_id:global_id+nx*ny].reshape(nx,ny)
            struct_mesh_dict[component]['y'] = ys[global_id:global_id+nx*ny].reshape(nx,ny)
            struct_mesh_dict[component]['z'] = zs[global_id:global_id+nx*ny].reshape(nx,ny)
            global_id += nx*ny
    struct_mesh_dict['full_set'] = {}
    struct_mesh_dict['full_set']['global_index'] = len(xs)
    struct_mesh_dict['full_set']['grid_map'] = grid_map
    struct_mesh_dict['full_set']['x'] = xs
    struct_mesh_dict['full_set']['y'] = ys
    struct_mesh_dict['full_set']['z'] = zs
    return M,C,K,struct_mesh_dict

def create_fem(struct_dict,ndof):
    struct_mesh_dict = {}
    M = numpy.array([])
    K = numpy.array([])
    C = numpy.array([])
    global_id = 0
    grid_map = []
    xs = []
    ys = []
    zs = []
    for component in struct_dict.keys():
        if component != 'joints':
            rho_m = float(struct_dict[component]['rho_m'][0])
            E = float(struct_dict[component]['E'][0])
            nu = float(struct_dict[component]['nu'][0])
           #damping = [float(struct_dict[component]['damping'][0]),float(struct_dict[component]['damping'][1])]
            damping = [0,0]
            shape_type = struct_dict[component]['type'][0]
            x_0 = float(struct_dict[component]['x_0'][0])
            y_0 = float(struct_dict[component]['y_0'][0])
            z_0 = float(struct_dict[component]['z_0'][0])
            sweep = float(struct_dict[component]['sweep_angle'][0])*numpy.pi/180
            x_lenght = float(struct_dict[component]['x_lenght'][0])
            y_lenght = float(struct_dict[component]['y_lenght'][0])
            z_lenght = float(struct_dict[component]['z_lenght'][0])
            nx = int(struct_dict[component]['nx'][0])
            ny = int(struct_dict[component]['ny'][0])
            sym = struct_dict[component]['sym'][0]
            clamped = struct_dict[component]['clamped'][0]
            spline = struct_dict[component]['spline'][0]
            section = struct_dict[component]['section'][0]
            
            if shape_type == 'plate':
                Lx = x_lenght
                Ly = y_lenght
                Lz = z_lenght
                rho_A = rho_m*Lz #mass per area
                if clamped == '1':
                    phi, omega_n = love_kirchhoff_cf(Ly,Lx,Lz,E,rho_A,nu,ny,nx,modesx=ny,modesy=nx) #x and y inverted because here the plate is clamped in y=0
                    ind_sort=numpy.argsort(omega_n)
                    omega_n = omega_n[ind_sort]
                    phi = phi[:,ind_sort]
                elif clamped == '0':
                    phi, omega_n = love_kirchhoff_ff(Ly,Lx,Lz,E,rho_A,nu,ny,nx,modesx=ny+1,modesy=nx+1) #x and y inverted because here the plate is clamped in y=0
                   #ind_sort=numpy.argsort(omega_n)
                   #omega_n = omega_n[ind_sort][4:nx*ny+4]
                   #phi = phi[:,ind_sort][:,4:nx*ny+4]
                    
                for mode in range(ny*nx):
                    phi[:,mode] = phi[:,mode].reshape(ny,nx).T.flatten() #another fix due to clamping in y=0
            elif shape_type == 'beam':
                if section == 'rectangular':
                    EI = E*x_lenght*z_lenght**3/12
                    mi = rho_m*x_lenght*z_lenght #mass per lenght
                    y = numpy.linspace(y_lenght*1.0/ny,y_lenght,ny)
                    L = y_lenght
                    Ly = L
                    Lx = 0
                if clamped == '1':
                    phi, omega_n = euler_modes_cf(L,EI,mi,y,nmodes=ny)
                elif clamped == '0':
                    phi, omega_n = euler_modes_ff(L,EI,mi,y,nmodes=ny)
            #getting the mass and stiffness matrix
            Mc,Kc,Mhhc,Khhc = struct_from_modes(phi, omega_n, rho_m, x_lenght, y_lenght, z_lenght, nx, ny)
            Cc = damping[0]*Mc + damping[1]*Kc
            Chhc = numpy.dot(numpy.dot(phi.T,Cc),phi)
            if M.shape[0] == 0:
                M = Mc
                K = Kc
                C = Cc
            else:
                M = numpy.vstack([numpy.hstack([M,numpy.zeros((M.shape[0],Mc.shape[1]))]),numpy.hstack([numpy.zeros((Mc.shape[0],M.shape[1])),Mc])])
                K = numpy.vstack([numpy.hstack([K,numpy.zeros((K.shape[0],Kc.shape[1]))]),numpy.hstack([numpy.zeros((Kc.shape[0],M.shape[1])),Kc])])
                C = numpy.vstack([numpy.hstack([C,numpy.zeros((C.shape[0],Cc.shape[1]))]),numpy.hstack([numpy.zeros((Cc.shape[0],C.shape[1])),Cc])])
            #in local sweep coordinates:
            x_prime = (numpy.ones((ny,nx))*numpy.linspace(x_0,x_0+Lx,nx)).T
            y_prime = numpy.ones((nx,ny))*numpy.linspace(y_0,y_0+Ly,ny)
            #in global coordinates:
            xc = x_prime*numpy.cos(sweep) + y_prime*numpy.sin(sweep)
            yc = -x_prime*numpy.sin(sweep) + y_prime*numpy.cos(sweep)
            zc = numpy.zeros_like(xc) + z_0
            xs.extend(xc.flatten())
            ys.extend(yc.flatten())
            zs.extend(zc.flatten())
            grid_mapc = numpy.arange(global_id+1,global_id+1+nx*ny).reshape(nx,ny)
            grid_map.extend(grid_mapc.flatten())
            if sym == '0':
                struct_mesh_dict[component] = {}
                struct_mesh_dict[component]['global_index'] = global_id
                struct_mesh_dict[component]['grid_map'] = grid_mapc
                struct_mesh_dict[component]['spline'] = spline
                struct_mesh_dict[component]['x'] = xc
                struct_mesh_dict[component]['dx'] = x_lenght/nx
                struct_mesh_dict[component]['dy'] = y_lenght/ny
                struct_mesh_dict[component]['y'] = yc
                struct_mesh_dict[component]['z'] = zc
            elif sym == '1':
                M = numpy.vstack([numpy.hstack([M,numpy.zeros((M.shape[0],Mc.shape[1]))]),numpy.hstack([numpy.zeros((Mc.shape[0],M.shape[1])),Mc])])
                K = numpy.vstack([numpy.hstack([K,numpy.zeros((K.shape[0],Kc.shape[1]))]),numpy.hstack([numpy.zeros((Kc.shape[0],K.shape[1])),Kc])])
                C = numpy.vstack([numpy.hstack([C,numpy.zeros((C.shape[0],Cc.shape[1]))]),numpy.hstack([numpy.zeros((Cc.shape[0],C.shape[1])),Cc])])
                struct_mesh_dict[component+'_right'] = {}
                struct_mesh_dict[component+'_right']['global_index'] = global_id
                struct_mesh_dict[component+'_right']['grid_map'] = grid_mapc
                struct_mesh_dict[component+'_right']['spline'] = spline
                struct_mesh_dict[component+'_right']['x'] = xc
                struct_mesh_dict[component+'_right']['dx'] = x_lenght/nx
                struct_mesh_dict[component+'_right']['dy'] = y_lenght/ny
                struct_mesh_dict[component+'_right']['y'] = yc
                struct_mesh_dict[component+'_right']['z'] = zc
                global_id += nx*ny
                xs.extend(xc.flatten())
                ys.extend(-yc.flatten())
                zs.extend(zc.flatten())
                grid_mapc = numpy.arange(global_id+1,global_id+1+nx*ny).reshape(nx,ny)
                grid_map.extend(grid_mapc.flatten())
                struct_mesh_dict[component+'_left'] = {}
                struct_mesh_dict[component+'_left']['global_index'] = global_id
                struct_mesh_dict[component+'_left']['grid_map'] = grid_mapc
                struct_mesh_dict[component+'_left']['spline'] = spline
                struct_mesh_dict[component+'_left']['x'] = xc
                struct_mesh_dict[component+'_left']['dx'] = x_lenght/nx
                struct_mesh_dict[component+'_left']['dy'] = y_lenght/ny
                struct_mesh_dict[component+'_left']['y'] = -yc
                struct_mesh_dict[component+'_left']['z'] = zc
            global_id += nx*ny
    for joint in struct_dict['joints'].keys():
        if joint != 'grids':
            comp1 = struct_dict['joints'][joint]['components'][0]
            comp2 = struct_dict['joints'][joint]['components'][1]
            for i,dof in enumerate(struct_dict['joints'][joint]['dof']):
                for j in range(len(struct_dict['joints'][joint]['grids1'])):
                    local_id1 = struct_dict['joints'][joint]['grids1'][j]
                    local_id2 = struct_dict['joints'][joint]['grids2'][j]
                    id1 = (struct_mesh_dict[comp1]['global_index']+int(local_id1)-1)*ndof + int(dof)
                    id2 = (struct_mesh_dict[comp2]['global_index']+int(local_id2)-1)*ndof + int(dof)
                    K[id1,id1] += float(struct_dict['joints'][joint]['stiff'][i])
                    K[id1,id2] -= float(struct_dict['joints'][joint]['stiff'][i])
                    K[id2,id1] -= float(struct_dict['joints'][joint]['stiff'][i])
                    K[id2,id2] += float(struct_dict['joints'][joint]['stiff'][i])
        
    struct_mesh_dict['full_set'] = {}
    struct_mesh_dict['full_set']['global_index'] = len(xs)
    struct_mesh_dict['full_set']['grid_map'] = grid_map
    struct_mesh_dict['full_set']['x'] = xs
    struct_mesh_dict['full_set']['y'] = ys
    struct_mesh_dict['full_set']['z'] = zs
    return M,C,K,struct_mesh_dict

def struct_from_modes(phi, omega_n, rho_m, x_lenght, y_lenght, z_lenght, nx, ny):
    omega_n = numpy.diag(omega_n)
    mtot = rho_m*x_lenght*y_lenght*z_lenght
    m = numpy.ones((nx,ny))
    for i in range(nx):
        for j in range(ny):
            if i == 0 or i == nx-1:
                m[i,j] /= 2
            if j == 0 or j == ny-1:
                m[i,j] /= 2
                
   #pdb.set_trace()
   #m = 1
   #M = numpy.diag(numpy.ones(nx*ny)*m)
    M = numpy.diag(m.flatten())*mtot/numpy.sum(m) 
    Mhh = numpy.dot(numpy.dot(phi.T,M),phi)
    Khh = omega_n**2*numpy.diag(Mhh)
    K = numpy.dot(numpy.dot(numpy.linalg.pinv(phi.T),Khh),numpy.linalg.pinv(phi))

    return M,K,Mhh,Khh

def clamp_nodes(struct_mesh_dict,M,C,K,ndof,clamped_nodes):
    for component in struct_mesh_dict.keys():
        if component != 'full_set':
            inds2removex = []
            inds2removey = []
            indx = 0
            for i in range(struct_mesh_dict[component]['x'].shape[0]):
                indy = 0
                for j in range(struct_mesh_dict[component]['x'].shape[1]):
                    if struct_mesh_dict[component]['grid_map'][indx,indy] in clamped_nodes:
                        inds2removex.append(indx)
                        inds2removey.append(indy)
                    indy += 1
                indx += 1
            try:
                if numpy.linalg.norm(numpy.array(inds2removex)-inds2removex[-1]) == 0:
                    struct_mesh_dict[component]['x'] = numpy.delete(struct_mesh_dict[component]['x'],inds2removex[0],0)
                    struct_mesh_dict[component]['y'] = numpy.delete(struct_mesh_dict[component]['y'],inds2removex[0],0)
                    struct_mesh_dict[component]['z'] = numpy.delete(struct_mesh_dict[component]['z'],inds2removex[0],0)
                    struct_mesh_dict[component]['grid_map'] = numpy.delete(struct_mesh_dict[component]['grid_map'],inds2removex[0],0)
                    struct_mesh_dict[component]['nclamped'] = len(inds2removex)
                    
                elif numpy.linalg.norm(numpy.array(inds2removey)-inds2removey[-1]) == 0:
                    struct_mesh_dict[component]['x'] = numpy.delete(struct_mesh_dict[component]['x'],inds2removey[0],1)
                    struct_mesh_dict[component]['y'] = numpy.delete(struct_mesh_dict[component]['y'],inds2removey[0],1)
                    struct_mesh_dict[component]['z'] = numpy.delete(struct_mesh_dict[component]['z'],inds2removey[0],1)
                    struct_mesh_dict[component]['grid_map'] = numpy.delete(struct_mesh_dict[component]['grid_map'],inds2removey[0],1)
                    struct_mesh_dict[component]['nclamped'] = len(inds2removey)
            except:
                continue
        else:
            inds2remove = []
            ind = 0
            for i in range(len(struct_mesh_dict[component]['x'])):
                if struct_mesh_dict[component]['grid_map'][ind] in clamped_nodes:
                    inds2remove.append(ind)
                ind += 1
            struct_mesh_dict[component]['x'] = numpy.delete(struct_mesh_dict[component]['x'],inds2remove,0)
            struct_mesh_dict[component]['y'] = numpy.delete(struct_mesh_dict[component]['y'],inds2remove,0)
            struct_mesh_dict[component]['z'] = numpy.delete(struct_mesh_dict[component]['z'],inds2remove,0)
            struct_mesh_dict[component]['grid_map'] = numpy.delete(struct_mesh_dict[component]['grid_map'],inds2remove,0)
    
    ind2remove = []
    for grid in clamped_nodes:
        ind = []
        for i in range(ndof):
            ind.append(ndof*(grid-1)+i) # -1 because grid starts from 1 and counting from 0
        ind2remove.extend(ind)
    K = numpy.delete(K,ind2remove,0) 
    K = numpy.delete(K,ind2remove,1) 
    C = numpy.delete(C,ind2remove,0) 
    C = numpy.delete(C,ind2remove,1) 
    M = numpy.delete(M,ind2remove,0) 
    M = numpy.delete(M,ind2remove,1)
    ngrids = K.shape[0]/ndof

    return struct_mesh_dict, M, C, K, ngrids

def spline_matrix(struct_dict,struct_mesh_dict,aero_mesh_dict,ndof,ngrids,npan,aoa):
    #spline matrix
    ncomp = len(struct_mesh_dict.keys())-1
    GG_SS = numpy.zeros((ngrids*ndof,npan))
    D1_GG = numpy.zeros((npan,ngrids*ndof))
    D2_GG = numpy.zeros((npan,ngrids*ndof))
        
        
    nclamped = 0
    for i,component in enumerate(struct_mesh_dict.keys()):
        if component != 'full_set':
            if component not in aero_mesh_dict.keys():
                nclamped += len(struct_mesh_dict[component]['x'].flatten())
            if component in aero_mesh_dict.keys():
                xstr = struct_mesh_dict[component]['x']
                ystr = struct_mesh_dict[component]['y']
                xc_c = aero_mesh_dict[component]['ac_x']
                yc_c = aero_mesh_dict[component]['ac_y']
                xpc_c = aero_mesh_dict[component]['collocation_x']
                ypc_c = aero_mesh_dict[component]['collocation_y']
                area_c = aero_mesh_dict[component]['area'].flatten()
                if struct_mesh_dict[component]['spline'] == 'IPS':
                    GG_comp,SS_comp,D1_comp,D2_comp = IPS(xstr,ystr,xc_c,yc_c,xpc_c,ypc_c,area_c,ndof,opt=1)
                elif struct_mesh_dict[component]['spline'] == 'RBA':
                    GG_comp,SS_comp,D1_comp,D2_comp = RBA(xstr,ystr,xc_c,yc_c,xpc_c,ypc_c,area_c)
                elif struct_mesh_dict[component]['spline'] == 'beam':
                    try:
                        EI = float(struct_dict[component]['E'][0])*float(struct_dict[component]['x_lenght'][0])*float(struct_dict[component]['z_lenght'][0])**3/12
                    except:
                        EI = float(struct_dict[component.split('_')[0]]['E'][0])*float(struct_dict[component.split('_')[0]]['x_lenght'][0])*float(struct_dict[component.split('_')[0]]['z_lenght'][0])**3/12
                    GG_comp,SS_comp,D1_comp,D2_comp = beam(xstr,ystr,xc_c,yc_c,xpc_c,ypc_c,area_c,EI,ndof,aoa)
                for j,aerocomp in enumerate(aero_mesh_dict.keys()):
                    if aerocomp == component:
                        break
                aeroind1 = aero_mesh_dict[aero_mesh_dict.keys()[j]]['global_index']
                aeroind2 = aero_mesh_dict[aero_mesh_dict.keys()[j+1]]['global_index']
                structind1 = struct_mesh_dict[struct_mesh_dict.keys()[i]]['global_index']-nclamped
                try:
                    nclamped += struct_mesh_dict[struct_mesh_dict.keys()[i]]['nclamped']
                except:
                    nclamped += 0
                structind2 = struct_mesh_dict[struct_mesh_dict.keys()[i+1]]['global_index']-nclamped
                GG_SS[structind1*ndof:structind2*ndof,aeroind1:aeroind2] = numpy.dot(GG_comp.T,SS_comp)
                D1_GG[aeroind1:aeroind2,structind1*ndof:structind2*ndof] = numpy.dot(D1_comp,GG_comp)
                D2_GG[aeroind1:aeroind2,structind1*ndof:structind2*ndof] = numpy.dot(D2_comp,GG_comp)
    return GG_SS, D1_GG, D2_GG

def mp_GAF(GAF_list):
    xl = GAF_list[0]
    yl = GAF_list[1]
    xc = GAF_list[2]
    yc = GAF_list[3]
    xr = GAF_list[4]
    yr = GAF_list[5]
    xpc = GAF_list[6]
    ypc = GAF_list[7]
    mach = GAF_list[8]
    k = GAF_list[9]
    semi_chord = GAF_list[10]
    phi = GAF_list[11]
    GG_SS = GAF_list[12]
    D1_GG = GAF_list[13]
    D2_GG = GAF_list[14]
    aero_sym = GAF_list[15]
    ndof = phi.shape[0]/len(xl)
    print 'Calculating AIC for reduced frequency {}'.format(k)
    AIC = kernel(xl,yl,xc,yc,xr,yr,xpc,ypc,mach,k,semi_chord,aero_sym)
    if k == 0:
        GAF = numpy.dot(numpy.dot(numpy.dot(numpy.dot(phi.T,GG_SS),numpy.linalg.pinv(AIC)),(D1_GG)),phi) # Physical domain.
    else:
        GAF = numpy.dot(numpy.dot(numpy.dot(numpy.dot(phi.T,GG_SS),numpy.linalg.inv(AIC)),(D1_GG + 1j*k*D2_GG/semi_chord)),phi) # Physical domain.
    return GAF

def gpu_GAF(xl,yl,xc,yc,xr,yr,xpc,ypc,mach,ks,semi_chord,phi,GG_SS,D1_GG,D2_GG,aero_sym):
    for i,k in enumerate(ks):
        print 'Calculating AIC for reduced frequency {}'.format(k)
        AIC = kernel(xl,yl,xc,yc,xr,yr,xpc,ypc,mach,k,semi_chord,aero_sym)
        if k == 0:
            GAF = numpy.dot(numpy.dot(numpy.dot(numpy.dot(phi.T,GG_SS),numpy.linalg.pinv(AIC)),(D1_GG)),phi) # Physical domain.
        else:
            GAF = numpy.dot(numpy.dot(numpy.dot(numpy.dot(phi.T,GG_SS),numpy.linalg.inv(AIC)),(D1_GG + 1j*k*D2_GG)),phi) # Physical domain.
    return GAF

def mp_frf(frf_list):
    struct_mesh_dict = frf_list[0] 
    GAF = frf_list[1]
    Mtot = frf_list[2]
    Mhh = frf_list[3]
    Chh = frf_list[4]
    Khh = frf_list[5]
    k = frf_list[6]
    v = frf_list[7]
    rho = frf_list[8]
    semi_chord = frf_list[9]
    M = frf_list[10]
    C = frf_list[11]
    K = frf_list[12]
    omega_n = frf_list[13]
    freqlen = frf_list[14]
    wall_nodes = frf_list[15]
    sgrids = frf_list[16]
    ref_grid = frf_list[17]
    phi = frf_list[18]
    miabs = frf_list[19]
    abs_damping = frf_list[20]
    tmode = frf_list[21]
    ndof = frf_list[22]
    nmodes = frf_list[23]
    damp = frf_list[24]
    ext_shape = frf_list[25]
    x_rot = frf_list[26]
    plot_path  = frf_list[27]
    infabs  = frf_list[28]
    print 'Calculating frf for velocity {}m/s'.format(v)
    frf_results = frf_analysis(struct_mesh_dict,GAF,Mtot,Mhh,Chh,Khh,k,v,rho,semi_chord,M,C,K,omega_n,freqlen,wall_nodes,sgrids,infabs,ref_grid,phi,miabs,abs_damping,tmode,ndof,nmodes,damp,ext_shape,x_rot,plot_path)
    return frf_results

def mp_nl(nl_list):
    struct_mesh_dict = nl_list[0]
    Ri = nl_list[1]
    PP_GAF = nl_list[2]
    Mtot = nl_list[3]
    Mhh = nl_list[4]
    Chh = nl_list[5]
    Khh = nl_list[6]
    k = nl_list[7]
    v = nl_list[8]
    rho = nl_list[9]
    semi_chord = nl_list[10]
    M = nl_list[11]
    C = nl_list[12]
    K = nl_list[13]
    omega_n = nl_list[14]
    amplitude = nl_list[15]
    freqlen = nl_list[16]
    sgrids = nl_list[17]
    ref_grid = nl_list[18]
    phi = nl_list[19]
    miabs = nl_list[20]
    K1 = nl_list[21]
    K3 = nl_list[22]
    ndof = nl_list[23]
    nmodes = nl_list[24]
    damp = nl_list[25]
    ext_shape = nl_list[26]
    x_rot = nl_list[27]
    plot_path = nl_list[28]
    print 'Performing harmonic balance analysis for V={}m/s'.format(v)
    nl_results = non_linear_analysis(struct_mesh_dict,Ri,PP_GAF,Mtot,Mhh,Chh,Khh,k,v,rho,semi_chord,M,C,K,omega_n,amplitude,freqlen,sgrids,ref_grid,phi,miabs,K1,K3,ndof,nmodes,damp,ext_shape,x_rot,plot_path)
    return nl_results

#main
def DLM(aero,struct,param,out,project_dir):
    path = os.path.join(out,'results')
    if not os.path.isdir(path):
        os.mkdir(path)
    aero_dict = read_aero_dat(aero)
    struct_dict = read_struct_dat(struct)
    param_dict = read_param_dat(param)
    aero_mesh_dict = generate_mesh(aero_dict,project_dir)
    plot_path = os.path.join(path,'plots')
    if not os.path.isdir(plot_path):
        os.mkdir(plot_path)
    plot_geom(aero_mesh_dict,plot_path)
   #window_plot(mesh_dict)

    #Other input variables:
    model = param_dict['model'][0]
    ndof = int(param_dict['ndof'][0])
    ncores = numpy.min([mp.cpu_count(),int(param_dict['ncores'][0])])
    nmodes = int(param_dict['n_modes'][0])
    modes = numpy.arange(0,nmodes)
    altitude = float(param_dict['h'][0])
    gaf_interp = param_dict['gaf_interp'][0]
    flutter_v = param_dict['flutter_video'][0]
    time_vd = param_dict['time_video'][0]
    div_video = param_dict['div_video'][0]
    clamped_model = param_dict['clamped_model'][0]
    damp = param_dict['damping']
    try:
        abs_comp = param_dict['abs_comp'][0]
    except:
        abs_comp = 'full_set'
    try:
        aoa = float(param_dict['AOA'][0])
    except:
        aoa = 0.0
    try:
        x_rot = float(param_dict['x_rot'][0])
    except:
        x_rot = 0.0
    semi_chord = float(param_dict['c_bar'][0])/2
    try:
        V = numpy.arange(float(param_dict['V'][0].split(':')[0]),float(param_dict['V'][0].split(':')[2]),float(param_dict['V'][0].split(':')[1])) #range for flutter
    except:
        V = numpy.array([])
    try:
        freqlen = float(param_dict['freqlen'][0]) #resolution for frf
    except:
        freqlen = 0
    try:
        tmode = []
        for gset in param_dict['tmode']:
            if 'Hz' in gset:
                tmode.append(gset)
            else:
                tmode.append(int(gset) - 1) #target mode
    except:
        tmode = 0
    try:
        abs_damping = []
        for gset in param_dict['abs_damping']:
            abs_damping.append(float(gset))
    except:
        abs_damping = []
        for gset in param_dict['tmode']:
            abs_damping.append(0)
        
    try:
        ext_shape = param_dict['ext_shape'][0]
    except:
        ext_shape = 'bending'
    try:
        infabs = param_dict['inf_abs'][0]
    except:
        infabs = '1'
    try:
        ref_grid = int(param_dict['ref_grid'][0]) #reference grid
    except:
        ref_grid = 1
    try:
        miabs = [] #mass ratio
        for gset in param_dict['mi']:
            miabs.append(gset)
    except:
        miabs = ['0']
    try:
        sgrids = []#grids IDs with absorbers
        elemlist = []
        for elem in param_dict['sgrids']:
            if elem == ',':
                sgrids.append(elemlist)
                elemlist = []
            else:
                elemlist.append(int(elem))
        sgrids.append(elemlist)
    except:
        sgrids = numpy.array([])
    try:
        v = numpy.zeros(len(param_dict['v'])) #static ans frf
        for i in range(len(v)):
            v[i] = float(param_dict['v'][i])
    except:
        v = 0
   #k = numpy.zeros(len(param_dict['k']))
   #for i in range(len(k)):
   #    k[i] = float(param_dict['k'][i])
    try:
        FEM_path = param_dict['FEM_path'][0]
        FEM_unit = param_dict['FEM_unit'][0]
    except:
        FEM_path = []
        FEM_unit = []
    try:
        clamped_nodes = numpy.zeros(len(param_dict['clamped_nodes']))
        for i in range(len(clamped_nodes)):
            clamped_nodes[i] = float(param_dict['clamped_nodes'][i])
    except:
        clamped_nodes = []
    try:
        wall_nodes = numpy.zeros(len(param_dict['wall_nodes']))
        for i in range(len(wall_nodes)):
            wall_nodes[i] = float(param_dict['wall_nodes'][i])
    except:
        wall_nodes = []
    try:
        gust = param_dict['gust'][0]
    except:
        gust = '0'
    try:
        gust_shape = param_dict['gust_shape'][0]
    except:
        gust_shape = '0'
    try:
        Lgust = float(param_dict['Lgust'][0])
    except:
        Lgust = 0.0
    try:
        Vgust = float(param_dict['Vgust'][0])
    except:
        Vgust = 0.0
    try:
        x0gust = float(param_dict['x0gust'][0])
    except:
        x0gust = 0.0
    static = param_dict['static'][0]
    flutter = param_dict['flutter'][0]
    frf = param_dict['frf'][0]
    time_solve = param_dict['time_solve'][0]
    rho = density_SI(altitude)
    Sref = float(param_dict['Sref'][0])
    tmax = float(param_dict['tmax'][0])
    dt = float(param_dict['dt'][0])
    amplitude = float(param_dict['amplitude'][0])
    try:
        freq_t = float(param_dict['freq_t'][0])
    except:
        freq_t = param_dict['freq_t'][0]
    vtime = float(param_dict['vtime'][0])
    time_model = param_dict['time_model'][0]
    try:
        nl_abs = param_dict['nl_abs'][0]
    except:
        nl_abs = '0'
    try:
        Br = float(param_dict['Br'][0])
    except:
        Br = 0
    try:
        tk = [] 
        for gset in param_dict['tk']:
            tk.append(float(gset))
    except:
        tk = [0]
    try:
        r1 = [] 
        for gset in param_dict['r1']:
            r1.append(float(gset))
    except:
        r1 = [0]
    try:
        r2 = [] 
        for gset in param_dict['r2']:
            r2.append(float(gset))
    except:
        r2 = [0]
    try:
        a1 = [] 
        for gset in param_dict['a1']:
            a1.append(float(gset))
    except:
        a1 = [0]
    try:
        a2 = [] 
        for gset in param_dict['a2']:
            a2.append(float(gset))
    except:
        a2 = [0]

    T = temperature_SI(altitude)
    sound_speed = sound_speed_SI(T)
   #mach = v/sound_speed #static
    mach = 0 

    inputdict = {'ndof':ndof,'nmodes':nmodes,'damping_data':damp,'aoa':aoa,'x_rotation_axis':x_rot,'semi_chord':semi_chord,'Sref':Sref,'V_range':V,'V_static':v,'frf_target_mode':tmode,'ref_grid':ref_grid,'miabs':miabs,'absorbers_location_grids':sgrids,'wall_nodes':wall_nodes,'rho':rho}
    scipy.io.savemat(path+'/inputs.mat',inputdict)
    
    if model == 'fem':
        #sorting dictionaries by grid index variable
        struct_dict = OrderedDict(sorted(struct_dict.items(), key=lambda kv: kv[1]['grids'][0]))
        M,C,K,struct_mesh_dict = load_fem(FEM_path,FEM_unit,ndof,struct_dict,damp)
        
    elif model == 'modal':
        M,C,K,struct_mesh_dict = create_fem(struct_dict,ndof)

    #Aero mesh data extraction:
    xl = aero_mesh_dict['full_set']['xl']
    yl = aero_mesh_dict['full_set']['yl']
    xc = aero_mesh_dict['full_set']['ac_x']
    yc = aero_mesh_dict['full_set']['ac_y']
    xr = aero_mesh_dict['full_set']['xr']
    yr = aero_mesh_dict['full_set']['yr']
    xpc = aero_mesh_dict['full_set']['collocation_x']
    ypc = aero_mesh_dict['full_set']['collocation_y']
    area = aero_mesh_dict['full_set']['area']
    aero_sym = aero_dict[aero_dict.keys()[0]]['sym']
    npan = len(xl)

    #Calculating total mass
    struct_mesh_dict = OrderedDict(sorted(struct_mesh_dict.items(), key=lambda kv: kv[1]['global_index']))
    Mtot = {}
    for i,component in enumerate(struct_mesh_dict.keys()):
        if component != 'full_set':
            nx = struct_mesh_dict[component]['x'].shape[0]
            ny = struct_mesh_dict[component]['x'].shape[1]
            structind1 = struct_mesh_dict[struct_mesh_dict.keys()[i]]['global_index']
            structind2 = struct_mesh_dict[struct_mesh_dict.keys()[i+1]]['global_index']
            Mtot[component] = numpy.sum(M[structind1*ndof:structind2*ndof:ndof,:][:,structind1*ndof:structind2*ndof:ndof])/ndof
    Mtot = Mtot[abs_comp]

    #clamping the structural nodes
    struct_mesh_dict, M, C, K, ngrids = clamp_nodes(struct_mesh_dict,M,C,K,ndof,clamped_nodes)
    #Converting to modal space
    phi,Mhh,Chh,Khh,omega_n = modal(M,C,K,modes,ndof,clamped_model)
    if damp[0] == 'flat':
        Chh = numpy.diag(2*numpy.sqrt(numpy.diag(Khh)*numpy.diag(Mhh)))*float(damp[1])
        C = numpy.dot(numpy.dot(numpy.linalg.pinv(phi.T),Chh),numpy.linalg.pinv(phi))
    elif damp[0] == 'proportional':
        Chh = Mhh*float(damp[1]) + Khh*float(damp[2])
        C = numpy.dot(numpy.dot(numpy.linalg.pinv(phi.T),Chh),numpy.linalg.pinv(phi))
    print 'Natural frequencies found:'
    for om in omega_n:
        print om, 'rad/s'

    #sorting dictionaries by global index variable
    aero_mesh_dict = OrderedDict(sorted(aero_mesh_dict.items(), key=lambda kv: kv[1]['global_index']))
    struct_mesh_dict = OrderedDict(sorted(struct_mesh_dict.items(), key=lambda kv: kv[1]['global_index']))
    
    #calculating spline matricies:
    print 'Assembling spline matrix...'
    GG_SS, D1_GG, D2_GG = spline_matrix(struct_dict,struct_mesh_dict,aero_mesh_dict,ndof,ngrids,npan,aoa)

    if static == '1':
        #generating static AIC
        err = 1
        machd = mach


        #divergence
        while err>0.01:
            AIC0 = kernel(xl,yl,xc,yc,xr,yr,xpc,ypc,machd,0,semi_chord,aero_sym)
            GAF0 = numpy.dot(numpy.dot(numpy.dot(numpy.dot(phi.T,GG_SS),numpy.linalg.pinv(AIC0)),(D1_GG)),phi) # Physical domain.
            val, vec = scipy.linalg.eig(Khh,GAF0)
            div_sort = numpy.argsort(val)
            div_ind = numpy.argmin(val)
            for index in div_sort:
                if val[index]>0.1 and val[index].real*2 != val[index].real:
                    div_ind = index
                    break
            if val[div_ind].real >= 0 and val[div_ind].real*2 != val[div_ind].real:
                div_V = numpy.sqrt(val[div_ind]/(0.5*rho)).real
            else:
                div_V = 0
            err = numpy.absolute(div_V/sound_speed-machd)/machd
            machd = div_V/sound_speed
            if machd >= 1:
                break
        div_mode = numpy.zeros(phi.shape[0])
        for mode in range(nmodes):
            div_mode += (vec[mode,div_ind]*phi[:,mode]).real
        div_mode /= numpy.max(numpy.absolute(div_mode))
        if div_V == 0:
            print 'Divergence not found'
            div_V = 1e10
            static_dict = {}
            static_dict['divergence'] = 'No divergence'
        else:
            print 'Divergence found at V = {0:.4g}m/s'.format(div_V)
            static_dict = {}
            static_dict['divergence'] = {}
            static_dict['divergence']['div_V'] = div_V
            static_dict['divergence']['div_mode'] = div_mode
            if div_video == '1':
                divergence_video(plot_path,div_V,div_mode,struct_mesh_dict,aero_mesh_dict,ndof)
        if v[0] >= div_V:
            print 'Static calculation will not be performed because V is above divergence.'
        else:
            static_dict = static_calculation(AIC0.real,struct_mesh_dict,aero_mesh_dict,GG_SS,D1_GG,ngrids,npan,ndof,x_rot,aoa,M,K,rho,Sref,v[0],path,plot_path,static_dict)
        scipy.io.savemat(path+'/static_results.mat',static_dict) 

   #pdb.set_trace()
    sgridsid = [] 
    for gset in sgrids:
        sgidlist = []
        for i in range(len(gset)):
            sgidlist.append(numpy.argwhere(struct_mesh_dict['full_set']['grid_map']==gset[i])[0][0])
        sgridsid.append(sgidlist)
    sgrids = sgridsid
    if frf == '1' or flutter == '1' or time_solve == '1':
        ## Obtaining the aerodynamic matrices for the known reduced frequencies.
        
        #reduced frequencies
        try: 
            kmax_f = numpy.max(omega_n.real)*semi_chord/V[0] 
        except:
            kmax_f = 0
        try: 
            kmax_frf = (2*max(omega_n[tmode]))*semi_chord/v[0]
        except:
            kmax_frf = 0
        k = [0,numpy.pi/2/1000]
        q = 2.0
       #while k[-1] < numpy.min([numpy.max([kmax_f,kmax_frf]),4]):
        while k[-1] < numpy.max([kmax_f,kmax_frf]):
            k.append(numpy.min([k[-1]*q,k[-1]+2*semi_chord]))
        k = numpy.array(k)

        beta = numpy.sqrt(1 - mach**2)  # Beta
        try:
            npzgaf = numpy.load(path+'/GAF.npz')
            GAF = npzgaf['GAF']
            print 'Loading available GAF.'
        except:
            GAF_input = []
            pool = mp.Pool(processes=ncores)
            print 'Calculating GAF for {} reduced frequencies.'.format(len(k))
            for ii in range(len(k)):
                GAF_input.append([xl,yl,xc,yc,xr,yr,xpc,ypc,mach,k[ii],semi_chord,phi,GG_SS,D1_GG,D2_GG,aero_sym])
            GAF_list = pool.map(mp_GAF,GAF_input)
            pool.close()
            GAF = numpy.zeros((nmodes,nmodes,len(k)),dtype=complex)
            for ii in range(len(k)):
                GAF[:,:,ii] = GAF_list[ii]
           #GAF = gpu_GAF(xl,yl,xc,yc,xr,yr,xpc,ypc,mach,k,semi_chord,phi,GG_SS,D1_GG,D2_GG,aero_sym)
            numpy.savez(path+'/GAF.npz',GAF=GAF)
        if gaf_interp == '1':
            #Checking interpolation
            if not os.path.isdir(path+'/interpolation_check'):
                os.mkdir(path+'/interpolation_check')
            newks = numpy.linspace(0,numpy.max([kmax_f,kmax_frf]),5000)
            newGAF = numpy.zeros((nmodes,nmodes,5000),dtype=complex)
            for i in range(5000):
               if newks[i] <= k[-1]:
                   newGAF[:,:,i] = interp_GAF(GAF,k,newks[i])
               else:
                   newGAF[:,:,i] = (GAF[:,:,-1]-GAF[:,:,-2])*(newks[i]-k[-1])/(k[-1]-k[-2]) + GAF[:,:,-1]

            for i in range(nmodes):
                for j in range (nmodes):
                    plt.figure()
                    plt.plot(k,GAF[i,j,:].real,'o')
                    plt.plot(newks,newGAF[i,j,:].real)
                    plt.plot(k,GAF[i,j,:].imag,'x')
                    plt.plot(newks,newGAF[i,j,:].imag)
                    plt.legend(['real','real spline','imag','imag spline'])
                    plt.savefig(path+'/interpolation_check'+'/Q{}.png'.format(str(i+1)+str(j+1)))
                    plt.close()
    if nl_abs == '1':
        mu0 = 1.257e-06
        #magnet areas
        A1 = math.pi*r1[0]**2
        A2 = math.pi*r2[0]**2
        xinterp = numpy.linspace(-0.6*a1[0],0.6*a2[0],500)
        poly = numpy.polyfit(xinterp,magnet_force(xinterp.reshape(500,1),mu0,Br,A1,A2,a1[0],a2[0],r1[0],r2[0],tk[0]),3)
        K3 = poly[0][0]
        K1 = poly[2][0]
        try:
            K3 = numpy.loadtxt(path+'/K3.txt')
        except:
            ###Optimizing K3 parameter
            print "Optimizing cubic stiffness. Initial guess K3={}N/m3".format(K3)
            K3 = optimize_K3(struct_mesh_dict,Mtot,Mhh,Chh,Khh,k,rho,semi_chord,M,C,K,omega_n,amplitude,freqlen,sgrids,ref_grid,phi,miabs,K1,K3,ndof,nmodes,damp,ext_shape,x_rot)
            numpy.savetxt(path+'/K3.txt',K3)
    if frf == '1':
        if nl_abs != '1':
            print 'Starting FRF analysis.'
            frf_input = []
            for ii in range(len(v)):
                frf_input.append([struct_mesh_dict,GAF,Mtot,Mhh,Chh,Khh,k,v[ii],rho,semi_chord,M,C,K,omega_n,freqlen,wall_nodes,sgrids,ref_grid,phi,miabs,abs_damping,tmode,ndof,nmodes,damp,ext_shape,x_rot,plot_path,infabs])
            if len(v) == 1:
                frf_list = mp_frf(frf_input[0])
                frf_list = [frf_list]
            else:
                pool = mp.Pool(processes=ncores)
                frf_list = pool.map(mp_frf,frf_input)
                pool.close()
            frf_results = {}
            for key in frf_list[0].keys():
                frf_results[key] = []
            for ii in range(len(v)):
                for key in frf_list[0].keys():
                    frf_results[key].append(frf_list[ii][key])
            for key in frf_list[0].keys():
                frf_results[key] = numpy.array(frf_results[key])
            frf_results['V'] = v
            plot_frf(frf_results,v,freqlen,miabs,plot_path)

            scipy.io.savemat(path+'/frf_results.mat',frf_results)

        ###### harmonic balance
        else:
            print 'Starting non-linear analysis'
           #Rogers aproximation:
           #print 'Performing Rogers approximation...'
            Ri,PP_GAF = rogers(GAF,k)
            nl_input = []
            nl_input.append([struct_mesh_dict,Ri,PP_GAF,Mtot,Mhh,Chh,Khh,k,0,rho,semi_chord,M,C,K,omega_n,amplitude,freqlen,sgrids,ref_grid,phi,miabs,K1,K3,ndof,nmodes,abs_damping,ext_shape,x_rot,plot_path])
            for ii in range(len(v)):
                nl_input.append([struct_mesh_dict,Ri,PP_GAF,Mtot,Mhh,Chh,Khh,k,v[ii],rho,semi_chord,M,C,K,omega_n,amplitude,freqlen,sgrids,ref_grid,phi,miabs,K1,K3,ndof,nmodes,abs_damping,ext_shape,x_rot,plot_path])
            pool = mp.Pool(processes=ncores)
           #mp_nl(nl_input[1])
            nl_results_list = pool.map(mp_nl,nl_input)
            pool.close()
            nl_freq = []
            nl_rms = []
            nl_rms_abs = []
            v_full = []
            v_full.append(0)
            for i in range(len(nl_results_list)):
                nl_freq.append(nl_results_list[i][0])
                nl_rms.append(nl_results_list[i][1])
                nl_rms_abs.append(nl_results_list[i][2])
                plt.figure()
                legend = []
                if all(nl_rms[-1])>0:  
                    plt.semilogy(nl_freq[-1],nl_rms[-1])
                    legend.append('Original plate')
                if all(nl_rms_abs[-1])>0:  
                    plt.semilogy(nl_freq[-1],nl_rms_abs[-1])
                    legend.append('Metamaterial')
                plt.xlabel('Frequency [Hz]')
                plt.ylabel('RMS amplitude [m]')
                if len(legend)>0:
                    plt.legend(legend)
                plt.savefig(plot_path+'/Nonlinear_a_{}_v{}.png'.format(amplitude,v_full[i]))
                try:
                    v_full.append(v[i])
                except:
                    continue
                plt.close()
            if len(v_full)> 1:
                plt.figure()
                X = []
                for i in range(int(freqlen)):
                    X.append(v_full)
                Xi = numpy.array(X).T
                X = []
                Y = []
                for i in range(int(freqlen)):
                    X.append(numpy.linspace(0,v[-1],freqlen))
                X=numpy.array(X).T
                Y=numpy.array(nl_freq).T[:,0:freqlen]
                Z = numpy.zeros((int(freqlen),int(freqlen)))
                Zi = numpy.log10(numpy.absolute(numpy.array(nl_rms)[:,0:freqlen]+amplitude/math.sqrt(2)))          
                for i in range(int(freqlen)):
                    func = scipy.interpolate.interp1d(Xi[:,i],Zi[:,i])
                    Z[:,i] = func(X[:,i])
                plt.imshow(Z.T, vmin=Z.min(), vmax=Z.max(), origin='lower',extent=[X.min(), X.max(), Y.min(), Y.max()])
                plt.ylabel('frequency [Hz]')
                plt.xlabel('Velocity [m/s]')
                plt.savefig(plot_path+'/nl_aero_colormap.png')
                plt.close()
                plt.figure()
                Z = numpy.zeros((int(freqlen),int(freqlen)))
                Zi = numpy.log10(numpy.absolute(numpy.array(nl_rms_abs)[:,0:freqlen]+amplitude/math.sqrt(2)))          
                for i in range(int(freqlen)):
                    func = scipy.interpolate.interp1d(Xi[:,i],Zi[:,i])
                    Z[:,i] = func(X[:,i])
                plt.imshow(Z.T, vmin=Z.min(), vmax=Z.max(), origin='lower',extent=[X.min(), X.max(), Y.min(), Y.max()])
                plt.ylabel('frequency [Hz]')
                plt.xlabel('Velocity [m/s]')
                plt.savefig(plot_path+'/nl_abs_aero_colormap.png')
                plt.close()
            
                
            nl_results = {'freq':nl_freq,'rms':nl_rms,'rms_abs':nl_rms_abs,'v':v_full}
                
            scipy.io.savemat(path+'/nl_results.mat',nl_results)
 
    #creating matrices with absorbers    
    try:
        miabs = [] #mass ratio
        for gset in param_dict['mi']:
            miabs.append(gset)
    except:
        miabs = [0]
    if miabs != [0]:

        nabs = 0
        grid_ids = []
        grid_idst = []
        grid_idsf = []
        nabst = 0
        nabsf = 0
        omegat = []
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
                infabs = '0'
                miabs[i] = float(miabs[i].replace('t',''))
                if type(tmode[i]) != type('Hz'):
                    omegat.append(omega_n[tmode[i]]*2/(1+numpy.sqrt(1+miabs[i])))
                else:
                    omegat.append(2*numpy.pi*float(tmode[i].replace('Hz',''))*2/(1+numpy.sqrt(1+miabs[i])))
            else:
                continue
            mset = numpy.ones(len(gset))*Mtot*miabs[i]/len(gset)
            ms.extend(mset)
            ks.extend((omegat[-1])**2*mset)
            bs.extend(abs_damping[i]*2*numpy.sqrt(mset*((omegat[-1])**2*mset)))

        Knew,Cnew,Mnew = insert_tshock_absorbers(struct_mesh_dict,grid_idst,ks,bs,ms,M,C,K,ndof)
        ks = []
        ms = []
        bs = []
        for i,gset in enumerate(sgrids):
           #pdb.set_trace()
            if type(miabs[i]) == type('t'):
                miabs[i] = float(miabs[i])
                if type(tmode[i]) != type('Hz'):
                    omegat.append(omega_n[tmode[i]]*2/(1+numpy.sqrt(1+miabs[i])))
                else:
                    omegat.append(2*numpy.pi*float(tmode[i].replace('Hz',''))*2/(1+numpy.sqrt(1+miabs[i])))
            else:
                continue
            mset = numpy.ones(len(gset))*Mtot*miabs[i]/len(gset)
            ms.extend(mset)
            ks.extend((omegat[-1])**2*mset)
            bs.extend(abs_damping[i]*2*numpy.sqrt(mset*((omegat[-1])**2*mset)))

        Knew,Cnew,Mnew = insert_shock_absorbers(struct_mesh_dict,grid_idsf,ks,bs,ms,Mnew,Cnew,Knew,ndof)

        if infabs != '1':
           #Khhnew, Chhnew, Mhhnew, phinew, omega_new = insert_modal_absorbers(struct_mesh_dict,grid_ids,ks,bs,ms,Mhh,Chh,Khh,phi,omega_n)
           #nmodesabs = len(omega_new)
           #Mnew = numpy.dot(numpy.dot(numpy.linalg.pinv(phinew.T),Mhhnew),numpy.linalg.pinv(phinew))
    
            A = numpy.vstack([numpy.hstack([numpy.zeros_like(Mnew),numpy.eye(Mnew.shape[0])]),\
                numpy.hstack([numpy.dot(-numpy.linalg.pinv(Mnew),Knew),numpy.dot(-numpy.linalg.pinv(Mnew),Cnew)])])
            if ndof == 6:
                val,vec = numpy.linalg.eig(A)
            else:
                val,vec = numpy.linalg.eig(A)
            ind_sort = numpy.argsort(val.imag)[Mnew.shape[0]:]
            omega_new = []
            nmodesabs = 0
            while val.imag[ind_sort][nmodesabs] < omega_n[-1]*1.1:
                omega_new.append(val.imag[ind_sort][nmodesabs])
                nmodesabs += 1
            omega_new = numpy.array(omega_new)
            phinew = vec[0:Mnew.shape[0],ind_sort][:,0:nmodesabs]
            if ndof != 6:
                for i in range(phinew.shape[1]):
                    phinew[:,i] /= max(numpy.absolute(phinew[:,i]))#*numpy.cos(numpy.angle(phinew[0,i]))
            else:
                for i in range(phinew.shape[1]):
                    phinew[:,i] /= max(numpy.absolute(phinew[2::ndof,i]))#*numpy.cos(numpy.angle(phinew[0,i]))
            ybabs = numpy.zeros((phinew.shape[0],1))
            if ext_shape == 'bending':
                if ndof == 1:
                    ybabs += 1.0
                else:
                    ybabs[2:phi.shape[0]:ndof] = 1.0
                    ybabs[phi.shape[0]:] = 1.0
            elif ext_shape == 'torsion':
                x_abs = struct_mesh_dict['full_set']['x']
                for i,grid in enumerate(grid_ids):
                   #grid = numpy.argwhere(struct_mesh_dict['full_set']['grid_map']==grid)[0][0] +1
                    x_abs = numpy.hstack([x_abs,struct_mesh_dict['full_set']['x'][grid]])
                if ndof <= 3:
                    ybabs += 0.1*(x_rot - x_abs).reshape(phinew.shape[0],1)
                else:
                    ybabs[2:phi.shape[0]:ndof] += 0.1*(x_rot - x_abs).reshape(phinew.shape[0],1)[2:phi.shape[0]:ndof]
                    ybabs[4:phi.shape[0]:ndof] += 0.1
                    ybabs[phi.shape[0]:] += 0.1*(x_rot - x_abs).reshape(phinew.shape[0],1)[phi.shape[0]:]
                    ybabs[phi.shape[0]:] += 0.1
            ybhh = numpy.zeros_like(numpy.dot(numpy.linalg.pinv(phinew),ybabs))
            if ndof == 1:
                ybhh = numpy.dot(numpy.linalg.pinv(phinew),ybabs)
            else:
                for dof in [2]:
                   ybhh  += numpy.dot(numpy.linalg.pinv(phinew[dof::ndof,:]),ybabs[dof::ndof])
           #for i in range(nmodesabs):
           #    if abs(ybhh[i,0])<0.1:
           #        phinew[0:M.shape[0],i] *= 0

            #phinew = phinew.real
            #######old
           #val,vec = numpy.linalg.eig(numpy.dot(numpy.linalg.inv(Mnew),Knew))
           #ind_sort = numpy.argsort(val)
           #omega_new = []
           #nmodesabs = 0
           #while numpy.sqrt(val[ind_sort])[nmodesabs] < omega_n[-1]*1.1:
           #    omega_new.append(numpy.sqrt(val[ind_sort])[nmodesabs])
           #    nmodesabs += 1
           #omega_new = numpy.array(omega_new)
           #phinew = vec[:,ind_sort][:,0:nmodesabs]
           #for i in range(phinew.shape[1]):
           #    phinew[:,i] /= max(numpy.absolute(phinew[:,i]))
            ####damp resonators
           #rel_mov = numpy.zeros(phinew.shape[1]) #average relative distance between ressonators and plate 
           #for i in range(phinew.shape[1]):
           #    if ndof == 1:
           #        rel_mov[i] = numpy.sum(numpy.absolute(phinew[-nabs:,i]-phinew[sgrids,i]))/nabs
           #    else:
           #        rel_mov[i] = numpy.sum(numpy.absolute(phinew[-nabs:,i]-phinew[sgrids*ndof+2,i]))/nabs
    
            #####################
           #pdb.set_trace()
           #for i in range(phinew.shape[1]):
           #    if max(numpy.absolute(phinew[:-nabs,i])) <= 0.1*max(numpy.absolute(phinew[-nabs:,i])): 
           #        phinew[-nabs:,i] = numpy.ones_like(phinew[-nabs:,i])
         ###nmodesabs = 0
         ###count = 0
         ###phinew2 = []
         ###omega_new2 = []
         ###egvec = numpy.zeros_like(phinew[:,0])
         ###egval = 0
         ###for i in range(phinew.shape[1]):
         ###    if max(numpy.absolute(phinew[:-nabs,i])) > 0.1*max(numpy.absolute(phinew[-nabs:,i])):
         ###        phinew2.append(phinew[:,i])
         ###        omega_new2.append(omega_new[i])
         ###        nmodesabs += 1
         ###    elif max(numpy.absolute(phinew[:-nabs,i])) <= 0.1*max(numpy.absolute(phinew[-nabs:,i])): 
         ###        count += 1
         ###        egvec += phinew[:,i]
         ###        egval += omega_new[i]
         ###if count >= 1: 
         ###    nmodesabs += 1
         ###    phinew2.append(egvec/count)
         ###    omega_new2.append(egval/count)
         ###phinew = numpy.array(phinew2).T
         ###omega_new = numpy.array(omega_new2)
           #pdb.set_trace()
           #phinew = phinew[:,[0,1,2,3,126,127,128,129,130,131]]
           #nmodesabs = 10
           #omega_new = omega_new[[0,1,2,3,126,127,128,129,130,131]]
            #####################
           #sgrids = numpy.array(sgrids).reshape(nabs)
           #for mode in range(nmodesabs):
           #    fig = plt.figure()
           #    ax  = fig.add_subplot(111, projection='3d')
           #    nclamped = 0
           #    for i,component in enumerate(struct_mesh_dict.keys()):
           #        if component != 'full_set':
           #            nx = struct_mesh_dict[component]['x'].shape[0] 
           #            ny = struct_mesh_dict[component]['x'].shape[1]
           #            structind1 = struct_mesh_dict[struct_mesh_dict.keys()[i]]['global_index']-nclamped
           #            try:
           #                nclamped += struct_mesh_dict[struct_mesh_dict.keys()[i]]['nclamped']
           #            except:
           #                nclamped += 0
           #            structind2 = struct_mesh_dict[struct_mesh_dict.keys()[i+1]]['global_index']-nclamped
           #            if ndof == 1:
           #                disp = phinew[structind1:structind2,mode].imag
           #            elif ndof >= 3:
           #                disp = phinew[2+structind1*ndof:structind2*ndof:ndof,mode]
           #            if component in aero_mesh_dict.keys():
           #                for j,aerocomp in enumerate(aero_mesh_dict.keys()):
           #                    if aerocomp == component:
           #                        break
           #                npx = aero_mesh_dict[component]['area'].shape[0] 
           #                npy = aero_mesh_dict[component]['area'].shape[1]
           #                aeroind1 = aero_mesh_dict[aero_mesh_dict.keys()[j]]['global_index']
           #                aeroind2 = aero_mesh_dict[aero_mesh_dict.keys()[j+1]]['global_index']
           #            ax.plot_wireframe(struct_mesh_dict[component]['x'],struct_mesh_dict[component]['y'],struct_mesh_dict[component]['z'])
           #            ax.plot_wireframe(struct_mesh_dict[component]['x'],struct_mesh_dict[component]['y'],struct_mesh_dict[component]['z']+0.2*disp.reshape(nx,ny),color='green')
           #            maxlen = numpy.max([numpy.max(struct_mesh_dict['full_set']['x'])-numpy.min(struct_mesh_dict['full_set']['x']),numpy.max(struct_mesh_dict['full_set']['y'])-numpy.min(struct_mesh_dict['full_set']['y']),numpy.max(struct_mesh_dict['full_set']['z'])-numpy.min(struct_mesh_dict['full_set']['z'])])
           #            ax.plot(struct_mesh_dict['full_set']['x'][sgrids],struct_mesh_dict['full_set']['y'][sgrids],0.2*phinew.imag[-nabs:,mode],color='black')
           #    ax.set_xlim([numpy.min(struct_mesh_dict['full_set']['x']),numpy.min(struct_mesh_dict['full_set']['x'])+maxlen])
           #    dx = numpy.max(struct_mesh_dict['full_set']['x']) - numpy.min(struct_mesh_dict['full_set']['x'])
           #    ax.set_xlim([numpy.min(struct_mesh_dict['full_set']['x'])-dx/2,numpy.min(struct_mesh_dict['full_set']['x'])+1.5*dx])
           #    ax.set_ylim([numpy.min(struct_mesh_dict['full_set']['y']),numpy.min(struct_mesh_dict['full_set']['y'])+maxlen])
           #    ax.set_zlim([-maxlen/2,maxlen/2])
           #   #ax.view_init(0,90)
           #   #ax.set_axis_off()
           #    ax.set_title('Freq {0:.2f} Hz'.format(omega_new[mode]/2/numpy.pi))
           #    ax.legend(['undeformed','plate mode shape','abs mode shape'],loc='lower left')
           #    plt.savefig(plot_path+'/mode{}.png'.format(mode+1))
           #    plt.close()
           #pdb.set_trace()    
            Mhhnew = numpy.dot(numpy.dot(phinew.T,Mnew),phinew)
            Khhnew = numpy.dot(numpy.dot(phinew.T,Knew),phinew)
           #if damp[0] == 'flat':
           #    Chhnew = numpy.diag(2*numpy.sqrt(numpy.diag(Khhnew)*numpy.diag(Mhhnew)))*float(damp[1])
           #    Cnew = numpy.dot(numpy.dot(numpy.linalg.pinv(phinew.T),Chhnew),numpy.linalg.pinv(phinew))
           #elif damp[0] == 'proportional':
           #    Chhnew = Mhhnew*float(damp[1]) + Khhnew*float(damp[2])
           #    Cnew = numpy.dot(numpy.dot(numpy.linalg.pinv(phinew.T),Chhnew),numpy.linalg.pinv(phinew))
            Chhnew = numpy.dot(numpy.dot(phinew.T,Cnew),phinew)
    
            GAFabs = numpy.zeros((nmodesabs,nmodesabs,GAF.shape[2]),dtype=complex)
            for i in range(len(k)):
                AICabs = numpy.vstack([numpy.hstack([numpy.dot(numpy.dot(numpy.linalg.pinv(phi).T,GAF[:,:,i]),numpy.linalg.pinv(phi)),numpy.zeros((phi.shape[0],nabs))]),numpy.hstack([numpy.zeros((nabs,phi.shape[0])),numpy.zeros((nabs,nabs))])])
                GAFabs[:,:,i] = numpy.dot(numpy.dot(phinew.T,AICabs),phinew) 
    if flutter == '1':
        print 'Initializing flutter calculations with P-K method.'
        PK_results = {}
        g, k_flutter, pk_modes = flutter_calculation(Mhh,Chh,Khh,GAF,V,k,semi_chord,rho,omega_n,ncores)
        PK_results['g'] = g
        PK_results['k'] = k_flutter
        PK_results['mode_shapes'] = pk_modes
        print 'Ploting the results.'
        plot_flutter(plot_path,V,g,k_flutter,semi_chord,nmodes,omega_n)
        #Identifying flutter speed:
        v_flutter, flutter_mode, k_f = get_v_flutter(g,k_flutter,pk_modes,V,nmodes,k,Mhh,Chh,Khh,GAF,semi_chord,rho)
        PK_results['v_flutter'] = 'No flutter found'
        #Animating flutter
        if v_flutter != V[-1]:
            PK_results['v_flutter'] = v_flutter
            PK_results['flutter_mode'] = flutter_mode
            PK_results['k_flutter'] = k_f
            if flutter_v == '1':
                if not os.path.isfile(plot_path+'/'+'flutter_v_{0:.4g}.mp4'.format(v_flutter)):
                    flutter_video(struct_mesh_dict,aero_mesh_dict,plot_path,semi_chord,phi,omega_n,v_flutter, flutter_mode, k_f, ndof, ncores, nabs=0)
        if all(miabs) != 0 and infabs != '1':
            print 'Initializing flutter calculations with absorbers.'
            g, k_flutter, pk_modes = flutter_calculation(Mhhnew,Chhnew,Khhnew,GAFabs,V,k,semi_chord,rho,omega_new,ncores)
            PK_results['g_abs'] = g
            PK_results['k_abs'] = k_flutter
            PK_results['mode_shapes_abs'] = pk_modes
            print 'Ploting the results.'
            plot_flutter(plot_path,V,g,k_flutter,semi_chord,nmodesabs,omega_new,nabs)
            #Identifying flutter speed:
            v_flutter, flutter_mode, k_f = get_v_flutter(g,k_flutter,pk_modes,V,nmodesabs,k,Mhhnew,Chhnew,Khhnew,GAFabs,semi_chord,rho)
            PK_results['v_flutter_abs'] = 'No flutter found'
            #Animating flutter
            if v_flutter != V[-1]:
                PK_results['v_flutter_abs'] = v_flutter
                PK_results['flutter_mode_abs'] = flutter_mode
                PK_results['k_flutter_abs'] = k_f
                if flutter_v == '1':
                    if not os.path.isfile(plot_path+'/'+'flutter_v_{0:.4g}.mp4'.format(v_flutter)):
                        flutter_video(struct_mesh_dict,aero_mesh_dict,plot_path,semi_chord,phi,omega_new,v_flutter, flutter_mode, k_f, ndof, ncores, nabs, grid_ids)
        if all(miabs) != 0 and infabs == '1':
            print 'Initializing flutter calculations with absorbers.'
            g, k_flutter, pk_modes = flutter_calculation(Mhh,Chh,Khh,GAF,V,k,semi_chord,rho,omega_n,ncores,numpy.array(miabs),abs_damping,omegat)
            PK_results['g_abs'] = g
            PK_results['k_abs'] = k_flutter
            PK_results['mode_shapes_abs'] = pk_modes
            print 'Ploting the results.'
            plot_flutter(plot_path,V,g,k_flutter,semi_chord,nmodes,omega_n,nabs='inf')
            #Identifying flutter speed:
            v_flutter, flutter_mode, k_f = get_v_flutter(g,k_flutter,pk_modes,V,nmodes,k,Mhh,Chh,Khh,GAF,semi_chord,rho)
            PK_results['v_flutter_abs'] = 'No flutter found'
            #Animating flutter
            if v_flutter != V[-1]:
                PK_results['v_flutter_abs'] = v_flutter
                PK_results['flutter_mode_abs'] = flutter_mode
                PK_results['k_flutter_abs'] = k_f
                if flutter_v == '1':
                    if not os.path.isfile(plot_path+'/'+'flutter_v_{0:.4g}.mp4'.format(v_flutter)):
                        flutter_video(struct_mesh_dict,aero_mesh_dict,plot_path,semi_chord,phi,omega_n,v_flutter, flutter_mode, k_f, ndof, ncores, nabs, grid_ids)
        scipy.io.savemat(path+'/flutter_results.mat',PK_results)

    if time_solve == '1':
        engpath = plot_path+'/energy'
        if not os.path.isdir(engpath):
            os.mkdir(engpath)
        if time_model == 'original' or miabs == 0:
            print 'Starting time solver'
           #Rogers aproximation:
            print 'Performing Rogers approximation...'
            Ri,PP_GAF = rogers(GAF,k)
           #non_linear_analysis(struct_mesh_dict,Ri,PP_GAF,Mtot,Mhh,Chh,Khh,k,00,rho,semi_chord,M,C,K,omega_n,amplitude,freqlen,sgrids,ref_grid,phi,miabs,ndof,nmodes,damp,ext_shape,x_rot,plot_path)
            yb = numpy.zeros((phi.shape[0],1))
            if ext_shape == 'bending':
                if ndof == 1:
                    yb += 1.0
                else:
                    yb[2::ndof] = 1.0
            elif ext_shape == 'torsion':
                if ndof <= 3:
                    yb += 1*(x_rot - struct_mesh_dict['full_set']['x']).reshape(phi.shape[0]/ndof,1)
                else:
                    yb[2::ndof] += 1*(x_rot - struct_mesh_dict['full_set']['x']).reshape(phi.shape[0]/ndof,1)
                    yb[4::ndof] += 1
            if gust == '0':
                time, resp = time_solver(plot_path,PP_GAF,Ri,M,phi,Mhh,Khh,Chh,vtime,rho,nmodes,semi_chord,tmax,dt,amplitude,freq_t,yb,ndof)
            elif gust == '1':
                GAFg = []
                for i in range(len(k)):
                    if ndof == 1:
                        GAFg.append(numpy.dot(GAF[:,:,i],numpy.linalg.pinv(phi))*numpy.exp(-1j*k[i]*(struct_mesh_dict['full_set']['x']-x0gust)/semi_chord))
                    else:
                        dir_vec = numpy.zeros((1,ndof*len(struct_mesh_dict['full_set']['x'])))
                        dir_vec[2::ndof] = numpy.exp(-1j*k[i]*(struct_mesh_dict['full_set']['x']-x0gust)/semi_chord)
                        GAFg.append(numpy.dot(GAF,numpy.linalg.pinv(phi))*dir_vec)
                GAFg = numpy.array(GAFg).reshape(nmodes,len(struct_mesh_dict['full_set']['x']),len(k))
                Rg,PPg = rogers(GAFg,k)
                time, resp = time_solver_gust(plot_path,PP_GAF,PPg,Ri,Rg,M,phi,Mhh,Khh,Chh,vtime,rho,nmodes,semi_chord,tmax,dt,amplitude,freq_t,yb,gust_shape,x0gust,Vgust,Lgust,struct_mesh_dict['full_set']['x'],ndof)
            if time_vd == '1':
                time_video(struct_mesh_dict,aero_mesh_dict,plot_path,ref_grid,semi_chord,phi,time,vtime, resp, freq_t, ndof, ncores, path, ext_shape)
            K_eng,P_eng,G_eng = plot_energy(plot_path,time_model,0,miabs,phi,time,nmodes,Khh,Mhh,resp,ref_grid,ndof,vtime,freq_t)
            h = numpy.zeros((phi.shape[0],len(time)),dtype=complex)
            Vel = numpy.zeros((phi.shape[0],len(time)),dtype=complex)
            for j,t in enumerate(time):
                for mode in range(phi.shape[1]):
                    h[:,j] += resp[j][mode]*phi[:,mode]
                    Vel[:,j] += resp[j][nmodes+mode]*phi[:,mode]
        elif time_model == 'absorbers':
            Knl,Bnl,Mnl = insert_shock_absorbers(struct_mesh_dict,grid_ids,numpy.ones(nabs),numpy.zeros(nabs),numpy.zeros(nabs),M*0,C*0,K*0,ndof)
            if nl_abs == '1':
                print 'Starting time solver'
               #Rogers aproximation:
                print 'Performing Rogers approximation...'
                phinew = numpy.hstack([numpy.vstack([phi,numpy.zeros((nabs,nmodes))]),numpy.vstack([numpy.zeros((phi.shape[0],nabs)),numpy.eye(nabs)])])
                nmodesabs = nmodes+nabs
                Mhhnew = numpy.dot(phinew.T,numpy.dot(Mnew,phinew))
                Khhnew = numpy.dot(phinew.T,numpy.dot(Knew,phinew))
                Chhnew = numpy.dot(phinew.T,numpy.dot(Cnew,phinew))
                GAFabs = numpy.zeros((Mhhnew.shape[0],Mhhnew.shape[1],len(k)),dtype=complex)
                for i in range(len(k)):
                    GAFabs[:,:,i] = numpy.vstack([numpy.hstack([GAF[:,:,i],numpy.zeros((GAF.shape[0],nabs))]),numpy.zeros((nabs,nabs+GAF.shape[1]))])

                Ri,PP_GAF = rogers(GAFabs,k)
                ybabs = numpy.zeros((phinew.shape[0],1))
                if ext_shape == 'bending':
                    if ndof == 1:
                        ybabs += 1.0
                    else:
                        ybabs[2:-nabs:ndof] = 1.0
                        ybabs[-nabs:] = 1.0
                elif ext_shape == 'torsion':
                    x_abs = struct_mesh_dict['full_set']['x']
                    for i,grid in enumerate(grid_ids):
                       #grid = numpy.argwhere(struct_mesh_dict['full_set']['grid_map']==grid)[0][0] +1
                        x_abs = numpy.hstack([x_abs,struct_mesh_dict['full_set']['x'][grid]])
                    if ndof <= 3:
                        ybabs += (x_rot - x_abs).reshape(phinew.shape[0],1)
                    else:
                        ybabs[2::ndof] += 1*(x_rot - x_abs).reshape(phinew.shape[0],1)
                        ybabs[4::ndof] += 1
                if gust == '0':
                    time, resp = time_solver_nl(plot_path,PP_GAF,Ri,Mnew,phinew,Mhhnew,Khhnew,Chhnew,vtime,rho,nmodesabs,semi_chord,tmax,dt,amplitude,freq_t,ybabs,ndof,Knl,K3,miabs)
                elif gust == '1':
                    GAFg = []
                    for i in range(len(k)):
                        x_full_abs = numpy.hstack([struct_mesh_dict['full_set']['x'],struct_mesh_dict['full_set']['x'][sgrids]])
                        if ndof == 1:
                            GAFg.append(numpy.dot(GAFabs[:,:,i],numpy.linalg.pinv(phinew))*numpy.exp(-1j*k[i]*(x_full_abs-x0gust)/semi_chord))
                        else:
                            dir_vec = numpy.zeros((1,ndof*len(x_full_abs)))
                            dir_vec[2::ndof] = numpy.exp(-1j*k[i]*(x_full_abs-x0gust)/semi_chord)
                            GAFg.append(numpy.dot(GAFabs[:,:,i],numpy.linalg.pinv(phinew))*dir_vec)
                    GAFg = numpy.array(GAFg).reshape(nmodesabs,len(x_full_abs),len(k))
                    Rg,PPg = rogers(GAFg,k)
                    time, resp = time_solver_gust_nl(plot_path,PP_GAF,PPg,Ri,Rg,Mnew,phinew,Mhhnew,Khhnew,Chhnew,vtime,rho,nmodesabs,semi_chord,tmax,dt,amplitude,freq_t,ybabs,gust_shape,x0gust,Vgust,Lgust,x_full_abs,ndof,Knl,K3,miabs)
                if time_vd == '1':
                    time_video(struct_mesh_dict,aero_mesh_dict,plot_path,ref_grid,semi_chord,phinew,time,vtime, resp, freq_t, ndof, ncores, path, ext_shape, nabs, grid_ids)
                K_eng,P_eng,G_eng = plot_energy(plot_path,time_model,nabs,miabs,phinew,time,nmodesabs,Khhnew,Mhhnew,resp,ref_grid,ndof,vtime,freq_t)
                h = numpy.zeros((phinew.shape[0],len(time)),dtype=complex)
                Vel = numpy.zeros((phinew.shape[0],len(time)),dtype=complex)
                for j,t in enumerate(time):
                    for mode in range(phinew.shape[1]):
                        h[:,j] += resp[j][mode]*phinew[:,mode]
                        Vel[:,j] += resp[j][nmodesabs+mode]*phinew[:,mode]
            else:
                if infabs != '1':
                    print 'Starting time solver'
                   #Rogers aproximation:
                    print 'Performing Rogers approximation...'
                    Ri,PP_GAF = rogers(GAFabs,k)
                    ybabs = numpy.zeros((phinew.shape[0],1))
                    if ext_shape == 'bending':
                        if ndof == 1:
                            ybabs += 1.0
                        else:
                            ybabs[2::ndof] = 1.0
                    elif ext_shape == 'torsion':
                        x_abs = struct_mesh_dict['full_set']['x']
                        for i,grid in enumerate(grid_ids):
                           #grid = numpy.argwhere(struct_mesh_dict['full_set']['grid_map']==grid)[0][0] +1
                            x_abs = numpy.hstack([x_abs,struct_mesh_dict['full_set']['x'][grid]])
                        if ndof <= 3:
                            ybabs += (x_rot - x_abs).reshape(phinew.shape[0],1)
                        else:
                            ybabs[2::ndof] += 1*(x_rot - x_abs).reshape(phinew.shape[0],1)
                            ybabs[4::ndof] += 1
                    if gust == '0':
                        time, resp = time_solver(plot_path,PP_GAF,Ri,Mnew,phinew,Mhhnew,Khhnew,Chhnew,vtime,rho,nmodesabs,semi_chord,tmax,dt,amplitude,freq_t,ybabs,ndof)
                    elif gust == '1':
                        GAFg = []
                        for i in range(len(k)):
                            x_full_abs = numpy.hstack([struct_mesh_dict['full_set']['x'],struct_mesh_dict['full_set']['x'][sgrids]])
                            if ndof == 1:
                                GAFg.append(numpy.dot(GAFabs[:,:,i],numpy.linalg.pinv(phinew))*numpy.exp(-1j*k[i]*(x_full_abs-x0gust)/semi_chord))
                            else:
                                dir_vec = numpy.zeros((1,ndof*len(x_full_abs)))
                                dir_vec[2::ndof] = numpy.exp(-1j*k[i]*(x_full_abs-x0gust)/semi_chord)
                                GAFg.append(numpy.dot(GAFabs[:,:,i],numpy.linalg.pinv(phinew))*dir_vec)
                        GAFg = numpy.array(GAFg).reshape(nmodesabs,len(x_full_abs),len(k))
                        Rg,PPg = rogers(GAFg,k)
                        time, resp = time_solver_gust(plot_path,PP_GAF,PPg,Ri,Rg,Mnew,phinew,Mhhnew,Khhnew,Chhnew,vtime,rho,nmodesabs,semi_chord,tmax,dt,amplitude,freq_t,ybabs,gust_shape,x0gust,Vgust,Lgust,x_full_abs,ndof)
                    if time_vd == '1':
                        time_video(struct_mesh_dict,aero_mesh_dict,plot_path,ref_grid,semi_chord,phinew,time,vtime, resp, freq_t, ndof, ncores, path, ext_shape, nabs, grid_ids)
                    K_eng,P_eng,G_eng = plot_energy(plot_path,time_model,nabs,miabs,phinew,time,nmodesabs,Khhnew,Mhhnew,resp,ref_grid,ndof,vtime,freq_t)
                    h = numpy.zeros((phinew.shape[0],len(time)),dtype=complex)
                    Vel = numpy.zeros((phinew.shape[0],len(time)),dtype=complex)
                    for j,t in enumerate(time):
                        for mode in range(phinew.shape[1]):
                            h[:,j] += resp[j][mode]*phinew[:,mode]
                            Vel[:,j] += resp[j][nmodesabs+mode]*phinew[:,mode]
                   #abs_dict,phase,omega_resp,group_v,wave_vec,start_t = wave_propagation(h,phinew,time,dt,freq_t,struct_mesh_dict,ndof,nabs,grid_ids)
                   #group_v_video(struct_mesh_dict,aero_mesh_dict,plot_path,semi_chord,phinew,time,vtime, phase, freq_t, ndof, ncores, path, ext_shape, nabs, abs_dict)
                    
                   #plot_laststep(struct_mesh_dict,aero_mesh_dict,plot_path,semi_chord,phinew,time,vtime, resp, freq_t, ndof, ncores, ref_grid, nabs, grid_ids)
                else:
                    print 'Starting time solver'
                   #Rogers aproximation:
                    print 'Performing Rogers approximation...'
                    Ri,PP_GAF = rogers(GAF,k)
                    yb = numpy.zeros((phi.shape[0],1))
                    if ext_shape == 'bending':
                        if ndof == 1:
                            yb += 1.0
                        else:
                            yb[2::ndof] = 1.0
                    elif ext_shape == 'torsion':
                        if ndof <= 3:
                            yb += 1*(x_rot - struct_mesh_dict['full_set']['x']).reshape(phi.shape[0]/ndof,1)
                        else:
                            yb[2::ndof] += 1*(x_rot - struct_mesh_dict['full_set']['x']).reshape(phi.shape[0]/ndof,1)
                            yb[4::ndof] += 1
                    if gust == '0':
                        time, resp = time_solver(plot_path,PP_GAF,Ri,M,phi,Mhh,Khh,Chh,vtime,rho,nmodes,semi_chord,tmax,dt,amplitude,freq_t,yb,ndof,numpy.array(miabs),abs_damping,omegat)
                    elif gust == '1':
                        GAFg = []
                        for i in range(len(k)):
                            if ndof == 1:
                                GAFg.append(numpy.dot(GAF[:,:,i],numpy.linalg.pinv(phi))*numpy.exp(-1j*k[i]*(struct_mesh_dict['full_set']['x']-x0gust)/semi_chord))
                            else:
                                dir_vec = numpy.zeros((1,ndof*len(struct_mesh_dict['full_set']['x'])))
                                dir_vec[2::ndof] = numpy.exp(-1j*k[i]*(struct_mesh_dict['full_set']['x']-x0gust)/semi_chord)
                                GAFg.append(numpy.dot(GAF,numpy.linalg.pinv(phi))*dir_vec)
                        GAFg = numpy.array(GAFg).reshape(nmodes,len(struct_mesh_dict['full_set']['x']),len(k))
                        Rg,PPg = rogers(GAFg,k)
                        time, resp = time_solver_gust(plot_path,PP_GAF,PPg,Ri,Rg,M,phi,Mhh,Khh,Chh,vtime,rho,nmodes,semi_chord,tmax,dt,amplitude,freq_t,yb,gust_shape,x0gust,Vgust,Lgust,struct_mesh_dict['full_set']['x'],ndof,numpy.array(miabs),abs_damping,omegat)
                    if time_vd == '1':
                        time_video(struct_mesh_dict,aero_mesh_dict,plot_path,ref_grid,semi_chord,phi,time,vtime, resp, freq_t, ndof, ncores, path, ext_shape)
                    K_eng,P_eng,G_eng = plot_energy(plot_path,time_model,0,miabs,phi,time,nmodes,Khh,Mhh,resp,ref_grid,ndof,vtime,freq_t)
                    h = numpy.zeros((phi.shape[0],len(time)),dtype=complex)
                    Vel = numpy.zeros((phi.shape[0],len(time)),dtype=complex)
                    for j,t in enumerate(time):
                        for mode in range(phi.shape[1]):
                            h[:,j] += resp[j][mode]*phi[:,mode]
                            Vel[:,j] += resp[j][nmodes+mode]*phi[:,mode]
        else:
            print 'time_model variable must be \'original\' or \'absorbers\''
        timedict = {'time':time,'Ekinetic':K_eng,'Eelastic':P_eng,'Egravitational':G_eng,'nabs':nabs,'disp':h,'disp_vel':Vel,'v_flow':vtime,'freq':freq_t}
        scipy.io.savemat(path+'/time_response.mat',timedict)


    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to run a VLM for a specified geometry.')
    parser.add_argument('-aero', nargs='+', default=[project_dir+'/templates/aero.dat'], help='Path to the dat file containing the aerodynamics geometry.')
    parser.add_argument('-struct', nargs='+', default=[project_dir+'/templates/struct.dat'], help='Path to the dat file containing the structural geometry.')
    parser.add_argument('-par', nargs='+', default=[project_dir+'/templates/parameters.dat'], help='Path to the dat file containing the parameters for the simulation.')
    parser.add_argument('-o', nargs='+', default=['.'], help='Desired path to save the results.')
    args = parser.parse_args()

    DLM(args.aero[0], args.struct[0], args.par[0], args.o[0], project_dir)
