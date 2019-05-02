import argparse, numpy, pdb, scipy.linalg, scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def extract_frf(f06):
    disp_r = []
    disp_i = []
    grids = []
    freq = []
    f_old = -1
    read_next = 0 #flag to save next line data
    with open(f06,'r') as infile:
        for line in infile:
            if 'C O M P L E X   F O R C E S   O F   S I N G L E   P O I N T   C O N S T R A I N T' in line:
                freq = freq[0:-1]
                break
            data = line.split()
            if len(data) == 0:
                data = ['empty']
            if data[0] == 'FREQUENCY':
                if float(data[2]) != f_old:
                    freq.append(float(data[2]))
                    f_old = freq[-1]
            if data[0] == '0' and read_next == 1:
                read_next = 2
                grids.append(int(data[1]))
                disp_r.append([float(data[3]),float(data[4]),float(data[5]),float(data[6]),float(data[7]),float(data[8])])
            elif read_next == 2:
                read_next = 1
                disp_i.append([float(data[0]),float(data[1]),float(data[2]),float(data[3]),float(data[4]),float(data[5])])
            elif data[0] == '1':
                read_next = 0
            if data[0] == 'POINT':
                read_next = 1
    disp = numpy.array(disp_r) + 1j*numpy.array(disp_i)
    freq = numpy.array(freq) 
    return grids,freq, disp

def extract_data(f06,dof,wall,ref,out):
    grids, freq, disp = extract_frf(f06)
    ref_id = numpy.argwhere(numpy.array(grids)==int(ref))
    wall_id = numpy.argwhere(numpy.array(grids)==int(wall))
    dof = int(dof) - 1
   #plt.figure()
   #plt.plot(freq,numpy.log(numpy.absolute(disp[ref_id,dof]/disp[wall_id,dof])))
   #plt.show()
    frf_dict = {}
    frf_dict['wall_grid'] = int(wall)
    frf_dict['ref_grid'] = int(ref)
    frf_dict['freq'] = freq
    frf_dict['frf'] = disp[ref_id,dof]/disp[wall_id,dof]
    
    scipy.io.savemat(out+'/frf.mat',frf_dict)
    numpy.savetxt(out+'/freq.txt',freq)
    numpy.savetxt(out+'/frf.txt',disp[ref_id,dof]/disp[wall_id,dof])
    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script para extrair a frf do Nastran, usando o arquivo f06')
    parser.add_argument('-f06', nargs='+', default=['frf.f06'], help='Arquivo f06 do Nastran.')
    parser.add_argument('-dof', nargs='+', default=['3'], help='Grau de liberdade no qual se deseja extrair a frf 1=x, 2=y, 3=z')
    parser.add_argument('-wall', nargs='+', default=['frf.f06'], help='No onde eh aplicada a excitacao no Nastran.')
    parser.add_argument('-ref', nargs='+', default=['frf.f06'], help='No de referencia da frf Nastran.')
    parser.add_argument('-out' , nargs='+', default=['.'], help='Local desejado para salvar os dados.')
    args = parser.parse_args()

   #extract_data(args.bdf,args.pch,args.out)
    extract_data(args.f06[0],args.dof[0],args.wall[0],args.ref[0],args.out[0])
