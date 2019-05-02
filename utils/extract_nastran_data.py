import argparse, numpy, pdb, scipy.linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def extract_M_K(pch,out):
    count = 0 #para checar quando se le rigidez ou massa no arquivo pch.
    checked_clamped_elem = 0
    with open(pch,'r') as infile:
        for line in infile:
            data = line.split()
            if data[0] == 'DMIG':
                gridn = -1
                gridold = 0
                count += 1
                column_missing = {}
                if count == 1:
                    K = numpy.zeros((int(data[-1]),int(data[-1])))
                elif count == 2:
                    M = numpy.zeros((int(data[-1]),int(data[-1])))
                else:
                    break
                ndof = int(data[3])
                if checked_clamped_elem == 0:
                    clamped_elem = int(data[2])
                    checked_clamped_elem = 1
            elif data[0] == 'DMIG*':
                if data[2] != gridold:
                    gridn += 1
                gridold = data[2]
                linecount = (gridn-clamped_elem)*ndof + int(data[3])-1
                column_missing[int(data[2])] = int(data[2]) - gridn
            elif data[0] == '*':
                if count == 1:
                    if len(data[-1])<=16:
                        K[linecount,ndof*(int(data[1])-clamped_elem-column_missing[int(data[1])])+int(data[2][0])-1] = float(data[-1].replace('D','E'))
                    else:
                        K[linecount,ndof*(int(data[1])-clamped_elem-column_missing[int(data[1])])+int(data[2][0])-1] = float(data[-1][-16:].replace('D','E'))
                elif count == 2:
                    if len(data[-1])<=16:
                        M[linecount,ndof*(int(data[1])-clamped_elem-column_missing[int(data[1])])+int(data[2][0])-1] = float(data[-1].replace('D','E'))
                    else:
                        M[linecount,ndof*(int(data[1])-clamped_elem-column_missing[int(data[1])])+int(data[2][0])-1] = float(data[-1][-16:].replace('D','E'))

    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            if j>i:
                K[i,j] = K[j,i]
                M[i,j] = M[j,i]
    if ndof == 6:
        Kred = numpy.zeros((K.shape[0]/2,K.shape[1]/2))
        Mred = numpy.zeros((M.shape[0]/2,M.shape[1]/2))
        count = 0
        for i in range(Kred.shape[0]/3):
            for j in range(Kred.shape[1]/3):
                Kred[i*3:(i+1)*3,j*3:(j+1)*3] = K[i*6:(i+1)*6,j*6:(j+1)*6][0:3,0:3]
                Mred[i*3:(i+1)*3,j*3:(j+1)*3] = M[i*6:(i+1)*6,j*6:(j+1)*6][0:3,0:3]
    else:
        Kred = K
        Mred = M
    numpy.savetxt(out+'/K.txt',K)
    numpy.savetxt(out+'/M.txt',M)
   #numpy.savetxt(out+'/Kred.txt',Kred)
   #numpy.savetxt(out+'/Mred.txt',Mred)
    return K,M,Kred,Mred

def get_grid(bdf,out):
    #Le o arquivo bdf do nastran para criar as malhas para plotar graficos atravez da localizacao dos elementos.
    grid_map = []
    x = []
    y = []
    z = []
    count = 0
    check = 0
    with open(bdf,'r') as infile:
        for line in infile:
            data = line.split()
            datanew = []
            for item in data:
                if len(item)<=8:
                    datanew.append(item)
                else:
                    nitens = len(item)%8
                    floatn = 0
                    for i in range(nitens):
                        if '-' in item[i*8+1:(i+1)*8] or '+' in item[i*8+1:(i+1)*8]:
                            floatn += 1
                            datanew.append(item[i*(8+floatn-1):(i+1)*(8+floatn)])
                        datanew.append(item[i*(8+floatn):(i+1)*(8+floatn)])
                    if len(item[(i+1)*floatn+(i+1)*8:])>0:
                        datanew.append(item[(i+1)*floatn+(i+1)*8:])
            data = datanew
                
            if data[0] == 'GRID':
                if '' in data:
                    data.remove('')
                grid_map.append(int(data[1]))
                if data[2][-1] == '.':
                    data[2] = data[2].replace('.','')
                if data[3][-1] == '.':
                    data[3] = data[3].replace('.','')
                if data[4][-1] == '.':
                    data[4] = data[4].replace('.','')
                if '-' in data[2][1:]:
                    data[2]= data[2][0] + data[2][1:].replace('-','E-')
                if '+' in data[2][1:]:
                    data[2]= data[2][0] + data[2][1:].replace('+','E+')
                if '-' in data[3][1:]:
                    data[3]= data[3][0] + data[3][1:].replace('-','E-')
                if '+' in data[3][1:]:
                    data[3]= data[3][0] + data[3][1:].replace('+','E+')
                if '-' in data[2][1:]:
                    data[4]= data[4][0] + data[4][1:].replace('-','E-')
                if '+' in data[4][1:]:
                    data[4]= data[4][0] + data[4][1:].replace('+','E+')
                x.append(float(data[2]))
                y.append(float(data[3]))
                z.append(float(data[4]))
            elif data[0] == 'GRID*':
                tempdata = []
                tempdata.extend(data)
            elif data[0] == '*':
                try:
                    tempdata.extend(data[1:])
                    grid_map.append(int(tempdata[1]))
                    if tempdata[2][-1] == '.':
                        tempdata[2] = tempdata[2].replace('.','')
                    if tempdata[3][-1] == '.':
                        tempdata[3] = tempdata[3].replace('.','')
                    if tempdata[4][-1] == '.':
                        tempdata[4] = tempdata[4].replace('.','')
                    if '-' in tempdata[2][1:]:
                        tempdata[2]= tempdata[2][0] + tempdata[2][1:].replace('-','E-')
                    if '+' in tempdata[2][1:]:
                        tempdata[2]= tempdata[2][0] + tempdata[2][1:].replace('+','E+')
                    if '-' in tempdata[3][1:]:
                        tempdata[3]= tempdata[3][0] + tempdata[3][1:].replace('-','E-')
                    if '+' in tempdata[3][1:]:
                        tempdata[3]= tempdata[3][0] + tempdata[3][1:].replace('+','E+')
                    if '-' in tempdata[4][1:]:
                        tempdata[4]= tempdata[4][0] + tempdata[4][1:].replace('-','E-')
                    if '+' in tempdata[4][1:]:
                        tempdata[4]= tempdata[4][0] + tempdata[4][1:].replace('+','E+')
                    x.append(float(tempdata[2]))
                    y.append(float(tempdata[3]))
                    z.append(float(tempdata[4]))
                except:
                    continue
            
#               if count>=2:
#                   if check<1:
#                       if numpy.absolute(x[-1]-x[-2])/numpy.absolute(x[-2]-x[-3])>1.01:
#                           xsteps = grid_map[-1]-1
#                           check+=1
#               count += 1
#   try:
#       ysteps = grid_map[-1]/xsteps
#   except:
#       xsteps = grid_map[-1]
#       ysteps = grid_map[-1]/xsteps
    grid_map=numpy.array(grid_map)#.reshape((ysteps,xsteps))
    x=numpy.array(x)#.reshape((ysteps,xsteps)) 
    y=numpy.array(y)#.reshape((ysteps,xsteps)) 
    z=numpy.array(z)#.reshape((ysteps,xsteps)) 
                
    numpy.savetxt(out+'/grid_map.txt',grid_map)
    numpy.savetxt(out+'/x.txt',x)
    numpy.savetxt(out+'/y.txt',y)
    numpy.savetxt(out+'/z.txt',z)
    return grid_map, x, y, z

def extract_data(bdf,pch,out):
    K,M,Kred,Mred = extract_M_K(pch,out)
    grid_map, x, y, z = get_grid(bdf,out)
    return

def test(bdf,pch,out):

    K,M,Kred,Mred = extract_M_K(pch,out)
    grid_map, x, y, z = get_grid(bdf,out)
   #M = numpy.loadtxt('M.txt')
   #K = numpy.loadtxt('K.txt')
    #converter unidades K[N/mm]->[N/m]:
    #K*=1000

#   val, vec = scipy.linalg.eig(K,M)
#  #val, vec = scipy.linalg.eig(Kred,Mred)
#   ind = numpy.argsort(val)
#   vec = vec[:,ind]
#   val = val[ind]

#   #nastran_plot
#   xx = numpy.linspace(0,3,16)
#   yy = numpy.linspace(0,1,11)
#   x=((numpy.ones((16,11))).T*xx).T
#   y=numpy.ones((16,11))*yy
#   #analytical_plot
#   xsteps=15
#   ysteps=10
#   L1=3.0
#   L2=1.0
#   x1=((numpy.ones((xsteps,ysteps))).T*numpy.linspace(0,L1,xsteps)).T
#   y1=numpy.ones((xsteps,ysteps))*numpy.linspace(0,L2,ysteps)


#   cont = []
#   cont.append(numpy.loadtxt('analytical_modes/w_00.txt'))
#   cont.append(numpy.loadtxt('analytical_modes/w_01.txt'))
#   cont.append(numpy.loadtxt('analytical_modes/w_10.txt'))
#   freq_w = numpy.loadtxt('analytical_modes/f.txt')[0,0:3]
#   

#  #pdb.set_trace()
#   for mode in [0,1,2]:
#       fig = plt.figure()
#       ax = fig.add_subplot(111, projection='3d')
#       ax.plot_wireframe(x,y, numpy.vstack([numpy.zeros(11),0.25*vec[2::6,mode].reshape(15,11)/numpy.max([numpy.max(vec[2::6,mode]),-numpy.min(vec[2::6,mode])])]) ,color='blue')
#       ax.plot_wireframe(x1,y1, 0.25*cont[mode]/numpy.max([numpy.max(cont[mode]),-numpy.min(cont[mode])]) ,color='red')
#       ax.legend(['MEF','Analitico'],loc='lower left')
#      #ax.plot_wireframe(x,y, 0.25*vec[2::3,mode].reshape(11,15).T/numpy.max([numpy.max(vec[2::3,mode]),-numpy.min(vec[2::3,mode])]) ,color='blue')
#       ax.set_zlim([-1.5,1.5])
#       ax.set_ylim([-1.5,1.5])
#       ax.view_init(elev=45,azim=-45)
#       ax.set_title('f_MEF={}Hz, f_analitico={}Hz'.format(round(1/(2*numpy.pi)*numpy.sqrt(val[mode].real),2),round(freq_w[mode],2)))
#       fig.savefig('figuras/mode_{}.png'.format(mode))
#       plt.close()

#  #pdb.set_trace()
#   
    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script para extrair a matriz de massa, rigidez e dados de grid do modelo em Nastran, usando os arquivos bdf e pch')
    parser.add_argument('-bdf', nargs='+', default=['placa.bdf'], help='Arquivo bdf do Nastran.')
    parser.add_argument('-pch', nargs='+', default=['placa.pch'], help='Arquivo pch do Nastran.')
    parser.add_argument('-out' , nargs='+', default=['.'], help='Local desejado para salvar os dados.')
    args = parser.parse_args()

   #extract_data(args.bdf,args.pch,args.out)
    test(args.bdf[0],args.pch[0],args.out[0])
