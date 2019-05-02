import numpy, argparse, sys, os, pdb

def temperature_SI(h):
    '''Given an altitude in m, it returns the temperature in K'''
    a = -0.00356616 #F/ft
    a = (a*5/9)/0.3048 #K/m  
    T0 = 288.16 #K
    if h < 11000:
        T = T0 + a*h #K
    elif h < 20000:
        T = -69.7 #F
        T = (T-32)*5/9 + 273.15 #K
    else:
        sys.exit('Specified altitude out of Stratosphere range.')
    return T

def temperature_imperial(h):
    '''Given an altitude in ft, it returns the temperature in R'''
    a = -0.00356616 #F/ft = R/ft
    T0 = 518.69  #R
    if h < 36089:
        T = T0 + a*h #R
    elif h < 65617:
        T = -69.7 #F
        T += 459.7 #R
    else:
        sys.exit('Specified altitude out of Stratosphere range.')
    return T

def density_SI(h):
    '''Given an altitude in m, it returns the air density in kg/m3'''
    a = 0.00356616 #F/ft
    a = (a*5/9)/0.3048 #K/m  
    T0 = 288.16 #K
    rho0 = 1.225 #kg/m3
    if h < 11000:
        theta = 1 - a*h/T0
        rho = rho0*theta**4.2561
    elif h < 20000:
        rho = rho0*(0.2971)*numpy.exp(-(h/0.3048-36089)/20806.7)
    else:
        sys.exit('Specified altitude out of Stratosphere range.')
    return rho

def density_imperial(h):
    '''Given an altitude in ft, it returns the air density in slug/ft3'''
    a = 0.00356616 #F/ft = R/ft
    T0 = 518.69  #R
    rho0 = 2.3769e-3 #slug/ft3
    if h < 36089:
        theta = 1 - a*h/T0
        rho = rho0*theta**4.2561
    elif h < 65617:
        rho = rho0*(0.2971)*numpy.exp(-(h-36089)/20806.7)
    else:
        sys.exit('Specified altitude out of Stratosphere range.')
    return rho

def pressure_SI(h):
    '''Given an altitude in m, it returns the atmospheric pressure in Pa'''
    a = 0.00356616 #F/ft
    a = (a*5/9)/0.3048 #K/m  
    T0 = 288.16 #K
    p0 = 1.013e5 #Pa
    if h < 11000:
        theta = 1 - a*h/T0
        p = p0*theta**5.2561
    elif h < 20000:
        p = p0*(0.2234)*numpy.exp(-(h/0.3048-36089)/20806.7)
    else:
        sys.exit('Specified altitude out of Stratosphere range.')
    return p

def pressure_imperial(h):
    '''Given an altitude in ft, it returns the air density in lbs/ft2'''
    a = 0.00356616 #F/ft = R/ft
    T0 = 518.69  #R
    p0 = 2116.2 #lbs/ft2
    if h < 36089:
        theta = 1 - a*h/T0
        p = p0*theta**5.2561
    elif h < 65617:
        p = p0*(0.2234)*numpy.exp(-(h/0.3048-36089)/20806.7)
    else:
        sys.exit('Specified altitude out of Stratosphere range.')
    return p

def sound_speed_SI(T):
    '''Given a temperature in K, it returns the sound speed in m/s'''
    R = 287.1
    gama = 1.4
    a = numpy.sqrt(gama*R*T)
    return a
    
def sound_speed_imperial(T):
    '''Given a temperature in R, it returns the sound speed in ft/s'''
    R = 1715.7
    gama = 1.4
    a = numpy.sqrt(gama*R*T)
    return a

def viscosity_SI(T):
    '''Given a temperature in K, it returns the viscosity mu in kg*s/m'''
    mu0 = 1.789e-5
    T0 = 288.16
    S = 110.56
    mu = mu0*((T/T0)**(1.5))*(T0+S)/(T+S)
    return mu

def viscosity_imperial(T):
    '''Given a temperature in R, it returns the viscosity mu in slug*s/ft'''
    mu0 = 3.737e-7
    T0 = 518.69
    S = 110.56*1.8
    mu = mu0*((T/T0)**(1.5))*(T0+S)/(T+S)
    return mu

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to provide atmospheric parameters and save it into a text file. If the altitude is provided in metres, the data is given in SI. If it is provided in feet, the data is given in imperial units. The same idea works for temperature given in K or R.')
    parser.add_argument('-path', nargs='+', default='.', help='Desired path to save the data.')
    parser.add_argument('-m', type=float, nargs='+', default=None, help='Altitude in metres.')
    parser.add_argument('-ft', type=float, nargs='+', default=None, help='Altitude in feet.')
    parser.add_argument('-tk', type=float, nargs='+', default=None, help='Temperature in Kelvin to get sound speed and viscosity.')
    parser.add_argument('-tr', type=float, nargs='+', default=None, help='Temperature in Rankine to get sound speed and viscosity.')
    args = parser.parse_args()

    data_file = os.path.join(args.path,'atmospheric_data.txt')
    if args.m is not None:
        with open(data_file,'w') as outfile:
            header = ''
            for i in xrange(62):
                header += '#'
            header += '\n'
            outfile.write(header)
            outfile.write('#Atmospheric data\n')
            outfile.write(header)
           #outfile.write('    h(m)   rho(kg/m3)     T(K)      p(Pa)          {}   \n'.format(u"\u03BC"))
            outfile.write('   h(m)  rho(kg/m3)    T(K)       p(Pa)    {}(kgs/m)  a(m/s) \n'.format(unichr(0x3bc).encode('utf-8')))
            for h in args.m:
                rho = '{:.4E}'.format(density_SI(h))
                T = str(numpy.round(temperature_SI(h),2))
                a = str(numpy.round(sound_speed_SI(float(T)),2))
                p = '{:.4E}'.format(pressure_SI(h))
                mu = '{:.4E}'.format(viscosity_SI(float(T)))
                h = str(h)
                while len(h)<7:
                    h = ' '+h 
                while len(T)<6:
                    T = ' '+T 
                while len(a)<5:
                    a = ' '+a 
                while len(rho)<10:
                    rho = ' '+rho 
                while len(p)<10:
                    p = ' '+p 
                while len(mu)<10:
                    mu = ' '+mu 
                
                line = '  '.join([h,rho,T,p,mu,a]) + '\n'
                outfile.write(line)
    elif args.ft is not None:
        with open(data_file,'w') as outfile:
            header = ''
            for i in xrange(62):
                header += '#'
            header += '\n'
            outfile.write(header)
            outfile.write('#Atmospheric data\n')
            outfile.write(header)
           #outfile.write('   h(ft)  rho(sl/m3)     T(R) p(lbs/ft2)          {}   \n'.format(u"\u03BC"))
            outfile.write('  h(ft) rho(sl/ft3)    T(R)  p(lbs/ft2)  {}(sl*s/ft)  a(ft/s)\n'.format(unichr(0x3bc).encode('utf-8')))
            for h in args.ft:
                rho = '{:.4E}'.format(density_imperial(h))
                T = str(numpy.round(temperature_imperial(h),2))
                a = str(numpy.round(sound_speed_imperial(float(T)),2))
                p = '{:.4E}'.format(pressure_imperial(h))
                mu = '{:.4E}'.format(viscosity_imperial(float(T)))
                h = str(h)
                while len(h)<7:
                    h = ' '+h 
                while len(T)<6:
                    T = ' '+T 
                while len(a)<5:
                    a = ' '+a 
                while len(rho)<10:
                    rho = ' '+rho 
                while len(p)<10:
                    p = ' '+p 
                while len(mu)<10:
                    mu = ' '+mu 
                
                line = '  '.join([h,rho,T,p,mu,a]) + '\n'
                outfile.write(line)
    elif args.tk is not None:
        with open(data_file,'w') as outfile:
            header = ''
            for i in xrange(62):
                header += '#'
            header += '\n'
            outfile.write(header)
            outfile.write('#Atmospheric data\n')
            outfile.write(header)
            outfile.write(' T(K)  a(m/s)    {}(kgs/m)\n'.format(unichr(0x3bc).encode('utf-8')))
            for T in args.tk:
                T = str(T)
                a = str(numpy.round(sound_speed_SI(float(T)),2))
                mu = '{:.4E}'.format(viscosity_SI(float(T)))
                while len(T)<6:
                    T = ' '+T 
                while len(a)<5:
                    a = ' '+a 
                while len(mu)<10:
                    mu = ' '+mu 
                line = '  '.join([T,a,mu]) + '\n'
                outfile.write(line)
    elif args.tr is not None:
        with open(data_file,'w') as outfile:
            header = ''
            for i in xrange(62):
                header += '#'
            header += '\n'
            outfile.write(header)
            outfile.write('#Atmospheric data\n')
            outfile.write(header)
            outfile.write(' T(R) a(ft/s)  {}(sl*s/ft) \n'.format(unichr(0x3bc).encode('utf-8')))
            for T in args.tr:
                T = str(T)
                a = str(numpy.round(sound_speed_imperial(float(T)),1))
                mu = '{:.4E}'.format(viscosity_imperial(float(T)))
                while len(T)<6:
                    T = ' '+T 
                while len(a)<5:
                    a = ' '+a 
                while len(mu)<10:
                    mu = ' '+mu 
                line = '  '.join([T,a,mu]) + '\n'
                outfile.write(line)
    else:
        sys.exit('Please provide the altitude or the temperature.')
