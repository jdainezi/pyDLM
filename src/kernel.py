import numpy, pdb

def kernel(xli,yli,xci,yci,xri,yri,xpci,ypci,mach,kk,semi_chord,sym):

    # xli = input xl
    ## Comprimento do painel
    # Nesse passo computa-se o comprimento da corda do painel na metade da
    # envergadura do mesmo
  
    if sym == 2: #half-model symmetry
        xc = numpy.hstack([numpy.array(xci),numpy.array(xci)]) 
        xpc = numpy.hstack([numpy.array(xpci),numpy.array(xpci)])
        xl = numpy.hstack([numpy.array(xli),numpy.array(xri)])
        xr = numpy.hstack([numpy.array(xri),numpy.array(xli)])
        yc = numpy.hstack([numpy.array(yci),-numpy.array(yci)])
        ypc = numpy.hstack([numpy.array(ypci),-numpy.array(ypci)])
        yl = numpy.hstack([numpy.array(yli),-numpy.array(yri)])
        yr = numpy.hstack([numpy.array(yri),-numpy.array(yli)])
    else:
        xc = xci 
        xpc = xpci
        xl = xli
        xr = xri
        yc = yci
        ypc = ypci
        yl = yli
        yr = yri

    nxpc = len(xpci)
    XX = 2*(numpy.array(xpc)-numpy.array(xc))/semi_chord
    
    ## Area do painel
    
   #ee = area 
       
    beta = numpy.sqrt(1 - mach**2)
    xl1 = numpy.zeros((len(xpc),nxpc),dtype=complex)  
    yl1 = numpy.zeros_like(xl1)  
    xc1 = numpy.zeros_like(xl1)  
    yc1 = numpy.zeros_like(xl1)  
    xr1 = numpy.zeros_like(xl1)  
    yr1 = numpy.zeros_like(xl1) 
    kl1 = numpy.zeros_like(xl1)  
    kr1 = numpy.zeros_like(xl1)  
    kc1 = numpy.zeros_like(xl1) 
    ul1 = numpy.zeros_like(xl1)  
    ur1 = numpy.zeros_like(xl1)  
    uc1 = numpy.zeros_like(xl1) 
    ul11 = numpy.zeros_like(xl1)  
    ur11 = numpy.zeros_like(xl1)  
    uc11 = numpy.zeros_like(xl1) 
    ul12 = numpy.zeros_like(xl1)  
    ur12 = numpy.zeros_like(xl1)  
    uc12 = numpy.zeros_like(xl1) 
    Il1 = numpy.zeros_like(xl1)  
    Ic1 = numpy.zeros_like(xl1)  
    Ir1 = numpy.zeros_like(xl1) 
    Il11 = numpy.zeros_like(xl1)  
    Ic11 = numpy.zeros_like(xl1)  
    Ir11 = numpy.zeros_like(xl1) 
    Il12 = numpy.zeros_like(xl1)  
    Ic12 = numpy.zeros_like(xl1)  
    Ir12 = numpy.zeros_like(xl1) 
    Jl1 = numpy.zeros_like(xl1)  
    Jc1 = numpy.zeros_like(xl1)  
    Jr1 = numpy.zeros_like(xl1) 
    Jl11 = numpy.zeros_like(xl1)  
    Jc11 = numpy.zeros_like(xl1)  
    Jr11 = numpy.zeros_like(xl1) 
    Jl12 = numpy.zeros_like(xl1)  
    Jc12 = numpy.zeros_like(xl1)  
    Jr12 = numpy.zeros_like(xl1) 
    Kl1 = numpy.zeros_like(xl1)  
    Kc1 = numpy.zeros_like(xl1)  
    Kr1 = numpy.zeros_like(xl1) 
    A0 = numpy.zeros_like(xl1)  
    A1 = numpy.zeros_like(xl1)  
    A2 = numpy.zeros_like(xl1) 
    B0 = numpy.zeros_like(xl1)  
    B1 = numpy.zeros_like(xl1)  
    B2 = numpy.zeros_like(xl1) 
    
    ## Matriz de Influencia Vortex-lattice

   
    for ii in range(len(xpc)): 
        for jj in range(nxpc):
            
            if yl[jj] < 0: #trying to fix assymetry
                xl1[ii,jj] = (xpc[ii] - xr[jj])/semi_chord 
                yl1[ii,jj] = (ypc[ii] - yr[jj])/semi_chord 
                         
                xc1[ii,jj] = (xpc[ii] - xc[jj])/semi_chord 
                yc1[ii,jj] = (ypc[ii] - yc[jj])/semi_chord 
                         
                xr1[ii,jj] = (xpc[ii] - xl[jj])/semi_chord 
                yr1[ii,jj] = (ypc[ii] - yl[jj])/semi_chord 

            else:
                xl1[ii,jj] = (xpc[ii] - xl[jj])/semi_chord 
                yl1[ii,jj] = (ypc[ii] - yl[jj])/semi_chord 
                         
                xc1[ii,jj] = (xpc[ii] - xc[jj])/semi_chord 
                yc1[ii,jj] = (ypc[ii] - yc[jj])/semi_chord 
                         
                xr1[ii,jj] = (xpc[ii] - xr[jj])/semi_chord 
                yr1[ii,jj] = (ypc[ii] - yr[jj])/semi_chord 
    
#   FF1 = numpy.zeros_like(xl1)
    #comprimento da linha de dipolos 
    L = numpy.sqrt((numpy.array(yl)-numpy.array(yr))**2+(numpy.array(xl)-numpy.array(xr))**2)/(2*semi_chord)
 
    ## Calculo de u1 e k1
    
    # Essas sao variaveis advindas para o computos dos termos da matriz de
    # Kernel. u1 eh uma variavel geometrica e aerodinamica, e k esta relacionado
    # com a frequencia reduzida do sistema.
    
    # Existem problemas de singularidade quanto a u1, assim quando y1 eh zero,
    # examina-se a situacao limite.
    
    for ii in range(len(xpc)):
        for jj in range(nxpc):
            
            kl1[ii,jj] = kk*numpy.absolute(yl1[ii,jj])
            if yl1[ii,jj] == 0: 
                if xl1[ii,jj] > 0:
                    ul1[ii,jj] = -10**20 
                else:
                    ul1[ii,jj] = 10**20 
    
            else:
                ul1[ii,jj] = (mach*(numpy.sqrt(xl1[ii,jj]**2.0 + beta**2.0*yl1[ii,jj]**2.0)) - xl1[ii,jj])/(numpy.absolute(yl1[ii,jj])*beta**2.0) 
    
    
    # Para c. De fato, so se tem problemas de singularidade com c.
    
            kc1[ii,jj] = kk*numpy.absolute(yc1[ii,jj])
    
            if yc1[ii,jj] == 0: 
                if xc1[ii,jj] > 0:
                    uc1[ii,jj] = -10**20 
                else:
                    uc1[ii,jj] = 10**20 
    
            else:
                uc1[ii,jj] = (mach*(numpy.sqrt(xc1[ii,jj]**2.0 + beta**2.0*yc1[ii,jj]**2.0)) - xc1[ii,jj])/(numpy.absolute(yc1[ii,jj])*beta**2.0) 
    
            kr1[ii,jj] = kk*numpy.absolute(yr1[ii,jj])
            if yr1[ii,jj] == 0: 
                if xr1[ii,jj] > 0:
                    ur1[ii,jj] = -10**20 
                else:
                    ur1[ii,jj] = 10**20 
    
            else:
                ur1[ii,jj] = (mach*numpy.sqrt(xr1[ii,jj]**2.0 + beta**2.0*yr1[ii,jj]**2.0) - xr1[ii,jj])/(numpy.absolute(yr1[ii,jj])*beta**2.0) 
    
    
    ## Polinomio de J1
    
    #Esse eh um polinomio aproximador resultante de uma integral. Pode ser vista
    #na equacao 281, pag. 89 do relatorio WL-TR-95-3022 de Max Blair
    
    aa = numpy.array([0.24186198,-2.7918027,24.991079,-111.59196,271.43549,-305.75288,-41.18363,545.98537,-644.78155,328.72755,-64.279511]) 
    cc = 0.372 
    
    ## Calculo de Il1 , JL1 e Kl1
    
    for ii in range(len(xpc)):
        for jj in range(nxpc):
    
            if ul1[ii,jj]>0:

                if ul1[ii,jj]>10**10:
                    Il1[ii,jj] = 0 
                else:
    
                    Jl1[ii,jj] = 0 
    
                    for ll in range(1,len(aa)+1):
    
                          Jl1[ii,jj] = (aa[ll-1]*numpy.exp(-ll*cc*ul1[ii,jj]))/(ll**2.0*cc**2.0 + (kl1[ii,jj])**2.0)*(ll*cc - 1j*kl1[ii,jj]) + Jl1[ii,jj] 
    
                    Il1[ii,jj] = numpy.exp(-1j*kl1[ii,jj]*ul1[ii,jj])*(1 - (ul1[ii,jj]/(1 + ul1[ii,jj]**2.0)**(1/2.0)) - 1j*Jl1[ii,jj]*kl1[ii,jj]) 
    
            else:
                if ul1[ii,jj]<-10**10:
                    ul11[ii,jj] = 0 
                    Jl11[ii,jj] = 0 
    
                    for ll in range(1,len(aa)+1):
                        Jl11[ii,jj] = (aa[ll-1]*numpy.exp(-ll*cc*ul11[ii,jj]))/(ll**2.0*cc**2.0 + kl1[ii,jj]**2.0)*(ll*cc - 1j*kl1[ii,jj]) + Jl11[ii,jj] 
    
                    Il1[ii,jj] = 2.0*(numpy.exp(-1j*kl1[ii,jj]*ul11[ii,jj])*(1 - (ul11[ii,jj]/(1 + ul11[ii,jj]**2.0)**(1/2.0)) - 1j*Jl11[ii,jj]*kl1[ii,jj])) 
    
                else:
    
                    ul11[ii,jj] = 0 
                    Jl11[ii,jj] = 0 
    
                    for ll in range(1,len(aa)+1):
    
                        Jl11[ii,jj] = (aa[ll-1]*numpy.exp(-ll*cc*ul11[ii,jj]))/(ll**2.0*cc**2.0 + kl1[ii,jj]**2.0)*(ll*cc - 1j*kl1[ii,jj]) + Jl11[ii,jj] 
    
    
                    ul12[ii,jj] = -ul1[ii,jj] 
    
                    Jl12[ii,jj] = 0 
    
                    for ll in range(1,len(aa)+1):
    
                        Jl12[ii,jj] = (aa[ll-1]*numpy.exp(-ll*cc*ul12[ii,jj]))/(ll**2.0*cc**2.0 + kl1[ii,jj]**2.0)*(ll*cc - 1j*kl1[ii,jj]) + Jl12[ii,jj] 
    
    
                    Il11[ii,jj] = numpy.exp(-1j*kl1[ii,jj]*ul11[ii,jj])*(1 - (ul11[ii,jj]/numpy.sqrt(1 + ul11[ii,jj]**2.0)) - 1j*Jl11[ii,jj]*kl1[ii,jj]) 
                    Il12[ii,jj] = numpy.exp(-1j*kl1[ii,jj]*ul12[ii,jj])*(1 - (ul12[ii,jj]/numpy.sqrt(1 + ul12[ii,jj]**2.0)) - 1j*Jl12[ii,jj]*kl1[ii,jj]) 
    
                    Il1[ii,jj] = (2.0*(Il11[ii,jj].real) - (Il12[ii,jj].real) + 1j*(Il12[ii,jj].imag)) 
    
    ## Calculo de Ic1 , Jc1 e Kc1
    
            if uc1[ii,jj]>0:
    
                if uc1[ii,jj]>10**10:
                    Ic1[ii,jj] = 0 
                else:
    
                    Jc1[ii,jj] = 0 
                    for ll in range(1,len(aa)+1):
                        Jc1[ii,jj] = (aa[ll-1]*numpy.exp(-ll*cc*uc1[ii,jj]))/(ll**2.0*cc**2.0 + kc1[ii,jj]**2.0)*(ll*cc - 1j*kc1[ii,jj]) + Jc1[ii,jj] 
    
                    Ic1[ii,jj] = numpy.exp(-1j*kc1[ii,jj]*uc1[ii,jj])*(1 - (uc1[ii,jj]/(1 + uc1[ii,jj]**2.0)**(1/2.0)) - 1j*Jc1[ii,jj]*kc1[ii,jj]) 
    
            else:
                if uc1[ii,jj]<-10**10:
                    uc11[ii,jj] = 0 
                    Jc11[ii,jj] = 0 
    
                    for ll in range(1,len(aa)+1):
                        Jc11[ii,jj] = (aa[ll-1]*numpy.exp(-ll*cc*uc11[ii,jj]))/(ll**2.0*cc**2.0 + kc1[ii,jj]**2.0)*(ll*cc - 1j*kc1[ii,jj]) + Jc11[ii,jj] 
    
                    Ic1[ii,jj] = 2.0*(numpy.exp(-1j*kc1[ii,jj]*uc11[ii,jj])*(1 - (uc11[ii,jj]/(1 + uc11[ii,jj]**2.0)**(1/2.0)) - 1j*Jc11[ii,jj]*kc1[ii,jj])) 
    
                else:
                    uc11[ii,jj] = 0 
                    Jc11[ii,jj] = 0 
    
                    for ll in range(1,len(aa)+1):
                        Jc11[ii,jj] = (aa[ll-1]*numpy.exp(-ll*cc*uc11[ii,jj]))/(ll**2.0*cc**2.0 + kc1[ii,jj]**2.0)*(ll*cc - 1j*kc1[ii,jj]) + Jc11[ii,jj] 
    
                    uc12[ii,jj] = -uc1[ii,jj] 
                    Jc12[ii,jj] = 0 
    
                    for ll in range(1,len(aa)+1):
                        Jc12[ii,jj] = (aa[ll-1]*numpy.exp(-ll*cc*(uc12[ii,jj])))/(ll**2.0*cc**2.0 + kc1[ii,jj]**2.0)*(ll*cc - 1j*kc1[ii,jj]) + Jc12[ii,jj] 
    
                    Ic11[ii,jj] = numpy.exp(-1j*kc1[ii,jj]*uc11[ii,jj])*(1 - (uc11[ii,jj]/numpy.sqrt(1 + uc11[ii,jj]**2.0)) - 1j*Jc11[ii,jj]*kc1[ii,jj]) 
                    Ic12[ii,jj] = numpy.exp(-1j*kc1[ii,jj]*(uc12[ii,jj]))*(1 - (uc12[ii,jj]/numpy.sqrt(1 + (uc12[ii,jj])**2.0)) - 1j*Jc12[ii,jj]*kc1[ii,jj]) 
    
                    Ic1[ii,jj] = (2.0*(Ic11[ii,jj].real) - (Ic12[ii,jj].real) + 1j*(Ic12[ii,jj].imag)) 
    
    ## Calculo de Ir1 , Jr1 e Kr1
    
            if ur1[ii,jj]>0:
                if ur1[ii,jj]>10**10:
                    Ir1[ii,jj] = 0 
                else:
    
                    Jr1[ii,jj] = 0 
    
                    for ll in range(1,len(aa)+1):
    
                        Jr1[ii,jj] = (aa[ll-1]*numpy.exp(-ll*cc*ur1[ii,jj]))/(ll**2.0*cc**2.0 + kr1[ii,jj]**2.0)*(ll*cc - 1j*kr1[ii,jj]) + Jr1[ii,jj] 
    
    
                    Ir1[ii,jj] = numpy.exp(-1j*kr1[ii,jj]*ur1[ii,jj])*(1 - (ur1[ii,jj]/numpy.sqrt(1 + ur1[ii,jj]**2.0)) - 1j*Jr1[ii,jj]*kr1[ii,jj]) 
    
            else:
                if ur1[ii,jj]<-10**10:
                    ur11[ii,jj] = 0 
                    Jr11[ii,jj] = 0 
    
                    for ll in range(1,len(aa)+1):
                        Jr11[ii,jj] = (aa[ll-1]*numpy.exp(-ll*cc*ur11[ii,jj]))/(ll**2.0*cc**2.0 + kr1[ii,jj]**2.0)*(ll*cc - 1j*kr1[ii,jj]) + Jr11[ii,jj] 
    
                    Ir1[ii,jj] = 2.0*(numpy.exp(-1j*kr1[ii,jj]*ur11[ii,jj])*(1 - (ur11[ii,jj]/(1 + ur11[ii,jj]**2.0)**(1/2.0)) - 1j*Jr11[ii,jj]*kr1[ii,jj])) 
                else:    
                    ur11[ii,jj] = 0 
    
                    Jr11[ii,jj] = 0 
    
                    for ll in range(1,len(aa)+1):
    
                        Jr11[ii,jj] = (aa[ll-1]*numpy.exp(-ll*cc*ur11[ii,jj]))/(ll**2.0*cc**2.0 + kr1[ii,jj]**2.0)*(ll*cc - 1j*kr1[ii,jj]) + Jr11[ii,jj] 
    
    
                    ur12[ii,jj] = -ur1[ii,jj] 
    
                    Jr12[ii,jj] = 0 
    
                    for ll in range(1,len(aa)+1):
    
                        Jr12[ii,jj] = (aa[ll-1]*numpy.exp(-ll*cc*ur12[ii,jj]))/(ll**2.0*cc**2.0 + kr1[ii,jj]**2.0)*(ll*cc - 1j*kr1[ii,jj]) + Jr12[ii,jj] 
    
    
                    Ir11[ii,jj] = numpy.exp(-1j*kr1[ii,jj]*ur11[ii,jj])*(1 - (ur11[ii,jj]/numpy.sqrt(1 + ur11[ii,jj]**2.0)) - 1j*Jr11[ii,jj]*kr1[ii,jj]) 
                    Ir12[ii,jj] = numpy.exp(-1j*kr1[ii,jj]*ur12[ii,jj])*(1 - (ur12[ii,jj]/numpy.sqrt(1 + ur12[ii,jj]**2.0)) - 1j*Jr12[ii,jj]*kr1[ii,jj]) 
    
                    Ir1[ii,jj] = (2.0*(Ir11[ii,jj].real) - (Ir12[ii,jj].real) + 1j*(Ir12[ii,jj].imag)) 
    
    ## Calculo de K
    
    #Kc1 apresenta um tratamento diferenciado jah que tem problemas quanto a
    #existencia de singularidades.
    
            if uc1[ii,jj]>0:
                if uc1[ii,jj]>10**10:
                    
                    Kc1[ii,jj] = -Ic1[ii,jj]*numpy.exp(-1j*kk*xc1[ii,jj]) ## Mudanca aqui #0
                     
                else:
                    Kc1[ii,jj] = (-Ic1[ii,jj] - (mach*numpy.absolute(yc1[ii,jj]))/(numpy.sqrt(xc1[ii,jj]**2.0 + beta**2.0*yc1[ii,jj]**2.0))*numpy.exp(-1j*kc1[ii,jj]*uc1[ii,jj])/numpy.sqrt(1 + uc1[ii,jj]**2.0))*numpy.exp(-1j*kk*xc1[ii,jj]) 
            else:
                if uc1[ii,jj]<-10**10:
                    Kc1[ii,jj] = -Ic1[ii,jj]*numpy.exp(-1j*kk*xc1[ii,jj]) 
                else:
                    Kc1[ii,jj] = (-Ic1[ii,jj] - (mach*numpy.absolute(yc1[ii,jj]))/(numpy.sqrt(xc1[ii,jj]**2.0 + beta**2.0*yc1[ii,jj]**2.0))*numpy.exp(-1j*kc1[ii,jj]*uc1[ii,jj])/(1 + uc1[ii,jj]**2.0)**(1/2.0))*numpy.exp(-1j*kk*xc1[ii,jj]) 
    
    
            Kl1[ii,jj] = (-Il1[ii,jj]-numpy.exp(-1j*kl1[ii,jj]*ul1[ii,jj])*(mach*numpy.absolute(yl1[ii,jj]))/(numpy.sqrt(xl1[ii,jj]**2.0 + beta**2.0*yl1[ii,jj]**2.0)*numpy.sqrt(1 + ul1[ii,jj]**2.0)))*numpy.exp(-1j*kk*xl1[ii,jj]) 
            Kr1[ii,jj] = (-Ir1[ii,jj]-numpy.exp(-1j*kr1[ii,jj]*ur1[ii,jj])*(mach*numpy.absolute(yr1[ii,jj]))/(numpy.sqrt(xr1[ii,jj]**2.0 + beta**2.0*yr1[ii,jj]**2.0)*numpy.sqrt(1 + ur1[ii,jj]**2.0)))*numpy.exp(-1j*kk*xr1[ii,jj]) 
            
    
    ## Calculo do A'
    
    # Como feito nas equacoes tal do relatorio (...).
    
            A0[ii,jj] = (Kc1[ii,jj]) 
            A1[ii,jj] = (Kr1[ii,jj] - Kl1[ii,jj])/(2.0*L[jj])
            A2[ii,jj] = (Kl1[ii,jj] - 2.0*Kc1[ii,jj] + Kr1[ii,jj])/(2.0*L[jj]**2.0)
    
    ## Calculo dos B
    
    # Como feito nas equacoes tal do relatorio (...).
    
    for ii in range(len(xpc)):
        for jj in range(nxpc):
    
           #yc1[ii,jj] = ((ypc[ii]/semi_chord) - yc[jj]/semi_chord)
                #In order to fix assymetry when y<0: 
            if ypc[ii] >= 0:
                yc1 = (ypc[ii] - yc[jj])/semi_chord
            else:
                yc1 = (-ypc[ii] + yc[jj])/semi_chord
            if ypc[ii] >= 0:
                if xr[ii]==xl[ii]:
                    sweep = numpy.pi/2
                else:
                    sweep = numpy.arctan((yr[ii]-yl[ii])/(xr[ii]-xl[ii]))
            else:
                if xr[ii]==xl[ii]:
                    sweep = numpy.pi/2
                else:
                    sweep = -numpy.arctan((yr[ii]-yl[ii])/(xr[ii]-xl[ii]))
    
            B0[ii,jj] = 2.0*L[jj]*A0[ii,jj]/(yc1**2.0 - L[jj]**2.0*numpy.sin(sweep)**2) 
            B1[ii,jj] = A1[ii,jj]*(1/(2.0*numpy.sin(sweep)**2))*numpy.log(((L[jj]*numpy.sin(sweep) - yc1)**2.0)/((L[jj]*numpy.sin(sweep) + yc1)**2.0))+(2.0*L[jj]*yc1*A1[ii,jj])/(yc1**2.0 - L[jj]**2.0*numpy.sin(sweep)**2)/numpy.sin(sweep)
            B2[ii,jj] = A2[ii,jj]*(2.0*L[jj]/numpy.sin(sweep)**2 + (yc1*numpy.log((L[jj]*numpy.sin(sweep) - yc1)**2.0/(yc1 + L[jj]*numpy.sin(sweep))**2.0)/numpy.sin(sweep))+(2.0*L[jj]*yc1**2.0)/(yc1**2.0 - L[jj]**2.0*numpy.sin(sweep)**2)/numpy.sin(sweep)**2) 
            
    ## Matriz de Kernel
    
    #Relaciona w=K*delta(p). Do lado esquerdo o vetor de downwash, K a matriz
    #de Kernel com a pressao em um box.
    
    KK = numpy.zeros_like(xl1) 
    for ii in range(len(xpc)):
        for jj in range(nxpc):
            KK[ii,jj] = -(XX[ii]/(8*numpy.pi))*(B0[ii,jj] + B1[ii,jj] + B2[ii,jj]) 

    if sym == 2: #half-model symmetry
        AIC = KK[0:len(xpc)/2,:][:,0:len(xpc)/2] + KK[len(xpc)/2:,:][:,0:len(xpc)/2]
    else:
        AIC = KK 
#   FF = FF1
    return AIC

#AIC = kernel(xl=[0.25,0.25,1.25,1.25],yl=[0,1,0,1],xc=[0.25,0.25,1.25,1.25],yc=[0.5,1.5,0.5,1.5],xr=[0.25,0.25,1.25,1.25],yr=[1,2,1,2],xpc=[0.75,0.75,1.75,1.75],ypc=[0.5,1.5,0.5,1.5],mach=0.5,kk=0.1,semi_chord=0.5)
