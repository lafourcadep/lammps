from numpy import *
Nx=21
Ny=11
Nz=11

ofile = open('ttm_grid.in','w')
ofile.write('# UNITS: metal COMMENT: initial electron temperature \n')
conv_K_to_eV = 8.61732814974056e-5
conv_eV_to_K = 1./conv_K_to_eV
Tmin=1.2
Tmax=8.5
for iz in range(Nz):
    for iy in range(Ny):
        for ix in range(Nx):
            ofile.write('%d %d %d %5.4f\n' %(ix+1,iy+1,iz+1,conv_eV_to_K*(Tmin+ix*(Tmax-Tmin)/(Nx-1))))
#            ofile.write('%d %d %d %5.4f\n' %(ix,iy,iz,conv_eV_to_K*Tmax))
ofile.close()
