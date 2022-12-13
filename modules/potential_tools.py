#Contains functions related to potential, density

import numpy as np
from scipy.integrate import dblquad

def MN_potential(R, z, a, b, M, G):
    '''
    :returns the Miyamoto - Nagai potential at (R, z)
    :inputs:
        R -> radial coordinate
        z -> vertical coordinate
        a -> radial length scale
        b -> vertical length scale
        M -> stellar mass of the disk
        G -> gravitational constant (give in appropriate units)
    '''
    
    A = G*M/a
    B = R/a
    C = 1 + np.sqrt((z/a)**2 + (b/a)**2)
    
    Phi = A/np.sqrt(B**2 + C**2)
    
    return Phi

def MN_rho(R, z, a, b, logM):
    '''
    :returns the Miyamoto - Nagai mass density at (R, z)
    :inputs:
        R -> radial coordinate
        z -> vertical coordinate
        a -> radial length scale
        b -> vertical length scale
        logM -> log of the stellar mass of the disk
    '''    
    
    A = (10**logM)/(4*np.pi*(a**2)*b)
    B = R/a
    C = 1 + 3*np.sqrt((z/a)**2 + (b/a)**2)
    D = 1 + np.sqrt((z/a)**2 + (b/a)**2)
    E = np.sqrt(1 + (z/b)**2)
    
    rho = A*(B**2 + C*(D**2))/(((B**2 + D**2)**(5/2))*(E**3))
    
    return rho

def MN_mass(Ri, Rf, zi, zf, a, b, logM):
    '''
    Integrate the MN density from Ri to Rf, zi to zf to obtain the mass contained in the cylinder defined by Ri, Rf and +-zi, +-zf
    :Ri, Rf -> radial boundary of the cylinder
    :zi, zf -> vertical boundary of the cylinder
    :a, b, logM -> parameters of the MN density profile
    returns: mass contained in the cylinder
    '''
    
    def lower_boundary(x): #helper function
        return Ri
    
    def upper_boundary(x): #helper function
        return Rf
    
    sol = dblquad(MN_rho, zi, zf, lower_boundary, upper_boundary, args = (a, b, logM))
    mass = 2*np.pi*sol[0] #2pi comes from the theta integral. Assuming axisymmetry
    
    return 2*mass #since we are considering the cylinder to be symmetric about the xy plane

def exp_rho(R, z, H, z0, logM):
    '''
    :returns the exponential mass density at (R, z)
    :inputs:
        R -> radial coordinate
        z -> vertical coordinate
        H -> radial length scale
        z0 -> vertical length scale
        logM -> log of the stellar mass of the disk
    '''    
    A = (10**logM)/(4*np.pi*(abs(H)**2)*abs(z0))
    B = abs(R)/H
    C = abs(z)/z0
    
    rho = A*np.exp(-1*B)*((np.cosh(C))**(-2))
    
    return rho    
    
def get_fit_residual(actual_values, fit_values, err_actual_values, err_fit_values):
	'''
	:returns the residual of a fit as well as the 1 sigma error bar on the residual
	:inputs:
		actual_values -> the actual values of the quantity
		fit_values -> the values of the quantity obtained by model fitting
		err_actual_values -> 1 sigma error on the actual values
		err_fit_values -> 1 sigma error on the fit values
	'''
	
	res = fit_values - actual_values
	err_res = np.sqrt(err_fit_values**2 + err_actual_values**2)
	
	return res, err_res

def get_cylindrical_coordinates(cart_pos):
    #Converts cartesian coordinates to cylindrical
    #cart_pos should be a 3*n_particles matrix of cartesian positions of each particle
    #returns a 3*n_particles matrix of (R, theta, Z) coordinates
    
    x = cart_pos[0, :]
    y = cart_pos[1, :]
    z = cart_pos[2, :]
    
    R = np.sqrt(x**2 + y**2) #radial coordinate
    theta = np.arctan(y/x) #polar angle in radian
    Z = z #vertical coordinate
    
    cyl_pos = np.array([R, theta, Z])
    
    return cyl_pos

def get_counts_cylinder(pos, Ri, Rf, zi, zf):
    
    '''
    get particle counts in a cylindrical volume defined by Ri, Rf and +-zi, +-zf. Assumes disk to be aligned in xy plane and COM coincides with origin
    :pos -> 3*n_particles array of cartesian coordinates of star particles
    :Ri, Rf -> radial boundary of the cylinder
    :zi, zf -> vertical boundary of the cylinder
    return: counts
    '''
    
    #convert cartesian coordinates to cylindrical
    cyl_pos = get_cylindrical_coordinates(pos)
    
    R, theta, z = cyl_pos[0, :], cyl_pos[1, :], cyl_pos[2, :]
    
    #define cylinder
    in_cylinder = np.logical_and(np.logical_and(R >= Ri, R <= Rf), 
                              np.logical_and(z >= zi, z <= zf))
    
    #count particles
    counts = np.sum(in_cylinder)
    
    return 2*counts #since we are considering the cylinder symmetric about the xy plane
 
#Other miscellaneous functions


#def log_rad_density(r, logrho0, H): 
#    '''
#    :returns the log of the radial density
#    :inputs:
#	r -> radial coordinate
#	logrho0 -> log of the central density
#	H -> radial scale length
#    '''
#    return logrho0 - (r/H)*(1/np.log(10))
#
#def log_vert_density(z, logrho0, z0):
#    '''
#    :returns the log of the vertical density
#    :inputs:
#	z -> vertical coordinate
#	logrho0 -> log of the central density
#	z0 -> radial scale length
#    '''    
#    return logrho0 + 2*np.log10((np.cosh(z/z0))**(-1))
#
#def exp_potential(R, z, Rd, z0, Sigma0, G):
#    '''
#    :returns the exponential potential at (R, z)
#    :inputs:
#	R -> radial coordinate
#	z -> vertical coordinate
#	Rd -> radial length scale
#	z0 -> vertical length scale
#	Sigma0 -> Constant surface density multiplier
#	G -> gravitational constant (give in appropriate units)
#    '''    
#    
#    A = 2*np.pi*G*Sigma0
#    B = np.exp(-R/Rd)
#    C = z0*np.exp(-abs(z)/z0)
#    D = abs(z)
#    
#    return A*B*(C + D)
#
#def rho_exp_pot(R, z, Rd, z0, rho_d):
#    '''
#    :returns the mass density corresponding to the exponential potential profile at (R, z)
#    :inputs:
#	R -> radial coordinate
#	z -> vertical coordinate
#	Rd -> radial length scale
#	z0 -> vertical length scale
#	rho_d -> mass density at (Rd, 0)
#    '''        
#    
#    A = rho_d/np.e
#    B = np.exp(-R/Rd)
#    C = np.exp(-abs(z)/z0)
#    D = z0/Rd
#    E = z0/R
#    F = abs(z)/R
#    G = (R/Rd) - 1
#    
#    return A*B*(C + (D*E*C + D*F)*G)
