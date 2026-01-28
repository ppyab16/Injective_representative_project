import numpy as np
from fbpic.lpa_utils.boosted_frame import BoostConverter
from scipy.constants import c, m_e, epsilon_0, e
'''
Really, there is no real need for the creation of any of these classes since the simulation is not complicated enough
to require class dependent attributes. It just makes the main script slightly less messy and easier for me to read. If
someone in future is reading this and judging it, then yes I agree that this was needless but I hope it is clear.
'''

class Laser_param:
    '''
    This class defines the laser parameters, it requires the following parameters when initialised:
    :param a0: Laser amplitude a0
    :param laser_waist_FWHM: Laser waist FWHM
    :param temporal_pulse_length_FWHM: Laser duration FWHM
    :param initialisation_z_position: The point at which you initialise the laser pulse to. Normally, we want to set this before the front
    tail of the laser pulse interacts with the gas cell since we want to see acceleration stemming from the entire
    laser pulse.
    :param z_focal_position: Focal position (again, need to check what this is)
    :param laser_wavelength: Laser wavelength (metres)
    :param pulse_Energy: The laser pulse energy

    :return: Laser object with all the desired parameters that can be put straight into the simulation class
    '''
    def __init__(self, laser_waist_FWHM, temporal_pulse_length_FWHM, initialisation_z_position, z_focal_position,
                 laser_wavelength, pulse_Energy):
        self.laser_waist_FWHM = laser_waist_FWHM
        self.laser_wavelength = laser_wavelength
        self.initialisation_z_position = initialisation_z_position
        self.temporal_pulse_length_FWHM = temporal_pulse_length_FWHM
        self.pulse_Energy = pulse_Energy
        self.z_focal_position = z_focal_position
        #intensity is required to calculate a0
        self.intensity = 0.83 * self.pulse_Energy / (self.temporal_pulse_length_FWHM * (self.laser_waist_FWHM) ** 2)
        self.a0 = 0.86 * self.laser_wavelength * 1e6 * np.sqrt(self.intensity * 1e-22)
        self.z0 = self.initialisation_z_position
        self.zfoc = self.z_focal_position
        self.lambda0 = self.laser_wavelength
        #conversion from FWHM to 1/e^2
        self.ctau = self.temporal_pulse_length_FWHM * c / (np.sqrt(2 * np.log(2)))
        self.w0 = self.laser_waist_FWHM / (np.sqrt(2 * np.log(2)))


class Sim_box2:
    '''
    This class defines the simulation box. The box needs to be the right size to capture the entire interaction, 
    without being too big so that computation time slows down. Convergence scanning will be done in order to
    determine the correct size of the box.
    '''
    def __init__(self, Nm, boost, laser: Laser_param, density, z_min_plasmaWLscaling, rmax_rbscaling, Nr_rbscaling,
                 dz_lambda0scaling):
        '''
        :param Nm: Number of modes used (see documentation for more details)
        :param boost: the desired value for the Lorentz boost. The higher the value, the faster the simulation will
        run but you risk numerical heating when it the lorentz boost is too high. See the fbpic documentation for
        more details
        :param laser: An instance of the laser class
        :param density: The plasma number density
        :param z_min_plasmaWLscaling: Zmin in units of plasma wave length (calculated analytically)
        :param rmax_rbscaling: Rmax in units of the bubble radius (calculated analytically)
        :param Nr_rbscaling: Number of particles per cell in r in units of bubble radius (calculated analytically)
        :param dz_lambda0scaling: dz expressed as a fraction of the laser wavelength. 32-40 is the standard default
        for high resolution, going down to 16 for high speed. Of course convergence scanning is still required.
        '''
        self.plasmaFreq = np.sqrt((density * e ** 2) / (m_e * epsilon_0))
        print(self.plasmaFreq)
        self.plasmaWL = 2 * np.pi * c / self.plasmaFreq
        print(self.plasmaWL)
        self.a0 = laser.a0
        self.rb = rb = 2*np.sqrt(self.a0)*c/self.plasmaFreq
        print(self.rb)

        self.zmax = 0.5e-5
        self.zmin = self.zmax - 1 * self.plasmaWL * z_min_plasmaWLscaling
        self.rmax = self.rb * rmax_rbscaling
        print(self.rmax)
        self.dr = rb / Nr_rbscaling
        self.Nr = int(np.round(self.rmax/self.dr))
        print(self.Nr)
        self.Nm = Nm
        self.dz = laser.lambda0 / dz_lambda0scaling
        self.Nz = int(np.round((self.zmax - self.zmin) / self.dz))
        self.n = density
        self.v_window = 1 * c *  np.sqrt(1 - self.n / 1.75e27)
        if boost == 1:
            self.dt = min(self.rmax / (2 * self.Nr) / c, (self.zmax - self.zmin) / self.Nz / c)
        else:
            self.Boost = BoostConverter(boost)
            self.v_comoving = - c * np.sqrt(1. - 1. / (self.Boost.gamma0 ** 2))
            self.dt = min(self.rmax / (2 * self.Boost.gamma0 * self.Nr) / c, (self.zmax - self.zmin) / self.Nz / c)
            
class Sim_box:
    '''
    This class defines the simulation box. The box needs to be the right size to capture the entire 
    interaction, without being too big so that computation time slows down. Convergence scanning will
    be done in order to determine the correct size of the box.
    '''

    def __init__(self, Nm, boost, laser: Laser_param, density, z_min_LaserWLscaling, rmax_spotscaling, 
                 Nr_spotscaling, dz_lambda0scaling):
        '''
        :param Nm: Number of modes used (see documentation for more details)
        :param boost: the desired value for the Lorentz boost. The higher the value, the faster the simulation 
        will run but you risk numerical heating when it the lorentz boost is too high. See the fbpic 
        documentation for more details
        :param laser: An instance of the laser class
        :param density: The plasma number density
        :param z_min_LaserWLscaling: Zmin in units of laser wavelength
        :param rmax_spotscaling: Rmax in units of the spot size
        :param Nr_spotscaling: Number of particles per cell in r in units of spot size
        :param dz_lambda0scaling: dz expressed as a fraction of the laser wavelength. 32-40 is the standard 
        default for high resolution, going down to 16 for high speed. Of course convergence scanning is still 
        required.
        '''
        self.zmax = -0.0e-5
        self.zmin = self.zmax - 1 * laser.lambda0 * z_min_LaserWLscaling
        self.rmax = laser.w0 * rmax_spotscaling
        print(self.rmax)
        self.dr = laser.w0 / Nr_spotscaling
        self.Nr = int(np.round(self.rmax/self.dr))
        print(self.Nr)
        self.Nm = Nm
        self.dz = laser.lambda0 / dz_lambda0scaling
        self.Nz = int(np.round((self.zmax - self.zmin) / self.dz))
        self.n = density
        self.v_window = 1 * c *  np.sqrt(1 - self.n / 1.75e27)
        if boost == 1:
            self.dt = min(self.rmax / (2 * self.Nr) / c, (self.zmax - self.zmin) / self.Nz / c)
        else:
            self.Boost = BoostConverter(boost)
            self.v_comoving = - c * np.sqrt(1. - 1. / (self.Boost.gamma0 ** 2))
            self.dt = min(self.rmax / (2 * self.Boost.gamma0 * self.Nr) / c, (self.zmax - self.zmin) / self.Nz / c)

            
class density_profile:
    '''
    This class defines the density profile, it requires the following parameters when initialised:
    :boost: the gamma boost factor
    :param up_ramp_length: distance corresponding to linear ramp up in density
    :param plateau_length: distance corresponding to the flat section of the density profile
    :param down_ramp_length: distance corresponding to linear ramp down in density
    '''
    def __init__(self, boost, up_ramp_length, plateau_length, down_ramp_length):
        self.ramp_up = up_ramp_length
        self.plateau = plateau_length
        self.ramp_down = down_ramp_length
        self.boost = BoostConverter(boost)
        self.ramp_up_b, self.plateau_b, self.ramp_down_b = self.boost.static_length([self.ramp_up, self.plateau, self.ramp_down])
    def dens_func(self, z, r):
        '''
        User-defined function: density profile of the plasma

        It should return the relative density with respect to n_plasma,
        at the position x, y, z (i.e. return a number between 0 and 1)
        :param z, r: 1d array of floats with one element per macroparticle corresponding to the longitudinal and
        transverse displacements
        :return: a 1d array of floats containing the relative density with one element per macroparticle
        '''
    # Allocate relative density
        n = np.ones_like(z)
        # Make ramp up (note: use boosted-frame values of the ramp length)
        n = np.where( z<self.ramp_up_b, -(np.cos((np.pi)*(z/self.ramp_up_b))-1)/2, n) #
        n = np.where( z>self.ramp_up_b+self.plateau_b, (np.cos(np.pi*(z-(self.ramp_up_b+self.plateau_b))/self.ramp_down_b)+1)/2, n)
        n = np.where( z >= self.ramp_up_b+self.plateau_b+self.ramp_down_b, 0, n)
        return(n)

class plasma_param:
    '''
    This class defines the plasma profile, it requires the following parameters when initialised:

    Note that the following params govern the position of the plasma particles within the moving window. It is
    advisable to NOT define the plasma as extending to the RADIAL extreme of the cell. Allow some tolerance.
    :param minimum_z_position: The minimum z position in the box that the plasma occupies
    :param maximum_r_position: The maximum radial position in the box that the plasma occupies

    These params determine the number of plasma particles along cylindrical coordinate
    :param particles_per_cell_in_z: Plasma particles per cell along z
    :param particles_per_cell_in_r: Plasma particles per cell along r
    :param particles_per_cell_in_theta: Plasma particles per cell along theta

    :param gas_density: The gas density

    :param N2_percentage_by_mass: The percentage of the Nitrogen in the gas (set to 2 by default)
    '''
    def __init__(self, Nm, box:Sim_box, minimum_z_position, particles_per_cell_in_z, particles_per_cell_in_r,
                 modal_ntheta_scaling, N2_percentage_by_mass):
        #Position of the plasma
        self.p_zmin = minimum_z_position
        self.p_rmax = box.rmax# - box.rmax/10
        #Particles per cell (see documentation)
        self.nz = particles_per_cell_in_z
        self.nr = particles_per_cell_in_r
        self.ntheta = Nm * modal_ntheta_scaling
        #gas density
        self.n = box.n
        #
        self.dp = N2_percentage_by_mass

    #Doping helium gas with 2 percent N2 (ie 2 percent nitrogen by molecule, need to check that it's not by
    #atom)
    #2 percent by mass diatomic nitrogen
    def doping_densities2(self):
        '''
        We just need the gas density which is inputted into the plasma param class. This function returns the
        number densities of the N2 and He
        :return: number densities of N2 and He
        '''
        N2_percentage_by_mass = 1/7
        #aka number density
        N_percentage_by_atom = self.dp*N2_percentage_by_mass
        n_N = self.n*N_percentage_by_atom
        He_percentage_by_atom= 1 - N_percentage_by_atom
        n_He = self.n*He_percentage_by_atom
        return n_N, n_He
    
    def doping_densities(self):
        N_Percentage = float(self.dp)
        if N_Percentage > 1:
            N_Percentage = N_Percentage/100
        N_Percentage_By_Atom = N_Percentage * 4/14 # convert gas percentage by mass to percentage by atom
        n_He = self.n*(1-(N_Percentage_By_Atom)/100)   # Hydrogen density in mixed gas
        n_N = self.n*N_Percentage_By_Atom/100
        return n_N, n_He

def T_interaction(box, L_interact, N_lab_diag, gamma_boost):
    '''
    Function gives the total interaction time and time between diagnostic snapshots
    :param box: an object of the Sim_box class
    :param L_interact: The length of the plasma
    :return T_interact:
    :return dt_lab_diag_period:
    '''
    T_interact = gamma_boost.interaction_time(L_interact, (box.zmax - box.zmin), box.v_window)
    dt_lab_diag_period = (L_interact + (box.zmax - box.zmin)) / box.v_window / (N_lab_diag - 1)

    return T_interact, dt_lab_diag_period

def t_interaction_non_rel(box, L_interact, N_lab_diag):
    T_interact = (L_interact - box.zmin)/box.v_window
    #dt_lab_diag_period = T_interact / (N_lab_diag -1)
    return T_interact#, dt_lab_diag_period

