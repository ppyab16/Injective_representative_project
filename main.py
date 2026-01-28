# runner.py  (updated portion)
import yaml, sys, os
#need to change the path here
sys.path.append('/g/g92/bennett83/TA2')
from Optimisation_classes import Sim_box, Laser_param, plasma_param, density_profile
from Sim_function import simulator
from scipy.constants import c

with open("/g/g92/bennett83/TA2/test.yaml", "r") as f:
    config = yaml.safe_load(f)

MP = config['misc_params']
SB = config['Sim_box']
LP = config['laser_params']
PP = config['plasma_param']
DP = config['density_profile']

Laser = Laser_param(**LP)
box = Sim_box(MP["Nmodes"], MP["boost"], Laser, **SB)
plasma = plasma_param(MP["Nmodes"], box, **PP)

length = DP['up_ramp_length'] + DP['down_ramp_length'] + DP['plateau_length']
profile = density_profile(MP["boost"], **DP)

# optional LASY file handling controlled in YAML
lasy_file = MP.get('lasy_file') if MP.get('use_lasy', False) else None

#'''
if __name__ == '__main__':
    simulator(
        box, plasma, MP["boost"], MP['N_lab_diag'], MP['write_period'],
        MP['write_dir'], MP['track_Electrons'], -1, length, Lasy_t_start=MP['lasy_t_start'],
        insight=None, Laser=None, profile=profile, dens_func=None,
        use_restart=MP.get('restart_from_checkpoint', False),
        checkpoint_period=MP.get('checkpoint_period', None),
        checkpoint_dir=MP.get('checkpoint_dir', None),
        lasy_file=lasy_file,
        Lasy_z0=LP["initialisation_z_position"]
    )
'''

if __name__ == '__main__':
    simulator(
        box, plasma, MP["boost"], MP['N_lab_diag'], MP['write_period'],
        MP['write_dir'], MP['track_Electrons'], -1, length,
        insight=None, Laser=Laser, profile=profile, dens_func=None,
        use_restart=MP.get('restart_from_checkpoint', False),
        checkpoint_period=MP.get('checkpoint_period', None),
        checkpoint_dir=MP.get('checkpoint_dir', None),
    )
'''