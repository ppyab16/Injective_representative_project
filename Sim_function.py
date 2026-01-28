# Sim_function.py  (updated simulator)
import os
import sys
import numpy as np
from scipy.constants import c, e, m_e, m_p, epsilon_0

from fbpic.main import Simulation
from fbpic.lpa_utils.laser import add_laser, add_laser_pulse
# FromLasyFileLaser imported on demand (see below)
from fbpic.lpa_utils.boosted_frame import BoostConverter
from fbpic.openpmd_diag import (
    FieldDiagnostic, ParticleDiagnostic,
    BackTransformedFieldDiagnostic, BackTransformedParticleDiagnostic,
    set_periodic_checkpoint, restart_from_checkpoint
)

# Keep your existing sys.path usage if you need the local module import
sys.path.append('/g/g92/bennett83/Convergence')
from Optimisation_classes import T_interaction, t_interaction_non_rel

def simulator(box, plasma, boost, N_lab_diag, write_period, write_dir, track_Electrons,
              n_order, length, Lasy_t_start=0, insight=None, Laser=None, profile=None, dens_func=None,
              use_restart=False, checkpoint_period=None, checkpoint_dir=None,
              lasy_file=None, Lasy_z0=None):
    """
    Runs an FBPIC simulation with optional LASY input and checkpoint/restart support.

    New optional args:
      - use_restart (bool): if True, load the latest checkpoint from `checkpoint_dir` and continue
      - checkpoint_period (int): if provided (>0), set periodic checkpoints every N steps
      - checkpoint_dir (str): directory in which checkpoints are stored. Defaults to write_dir + '/checkpoints'
      - lasy_file (str): path to a LASY HDF5 file to build the laser profile with FromLasyFileLaser
      - lasy_z0 (float): (optional) Initialisation z position of the antenna

    Notes:
      - Exactly one of (Laser, insight, lasy_file) must be provided.
      - Exactly one of (profile, dens_func) must be provided.
    """

    print("Passed boost:", boost)

    # --- argument checks ---
    laser_sources = [Laser is not None, insight is not None, lasy_file is not None]
    if sum(laser_sources) != 1:
        raise ValueError("Exactly one of 'Laser', 'insight', or 'lasy_file' must be provided.")

    if (dens_func is None) == (profile is None):
        raise ValueError("Exactly one of 'dens_func' or 'profile' must be provided.")

    # default checkpoint dir
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(write_dir, "checkpoints")

    # If a LASY file path is given, create the laser-profile object (FromLasyFileLaser)
    if lasy_file is not None:
        try:
            from fbpic.lpa_utils.laser import FromLasyFileLaser
            print("using LASY pulse")
        except Exception as exc:
            raise ImportError(
                "Failed to import FromLasyFileLaser from fbpic.lpa_utils.laser. "
                "Ensure your fbpic installation exposes FromLasyFileLaser. "
                "Original error: {}".format(exc)
            )
        tau = (40e-15)/np.sqrt(2*np.log(2))
        insight = FromLasyFileLaser(lasy_file, t_start=Lasy_t_start*tau)
        # now insight is set; antenna method will be used below

    # Instantiate the Simulation object (but do NOT set the moving window if we will restart)
    if boost == 1:
        sim = Simulation(box.Nz, box.zmax, box.Nr, box.rmax, box.Nm, box.dt,
                         zmin=box.zmin, n_order=n_order, n_damp={'r': 128, 'z': 64},
                         use_cuda=True, boundaries={'z': 'open', 'r': 'open'},
                         particle_shape='cubic')
        # we will call sim.set_moving_window(...) later unless not restarting
    else:
        # create BoostConverter instance (if not already created)
        if not isinstance(boost, BoostConverter):
            Boost = BoostConverter(boost)
        print("creating sim")
        # Pass gamma_boost and v_comoving to Simulation constructor (this does not initialize moving window)
        sim = Simulation(box.Nz, box.zmax, box.Nr, box.rmax, box.Nm, box.dt, zmin=box.zmin,
                         v_comoving=box.v_comoving, gamma_boost=Boost.gamma0,
                         n_order=n_order, n_damp={'r': 128, 'z': 64}, use_cuda=True,
                         boundaries={'z': 'open', 'r': 'open'}, particle_shape='cubic')
        print("sim object complete")

    N_lab_diag = int(N_lab_diag)

    # Compute interaction time & diagnostic period depending on boost
    if boost == 1:
        #t_interact, dt_lab_diag_period = t_interaction_non_rel(box, length, N_lab_diag)
        t_interact = t_interaction_non_rel(box, length, N_lab_diag)
        Nsteps = int(np.round(t_interact / sim.dt))
        dt_period = np.floor(Nsteps/(N_lab_diag-1))
        print("dt_period = " + str(dt_period))
        #dt_period = 333
    else:
        Boost = BoostConverter(boost)
        t_interact, dt_lab_diag_period = T_interaction(box, length, N_lab_diag, Boost)

        print('Lab-frame diagnostic period (time):', dt_lab_diag_period)

    # --- Add species placeholders (must exist before restart_from_checkpoint) ---
    # Get densities (doping)
    n_N, n_He = plasma.doping_densities()
    dens_func_to_use = profile.dens_func if dens_func is None else dens_func

    atoms_He = sim.add_new_species(q=e, m=4*m_p, n=n_He, dens_func=dens_func_to_use,
                                   p_nz=plasma.nz, p_nr=plasma.nr, p_nt=plasma.ntheta,
                                   p_zmin=plasma.p_zmin, p_rmax=plasma.p_rmax)
    atoms_N = sim.add_new_species(q=5 * e, m=14. * m_p, n=n_N, dens_func=dens_func_to_use,
                                  p_nz=plasma.nz, p_nr=plasma.nr, p_nt=plasma.ntheta,
                                  p_zmin=plasma.p_zmin, p_rmax=plasma.p_rmax)

    electron_number_density = 5 * n_N + n_He
    elec = sim.add_new_species(q=-e, m=m_e, n=electron_number_density, dens_func=dens_func_to_use,
                               p_nz=plasma.nz, p_nr=plasma.nr, p_nt=plasma.ntheta,
                               p_zmin=plasma.p_zmin, p_rmax=plasma.p_rmax)

    elecNIonised = sim.add_new_species(q=-e, m=m_e)  # empty placeholder; restart will fill arrays if present

    # make ionizable as in original
    atoms_He.make_ionizable('He', target_species=elec, level_start=1)
    atoms_N.make_ionizable('N', target_species=elecNIonised, level_start=5)

    if track_Electrons:
        elecNIonised.track(sim.comm)

    # --- Restart path: if requested, load the checkpoint into the Simulation object ---
    if use_restart:
        print("Restart requested. Loading checkpoint from:", checkpoint_dir)
        # restart_from_checkpoint fills sim.iteration and sim.time etc.
        restart_from_checkpoint(sim, iteration=None, checkpoint_dir=checkpoint_dir)
        print("Restarted: current sim.iteration =", getattr(sim, 'iteration', None),
              "sim.time =", getattr(sim, 'time', None))

        # Now set the moving window (after restart)
        if boost == 1:
            sim.set_moving_window(v=box.v_window)
        else:
            # compute boosted v_window
            v_window_boosted, = Boost.velocity([box.v_window])
            sim.set_moving_window(v=v_window_boosted)

        # Reattach diagnostics (they are not restored automatically)
        #ParticleDiagnostic(period=N_lab_diag, species={"Electrons from Ionised N": elecNIonised},
        #                         select={'uz': [0., None]}, comm=sim.comm, write_dir=write_dir)
        if boost == 1:
            sim.diags = [
                FieldDiagnostic(dt_period=dt_period, fieldtypes=["E, rho"],
                                fldobject=sim.fld, comm=sim.comm, write_dir=write_dir),
                ParticleDiagnostic(dt_period=dt_period, species={"Electrons from Ionised N": elecNIonised},
                                   select={'uz': [0., None]}, comm=sim.comm, write_dir=write_dir)
            ]
        else:
            sim.diags = [
                BackTransformedFieldDiagnostic(box.zmin, box.zmax, box.v_window,
                                               dt_lab_diag_period, N_lab_diag, Boost.gamma0,
                                               fieldtypes=['E, rho'], period=write_period,
                                               fldobject=sim.fld, comm=sim.comm, write_dir=write_dir),
                BackTransformedParticleDiagnostic(box.zmin, box.zmax, box.v_window,
                                                  dt_lab_diag_period, N_lab_diag, Boost.gamma0,
                                                  write_period, sim.fld, select={'uz': [0., None]},
                                                  species={"Electrons from Ionised N": elecNIonised}, comm=sim.comm,
                                                  write_dir=write_dir)
            ]

        # If the original run used an antenna (i.e. `insight` / LASY), reattach it so emission continues.
        # We only re-add the antenna for `insight` (or lasy_file) cases. For Gaussian direct injection we do not re-add.
        if insight is not None:
            # add antenna. Use gamma_boost if available for boosted frame
            if boost == 1:
                add_laser_pulse(sim, insight, method='antenna', z0_antenna=Lasy_z0)
            else:
                add_laser_pulse(sim, insight, method='antenna', z0_antenna=Lasy_z0, gamma_boost=Boost.gamma0)

        # Re-enable periodic checkpointing if requested
        '''
        checkpoint_period = None
        if checkpoint_period is not None and checkpoint_period > 0:
            print("Registering periodic checkpoints (restart) -> period:", checkpoint_period)
            set_periodic_checkpoint(sim, checkpoint_period, checkpoint_dir)
        '''

        # Determine remaining steps
        total_steps = int(np.round(t_interact / sim.dt))
        print(total_steps)
        steps_done = int(getattr(sim, 'iteration', 0))
        N_step_remaining = max(total_steps - steps_done, 0)
        print("Total steps:", total_steps, "Already done:", steps_done, "Remaining:", N_step_remaining)
        if N_step_remaining > 0:
            sim.step(N_step_remaining)
        else:
            print("Checkpoint already at/after intended end of simulation. Nothing to run.")
        return dt_lab_diag_period

    # no restart
    # Now set moving window
    if boost == 1:
        sim.set_moving_window(v=box.v_window)
    else:
        v_window_boosted, = Boost.velocity([box.v_window])
        sim.set_moving_window(v=v_window_boosted)

    # Add laser depending on whether or not insight is defined
    if insight is None:
        if boost == 1:
            add_laser(sim, Laser.a0, Laser.w0, Laser.ctau, Laser.z0, lambda0=Laser.lambda0, zf=Laser.zfoc)
        else:
            add_laser(sim, Laser.a0, Laser.w0, Laser.ctau, Laser.z0, lambda0=Laser.lambda0,
                      zf=Laser.zfoc, gamma_boost=Boost.gamma0)
    else:
        # insight provided, using LASY file + antenna
        if boost == 1:
            add_laser_pulse(sim, insight, method='antenna', z0_antenna=Lasy_z0)
        else:
            add_laser_pulse(sim, insight, method='antenna', z0_antenna=Lasy_z0, gamma_boost=Boost.gamma0)

    # diagnostics for no restart
    if boost == 1:
        print("using lab frame diagnostics")
        sim.diags = [
            FieldDiagnostic(period=dt_period, fieldtypes=["E", "rho"],
                            fldobject=sim.fld, comm=sim.comm, write_dir=write_dir),
            ParticleDiagnostic(period=dt_period, species={"Electrons from Ionised N": elecNIonised},
                               select={'uz': [0., None]}, comm=sim.comm, write_dir=write_dir)
        ]
    else:
        print("using boosted frame diagnostics")
        sim.diags = [
            BackTransformedFieldDiagnostic(box.zmin, box.zmax, box.v_window,
                                           dt_lab_diag_period, N_lab_diag, Boost.gamma0,
                                           fieldtypes=['E', 'rho'], period=write_period,
                                           fldobject=sim.fld, comm=sim.comm, write_dir=write_dir),
            BackTransformedParticleDiagnostic(box.zmin, box.zmax, box.v_window,
                                              dt_lab_diag_period, N_lab_diag, Boost.gamma0,
                                              write_period, sim.fld, select={'uz': [0., None]},
                                              species={"Electrons from Ionised N": elecNIonised}, comm=sim.comm,
                                              write_dir=write_dir)
        ]

    # Periodic checkpointing (fresh run)
    '''
    if checkpoint_period is not None and checkpoint_period > 0:
        print("Setting periodic checkpoints every {} iterations (dir={})".format(checkpoint_period, checkpoint_dir))
        set_periodic_checkpoint(sim, checkpoint_period, checkpoint_dir)
    '''
    # Run full simulation
    N_step = int(np.round(t_interact / sim.dt))
    print("Total simulation steps:", N_step)
    sim.step(N_step)
    print('Sim Complete')
    return 
