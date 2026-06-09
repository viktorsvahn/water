import numpy as np
from ase import units


def split_blocks(x, block_size):
    """
    Split a 1D array into blocks.

    Returns
    -------
    ndarray, shape (n_blocks, block_size)
    """
    n_blocks = len(x) // block_size

    if n_blocks < 2:
        raise ValueError(
            f"Need at least 2 blocks, got {n_blocks}. "
            f"Increase trajectory length or decrease block size."
        )

    x = np.asarray(x[: n_blocks * block_size])

    return x.reshape(n_blocks, block_size)


def summarize_blocks(values):
    """
    Compute mean and standard error from block values.
    """
    values = np.asarray(values)

    return {
        "mean": values.mean(),
        "stderr": values.std(ddof=1) / np.sqrt(len(values)),
        "n_blocks": len(values),
    }


def compute_enthalpy_series(traj, pressure=1.0 * units.bar):
    """
    Instantaneous enthalpy time series in eV.
    """
    enthalpy = []

    for atoms in traj:
        energy = (
            atoms.get_potential_energy()
            + atoms.get_kinetic_energy()
        )

        volume = atoms.get_volume()

        enthalpy.append(energy + pressure * volume)

    return np.asarray(enthalpy)


def compute_density(traj, block_size=None):
    """
    Density in g/cm^3.
    """
    volumes = np.array([atoms.get_volume() for atoms in traj])

    mass = traj[0].get_masses().sum()

    density = mass / volumes
    density *= 1.66053906660

    if block_size is None:
        return density.mean()

    blocks = split_blocks(density, block_size)
    block_values = blocks.mean(axis=1)

    return summarize_blocks(block_values)


def compute_molar_enthalpy(traj, block_size=None):
    """
    Molar enthalpy in kJ/mol (per atom basis).
    """
    enthalpy = compute_enthalpy_series(traj)

    n_atoms = len(traj[0])

    enthalpy = (
        enthalpy / n_atoms
        * (units.mol / units.kJ)
    )

    if block_size is None:
        return enthalpy.mean()

    blocks = split_blocks(enthalpy, block_size)
    block_values = blocks.mean(axis=1)

    return summarize_blocks(block_values)


def compute_heat_capacity(
    traj,
    temperature,
    block_size=None,
):
    """
    Cp in kJ/mol/K.
    """
    enthalpy = compute_enthalpy_series(traj)

    k_b = units.kB

    if block_size is None:

        cp = (
            np.var(enthalpy, ddof=1)
            / (k_b * temperature**2)
        )

        # Convert eV/K to kJ/mol/K
        cp *= units.mol / units.kJ

        return cp

    cp_blocks = []

    for block in split_blocks(enthalpy, block_size):

        cp = (
            np.var(block, ddof=1)
            / (k_b * temperature**2)
        )

        # Convert eV/K to kJ/mol/K
        cp *= units.mol / units.kJ

        cp_blocks.append(cp)

    return summarize_blocks(np.array(cp_blocks))


def compute_compressibility(
    traj,
    temperature,
    block_size=None,
):
    """
    Isothermal compressibility in 1/Pa.
    """
    volumes = np.array(
        [atoms.get_volume() for atoms in traj]
    )

    k_b = units.kB
    
    # Conversion factor: 1 eV/Ų = 1.602176634e11 Pa (SI conversion)
    pressure_unit = 1.602176634e11

    if block_size is None:

        v_mean = volumes.mean()

        kappa = (
            np.var(volumes, ddof=1)
            / (k_b * temperature * v_mean)
        )
        
        return kappa / pressure_unit

    kappa_blocks = []

    for block in split_blocks(volumes, block_size):

        v_mean = block.mean()

        kappa = (
            np.var(block, ddof=1)
            / (k_b * temperature * v_mean)
        ) / pressure_unit

        kappa_blocks.append(kappa)

    return summarize_blocks(kappa_blocks)


def compute_bulk_modulus(
    traj,
    temperature,
    block_size=None,
):
    """
    Bulk modulus in GPa.
    """
    volumes = np.array(
        [atoms.get_volume() for atoms in traj]
    )

    k_b = units.kB
    
    # Conversion factor: 1 eV/Ų = 1.602176634e11 Pa (SI conversion)
    pressure_unit = 1.602176634e11

    if block_size is None:

        v_mean = volumes.mean()

        kappa = (
            np.var(volumes, ddof=1)
            / (k_b * temperature * v_mean)
        )

        # B = 1/κT, convert from Pa to GPa
        return (pressure_unit / kappa) / 1e9

    bulk_blocks = []

    for block in split_blocks(volumes, block_size):

        v_mean = block.mean()

        kappa = (
            np.var(block, ddof=1)
            / (k_b * temperature * v_mean)
        )

        # B = 1/κT, convert from Pa to GPa
        bulk_blocks.append((pressure_unit / kappa) / 1e9)

    return summarize_blocks(bulk_blocks)


def compute_thermal_expansion(
    traj,
    temperature,
    block_size=None,
):
    """
    Thermal expansion coefficient in 1/K.
    """
    volumes = np.array(
        [atoms.get_volume() for atoms in traj]
    )

    enthalpy = compute_enthalpy_series(traj)

    k_b = units.kB

    if block_size is None:

        d_v = volumes - volumes.mean()
        d_h = enthalpy - enthalpy.mean()

        cov = np.mean(d_v * d_h)

        return (
            cov
            / (
                k_b
                * temperature**2
                * volumes.mean()
            )
        )

    alpha_blocks = []

    volume_blocks = split_blocks(
        volumes,
        block_size,
    )

    enthalpy_blocks = split_blocks(
        enthalpy,
        block_size,
    )

    for v_block, h_block in zip(
        volume_blocks,
        enthalpy_blocks,
    ):

        d_v = v_block - v_block.mean()
        d_h = h_block - h_block.mean()

        cov = np.mean(d_v * d_h)

        alpha = (
            cov
            / (
                k_b
                * temperature**2
                * v_block.mean()
            )
        )

        alpha_blocks.append(alpha)

    return summarize_blocks(alpha_blocks)


def get_thermo_props(
    data_dict,
    temperature=350.0,
    block_sizes=None,
):
    """
    Parameters
    ----------
    block_sizes : dict or None

    Example
    -------
    block_sizes = {
        "density": 100,
        "enthalpy": 500,
        "heat_capacity": 1000,
        "compressibility": 1000,
        "bulk_modulus": 1000,
        "thermal_expansion": 1000,
    }
    """
    block_sizes = block_sizes or {}

    results = {}

    for key, traj in data_dict.items():

        results[key] = {
            "Density /g cm-3": compute_density(
                traj,
                block_sizes.get("density"),
            ),
            #"Enthalpy /kJ mol-1": compute_molar_enthalpy(
            #    traj,
            #    block_sizes.get("enthalpy"),
            #),
            "Heat Capacity /kJ mol-1 K-1": compute_heat_capacity(
                traj,
                temperature,
                block_sizes.get("heat_capacity"),
            ),
            "Compressibility /Pa-1": compute_compressibility(
                traj,
                temperature,
                block_sizes.get("compressibility"),
            ),
            "Bulk Modulus /GPa": compute_bulk_modulus(
                traj,
                temperature,
                block_sizes.get("bulk_modulus"),
            ),
            "Thermal Expansion /K-1": compute_thermal_expansion(
                traj,
                temperature,
                block_sizes.get("thermal_expansion"),
            ),
        }

    return results
