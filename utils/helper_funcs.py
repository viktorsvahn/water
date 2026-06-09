import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ase.units as units
from tol_colors import data



def load_data(path_dict, fname, root_dir, cache_path=None):
    """
    Load data files from a set of directories.

    Each entry in ``path_dict`` is searched for a file matching
    ``fname``. Wildcards are allowed, but exactly one file must match
    per directory.
    """
    import glob
    import os

    import pandas as pd
    from ase.io import read

    loaded_data = {}

    for key, path in path_dict.items():

        # Construct full search pattern.
        pattern = os.path.join(root_dir + path, fname)

        # Use glob only when the filename contains wildcards.
        if "*" in fname or "?" in fname:
            matches = sorted(glob.glob(pattern))
        else:
            matches = [pattern] if os.path.isfile(pattern) else []

        if len(matches) == 0:
            raise FileNotFoundError(
                f"No files match '{fname}' in '{root_dir + path}'."
            )

        if len(matches) > 1:
            raise ValueError(
                f"Multiple files match '{fname}' in '{root_dir + path}':\n"
                + "\n".join(matches)
            )

        file_path = matches[0]

        print(f"Loading data for '{key}' from '{file_path}'")

        if file_path.endswith(".csv"):
            loaded_data[key] = pd.read_csv(file_path, sep=";")

        elif file_path.endswith((".pdb", ".xyz", ".traj")):
            loaded_data[key] = read(file_path, ":")

        else:
            raise ValueError(
                f"Unsupported file type: '{file_path}'."
            )
        """
        if cache_path is not None and file_path.endswith((".pdb", ".xyz", ".traj")):
            import zstandard as zstd
            import pickle

            with open(cache_path + ".zst", "wb") as f:
                cctx = zstd.ZstdCompressor(level=10)
                with cctx.stream_writer(f) as compressor:
                    pickle.dump(
                        loaded_data,
                        compressor,
                        protocol=pickle.HIGHEST_PROTOCOL
                    )
        """
    return loaded_data


def create_thermo_df(data_dict):
    """
    Convert:

        {system: {property: {mean, stderr, n_blocks}}}

    into a styled MultiIndex DataFrame.

    Returns
    -------
    pandas.io.formats.style.Styler
    """
    import pandas as pd
    import numpy as np

    records = []

    def format_number(x):
        """
        Use regular notation for exponents within ±3,
        otherwise scientific notation.
        """
        if x == 0:
            return "0.000000"

        exponent = np.floor(np.log10(np.abs(x)))

        if -3 <= exponent <= 3:
            return f"{x:.6f}"

        return f"{x:.6e}"

    for system, props in data_dict.items():

        for prop, stats in props.items():

            # Block-averaged result
            if isinstance(stats, dict):

                records.append({
                    "system": system,
                    "property": prop,
                    "mean": stats.get("mean"),
                    "stderr": stats.get("stderr"),
                    "n_blocks": stats.get("n_blocks"),
                })

            # Scalar result
            else:

                records.append({
                    "system": system,
                    "property": prop,
                    "mean": stats,
                    "stderr": np.nan,
                    "n_blocks": np.nan,
                })

    df = pd.DataFrame(records)

    df = (
        df
        .set_index(["system", "property"])
        .sort_index()
    )

    return (
        df.style
        .format({
            "mean": format_number,
            "stderr": format_number,
            "n_blocks": "{:.0f}",
        })
        .set_properties(**{
            "text-align": "center"
        })
    )


def cached_thermo_props(
    cache_file,
    paths_dict,
    data_path,
    temperature,
    correlation_times,
    block_factor=10,
):
    """
    Load or compute thermodynamic properties
    with caching.

    Parameters
    ----------
    cache_file : str
        JSON cache filename.

    paths_dict : dict
        Mapping of system names to paths.

    data_path : str
        Root directory for trajectory data.

    temperature : float
        Simulation temperature in K.

    correlation_times : dict
        Correlation times (in frames) for each property.

    block_factor : int, default=10
        Multiplier used to determine block sizes:

            block_size = correlation_time * block_factor
    """
    import json
    import os

    import utils.thermo_funcs as tf

    # Load cached data
    if os.path.exists(cache_file):

        with open(cache_file, "r") as f:

            print(
                f"Loaded thermodynamic properties "
                f"from {cache_file}"
            )

            thermo_props = json.load(f)

    else:

        print(
            f"{cache_file} not found. "
            f"Computing thermodynamic properties..."
        )

        print("Loading trajectory data...")

        traj_data = load_data(
            paths_dict,
            fname="*.traj",
            root_dir=data_path,
        )

        print("Finished loading trajectory data.")

        print(
            "Computing thermodynamic properties..."
        )

        thermo_props = tf.get_thermo_props(
            traj_data,
            temperature=temperature,
            correlation_times=correlation_times,
            block_factor=block_factor,
        )

        with open(cache_file, "w") as f:

            json.dump(
                thermo_props,
                f,
                indent=2,
            )

        print(
            f"Saved thermodynamic properties "
            f"to {cache_file}"
        )

    return create_thermo_df(thermo_props)


def cached_hop_props(
    cache_file,
    paths_dict,
    data_path,
    correlation_times,
    block_factor=10,
    cutoff=1.25,
):
    """
    Load or compute proton hopping properties with caching.

    Computes:
    - Proton hop rate (hops/ps)
    - Average proton residence time (ps)

    Parameters
    ----------
    cache_file : str
        JSON cache filename.

    paths_dict : dict
        Mapping of system names to paths.

    data_path : str
        Root directory for trajectory data.

    correlation_times : dict
        Correlation times (in frames) for each property.
        Expected keys: 'hop_rate', 'residence_time'

    block_factor : int, default=10
        Multiplier used to determine block sizes:

            block_size = correlation_time * block_factor

    cutoff : float, default=1.25
        Distance cutoff for O-H assignment (Å).
    """
    import json
    import os

    import utils.proton_hop as ph

    # Load cached data
    if os.path.exists(cache_file):

        with open(cache_file, "r") as f:

            print(
                f"Loaded proton hopping properties "
                f"from {cache_file}"
            )

            hop_props = json.load(f)

    else:

        print(
            f"{cache_file} not found. "
            f"Computing proton hopping properties..."
        )

        print("Loading trajectory data...")

        traj_data = load_data(
            paths_dict,
            fname="*.traj",
            root_dir=data_path,
        )

        print("Finished loading trajectory data.")

        print(
            "Computing proton hopping properties..."
        )

        hop_props = {}

        for system_name, traj in traj_data.items():

            print(f"  Processing {system_name}...")

            hop_props[system_name] = {}

            # Hop rate
            if 'hop_rate' in correlation_times:
                hop_props[system_name]['hop_rate'] = (
                    ph.compute_hop_rate_blocks(
                        traj,
                        correlation_time=correlation_times['hop_rate'],
                        block_factor=block_factor,
                        cutoff=cutoff,
                    )
                )

            # Residence time
            if 'residence_time' in correlation_times:
                hop_props[system_name]['residence_time'] = (
                    ph.compute_residence_time_blocks(
                        traj,
                        correlation_time=correlation_times['residence_time'],
                        block_factor=block_factor,
                        cutoff=cutoff,
                    )
                )

        with open(cache_file, "w") as f:

            json.dump(
                hop_props,
                f,
                indent=2,
            )

        print(
            f"Saved proton hopping properties "
            f"to {cache_file}"
        )

    return create_thermo_df(hop_props)
