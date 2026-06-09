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

    into a MultiIndex DataFrame:
        index = (system, property)
        columns = mean, stderr, n_blocks
    """
    import pandas as pd
    import numpy as np
    records = []


    def format_number(x):
        """Format number using regular notation if exponent is within ±3, else scientific."""
        if x == 0:
            return '0.000000'
        
        exponent = np.floor(np.log10(np.abs(x)))
        
        # Use regular notation for exponents between -3 and +3
        if -3 <= exponent <= 3:
            return f'{x:.6f}'
        else:
            return f'{x:.6e}'


    for system, props in data_dict.items():
        for prop, stats in props.items():

            # allow both raw scalars and dict outputs
            if isinstance(stats, dict):
                records.append({
                    "system": system,
                    "property": prop,
                    **stats
                })
            else:
                records.append({
                    "system": system,
                    "property": prop,
                    "mean": stats,
                    "stderr": 0.0,
                    "n_blocks": 1,
                })

    df = pd.DataFrame(records)
    df = df.set_index(["system", "property"]).sort_index()
    
    # Display with styling for better readability
    return df.style.format({
        'mean': format_number,
        'stderr': format_number,
    }).set_properties(**{'text-align': 'center'})


def cached_thermo_props(cache_file, paths_dict, data_path, temperature, block_sizes):
    """
    Load or compute thermodynamic properties with caching and return styled DataFrame.
    """
    import json
    import os
    import utils.thermo_funcs as tf
    
    # Try to load from cache
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            print(f'Loaded thermodynamic properties from {cache_file}')
            thermo_props = json.load(f)
    else:
        # Compute if not cached
        print(f'{cache_file} not found. Computing thermodynamic properties...')
        print('Loading trajectory data...')
        traj_data = load_data(paths_dict, fname='*.traj', root_dir=data_path)
        print('Finished loading trajectory data.')
        
        print('Computing thermodynamic properties...')
        thermo_props = tf.get_thermo_props(
            traj_data,
            temperature=temperature,
            block_sizes=block_sizes,
        )
        
        # Save to cache
        with open(cache_file, 'w') as f:
            json.dump(thermo_props, f, indent=2)
        print(f'Saved thermodynamic properties to {cache_file}')
    
    # Return styled DataFrame
    return create_thermo_df(thermo_props)