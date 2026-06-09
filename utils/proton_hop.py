import numpy as np
from ase.io import Trajectory


# =============================================================================
# CONFIGURATION
# =============================================================================

O_SYMBOL = "O"
H_SYMBOL = "H"
CUTOFF = 1.25  # Å, adjust depending on system


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def get_indices(atoms):
    """
    Extract O and H atom indices from an ASE atoms object.
    """
    symbols = atoms.get_chemical_symbols()
    o_idx = [i for i, s in enumerate(symbols) if s == O_SYMBOL]
    h_idx = [i for i, s in enumerate(symbols) if s == H_SYMBOL]
    return o_idx, h_idx


def assign_protons(atoms, o_idx, h_idx, cutoff=CUTOFF):
    """
    Return dict: H index -> O index (ownership).
    
    Parameters
    ----------
    atoms : ase.Atoms
        Structure for a single frame.
    o_idx : list
        Indices of oxygen atoms.
    h_idx : list
        Indices of hydrogen atoms.
    cutoff : float
        Distance cutoff for O-H assignment (Å).
        
    Returns
    -------
    dict
        Mapping of H index -> closest O index (or None if unbound).
    """
    pos = atoms.get_positions()
    mapping = {}

    for h in h_idx:
        dists = []
        for o in o_idx:
            d = np.linalg.norm(pos[h] - pos[o])
            if d < cutoff:
                dists.append((d, o))

        if len(dists) > 0:
            _, o_best = min(dists)
            mapping[h] = o_best
        else:
            mapping[h] = None

    return mapping


def detect_hop_events(traj, o_symbol=O_SYMBOL, h_symbol=H_SYMBOL, cutoff=CUTOFF):
    """
    Detect proton hopping events across a trajectory.
    
    Parameters
    ----------
    traj : list of ase.Atoms
        Trajectory frames.
    o_symbol : str
        Element symbol for oxygen.
    h_symbol : str
        Element symbol for hydrogen.
    cutoff : float
        Distance cutoff for O-H assignment (Å).
        
    Returns
    -------
    dict
        Contains:
        - 'hop_events': List of (frame, h_idx, from_o_idx, to_o_idx)
        - 'residence': Dict mapping h_idx -> list of [start_frame, o_idx]
    """
    previous = None
    hop_events = []
    residence = {}

    for t, atoms in enumerate(traj):
        o_idx, h_idx = get_indices(atoms)
        current = assign_protons(atoms, o_idx, h_idx, cutoff=cutoff)

        if previous is None:
            # Initialize residence tracking
            for h, o in current.items():
                residence.setdefault(h, [])
                residence[h].append([t, o])
        else:
            for h in h_idx:
                prev_o = previous.get(h, None)
                curr_o = current.get(h, None)

                # Detect hop
                if prev_o is not None and curr_o is not None:
                    if prev_o != curr_o:
                        hop_events.append((t, h, prev_o, curr_o))

                # Update residence tracking
                if h not in residence:
                    residence[h] = [[t, curr_o]]
                else:
                    last_o = residence[h][-1][1]
                    if curr_o != last_o:
                        residence[h].append([t, curr_o])

        previous = current

    return {
        'hop_events': hop_events,
        'residence': residence,
    }


def compute_hop_rate(traj, o_symbol=O_SYMBOL, h_symbol=H_SYMBOL, cutoff=CUTOFF):
    """
    Compute proton hop rate (hops per picosecond).
    
    Assumes trajectory has constant timestep (1 fs per frame).
    
    Parameters
    ----------
    traj : list of ase.Atoms
        Trajectory frames.
    o_symbol : str
        Element symbol for oxygen.
    h_symbol : str
        Element symbol for hydrogen.
    cutoff : float
        Distance cutoff for O-H assignment (Å).
        
    Returns
    -------
    float
        Hop rate in hops/ps.
    """
    data = detect_hop_events(traj, o_symbol, h_symbol, cutoff)
    n_hops = len(data['hop_events'])
    
    # Assume 1 fs per frame
    sim_time_ps = len(traj) * 1e-3
    
    if sim_time_ps == 0:
        return 0.0
    
    return n_hops / sim_time_ps


def compute_avg_residence_time(traj, o_symbol=O_SYMBOL, h_symbol=H_SYMBOL, cutoff=CUTOFF):
    """
    Compute average proton residence time on an oxygen (in ps).
    
    Assumes trajectory has constant timestep (1 fs per frame).
    
    Parameters
    ----------
    traj : list of ase.Atoms
        Trajectory frames.
    o_symbol : str
        Element symbol for oxygen.
    h_symbol : str
        Element symbol for hydrogen.
    cutoff : float
        Distance cutoff for O-H assignment (Å).
        
    Returns
    -------
    float
        Average residence time in ps.
    """
    data = detect_hop_events(traj, o_symbol, h_symbol, cutoff)
    residence = data['residence']
    
    residence_times = []
    
    for h_idx, residence_events in residence.items():
        for i in range(len(residence_events) - 1):
            start_frame = residence_events[i][0]
            end_frame = residence_events[i + 1][0]
            duration_ps = (end_frame - start_frame) * 1e-3
            residence_times.append(duration_ps)
    
    if len(residence_times) == 0:
        return np.nan
    
    return np.mean(residence_times)


# =============================================================================
# BLOCK STATISTICS
# =============================================================================

def split_blocks(
    x,
    correlation_time,
    block_factor=10,
):
    """
    Split a 1D array into statistically independent blocks.
    
    Parameters
    ----------
    x : array-like
        Time series data.
    correlation_time : int
        Estimated correlation time in frames.
    block_factor : int, default=10
        Block size multiplier.
        
    Returns
    -------
    ndarray
        Shape (n_blocks, block_size)
    """
    block_size = int(
        np.ceil(correlation_time * block_factor)
    )

    n_blocks = len(x) // block_size

    if n_blocks < 2:
        raise ValueError(
            f"Need at least 2 blocks, got "
            f"{n_blocks}. Increase trajectory "
            f"length or decrease block_factor."
        )

    x = np.asarray(
        x[: n_blocks * block_size]
    )

    return x.reshape(n_blocks, block_size)


def summarize_blocks(values):
    """
    Compute mean and standard error from block-averaged values.
    """
    values = np.asarray(values)

    return {
        "mean": float(values.mean()),
        "stderr": float(
            values.std(ddof=1)
            / np.sqrt(len(values))
        ),
        "n_blocks": int(len(values)),
    }


def compute_hop_rate_blocks(
    traj,
    correlation_time,
    block_factor=10,
    o_symbol=O_SYMBOL,
    h_symbol=H_SYMBOL,
    cutoff=CUTOFF,
):
    """
    Compute block-averaged proton hop rate (hops/ps).
    
    Parameters
    ----------
    traj : list of ase.Atoms
        Trajectory frames.
    correlation_time : int
        Estimated correlation time in frames.
    block_factor : int, default=10
        Block size multiplier.
    o_symbol : str
        Element symbol for oxygen.
    h_symbol : str
        Element symbol for hydrogen.
    cutoff : float
        Distance cutoff for O-H assignment (Å).
        
    Returns
    -------
    dict
        With keys 'mean', 'stderr', 'n_blocks'.
    """
    data = detect_hop_events(traj, o_symbol, h_symbol, cutoff)
    hop_events = data['hop_events']
    
    # Bin hops into blocks
    blocks = split_blocks(
        np.arange(len(traj)),
        correlation_time,
        block_factor,
    )
    
    hop_rates = []
    
    for block_frames in blocks:
        # Count hops in this block
        n_hops_in_block = sum(
            1 for frame, _, _, _ in hop_events
            if frame in block_frames
        )
        
        # Time span of block in ps (1 fs per frame)
        block_time_ps = len(block_frames) * 1e-3
        
        # Hop rate for this block
        rate = n_hops_in_block / block_time_ps if block_time_ps > 0 else 0.0
        hop_rates.append(rate)
    
    return summarize_blocks(np.array(hop_rates))


def compute_residence_time_blocks(
    traj,
    correlation_time,
    block_factor=10,
    o_symbol=O_SYMBOL,
    h_symbol=H_SYMBOL,
    cutoff=CUTOFF,
):
    """
    Compute block-averaged proton residence time (ps).
    
    For each block, computes the average residence time of protons
    that end during that block.
    
    Parameters
    ----------
    traj : list of ase.Atoms
        Trajectory frames.
    correlation_time : int
        Estimated correlation time in frames.
    block_factor : int, default=10
        Block size multiplier.
    o_symbol : str
        Element symbol for oxygen.
    h_symbol : str
        Element symbol for hydrogen.
    cutoff : float
        Distance cutoff for O-H assignment (Å).
        
    Returns
    -------
    dict
        With keys 'mean', 'stderr', 'n_blocks'.
    """
    data = detect_hop_events(traj, o_symbol, h_symbol, cutoff)
    residence = data['residence']
    
    blocks = split_blocks(
        np.arange(len(traj)),
        correlation_time,
        block_factor,
    )
    
    residence_times_per_block = []
    
    for block_frames in blocks:
        block_frames_set = set(block_frames)
        residence_times = []
        
        for h_idx, residence_events in residence.items():
            for i in range(len(residence_events) - 1):
                start_frame = residence_events[i][0]
                end_frame = residence_events[i + 1][0]
                
                # Only count if residence ends in this block
                if end_frame in block_frames_set:
                    duration_ps = (end_frame - start_frame) * 1e-3
                    residence_times.append(duration_ps)
        
        # Average residence time in this block
        if len(residence_times) > 0:
            avg_time = np.mean(residence_times)
        else:
            avg_time = np.nan
        
        residence_times_per_block.append(avg_time)
    
    # Filter out NaN values
    valid_times = [t for t in residence_times_per_block if not np.isnan(t)]
    
    if len(valid_times) == 0:
        return {
            "mean": np.nan,
            "stderr": np.nan,
            "n_blocks": 0,
        }
    
    return summarize_blocks(np.array(valid_times))

    previous = current
    previous_frame = t

