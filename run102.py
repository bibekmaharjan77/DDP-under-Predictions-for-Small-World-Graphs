# good version till now

import networkx as nx
import math, os, random
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
from matplotlib.ticker import FixedLocator
import datetime as dt

# ---------------------------- CONFIGURATION ----------------------------
PREDICTION_FRACTIONS = [0.03125, 0.0625, 0.125, 0.25, 0.5]
ERROR_VALUES = [999999999999999999999999999999999999, 10, 5, 3.33333333333333333333333333, 2.5, 2]
ERROR_VALUES_2       = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
NUM_TRIALS           = 50
DEBUG                = True
down_link = {}

# ---------------------------- SUBMESH HIERARCHY ----------------------------

# --- NEW: a layout-aware id mapper ---
def xy_to_id_layout(x, y, size, layout=None):
    """
    If layout is None: row-major id = x*size + y (old behavior).
    If layout is a list of length n: returns layout[x*size + y].
    """
    idx = x*size + y
    if layout is None:
        return idx
    return layout[idx]


# --- modify the submesh generators to accept a mapper ---
def generate_type1_submeshes(size, layout=None):
    levels = int(math.log2(size)) + 1
    hierarchy = defaultdict(list)
    for level in range(levels):
        b = 2**level
        for i in range(0, size, b):
            for j in range(0, size, b):
                nodes = {
                    xy_to_id_layout(x, y, size, layout)
                    for x in range(i, min(i+b, size))
                    for y in range(j, min(j+b, size))
                }
                hierarchy[(level,2)].append(nodes)
    return hierarchy

def generate_type2_submeshes(size, layout=None):
    levels = int(math.log2(size))
    hierarchy = defaultdict(list)
    for level in range(1, levels):
        b = 2**level
        off = b//2
        for i in range(-off, size, b):
            for j in range(-off, size, b):
                nodes = {
                    xy_to_id_layout(x, y, size, layout)
                    for x in range(i, i+b)
                    for y in range(j, j+b)
                    if 0 <= x < size and 0 <= y < size
                }
                if nodes:
                    hierarchy[(level,1)].append(nodes)
    return hierarchy

def build_mesh_hierarchy(size, layout=None):
    """
    Optional layout: a permutation list of length n=size*size that
    says which node id occupies each row-major (x,y) cell.
    layout=None keeps old behavior.
    """
    H = generate_type1_submeshes(size, layout=layout)
    H.update(generate_type2_submeshes(size, layout=layout))

    # level-0 equal
    H[(0,1)] = list(H[(0,2)])

    # sanity checks (unchanged)
    levels = sorted({lvl for (lvl,_) in H})
    lo, hi = levels[0], levels[-1]
    if H[(lo,1)] != H[(lo,2)]:
        raise RuntimeError("Level 0 must have identical Type-1/2 clusters")
    for lvl in levels[1:-1]:
        if H[(lvl,1)] == H[(lvl,2)]:
            raise RuntimeError(f"Level {lvl} Type-1 == Type-2; they must differ")

    # root with *all current nodes* (layout or not)
    n = size*size
    all_nodes = set(layout if layout is not None else range(n))
    root_level = hi + 1
    H[(root_level,1)].append(all_nodes)
    H[(root_level,2)].append(all_nodes)
    if H[(root_level,1)] != H[(root_level,2)]:
        raise RuntimeError("Root level must have identical Type-1/2 clusters")

    return H

# heirarchy for small world graphs
def build_smallworld_hierarchy(G):
    """
    Hierarchy for small-world graphs using a ring over sorted node IDs.
    Levels:
      - level 0: singletons (Type-1 == Type-2)
      - level 1..L-1: blocks of size 2,4,8,... with a rotated (offset) tiling
      - level L: root (all nodes), Type-1 == Type-2
    This matches the interface of build_mesh_hierarchy(size).
    """
    H = defaultdict(list)
    nodes = sorted(G.nodes())
    n = len(nodes)

    # ----- level 0: singletons, Type-1 == Type-2 -----
    singletons = [{v} for v in nodes]
    H[(0, 1)] = list(singletons)
    H[(0, 2)] = list(singletons)

    # ----- coarser levels with block sizes 2,4,8,... < n -----
    level = 1
    block_size = 2
    while block_size < n:
        # Type-2: “base” tiling
        clusters_t2 = []
        for start in range(0, n, block_size):
            cl = set(nodes[start : min(start + block_size, n)])
            if cl:
                clusters_t2.append(cl)
        H[(level, 2)] = clusters_t2

        # Type-1: same block size, but ring-rotated by half a block
        shift = block_size // 2
        rotated = nodes[shift:] + nodes[:shift]
        clusters_t1 = []
        for start in range(0, n, block_size):
            cl = set(rotated[start : min(start + block_size, n)])
            if cl:
                clusters_t1.append(cl)
        H[(level, 1)] = clusters_t1

        level += 1
        block_size *= 2

    # ----- root level: all nodes, Type-1 == Type-2 -----
    root_level = level
    all_nodes = set(nodes)
    H[(root_level, 1)].append(all_nodes)
    H[(root_level, 2)].append(all_nodes)

    # (optional) sanity checks similar to your grid version:
    levels = sorted({lvl for (lvl, _) in H})
    lo, hi = levels[0], levels[-1]
    assert H[(lo,1)] == H[(lo,2)], "Level 0 must have identical Type-1/2 clusters"
    assert H[(hi,1)] == H[(hi,2)], "Root level must have identical Type-1/2 clusters"
    for lvl in levels[1:-1]:
        if H[(lvl,1)] == H[(lvl,2)]:
            raise RuntimeError(f"Level {lvl} Type-1 == Type-2; they must differ")

    return H


def make_pred_first_layout(all_nodes, predicted):
    """
    Returns a permutation list L of all_nodes, where elements of `predicted`
    (deduped, in their current order) appear first, followed by the remaining nodes.
    The length of L must be exactly n and contain each node id exactly once.
    """
    seen = set()
    P = [v for v in predicted if v not in seen and not seen.add(v)]
    rest = [v for v in all_nodes if v not in seen]
    return P + rest



def print_clusters(H):
    print("=== Cluster Hierarchy ===")
    levels = sorted(set(lvl for (lvl, _) in H))
    for lvl in levels:
        for t in [1, 2]:
            key = (lvl, t)
            if key in H:
                print(f"Level {lvl} Type-{t}: {len(H[key])} clusters")
                for idx, cl in enumerate(H[key]):
                    print(f"  Cluster {idx}: {sorted(cl)}")
    print("="*40)



def assign_cluster_leaders(H, seed=None, prefer=None):
    """
    Choose one leader per cluster.
    - If `prefer` (an iterable of node ids) is given, we pick the leader
      from (cluster ∩ prefer) when that intersection is non-empty.
    - Otherwise we fall back to uniform random in the cluster.
    """
    rng = random.Random(seed) if seed is not None else random.Random()
    prefer = set(prefer) if prefer is not None else None

    M = defaultdict(list)
    for lvl_type, clusters in H.items():
        for cl in clusters:
            if prefer:
                cand = [v for v in cl if v in prefer]
                if cand:          # prefer a predicted node inside this cluster
                    leader = rng.choice(cand)
                else:
                    leader = rng.choice(tuple(cl))
            else:
                leader = rng.choice(tuple(cl))
            M[lvl_type].append((leader, cl))
    return M



# ---------------------------- PUBLISH / DOWNWARD LINKS ----------------------------

def get_spiral(node, leader_map, verbose=False):
    path = [node]
    seen = {node}
    # iterate by level, and inside each level visit Type-1 (2) then Type-2 (1)
    levels = sorted({lvl for (lvl, _) in leader_map})
    for lvl in levels:
        for t in (2, 1):  # Type-1, then Type-2
            key = (lvl, t)
            if key not in leader_map: 
                continue
            for leader, cl in leader_map[key]:
                if node in cl:
                    if leader not in seen:
                        path.append(leader)
                        seen.add(leader)
                    break  # there is only one containing cluster per (lvl,t)

    # force-append the root leader so publish always reaches the root
    top_level = max(lvl for (lvl, _) in leader_map)
    root_key  = (top_level, 2) if (top_level, 2) in leader_map else (top_level, 1)
    root_leader = leader_map[root_key][0][0]
    if path[-1] != root_leader:
        path.append(root_leader)

    
    if verbose:
        print(f"Spiral upward path for {node}: {path}")
    return path


def publish(owner, leader_map):
    """
    Store downward pointers at every cluster leader the owner belongs to.
    """
    global down_link
    down_link.clear()
    sp = get_spiral(owner, leader_map) # spiral is bottom-up
    # Store a pointer at every leader in the spiral (except the last, which is the owner)
    for i in range(len(sp)-1, 0, -1):
        down_link[sp[i]] = sp[i-1]

# ---------------------------- STRETCH MEASUREMENT ----------------------------


def _single_request_costs(r, owner, leader_map, G, weight=None, trace=False):
    """
    Return (up_hops, down_hops, obj) for a single request r given current owner.
    obj is the shortest-path distance from owner -> r.
    """
    # publish at the current owner
    publish(owner, leader_map)

    # ---- UP: climb r's spiral until it hits a published pointer
    sp = get_spiral(r, leader_map)
    up_hops = 0
    intersection = None
    for i in range(len(sp) - 1):
        u, v = sp[i], sp[i+1]
        d = nx.shortest_path_length(G, u, v, weight=weight)
        up_hops += d
        if v in down_link:
            intersection = v
            break
    if intersection is None:
        intersection = sp[-1]  # root leader

    # ---- DOWN: follow pointers; if missing, jump directly to owner
    down_hops = 0
    cur = intersection
    seen = {cur}
    while cur != owner:
        nxt = down_link.get(cur)
        if nxt is None or nxt in seen:
            down_hops += nx.shortest_path_length(G, cur, owner, weight=weight)
            break
        down_hops += nx.shortest_path_length(G, cur, nxt, weight=weight)
        seen.add(nxt)
        cur = nxt

    # ---- object forwarding hop
    obj = nx.shortest_path_length(G, owner, r, weight=weight)

    if trace:
        print(f"[costs] r={r} up={up_hops} down={down_hops} obj={obj}")

    return up_hops, down_hops, obj



def measure_stretch(requesters, owner, leader_map, G, weight=None, trace=True):
    """
    LOOKUP stretch (your current metric):
      sum(UP+DOWN) / sum(OPT), where OPT = shortest(owner, requester).
    Kept for backward compatibility.
    """
    total_up_down = 0
    total_opt     = 0

    for r in requesters:
        if r == owner:
            continue
        up, down, obj = _single_request_costs(r, owner, leader_map, G, weight, trace and DEBUG)
        total_up_down += (up + down)
        total_opt     += obj
        owner = r  # move ownership for next step (your model does this)
        publish(owner, leader_map)

    return (total_up_down / total_opt) if total_opt > 0 else 1.0



def measure_stretch_move(requesters, owner, leader_map, G, weight=None, trace=True):
    """
    MOVE stretch (what MultiBend reports):
      sum(UP + DOWN + OBJ) / sum(OBJ)  ==  1 + sum(UP+DOWN)/sum(OBJ)
    """
    total_alg = 0
    total_opt = 0

    for r in requesters:
        if r == owner:
            continue
        up, down, obj = _single_request_costs(r, owner, leader_map, G, weight, trace and DEBUG)
        total_alg += (up + down + obj)
        total_opt += obj
        owner = r
        publish(owner, leader_map)

    return (total_alg / total_opt) if total_opt > 0 else 1.0




# ---------------------------- GRAPH LOADING ----------------------------

def load_graph(dfile):
    G = nx.read_graphml(os.path.join("graphs","small_world",dfile))
    return nx.relabel_nodes(G, lambda x:int(x))

# ---------------------------- ERRORS & HELPERS ----------------------------

def choose_Vp(G, fraction):
    G = nx.relabel_nodes(G, lambda x: int(x))
    nodes = list(G.nodes())
    random.shuffle(nodes)  # Shuffle the nodes to ensure randomness
    total_nodes = len(nodes)
    vp_size = int(total_nodes * fraction) # Fraction of nodes to be chosen as Vp
    original_Vp = list(random.choices(nodes, k=vp_size))
    random.shuffle(original_Vp)  # Shuffle Vp to ensure randomness

    reduced_Vp = set(original_Vp)

    reduced_Vp = list(reduced_Vp)  # Convert back to a list for indexing
    random.shuffle(reduced_Vp)  # Shuffle Vp to ensure randomness

    # Choose an owner node that is not in Vp
    remaining = set(nodes) - set(reduced_Vp)
    owner = random.choice(list(remaining))

    # Insert owner to reduced_Vp list at a random position
    insert_position = random.randint(0, len(reduced_Vp))
    reduced_Vp.insert(insert_position, owner)
    S = reduced_Vp.copy()
    S = set(S)  # Convert to a set for uniqueness

    return original_Vp


def count_duplicates(input_list):
    """
    Checks for duplicate elements in a list and returns their counts.

    Args:
        input_list: The list to check for duplicates.

    Returns:
        A dictionary where keys are the duplicate elements and values are their counts.
        Returns an empty dictionary if no duplicates are found.
    """
    counts = Counter(input_list)
    duplicates = {element: count for element, count in counts.items() if count > 1}
    return duplicates


def sample_Q_within_diameter(G, Vp, error_cutoff):
    diam = nx.diameter(G, weight='weight')
    max_iter = 100000  # Maximum number of iterations to avoid infinite loop

    for attempt in range(1, max_iter+1):
        # 1) sample one random reachable node per v
        Q = []
        for v in Vp:
            dist_map = nx.single_source_dijkstra_path_length(G, v, cutoff=float(diam/error_cutoff), weight="weight")
            Q.append(random.choice(list(dist_map.keys())))

        # 2) compute overlap
        dup_counts = count_duplicates(Q)
        # extra dups = sum of (count - 1) for each duplicated element
        extra_dups = sum(cnt for cnt in dup_counts.values())
        current_overlap = extra_dups / len(Q) * 100

        # 3) check if within tolerance
        if current_overlap <= 100:
            return Q

    random.shuffle(Q)  # Shuffle the list to ensure randomness
    return Q

def sample_actual(G, Vp, error):
    diam = nx.diameter(G)
    act = []
    for v in Vp:
        cutoff = int(diam/error) if error>0 else diam
        lengths = nx.single_source_shortest_path_length(G, v, cutoff=cutoff)
        act.append(random.choice(list(lengths.keys())))
    return act

def calculate_error(Vp, Q, G_example):
    diameter_of_G = nx.diameter(G_example, weight='weight')  # Compute the diameter of the graph G_example
    errors = []
    for req, pred in zip(Q, Vp):
        # Using NetworkX to compute the shortest path length in tree T.
        dist = nx.shortest_path_length(G_example, source=req, target=pred, weight='weight')
        error = dist / diameter_of_G
        errors.append(error)
        # print(f"\nDistance between request node {req} and predicted node {pred} is {dist}, error = {error:.4f}")
    
    # print("Diameter of G:", diameter_of_G)
    # print("Diameter of T:", diameter_of_T)
    total_max_error = max(errors) if errors else 0
    total_min_error = min(errors) if errors else 0
    RED = "\033[91m"
    RESET = "\033[0m"
    print(f"{RED}\nOverall max error (max_i(distance_in_G / diameter_G)) = {total_max_error:.4f}{RESET}")
    print(f"{RED}\nOverall min error (min_i(distance_in_G / diameter_G)) = {total_min_error:.4f}{RESET}")
    return total_max_error


# -----calaulate error stats to get max, min and avg error-------------
def calculate_error_stats(Vp, Q, G):
    diam = nx.diameter(G, weight='weight')
    vals = []
    for req, pred in zip(Q, Vp):
        d = nx.shortest_path_length(G, req, pred, weight='weight')
        vals.append(d / diam)
    if not vals:
        return 0.0, 0.0, 0.0
    return max(vals), min(vals), float(sum(vals)/len(vals))



# ---------------------------- SIMULATION ----------------------------

def simulate(graph_file, use_move_stretch=False):
    G = load_graph(graph_file)
    size = int(math.sqrt(G.number_of_nodes()))
    H = build_mesh_hierarchy(size)
    print_clusters(H)
    leaders = assign_cluster_leaders(H)

    measure_fn = measure_stretch_move if use_move_stretch else measure_stretch
    results = []
    for error in ERROR_VALUES:
        for frac in PREDICTION_FRACTIONS:
            owner = random.choice(list(G.nodes()))
            publish(owner, leaders)

            for _ in range(NUM_TRIALS):
                pred = choose_Vp(G, frac)
                act  = sample_Q_within_diameter(G, pred, error)
                err  = calculate_error(pred, act, G)

                for req in act:
                    if req == owner:
                        continue
                    stretch = measure_fn([req], owner, leaders, G, trace=False)

                    err_rate = 0.0 if error > 15 else round(1.0 / error, 1)
                    results.append((frac, err_rate, err, stretch))

                    owner = req
                    publish(owner, leaders)
    return results


# ---------------------------- PLOTTING ----------------------------

def plot_results(results):
    df  = pd.DataFrame(results, columns=["Frac","ErrRate","Err","Str"])
    avg = df.groupby(["Frac","ErrRate"]).mean().reset_index()

    # use your global list here
    xvals = PREDICTION_FRACTIONS

    plt.figure(figsize=(12,6))

    # ---------------- Error vs Fraction ----------------
    plt.subplot(1,2,1)
    for e in ERROR_VALUES_2:
        sub = avg[ avg.ErrRate == e ]
        plt.plot(sub.Frac, sub.Err, '-o', label=f"{e:.1f} Error")
    plt.title("Error vs Fraction of Predicted Nodes")
    plt.xlabel("Fraction of Predicted Nodes")
    plt.ylabel("Error (Max)")
    plt.xticks(xvals, [f"{f:.4f}" for f in xvals], rotation=45)
    plt.ylim(0, max(ERROR_VALUES_2)*1.1)
    plt.grid(True)
    plt.legend(loc="upper right")

    # ---------------- Stretch vs Fraction ----------------
    plt.subplot(1,2,2)
    # for e in ERROR_VALUES_2:
    #     sub = avg[ avg.ErrRate == e ]
    #     plt.plot(sub.Frac, sub.Str, '-o', label=f"{e:.1f} Stretch")

    # loop over each unique ErrRate in your aggregated frame

    for err_rate, group in avg.groupby("ErrRate"):
        plt.plot(
            group.Frac, 
            group.Str, 
            "-o", 
            label=f"{err_rate:.1f} Stretch"
        )

    


    plt.title("Stretch vs Fraction of Predicted Nodes")
    plt.xlabel("Fraction of Predicted Nodes")
    plt.ylabel("Stretch")
    plt.xticks(xvals, [f"{f:.4f}" for f in xvals], rotation=45)
    # plt.ylim(0.95, 1.05)
    plt.grid(True)
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.show()

# ---------------------------- HALVING MODE ADD-ONS ----------------------------

def halving_counts(n: int):
    """Return [n/2, n/4, ..., 1] using integer division."""
    counts = []
    k = n // 2
    while k >= 1:
        counts.append(k)
        k //= 2
    return counts

def choose_Vp_halving(G, k: int):
    """
    Halving-mode version of choose_Vp. Intentionally mirrors the original:
    - uses random.choices (with replacement)
    - dedups to build reduced_Vp
    - selects an owner not in reduced_Vp and inserts it (for parity with original)
    - returns *original_Vp* (same as your original choose_Vp)
    """
    G = nx.relabel_nodes(G, lambda x: int(x))
    nodes = list(G.nodes())
    random.shuffle(nodes)  # Shuffle for randomness
    total_nodes = len(nodes)

    # mirror the original: clamp to [1, n] and use as-is
    vp_size = int(k)
    vp_size = max(1, min(vp_size, total_nodes))

    # with replacement (exactly like your original)
    original_Vp = list(random.choices(nodes, k=vp_size))
    random.shuffle(original_Vp)

    # build reduced_Vp & owner (same as original structure)
    reduced_Vp = set(original_Vp)
    reduced_Vp = list(reduced_Vp)
    random.shuffle(reduced_Vp)

    remaining = set(nodes) - set(reduced_Vp)
    owner = random.choice(list(remaining)) if remaining else random.choice(nodes)

    insert_position = random.randint(0, len(reduced_Vp))
    reduced_Vp.insert(insert_position, owner)
    S = set(reduced_Vp)  # kept for parity; not returned/used (same as original)

    # IMPORTANT: return signature matches your original choose_Vp
    return original_Vp


def simulate_halving(graph_file, use_move_stretch=False):
    G = load_graph(graph_file)
    n = G.number_of_nodes()
    size = int(math.sqrt(n))
    H = build_mesh_hierarchy(size)
    print_clusters(H)
    leaders = assign_cluster_leaders(H)

    measure_fn = measure_stretch_move if use_move_stretch else measure_stretch
    k_values = halving_counts(n)
    results = []

    for error in ERROR_VALUES:
        for k in k_values:
            frac = k / n
            owner = random.choice(list(G.nodes()))
            publish(owner, leaders)

            for _ in range(NUM_TRIALS):
                P = choose_Vp_halving(G, k)
                Q = sample_Q_within_diameter(G, P, error)
                err = calculate_error(P, Q, G)

                for req in Q:
                    if req == owner:
                        continue
                    stretch = measure_fn([req], owner, leaders, G, trace=False)
                    err_rate = 0.0 if error > 15 else round(1.0 / error, 1)
                    results.append((frac, err_rate, err, stretch))
                    owner = req
                    publish(owner, leaders)
    return results



def plot_results_halving_counts(results, n=None, title_suffix=" (halving mode)", use_log_x=True):
    """
    Same input schema as before. Shows x-axis as |P| counts (1,2,4,...,n/2).
    Left error subplot y-axis fixed to 0..0.5 with ticks at 0.1.
    """
    from matplotlib.ticker import FormatStrFormatter  # local import

    df  = pd.DataFrame(results, columns=["Frac","ErrRate","Err","Str"])

    # infer n if not provided
    if n is None:
        min_frac = df["Frac"].min()
        n = int(round(1.0 / min_frac)) if min_frac > 0 else None
        if not n:
            raise ValueError("Please pass n explicitly to plot_results_halving_counts().")

    # k = |P|
    df["Count"] = (df["Frac"] * n).round().astype(int)
    avg = df.groupby(["Count","ErrRate"]).mean().reset_index()

    # tick positions: 1, 2, 4, 8, ..., floor(n/2) (whatever appeared in results)
    xvals = sorted(df["Count"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), constrained_layout=True, sharex=True)

    # helper: style both subplots
    def prettify_axis(ax, title, ylabel):
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("|P| (n/2, n/4, ..., 1)")
        # use log2 x-scale to spread small counts
        if use_log_x:
            try:
                ax.set_xscale("log", base=2)   # mpl >=3.3
            except TypeError:
                ax.set_xscale("log", basex=2)  # older mpl
        # show only our counts as ticks (keeps labels clean)
        ax.xaxis.set_major_locator(FixedLocator(xvals))
        ax.set_xticklabels([str(x) for x in xvals], rotation=0)
        # margins + lighter grid only on Y
        xmin = max(min(xvals), 1)
        ax.set_xlim(xmin * (0.98 if use_log_x else 0.9), max(xvals) * 1.02)
        ax.margins(x=0.02, y=0.08)
        ax.grid(True, axis="y", alpha=0.35)

    # -------- Error vs |P| --------
    ax = axes[0]
    for e in ERROR_VALUES_2:
        sub = avg[avg.ErrRate == e].sort_values("Count", ascending=True)
        ax.plot(sub.Count, sub.Err, "-o", label=f"{e:.1f} Error")
    prettify_axis(ax, "Error vs |P| (halving)"+title_suffix, "Error")

    # --- lock y-axis to 0..0.5 with ticks every 0.1 ---
    ax.set_ylim(0, 0.5)
    ax.set_yticks(np.arange(0, 0.51, 0.1))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    ax.legend(loc="upper right")

    # -------- Stretch vs |P| --------
    ax = axes[1]
    for err_rate, group in avg.groupby("ErrRate"):
        sub = group.sort_values("Count", ascending=True)
        ax.plot(sub.Count, sub.Str, "-o", label=f"{err_rate:.1f} Stretch")
    prettify_axis(ax, "Stretch vs |P| (halving)"+title_suffix, "Stretch")
    ax.legend(loc="upper right")

    plt.show()

# -----halving mode add ons complete-------------------------------


# ---------multibend comparison add-ons----------------------------
def multibend_move_sequence_stretch(requesters, owner, leader_map, G, weight=None, trace=False):
    """
    MultiBend *move* stretch for a requester sequence.
    Algorithm cost per request = UP + DOWN + (owner -> requester shortest path).
    OPT lower bound per request = (owner -> requester shortest path).
    Returns (sum alg costs) / (sum OPT costs).
    """
    total_alg = 0
    total_opt = 0

    for r in requesters:
        if r == owner:
            continue

        # publish at current owner (directory path)
        publish(owner, leader_map)

        # ----- UP: climb r's spiral until it hits the published path
        sp = get_spiral(r, leader_map)
        up_hops = 0
        intersection = None
        for i in range(len(sp) - 1):
            u, v = sp[i], sp[i+1]
            d = nx.shortest_path_length(G, u, v, weight=weight)
            up_hops += d
            if v in down_link:
                intersection = v
                break
        if intersection is None:
            intersection = sp[-1]

        # ----- DOWN: follow directory pointers (fallback = direct)
        down_hops = 0
        cur = intersection
        seen = {cur}
        while cur != owner:
            nxt = down_link.get(cur)
            if nxt is None or nxt in seen:
                down_hops += nx.shortest_path_length(G, cur, owner, weight=weight)
                break
            down_hops += nx.shortest_path_length(G, cur, nxt, weight=weight)
            seen.add(nxt)
            cur = nxt

        # ----- object forwarding hop
        obj = nx.shortest_path_length(G, owner, r, weight=weight)

        total_alg += (up_hops + down_hops + obj)
        total_opt += obj

        if trace:
            print(f"[MB seq] r={r} up={up_hops} down={down_hops} obj={obj} "
                  f"⇒ {(up_hops+down_hops+obj)/obj:.3f}")

        # move object to r and continue
        owner = r

    return (total_alg / total_opt) if total_opt > 0 else 1.0



def simulate_halving_compare_multibend(graph_file):
    """
    Compare: Our (prediction-aware leaders) vs MB (baseline random leaders).
    Returns rows: (Frac, ErrRate, ErrMax, ErrMin, ErrAvg, OurStr, MBStr).
    """
    G = load_graph(graph_file)
    n = G.number_of_nodes()
    size = int(math.sqrt(n))

    # Build one SPATIAL hierarchy once (row-major); baseline leaders are random.
    H_base = build_mesh_hierarchy(size)

    k_values = halving_counts(n)
    results = []

    for error in ERROR_VALUES:
        for k in k_values:
            frac = k / n

            for _ in range(NUM_TRIALS):

                # (optional) independent owner each trial
                owner_start = random.choice(list(G.nodes()))

                # sample predicted set and request sequence
                # new predictions & requests each trial
                P = choose_Vp_halving(G, k)
                Q = sample_Q_within_diameter(G, P, error)

                # REBUILD leaders fresh each trial
                leaders_base = assign_cluster_leaders(H_base, seed=None, prefer=None)

                # OUR directory = same clusters, but leaders biased to P for this batch
                leaders_pred = assign_cluster_leaders(H_base, seed=None, prefer=P)

                # batch error stats (repeated per-row for grouping later)
                err_max, err_min, err_avg = calculate_error_stats(P, Q, G)
                err_rate = 0.0 if error > 15 else round(1.0 / error, 1)

                owner_ours = owner_start
                owner_mb   = owner_start
                publish(owner_ours, leaders_pred)
                publish(owner_mb,   leaders_base)

                for req in Q:
                    if req == owner_ours:
                        owner_ours = req
                        owner_mb   = req
                        publish(owner_ours, leaders_pred)
                        publish(owner_mb,   leaders_base)
                        continue

                    our_str = measure_stretch_move([req], owner_ours, leaders_pred, G, trace=False)
                    owner_ours = req
                    publish(owner_ours, leaders_pred)

                    mb_str  = multibend_move_sequence_stretch([req], owner_mb, leaders_base, G)
                    owner_mb = req
                    publish(owner_mb, leaders_base)

                    results.append((frac, err_rate, err_max, err_min, err_avg, our_str, mb_str))

                # align owners for next trial at this (error,k)
                # owner_start = owner_ours

    return results


# small world variant of the omparison simulation
def simulate_smallworld_compare_multibend(graph_file):
    """
    Same as simulate_halving_compare_multibend, but using a
    small-world hierarchy instead of the grid mesh hierarchy.
    """
    G = load_graph(graph_file)
    n = G.number_of_nodes()

    # hierarchy no longer uses sqrt(n); purely graph-based
    H_base = build_smallworld_hierarchy(G)

    k_values = halving_counts(n)
    results = []

    for error in ERROR_VALUES:
        for k in k_values:
            frac = k / n

            for _ in range(NUM_TRIALS):
                owner_start = random.choice(list(G.nodes()))

                # predictions + requests as before
                P = choose_Vp_halving(G, k)
                Q = sample_Q_within_diameter(G, P, error)

                leaders_base = assign_cluster_leaders(H_base, seed=None, prefer=None)
                leaders_pred = assign_cluster_leaders(H_base, seed=None, prefer=P)

                err_max, err_min, err_avg = calculate_error_stats(P, Q, G)
                err_rate = 0.0 if error > 15 else round(1.0 / error, 1)

                owner_ours = owner_start
                owner_mb   = owner_start
                publish(owner_ours, leaders_pred)
                publish(owner_mb,   leaders_base)

                for req in Q:
                    if req == owner_ours:
                        owner_ours = req
                        owner_mb   = req
                        publish(owner_ours, leaders_pred)
                        publish(owner_mb,   leaders_base)
                        continue

                    our_str = measure_stretch_move([req], owner_ours, leaders_pred, G, trace=False)
                    owner_ours = req
                    publish(owner_ours, leaders_pred)

                    mb_str  = multibend_move_sequence_stretch([req], owner_mb, leaders_base, G)
                    owner_mb = req
                    publish(owner_mb, leaders_base)

                    results.append((frac, err_rate, err_max, err_min, err_avg, our_str, mb_str))

    return results



def plot_mb_vs_ours_per_error(results, n, err_levels=None, use_log_x=False, save=False, prefix="mb_vs_ours"):
    """
    Makes 1x2 figures like your reference image, one figure per error cutoff.
    Left:  Error vs |P| (halving)  — Y-axis fixed to 0..0.5 with ticks at 0.1.
    Right: Our stretch vs MultiBend move stretch
    """
    from matplotlib.ticker import FormatStrFormatter  # local import

    if err_levels is None:
        err_levels = ERROR_VALUES_2

    df = pd.DataFrame(results, columns=["Frac","ErrRate","Err","OurStr","MBStr"])
    df["Count"] = (df["Frac"] * n).round().astype(int)

    for e in err_levels:
        sub = df[df.ErrRate == e].copy()
        if sub.empty:
            continue
        avg = sub.groupby("Count").mean().reset_index().sort_values("Count")
        xvals = avg["Count"].tolist()

        fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

        # -------- Left: Error vs |P| --------
        ax = axes[0]
        ax.plot(avg["Count"], avg["Err"], "-o", label=f"Prediction Error ≤ {e:.1f}")
        ax.set_title("Error vs Fraction of Predicted Nodes")
        ax.set_ylabel("Average of Max Error")
        ax.set_xlabel(f"Number of predicted nodes among {n} nodes")
        if use_log_x:
            try: ax.set_xscale("log", base=2)
            except TypeError: ax.set_xscale("log", basex=2)
        ax.set_xticks(xvals)
        ax.grid(True, axis="y", alpha=0.35)
        ax.legend(loc="upper left")

        # --- lock y-axis to 0..0.5 with ticks every 0.1 ---
        ax.set_ylim(0, 0.5)
        ax.set_yticks(np.arange(0, 0.51, 0.1))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        # -------- Right: Stretch comparison --------
        ax = axes[1]
        x = avg["Count"].to_numpy(dtype=float)

        ax.plot(x*0.985, avg["OurStr"], "-o", label=f"Our Stretch ≤ {e:.1f}", zorder=3)
        ax.plot(x*1.015, avg["MBStr"], "--s", label=f"MultiBend Stretch ≤ {e:.1f}", alpha=0.85, zorder=2)

        ymax = float(np.nanmax([avg["OurStr"].max(), avg["MBStr"].max()]))
        ax.set_ylim(0, ymax * 1.05)

        ax.set_title("Our Stretch vs Multibend Stretch")
        ax.set_ylabel("Stretch")
        ax.set_xlabel(f"Number of predicted nodes among {n} nodes")
        if use_log_x:
            try: ax.set_xscale("log", base=2)
            except TypeError: ax.set_xscale("log", basex=2)
        ax.set_xticks(x)
        ax.grid(True, axis="y", alpha=0.35)
        ax.legend(loc="best")

        if save:
            out = f"{prefix}_err_{str(e).replace('.','p')}.png"
            plt.savefig(out, dpi=180)
            print("saved:", out)
        else:
            plt.show()



# ---------multibend comparison add-ons complete----------------------------

# ========== Excel helpers (add these) ======================================

def save_compare_results_to_excel(results, n, filename, graph_file=None):
    """
    Save simulate_halving_compare_multibend() results to an .xlsx file.
    Sheets:
      - 'raw': one row per request; includes ErrMax, ErrMin, ErrAvg
      - 'avg': mean aggregated by (Count, ErrRate) for all metrics
      - 'meta': run metadata (n, graph, trials, timestamp, etc.)
    """
    cols = ["Frac","ErrRate","ErrMax","ErrMin","ErrAvg","OurStr","MBStr"]
    df   = pd.DataFrame(results, columns=cols)
    df["Count"] = (df["Frac"] * n).round().astype(int)

    # group means for every numeric column except Frac
    group_cols = ["Count","ErrRate"]
    avg = df.groupby(group_cols, as_index=False).mean(numeric_only=True)

    meta = {
        "n": n,
        "graph_file": graph_file or "",
        "num_trials": NUM_TRIALS,
        "prediction_fracs": ",".join(map(str, PREDICTION_FRACTIONS)),
        "error_levels": ",".join(map(str, ERROR_VALUES_2)),
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "notes": "ErrMax/ErrMin/ErrAvg are per-batch stats copied to each request row."
    }

    with pd.ExcelWriter(filename, engine="xlsxwriter") as xw:
        df.to_excel(xw, sheet_name="raw", index=False)
        avg.to_excel(xw, sheet_name="avg", index=False)
        pd.DataFrame([meta]).to_excel(xw, sheet_name="meta", index=False)
    print(f"saved Excel: {filename}")



def plot_mb_vs_ours_from_excel(filename, use_avg=True, err_levels=None,
                               use_log_x=False, save=False, prefix="mb_vs_ours_from_xlsx",
                               error_metric="ErrAvg"):
    from matplotlib.ticker import FormatStrFormatter

    meta = pd.read_excel(filename, sheet_name="meta")
    n = int(meta["n"].iloc[0])

    if use_avg:
        df = pd.read_excel(filename, sheet_name="avg")
    else:
        raw = pd.read_excel(filename, sheet_name="raw")
        if "Count" not in raw.columns:
            raw["Count"] = (raw["Frac"] * n).round().astype(int)
        df = raw.groupby(["Count","ErrRate"], as_index=False).mean(numeric_only=True)

    if err_levels is None:
        err_levels = ERROR_VALUES_2

    for e in err_levels:
        sub = df[df.ErrRate == e].copy()
        if sub.empty:
            continue
        avg = sub.sort_values("Count")
        x = avg["Count"].to_numpy(dtype=float)

        fig, ax = plt.subplots(1, 1, figsize=(7, 5), constrained_layout=True)

        # ---- Stretch only ----
        ax.plot(x*0.985, avg["OurStr"], "-o", label=f"PMultiBend (err ≤ {e:.1f})", zorder=3)
        ax.plot(x*1.015, avg["MBStr"], "--s", label="MultiBend", alpha=0.85, zorder=2)

        ymax = float(np.nanmax([avg["OurStr"].max(), avg["MBStr"].max()]))
        ax.set_ylim(0, ymax * 1.05)

        ax.set_ylabel("Stretch")
        ax.set_xlabel(f"Number of predicted nodes among {n} nodes")
        if use_log_x:
            try: ax.set_xscale("log", base=2)
            except TypeError: ax.set_xscale("log", basex=2)
        ax.set_xticks(x)
        ax.grid(True, axis="y", alpha=0.35)
        ax.legend(loc="best")

        if save:
            out = f"{prefix}_stretch_only_err_{str(e).replace('.','p')}.png"
            plt.savefig(out, dpi=180)
            print("saved:", out)
        else:
            plt.show()





#box and whisker plot
def plot_stretch_minmax_avg_from_excel(filename, err_levels=None,
                                       use_log_x=False, save=False,
                                       prefix="mb_vs_ours_minmax_band"):
    import pandas as pd, numpy as np, matplotlib.pyplot as plt

    meta = pd.read_excel(filename, sheet_name="meta")
    n = int(meta["n"].iloc[0])

    raw = pd.read_excel(filename, sheet_name="raw")
    if "Count" not in raw.columns:
        raw["Count"] = (raw["Frac"] * n).round().astype(int)

    if err_levels is None:
        err_levels = sorted(raw["ErrRate"].unique())

    for e in err_levels:
        sub = raw[raw.ErrRate == e].copy()
        if sub.empty:
            continue

        # per-|P| stats across trials
        g = sub.groupby("Count")
        stats = g.agg(
            OurMean=("OurStr", "mean"), OurMin=("OurStr", "min"), OurMax=("OurStr", "max"),
            MBMean =("MBStr", "mean"),  MBMin =("MBStr", "min"),  MBMax =("MBStr", "max")
        ).reset_index().sort_values("Count")

        x = stats["Count"].to_numpy(float)
        our_mean = stats["OurMean"].to_numpy(float)
        our_yerr = np.vstack([our_mean - stats["OurMin"].to_numpy(float),
                              stats["OurMax"].to_numpy(float) - our_mean])
        mb_mean  = stats["MBMean"].to_numpy(float)
        mb_yerr  = np.vstack([mb_mean  - stats["MBMin"].to_numpy(float),
                              stats["MBMax"].to_numpy(float) - mb_mean])

        fig, ax = plt.subplots(1, 1, figsize=(7,5), constrained_layout=True)
        ax.errorbar(x*0.985, our_mean, yerr=our_yerr, fmt='-o',
                    label=f"PMultiBend (err ≤ {e:.1f})", capsize=4, zorder=3)
        ax.errorbar(x*1.015, mb_mean,  yerr=mb_yerr,  fmt='--s',
                    label="MultiBend", capsize=4, alpha=0.9, zorder=2)

        ax.set_ylabel("Stretch")
        ax.set_xlabel(f"Number of predicted nodes among {n} nodes")
        if use_log_x:
            try: ax.set_xscale("log", base=2)
            except TypeError: ax.set_xscale("log", basex=2)
        ax.set_xticks(x)
        ax.grid(True, axis="y", alpha=0.35)
        ax.legend(loc="best")
        ax.set_title(f"Mean with min–max whiskers (err ≤ {e:.1f})")

        ymax = float(np.nanmax([stats["OurMax"].max(), stats["MBMax"].max()]))
        ax.set_ylim(0, ymax*1.05)

        if save:
            out = f"{prefix}_err_{str(e).replace('.','p')}.png"
            plt.savefig(out, dpi=180); print("saved:", out); plt.close(fig)
        else:
            plt.show()





def print_stretch_summary_from_excel(filename, err_levels=None, digits=3):
    """
    Pretty terminal summary per ErrRate and per |P| (Count).
    Reads the 'raw' sheet to compute stats across trials:
      - min/mean/max/std for OurStr (PMultiBend) and MBStr (MultiBend)
      - Δ mean = OurMean - MBMean (negative favors PMultiBend)
      - Win%  = fraction of trials where OurStr < MBStr
    """
    import pandas as pd
    import numpy as np

    # meta to get n (useful for headings)
    meta = pd.read_excel(filename, sheet_name="meta")
    n = int(meta["n"].iloc[0])

    raw = pd.read_excel(filename, sheet_name="raw")
    if "Count" not in raw.columns:
        raw["Count"] = (raw["Frac"] * n).round().astype(int)

    if err_levels is None:
        err_levels = sorted(raw["ErrRate"].unique())

    # helper to format a mean ± std [min,max] block
    def fmt_block(mean, std, vmin, vmax):
        if np.isnan(std):  # single trial case
            return f"{mean:.{digits}f}  [{vmin:.{digits}f},{vmax:.{digits}f}]"
        return f"{mean:.{digits}f} ± {std:.{digits}f}  [{vmin:.{digits}f},{vmax:.{digits}f}]"

    for e in err_levels:
        sub = raw[raw.ErrRate == e].copy()
        if sub.empty:
            continue

        # per-|P| stats across trials
        grp = sub.groupby("Count")
        stats = grp.agg(
            Trials = ("OurStr", "size"),
            OurMin = ("OurStr", "min"),
            OurMean= ("OurStr", "mean"),
            OurMax = ("OurStr", "max"),
            OurStd = ("OurStr", "std"),
            MBMin  = ("MBStr", "min"),
            MBMean = ("MBStr", "mean"),
            MBMax  = ("MBStr", "max"),
            MBStd  = ("MBStr", "std"),
        ).reset_index().sort_values("Count")

        # win% per |P| (fraction of trials with PMB < MB)
        def win_rate(df):
            return float((df["OurStr"] < df["MBStr"]).mean())
        wins = grp.apply(win_rate).reset_index(name="WinRate")
        stats = stats.merge(wins, on="Count", how="left")

        # build printable rows
        rows = []
        for _, r in stats.iterrows():
            rows.append({
                "|P|": int(r["Count"]),
                "Trials": int(r["Trials"]),
                "PMultiBend (mean±std [min,max])": fmt_block(r["OurMean"], r["OurStd"], r["OurMin"], r["OurMax"]),
                "MultiBend   (mean±std [min,max])": fmt_block(r["MBMean"], r["MBStd"], r["MBMin"], r["MBMax"]),
                "Δ mean (PMB−MB)": round(r["OurMean"] - r["MBMean"], digits),
                "Win% (PMB<MB)": f"{100.0*r['WinRate']:.1f}%"
            })
        table = pd.DataFrame(rows)

        # overall across |P|
        overall = {
            "PMB mean": stats["OurMean"].mean(),
            "MB mean":  stats["MBMean"].mean(),
            "Δ mean":   (stats["OurMean"].mean() - stats["MBMean"].mean()),
            "PMB win% (avg)": 100.0 * (sub.groupby(["Count"]).apply(lambda df: (df["OurStr"] < df["MBStr"]).mean()).mean())
        }

        print("\n" + "="*72)
        print(f"ErrRate ≤ {e:.1f}  |  n = {n} nodes")
        print("-"*72)
        print(table.to_string(index=False))
        print("-"*72)
        print(f"Overall across |P|:  PMB mean = {overall['PMB mean']:.{digits}f}, "
              f"MB mean = {overall['MB mean']:.{digits}f}, "
              f"Δ mean = {overall['Δ mean']:.{digits}f}, "
              f"PMB win% (avg) = {overall['PMB win% (avg)']:.1f}%")
        print("="*72)



# ========================================================================



# ---------------------------- MAIN ----------------------------

if __name__ == "__main__":
    
    # Small-world experiment on 64-node graph
    res_cmp = simulate_smallworld_compare_multibend("64small_world_diameter4test.edgelist")

    save_compare_results_to_excel(
        res_cmp, n=64,
        filename="whisker_64_smallworld.xlsx",
        graph_file="64small_world_diameter4test.edgelist"
    )

    print_stretch_summary_from_excel("whisker_64_smallworld.xlsx")
    plot_stretch_minmax_avg_from_excel("whisker_64_smallworld.xlsx", use_log_x=True)