
"""

TODO
----
* Measures that have needs
* Measures with different needs to accomplish and different weigths.
* Measures with dynamic change in incresing wealth (reaction of redistribution)

"""

import numpy as np


def simple_negative_inequality(dist):
    """Simple inequality measure for the systems in which consider a need which
    is totally necessary.

    This measure is measuring the proportion of extra money the society has in
    proportion of the basic needs not covered to other of there population.
    The measure goes from [0, inf).

    """

    pos_dist = dist[dist >= 0]
    neg_dist = dist[dist < 0]

    pos_dist_tot = np.sum(pos_dist)
    neg_dist_tot = np.sum(neg_dist)
    if np.sign(neg_dist_tot) == 1:
        neg_dist_tot = -neg_dist_tot

    # If there is enough wealth to cover the needs.
    if neg_dist_tot < pos_dist_tot:
        inequality_ratio = (pos_dist_tot-neg_dist_tot)/pos_dist_tot
    # if there is no enough wealth to cover the needs.
    else:
        inequality_ratio = 0.
    return inequality_ratio
