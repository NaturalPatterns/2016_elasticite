#!/usr/bin/env python
# -*- coding: utf8 -*-

"""

Sur une ligne de lames, on fait tourner les lames avec un mouvement émergent.

"""


import elasticite as el

e = el.EdgeGrid(N_lame=20, grid_type='line')
el.main(e)
