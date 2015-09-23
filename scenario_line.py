#!/usr/bin/env python
# -*- coding: utf8 -*-

"""

Sur une ligne de lames, on fait tourner les lames avec un mouvement émergent.

"""
import sys
if len(sys.argv)>1: mode = sys.argv[1]
else: mode = 'both'


import elasticite as el

e = el.EdgeGrid(N_lame=20, grid_type='line', mode=mode)
el.main(e)
