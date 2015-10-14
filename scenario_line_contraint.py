#!/usr/bin/env python
# -*- coding: utf8 -*-

"""

Sur une ligne de lames, on fait tourner les lames avec un mouvement relativement élastique mais exogene (prédeterminé, pas émergent)

"""
import sys
if len(sys.argv)>1: mode = sys.argv[1]
else: mode = 'both'

import elasticite as el
import numpy as np
class EdgeGrid(el.EdgeGrid):
    def update(self):
        self.dt = self.time() - self.t
        self.lames[2, :] = 20.*np.pi/180. * np.sin(2*np.pi*(self.t)/10.)
        self.t = self.time()

e = EdgeGrid(N_lame=20, grid_type='line', mode=mode, verb=True)
el.main(e)
