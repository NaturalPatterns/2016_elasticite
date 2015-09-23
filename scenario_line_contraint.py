#!/usr/bin/env python
# -*- coding: utf8 -*-

import elasticite as el
import numpy as np
class EdgeGrid(el.EdgeGrid):
    def champ(self):
        force = np.zeros_like(self.lames[2, :])
        angle = lambda t, x: np.pi*np.mod(t / 6.-x, 1.)
        damp = lambda t: 0.01

        force = angle(self.t, self.lames[0, :]) - self.lames[2, :]
        force -= damp(self.t) * self.lames[3, :]/self.dt
        return 10. * force


e = EdgeGrid(N_lame=20, grid_type='line')
el.main(e)
