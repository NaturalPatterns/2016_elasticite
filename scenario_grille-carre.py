"""

Dans une grille en tore (pacman) privilégie les co-circulrités en évitant les co-linéarités

Short-range pour donner des patterns locaux

"""

import elasticite as el
import numpy as np
class EdgeGrid(el.EdgeGrid):
    def champ(self):
        force = np.zeros_like(self.lames[2, :])
        noise = lambda t: 2* np.exp((np.cos(2*np.pi*(t-0.) / 6.)-1.)/ 1.**2)
        damp = lambda t: 0.001 #* np.exp(np.cos(t / 6.) / 3.**2)
        colin_t = lambda t: -2.*np.exp((np.cos(2*np.pi*(t-3.) / 6.)-1.)/ 1.**2)
        colin_d = lambda d: np.exp(-d/.05) #np.exp(-np.log((d+1.e-12)/.05)**2/2/1.5)

        #delta_angle = np.mod(self.angle_relatif()-np.pi/3., 2*np.pi/3)
        delta_angle = self.angle_relatif()-np.pi/2.
        #delta_angle *= np.sign(delta_angle)
        force += colin_t(self.t) * np.sum(np.sin(4*delta_angle)*colin_d(self.distance(do_torus=True)), axis=1)
        force += noise(self.t)*np.pi*np.random.randn(self.N_lame)
        force -= damp(self.t) * self.lames[3, :]/self.dt
        return 100. * force

e = EdgeGrid()
el.main(e)
