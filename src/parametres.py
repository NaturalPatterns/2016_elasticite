#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Projection information
----------------------
Les coordonnées dans la salle sont par convention définis par rapport à un plan de référence
 perpendiculaire à l'axe long de la salle:

- le point central  (0., 0., 0.) est le coin en bas à gauche de ce plan (donc leur position est positive: x,y,z > 0)
- l'axe x  perpendiculairement à ce plan, vers les VPs
- l'axe y court est l'axe transversal, horizontal
- l'axe z est la hauteur
- tout est physique, en mètres (29.7cm = 0.297m)

Par convention, la position spatiale des VPs par rapport au centre du plan de reference
- placement regulier en profondeur a equidistance du plan de ref (le long d'un mur)
- placement regulier, le centre en premier
- on place les VPs à 1m36 de haut

Par convention, la position de la croix est au centre de la salle: [d_x/2, d_y/2]

"""
from numpy import arctan2, pi
import numpy as np

# pour savoir si on imprime des messages d'erreur
DEBUG  = False
#DEBUG  = True

# taille de l'espace
d_y, d_z = 11.2, 6.
d_x = 11.92 # distance en metres du plan sur lequel se positionnent les fenetres des VPs
volume = np.array([d_x, d_y,d_z])
# mesures au telemetre

#largeur_ecran = 1.7 # ouvert à fond
# https://unspecified.wordpress.com/2012/06/21/calculating-the-gluperspective-matrix-and-other-opengl-matrix-maths/
#hauteur_ecran = 1.7*9./16. # ouvert à fond
#distance_ecran = 2.446
hauteur_ecran, distance_ecran  = 0.52, 1.35
# distance dans l'axe de visee (essentiellement x)  entre le point mesuré (la vitre du VP) et le centre théorique du VP
# x_shift = .022/hauteur_ecran*distance_ecran
# on calcule
foc_estim = 2. * arctan2(hauteur_ecran/2., distance_ecran) * 180. / pi # ref P101L1
#print foc_estim
# foc = 30 * 9 /16.
foc = foc_estim
# d'autres références http://www.glprogramming.com/red/chapter03.html ou http://www.songho.ca/opengl/gl_transform.html
pc_min, pc_max = 0.001, 1000000.0
cz = 1.36  # hauteur des VPs

# scenario qui est joué par le modele physique
#scenario = "fan" # une aura autour de la position du premier player
#scenario = 'rotating-circle'
#scenario = 'cristal'
#scenario = "croix" # calibration autour de la croix
scenario = "leapfrog" # integration d'Euler améliorée pour simuler le champ

# et direction d'angle de vue (cx, cy, cz) comme le point de fixation ainsi que le champ de vue (en deg)
# distance des VPs du plan de reference
# profondeur du plan de référence
# les VPs sont positionnés en rang (x constants) sur un coté de la salle (cx_0) ou l'autre de la salle (cx_1)
#cx_0, cx_1 = 1. - x_shift, d_x + x_shift
#cx_0, cx_1 = 1. - x_shift, d_x

# tous les VPs regardent vers le VP central (positionné à d_y /2) au fond opposé
cy = (d_y/2) # on regarde le centre du plan de reference
# une liste des video projs donnant:
# leur adresse, port, leurs parametres physiques
VPs = [
        {'address':'10.42.0.51',
            'x':0.42, 'y':2.02, 'z': cz,
            'cx':d_x, 'cy':cy, 'cz': cz,
         'foc': foc, 'pc_min': pc_min, 'pc_max': pc_max},
        {'address':'10.42.0.52',
             'x':0.42, 'y':5.52, 'z': cz,
             'cx':d_x, 'cy':cy, 'cz': cz,
             'foc': foc, 'pc_min': pc_min, 'pc_max': pc_max},
        {'address':'10.42.0.53',
             'x':0.42, 'y':9.02, 'z': cz,
             'cx':d_x, 'cy':cy, 'cz': cz,
             'foc': foc, 'pc_min': pc_min, 'pc_max': pc_max},
        #{'address':'10.42.0.54',
            #'x':d_x-0.4, 'y':6.37, 'z': cz,
            #'cx':0, 'cy':cy, 'cz': cz,
            #'foc': foc, 'pc_min': pc_min, 'pc_max': pc_max},
        #{'address':'10.42.0.55',
            #'x':d_x-0.4, 'y':3.77, 'z': cz,
            #'cx':0, 'cy':cy, 'cz': cz,
            #'foc': foc, 'pc_min': pc_min, 'pc_max': pc_max},
        #{'address':'10.42.0.56',
            #'x':d_x-0.4, 'y':1.2, 'z': cz,
            #'cx':0, 'cy':cy, 'cz': cz,
            #'foc': foc, 'pc_min': pc_min, 'pc_max': pc_max},
        ]
        #Calibration Visée VP
import numpy as np
calibration = {
        'center': np.array([d_x/2., d_y/2, VPs[0]['z']], dtype='f'), # central point of the room  / point focal, pour lequel on optimise kinect et VPs?
        'croix': np.array([d_x, VPs[1]['y'], VPs[0]['z']], dtype='f'), # Visée Croix fond de salle x= d_x au milieu y=d_y/2
        #'croix': np.array([6.65, 3.13, 1.36], dtype='f'), # definition de la position de la croix
#        'croix': np.array([11.95, 2.2, 1.36], dtype='f'), # definition de la position de la croix
#         'roger': np.array([10.91, 6.24, 1.37], dtype='f'), #  fixation dot  (AKA Roger?)
        'roger': np.array([d_x/2., VPs[1]['y'], VPs[0]['z']], dtype='f'), # central point of the room  / point focal, pour lequel on optimise kinect et VPs?
                }
print 'DEBUG parametres , position croix: ', calibration['croix']
# parametres du champ
p = {'N': 32,
     # parametres perceptifs
     'distance_m': 0.5, # distance d'équilibre des segments autour d'une position de player
     'G_gravite_perc': 4.0, # attraction globale vers les centres des positions
     'G_gravite_perc_G': 10.0, # attraction globale vers les centres des positions
     'G_gravite_perc_R': 1., # attraction globale vers les centres des positions
     'kurt_gravitation' : -2., # 0 is normal gravity, higher makes the attraction more local, lower more global, -2 is a spring
     'G_rot_perc': 2., # Ressort, permet d'axrt et d'avoir Plus ou moins de fan
     'G_rot_perc_G': 10.,
     'G_rot_perc_R': 5.,
     'distance_tabou': 0.25, # distance tabou (perpendiculairement à l'axe VP-player)
     'G_tabou': 30., # force tabou qui expulse tout segment qui rentre dans la zone tabou (je suis completment tabou)
     'G_gravite_axis': 8.0, # parametre d'attraction physique vers les players
     'G_gravite_axis_R': 5., # parametre d'attraction physique vers les players
     'G_gravite_axis_G': 15., # parametre d'attraction physique vers les players
     # parametres physiques
     'G_poussee': 0.15, # parametre de poussee créateur de vortex
     'G_poussee_break': 3.0, # parametre de poussee créateur de vortex
     'G_struct': 0.02, # force avec laquelle les bouts de segments s'attirent
     'G_struct_G': .0, # force avec laquelle les bouts de segments s'attirent
     'G_struct_R': .1, # force avec laquelle les bouts de segments s'attirent
     'distance_struct': .2, # distance pour laquelle l'attraction des bouts de segments s'inverse
     'distance_struct_R': .13,
     'G_repulsion': .1, # constante de répulsion entre les particules
     'G_repulsion_G': .3, # force avec laquelle les bouts de segments s'attirent
     'G_repulsion_R': .05, # constante de répulsion entre les particules
     'kurt_struct' : 0., # 1 is normal gravity, higher makes the attraction more local, lower more global, -2 is a spring
     'eps': 1.e-4, # longueur (en metres) minimale pour eviter les overflows: ne doit pas avoir de qualité au niveau de la dynamique
     'G_spring': 15., 'l_seg_min': 0.4, 'l_seg_max': 2.5, 'N_max': 2, # dureté et longueur des segments
     # parametres break
     'G_spring_pulse': 30., 'l_seg_pulse': 0.43, 'N_max_pulse': 2,  # dureté et longueur des segments dans un break
     'damp_break23': .05,  # facteur de damping / absorbe l'énergie / regle la viscosité  / absorbe la péchitude
     'damp_break1': .1,  # facteur de damping / absorbe l'énergie / regle la viscosité  / absorbe la péchitude
     'speed_break': .6, # facteur global (et redondant avec les G_*) pour régler la vitesse des particules
     'T_break': 6., # duration (secondes) of breaks 2&3
     'A_break': 4., # amplitude de l'amplification de speed_0 dans les break #2 et #3
     'tau_break': .103, # duration du transient dans les breaks #2 et #3
     # parametres globaux
     'damp': 0.1,  # facteur de damping / absorbe l'énergie / regle la viscosité
     'damp_G': 0.2,  # facteur de damping / absorbe l'énergie / regle la viscosité
     'damp_R': 0.08,  # facteur de damping / absorbe l'énergie / regle la viscosité
     'speed_0': 1., # facteur global (et redondant avec les G_*) pour régler la vitesse des particules
     'scale': 21., # facteur global régler la saturation de la force - inopérant au dessus de  scale_max
     'scale_max': 20., # facteur global régler la saturation de la force - inopérant au dessus de scale_max
     'line_width': 3, # line width of segments
     }
#position_repos = ([, position[1],position[2]])
from numpy import pi
#parametres des kinects
# une liste des kinects donnant leur adresse, port, position (x; y; z) et azimuth.
# pour des kinects dans le segment (0, d_y) --- (d_x, d_y) alors  az : 11*pi/6 = a gauche , 9*pi/6 = tout droit, 7*pi/6 = a droite
info_kinects = [
		# on tourne les numeros de kinect dans le sens des aiguilles d'une montre en commencant par
           #  le point (0, 0)- le point de vue (az) donne l'ordre dans une colonne de kinects(cf document "notice")

		{'address':'10.42.0.14', 'port': 0, 'x':3.87, 'y':0.15, 'z': 1.24, 'az':3*pi/6 ,'max':600},#1.1
		{'address':'10.42.0.14', 'port': 1, 'x':3.87, 'y':0.15, 'z': 1.34, 'az':5*pi/6 ,'max':300}, #1.2
 		{'address':'10.42.0.15', 'port': 0, 'x':3.87, 'y':0.15, 'z': 1.34, 'az':pi/6 ,'max':500},#1.3
 		{'address':'10.42.0.15', 'port': 1, 'x':8.76, 'y':0.15, 'z': 1.24, 'az':3*pi/6 ,'max':650},#1.3

		{'address':'10.42.0.13', 'port': 0, 'x':6.90, 'y':10.5, 'z': 1.34, 'az':7*pi/6 ,'max':500},#1.1
		{'address':'10.42.0.13', 'port': 1, 'x':2.60, 'y':10.5, 'z': 1.14, 'az':9*pi/6 ,'max':550}, #1.2
		{'address':'10.42.0.12', 'port': 0, 'x':6.90, 'y':10.5,  'z': 1.24, 'az':9*pi/6 ,'max':500},#1.3
  		{'address':'10.42.0.12', 'port': 1, 'x':6.90, 'y':10.5, 'z': 1.14, 'az':11*pi/6 ,'max':500},#1.3

		]

run_thread_network_config = {
    'port_to_line_res' : 8005,
    'ip_to_line_res' : "10.42.0.70",
}

kinects_network_config = {
    'UDP_IP' : "",
    'UDP_PORT' : 3003,
    'send_UDP_IP' : "10.42.0.70",
    'send_UDP_PORT' : 3005,
    'para_data' : [1 , 10, 50, 350, 5 ],
}
try:
    def sliders(p):
        import matplotlib as mpl
        #mpl.rcParams['interactive'] = True
        #mpl.rcParams['backend'] = 'macosx'
        #mpl.rcParams['backend_fallback'] = True
        mpl.rcParams['toolbar'] = 'None'
        import pylab as plt
        fig = plt.figure(1)
        f_manager = plt.get_current_fig_manager()
        # f_manager.window.move(0, 0) does not work on MacOsX
        f_manager.set_window_title(" Quand c'est trop c'est tropico, COCO ")
        plt.ion()
        # turn interactive mode on for dynamic updates.  If you aren't in interactive mode, you'll need to use a GUI event handler/timer.
        from matplotlib.widgets import Slider as slider_pylab
        ax, value = [], []
        n_key = len(p.keys())*1.
    #    print s.p.keys()
        liste_keys = p.keys()
        liste_keys.sort()
        for i_key, key in enumerate(liste_keys):
            ax.append(fig.add_axes([0.15, 0.05+i_key/(n_key-1)*.9, 0.6, 0.05], axisbg='lightgoldenrodyellow'))
            if p[key] > 0:
                value.append(slider_pylab(ax[i_key], key, 0., (p[key] + (p[key]==0)*1.)*10, valinit=p[key]))
            elif p[key] < 0:
                value.append(slider_pylab(ax[i_key], key,  -(p[key] + (p[key]==0)*1.)*10, 0., valinit=p[key]))
            else:
                value.append(slider_pylab(ax[i_key], key,  -(p[key] + (p[key]==0)*1.)*10, (p[key] + (p[key]==0)*1.)*10, valinit=p[key]))

        def update(val):
            print '-'*80
            for i_key, key in enumerate(liste_keys):
                p[key]= value[i_key].val
                print key, p[key]#, value[i_key].val
            plt.draw()

        for i_key, key in enumerate(liste_keys): value[i_key].on_changed(update)
        plt.show(block=False) # il faut pylab.ion() pour pas avoir de blocage
        return fig
except Exception, e:
    print('problem while importing sliders ! Error = ', e)

if __name__ == "__main__":
    import sys
#    print sys.argv[0], str(sys.argv[1]), sys.argv[2] # nom du fichier, param1 , param2
    import display_modele_dynamique
    #sys.path.append('network')
    #import modele_dynamique_server
