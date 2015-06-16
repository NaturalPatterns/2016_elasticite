#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
"""
Q
Modèle dynamique
================

Rassemble les fonctions:
    - de calcul de trigonométrie sphérique,
    - de calcul du champ d'interaction des segments
    - de l'intégration de ce champ en mouvement

"""
from parametres import DEBUG
import numpy as np
import time

def arcdistance(rae1, rae2):
    """
    renvoie l'angle sur le grand cercle (en radians)

    # rae1 ---> rae2

     r = distance depuis le centre des coordonnées sphériques (mètres)
     a =  azimuth = declinaison = longitude (radians)
     e =  elevation = ascension droite = lattitude (radians)

    http://en.wikipedia.org/wiki/Great-circle_distance
    http://en.wikipedia.org/wiki/Vincenty%27s_formulae
    """
    a =  (np.cos(rae2[2, ...]) * np.sin(rae2[1, ...] - rae1[1, ...]))**2
    a += (np.cos(rae1[2, ...]) * np.sin(rae2[2, ...]) -  np.sin(rae1[2, ...]) *  np.cos(rae2[2, ...]) * np.cos(rae2[1, ...] - rae1[1, ...]))**2
    b =   np.sin(rae1[2, ...]) * np.sin(rae2[2, ...]) +  np.cos(rae1[2, ...]) *  np.cos(rae2[2, ...]) * np.cos(rae2[1, ...] - rae1[1, ...])
#
    #return np.arctan(np.sqrt(a) / b)
    return np.arctan2(np.sqrt(a), b)
#     print b.min(), b.max()
#     return np.arccos(b)

def orientation(rae1, rae2):
    """
    renvoie le cap suivant le grand cercle (en radians)

     r = distance depuis le centre des coordonnées sphériques (mètres)
     a =  azimuth = declinaison = longitude (radians)
     e =  elevation = ascension droite = lattitude (radians)

     http://en.wikipedia.org/wiki/Great-circle_navigation

    """
    return np.arctan2(np.sin(rae2[1, ...] - rae1[1, ...]), np.cos(rae1[2, ...])*np.tan(rae2[2, ...]) - np.sin(rae1[2, ...])*np.cos(rae2[1, ...] - rae1[1, ...]))

def xyz2azel(xyz, OV = np.zeros((3,)), eps=1.e-3):
    """
    renvoie le vecteur de coordonnées perceptuelles en fonction des coordonnées physiques

    xyz = 3 x N x ...

    Le vecteur OV désigne le centre des coordonnées sphériques,
    - O est la référence des coordonnées cartésiennes et
    - V les coordonnées cartesiennes du centre (typiquement du videoprojecteur).

    cf. https://en.wikipedia.org/wiki/Spherical_coordinates

    """
    rae = np.zeros(xyz.shape)
#     print rae.shape, xyz, VP
    if (rae.ndim > 1): OV = OV[:, np.newaxis]
    if (rae.ndim > 2): OV = OV[:, np.newaxis]
    rae[0, ...] = np.sqrt(np.sum((xyz - OV)**2, axis=0))
#     rae[1, ...] = np.arctan2((xyz[1, ...] - OV[1, ...]), (xyz[0, ...] - OV[0, ...]))
    rae[1, ...] = np.arctan2(xyz[1, ...] - OV[1, ...], xyz[0, ...] - OV[0, ...])
#     rae[2, ...] = np.arctan2(xyz[2, ...] - OV[2], rae[0, ...])
    rae[2, ...] = np.arcsin((xyz[2, ...] - OV[2, ...])/(rae[0, ...] + eps))
    return rae

def rae2xyz(rae, OV = np.zeros((3,))):
    """
    renvoie le vecteur de coordonnées physiques en fonction des coordonnées perceptuelles

    cf. https://en.wikipedia.org/wiki/Spherical_coordinates

    """
    xyz = np.zeros(rae.shape)
    xyz[0, ...] = rae[0, ...] * np.cos(rae[2, ...])  * np.cos(rae[1, ...]) + OV[0]
    xyz[1, ...] = rae[0, ...] * np.cos(rae[2, ...])  * np.sin(rae[1, ...]) + OV[1]
    xyz[2, ...] = rae[0, ...] * np.sin(rae[2, ...]) + OV[2]
    return xyz


class Scenario:
    def __init__(self, N, scenario, volume, VPs, p, calibration):
        self.t = time.time()
        self.scenario = scenario
        self.volume = volume
        self.center = calibration['center'] # central point of the room  / point focal, pour lequel on optimise kinect et VPs?
        self.croix =  calibration['croix'] # definition de la position de la croix
        self.roger =  calibration['roger'] #  fixation dot  (AKA Roger?)
        self.VPs = VPs # le dictionnaire avec les characteristiques de tous les VPs
        self.vps = [] # la liste des VPs qui ont utilisés, et que ceux-là
        for VP in self.VPs:
            self.vps.append(np.array([VP['x'], VP['y'], VP['z']]))
        self.nvps = len(self.vps)
        self.p = p
        self.N = N
        self.l_seg = p['l_seg_min'] * np.ones(N*self.nvps)
        self.l_seg_normal = p['l_seg_min'] * np.ones(N*self.nvps)
        self.l_seg_pulse = p['l_seg_min'] * np.ones(N*self.nvps)
        for i_VP, OV in enumerate(self.vps[:]):
            self.l_seg_normal[i_VP*self.N:((i_VP+1)*self.N-self.p['N_max'])] = self.p['l_seg_min'] * np.ones(self.N-self.p['N_max'])
            self.l_seg_normal[((i_VP+1)*self.N-self.p['N_max']):(i_VP+1)*self.N] = self.p['l_seg_max']
            self.l_seg_pulse[i_VP*self.N:((i_VP+1)*self.N-self.p['N_max_pulse'])] = self.p['l_seg_pulse'] * np.ones(self.N-self.p['N_max_pulse'])
            self.l_seg_pulse[((i_VP+1)*self.N-self.p['N_max_pulse']):(i_VP+1)*self.N] = self.p['l_seg_max']
        self.order = 2
        self.t_break = 0.
        self.init()
        self.particles_init = self.particles.copy()

    def init(self):
        # initialisation des particules
        self.particles = np.zeros((6*self.order, self.N*self.nvps), dtype='f') # x, y, z, u, v, w
        self.particles[0:3, :] = self.center[:, np.newaxis]
        self.particles[3:6, :] = self.center[:, np.newaxis]
        d_x, d_y, d_z = self.volume
        self.particles[1, :] = np.linspace(0., d_y, self.N*self.nvps)
        self.particles[4, :] = self.particles[1, :]
        self.particles[2, :] = d_z # tombent du plafond
        self.particles[5, :] = d_z - self.l_seg

    def champ(self, positions, events):
        N = self.N
        force = np.zeros((6, self.N*self.nvps)) # one vector per point
        # parametres
        G_tabou = self.p['G_tabou']
        distance_tabou = self.p['distance_tabou']
        G_spring = self.p['G_spring']
        G_repulsion = self.p['G_repulsion']
        G_gravite_perc = self.p['G_gravite_perc']
        G_rot_perc = self.p['G_rot_perc']
        G_repulsion = self.p['G_repulsion']
        G_struct = self.p['G_struct']
        distance_struct = self.p['distance_struct']
        G_poussee = self.p['G_poussee']
        G_gravite_axis = self.p['G_gravite_axis']
        # phases
        if events == [0, 0, 0, 0, 1, 0, 0, 0] or (events == [1, 1, 1, 1, 1, 1, 0, 0]): # phase avec la touche G dans display_modele_dynamique.py
            G_rot_perc = self.p['G_rot_perc_G']
            G_gravite_perc = self.p['G_gravite_perc_G']
            G_struct = self.p['G_struct_G']
            #G_poussee = 0.
            G_gravite_axis = self.p['G_gravite_axis_G']
            G_repulsion = self.p['G_repulsion_G']
        elif events == [1, 0, 0, 0, 0, 0, 0, 0] or (events == [1, 1, 1, 1, 1, 1, 1, 0]): # phase avec la touche R dans display_modele_dynamique.py
            G_rot_perc = self.p['G_rot_perc_R']
            G_gravite_perc = self.p['G_gravite_perc_R']
            G_gravite_axis = self.p['G_gravite_axis_R']
            G_struct = self.p['G_struct_R']
            distance_struct = self.p['distance_struct_R']
            G_repulsion =  self.p['G_repulsion_R']
            #G_poussee = 0.

        speed_0 = self.p['speed_0']
        damp = self.p['damp']
        if events == [0, 0, 0, 0, 1, 0, 0, 0]: # phase avec la touche G dans display_modele_dynamique.py
            damp = self.p['damp_G']
        elif events == [1, 0, 0, 0, 0, 0, 0, 0]: # phase avec la touche R dans display_modele_dynamique.py
            damp = self.p['damp_R']
        # pulse
        if events[1] == 1 and not(events[:6] == [1, 1, 1, 1, 1, 1]):
            self.l_seg = self.l_seg_pulse
            #self.l_seg[-self.p['N_max_pulse']:] = self.p['l_seg_max']
            G_spring = self.p['G_spring_pulse']
            #force += 4. * np.random.randn(6, self.N*self.nvps)
        else:  # cas général
            # événement Pulse avec la touche P dans display_modele_dynamique.py (Pulse)
            self.l_seg = self.l_seg_normal
            G_spring = self.p['G_spring']

        # événements (breaks)
        # ===================

        # les breaks sont signés par events[:6] == [1, 1, 1, 1, 1, 1], puis =
        # 1 : events[6:] == [1, 1]
        # 2 : events[6:] == [1, 0]
        # 3 : events[6:] == [0, 0]

        # initialize t_break at its onset - touche B
        if (events[:6] == [1, 1, 1, 1, 1, 1]) and (self.t_break == 0.):
            self.t_break = self.t

        if (events == [1, 1, 1, 1, 1, 1, 1, 1]) : # break 1 - touche B
            if DEBUG: print 'DEBUG, on est dans le break 1, on compte à rebours ',  self.t - self.t_break, speed_0
            G_poussee = self.p['G_poussee_break']
            speed_0 = self.p['speed_break']
            damp = self.p['damp_break1']
#        if events[7] == 1  and not(events[:6] == [1, 1, 1, 1, 1, 1]): # event avec la touche S dans display_modele_dynamique.py
#            damp = self.p['damp_break1']

        if not(self.t_break == 0.):# and not(events[:6] == [0, 0, 0, 0, 0, 0]):
            #if (events[:6] == [1, 1, 1, 1, 1, 1]) and (events[-1] == 0): # break #2 or #3 - touche J ou O
            if (events[-1] == 0): # break #2 or #3 - touche J ou O
                modul = np.exp(-(self.p['T_break'] - (self.t - self.t_break)) / self.p['tau_break'])
                speed_0 = self.p['speed_0'] *((self.p['A_break']-1) * modul + 1)
                #modult = 1. - 4*(self.p['T_break'] - (self.t - self.t_break)) *  (self.t - self.t_break) / self.p['T_break']**2
                modult = -1. + 2.* (self.p['T_break'] - (self.t - self.t_break)) / self.p['T_break']
                self.l_seg = self.l_seg_normal * (modult*(modult>0) + self.p['A_break'] * modul)
                damp = self.p['damp_break23']
                if DEBUG: print 'DEBUG, on est dans le break 2&3, modul      , modult, lseg ', modul, modult, modult*(modult>0) + self.p['A_break'] * modul
            # reset the break after T_break seconds AND receiving the resetting signal
            if DEBUG: print 'DEBUG, on est dans le break, on compte à rebours, speed_0 ',  self.t - self.t_break, speed_0
            if self.t > self.t_break + self.p['T_break']: self.t_break = 0.

        #print 'DEBUG, damp, speed_0 ',  damp, speed_0
        #print 'DEBUG, self.l_seg ',  self.l_seg
        #print 'DEBUG, # player ', len(positions)

        n_s = self.p['kurt_struct']
        n_g = self.p['kurt_gravitation']
        #if DEBUG: print G_gravite_axis, G_gravite_perc, G_struct, G_rot_perc, G_repulsion
        ###################################################################################################################################
        for i_VP, OV in enumerate(self.vps[:]):
            # point C (centre) du segment
            OA = self.particles[0:3, i_VP*N:(i_VP+1)*N]#.copy()
            OB = self.particles[3:6, i_VP*N:(i_VP+1)*N]#.copy()
            OC = (OA+OB)/2
            AB = OB - OA #self.particles[3:6, i_VP*N:(i_VP+1)*N] - self.particles[0:3, i_VP*N:(i_VP+1)*N]# 3 x N
            VA = OA - OV[:, np.newaxis]
            VA_0 = VA / (np.sqrt((VA**2).sum(axis=0)) + self.p['eps']) # unit vector going from the player to the center of the segment
            VB = OB - OV[:, np.newaxis]
            VB_0 = VB / (np.sqrt((VB**2).sum(axis=0)) + self.p['eps']) # unit vector going from the player to the center of the segment
            VC = OC - OV[:, np.newaxis]
            VC_0 = VC / (np.sqrt((VC**2).sum(axis=0)) + self.p['eps']) # unit vector going from the player to the center of the segment
            # FORCES SUBJECTIVES  dans l'espace perceptuel
            #if not(G_gravite_perc==0.) or not(G_gravite_axis==0.) or not(G_rot_perc==0.) or not(G_tabou==0.):
            rae_VC = xyz2azel(OC, OV)
            rae_VA = xyz2azel(OA, OV) # 3 x N
            rae_VB = xyz2azel(OB, OV) # 3 x N
            # modulation des forces en fonction de la longueur des segments
            AB = self.particles[0:3, i_VP*N:(i_VP+1)*N]-self.particles[3:6, i_VP*N:(i_VP+1)*N] # 3 x N
            distance = np.sqrt(np.sum(AB**2, axis=0)) # en metres

            # HACK            else: # on veut avoir un etat quand ya personne
            if (positions == None) or (positions == []): positions = [self.center]

            # attraction / repulsion des angles relatifs des segments
            if not(positions == None) and not(positions == []):
                distance_min = 1.e6 * np.ones((self.N)) # very big to begin with
                rotation = np.zeros((3, self.N))
                gravity = np.zeros((3, self.N))
                gravity_axis_A = np.zeros((3, self.N))
                gravity_axis_B = np.zeros((3, self.N))
                for position in positions:
                    #print 'POS', position
                    rae_VS = xyz2azel(np.array(position), OV)
                    arcdis = np.min(np.vstack((arcdistance(rae_VS, rae_VA),\
                                                arcdistance(rae_VS, rae_VB),\
                                                arcdistance(rae_VS, rae_VC))), axis=0)
                    #print 'arc distance ', arcdis
                    distance_closer = rae_VS[0]*np.sin(arcdis)
                    SC = OC - np.array(position)[:, np.newaxis]
                    SC_0 = SC / (np.sqrt((SC**2).sum(axis=0)) + self.p['eps']) # unit vector going from the player to the center of the segment

                    if n_g==-2.:
                        tabou = SC_0 * (distance_closer < distance_tabou) # en metres
                    else:
                        tabou = SC_0 * (distance_closer < distance_tabou) / (distance_closer + self.p['eps'])**(n_g+2) # en metres

                    force[0:3, i_VP*N:(i_VP+1)*N] += G_tabou * tabou
                    force[3:6, i_VP*N:(i_VP+1)*N] += G_tabou * tabou

                    # gravitation et rotation
                    arcdis = arcdistance(rae_VS, rae_VC)
                    distance_SC = rae_VS[0]*np.sin(arcdis)

                    # réduit la dimension de profondeur à une simple convergence vers la position en x / reflète la perception
                    if n_g==-2.:
                        gravity_ = - SC_0 * (distance_SC - self.p['distance_m']) # en metres
                    else:
                        gravity_ = - SC_0 * (distance_SC - self.p['distance_m'])/(distance_SC + self.p['eps'])**(n_g+2) # en metres

                    VS = np.array(position) - OV
                    VS_0 = VS / (np.sqrt((VS**2).sum(axis=0)) + self.p['eps']) # unit vector going from the player to the center of the segment
                    gravity_axis_A_ = - VA_0 * (rae_VA[0]-rae_VS[0]) # en metres
                    gravity_axis_B_ = - VB_0 * (rae_VB[0]-rae_VS[0]) # en metres
                    if False: #if DEBUG:
                        print "Convergence dans l'axe - A: ", (rae_VA[0]-rae_VS[0]).mean(), " +/- ", (rae_VA[0]-rae_VS[0]).std()
                        print "Convergence dans l'axe - B: ", (rae_VB[0]-rae_VS[0]).mean(), " +/- ", (rae_VB[0]-rae_VS[0]).std()

                    # compute desired rotation
                    cap_SC = orientation(rae_VS, rae_VC)
                    cap_AB = orientation(rae_VA, rae_VB)
                    # produit vecoriel VS /\ AB
                    rotation_ = np.sin(cap_SC-cap_AB)[np.newaxis, :] * np.cross(VS_0, SC_0, axis=0)

                    # only assign on the indices that correspond to the minimal distance (on each voronoi cell)
                    ind_assign = (distance_closer < distance_min)
                    #print ' DEBUG, pour le VP ', i_VP, ' # de points assignés ',  ind_assign.sum()
                    gravity[:, ind_assign] = gravity_[:, ind_assign]
                    gravity_axis_A[:, ind_assign] = gravity_axis_A_[:, ind_assign]
                    gravity_axis_B[:, ind_assign] = gravity_axis_B_[:, ind_assign]
                    rotation[:, ind_assign] = rotation_[:, ind_assign]
                    distance_min[ind_assign] = distance_closer[ind_assign]

                force[0:3, i_VP*N:(i_VP+1)*N] += G_gravite_axis * gravity_axis_A
                force[3:6, i_VP*N:(i_VP+1)*N] += G_gravite_axis * gravity_axis_B
                force[0:3, i_VP*N:(i_VP+1)*N] += G_gravite_perc * gravity
                force[3:6, i_VP*N:(i_VP+1)*N] += G_gravite_perc * gravity
                force[0:3, i_VP*N:(i_VP+1)*N] += G_rot_perc * rotation
                force[3:6, i_VP*N:(i_VP+1)*N] -= G_rot_perc * rotation

                # FORCES GLOBALES  dans l'espace physique
                ## forces entres les particules
                #if not(G_repulsion==0.):
                CC = OC[:, :, np.newaxis]-OC[:, np.newaxis, :] # 3xNxN ; en metres
                CC_0 = CC / (np.sqrt((CC**2).sum(axis=0)) + self.p['eps'])
                #CC_proj = CC_0 # - (VC_0[:, :, np.newaxis] * CC_0).sum(axis=0) * VC_0[:, :, np.newaxis]
                arcdis = arcdistance(rae_VC[:, :, np.newaxis], rae_VC[:, np.newaxis, :])
                #print 'arc distance ', arcdis
                distance_CC = rae_VS[0]*np.sin(arcdis) + 1.e6 * np.eye(self.N)  # NxN ; en metres
                gravity_repulsion = - np.sum( CC_0 /(distance_CC + self.p['eps'])**(n_s+2), axis=1)  #  3 x N; en metres
                # repulsion entre les centres de chaque paire de segments
                #distance_CC = np.sqrt(np.sum(CC**2, axis=0)) + 1.e6 * np.eye(self.N)  # NxN ; en metres
                #ind_plus_proche = distance_CC.argmin(axis=1)
                #for i_N in range(self.N):
                    #if n_s==-2:
                        #gravity_repuls[:, i_N] = - SC_0 * (distance_SC - self.p['distance_m']) # en metres
                    #else:
                        #gravity_repuls[:, i_N] = - SC_0 * (distance_SC - self.p['distance_m'])/(distance_SC + self.p['eps'])**(n_g+2) # en metres
#
                    #gravity_repuls[:, i_N] = CC[:, i_N, ind_plus_proche[i_N]]/(distance_CC[i_N,ind_plus_proche[i_N]] + self.p['eps'])**(n_s+2)# # 3 x N; en metres
                force[0:3, i_VP*N:(i_VP+1)*N] += G_repulsion * gravity_repulsion
                force[3:6, i_VP*N:(i_VP+1)*N] += G_repulsion * gravity_repulsion

            if not(G_poussee==0.):
                CC = OC[:, :, np.newaxis]-OC[:, np.newaxis, :] # 3xNxN ; en metres
                distance = np.sqrt(np.sum(CC**2, axis=0))# NxN ; en metres
                # poussee entrainant une rotation lente et globale (cf p152)
                ind_min = np.argmin(distance + np.eye(self.N)*1e6, axis=0)
                speed_CC = (self.particles[6:9, i_VP*N:(i_VP+1)*N] + self.particles[6:9, i_VP*N + ind_min]) + (self.particles[9:12, i_VP*N:(i_VP+1)*N] + self.particles[9:12, i_VP*N + ind_min])
                poussee =  np.sign(np.sum(speed_CC * CC[:,ind_min,:].diagonal(axis1=1, axis2=2), axis=0)) * CC[:,ind_min,:].diagonal(axis1=1, axis2=2)
                poussee /= (distance[:,ind_min].diagonal() + self.p['eps'])**(n_s+2) # 3 x N; en metres
                force[0:3, i_VP*N:(i_VP+1)*N] += G_poussee * poussee
                force[3:6, i_VP*N:(i_VP+1)*N] += G_poussee * poussee

            # attraction des extremites des segments au dessous d'une distance
            # critique pour créer des clusters de lignes
            if not(G_struct==0.):
                AA_ = self.particles[0:3, i_VP*N:(i_VP+1)*N, np.newaxis]-self.particles[0:3, np.newaxis, i_VP*N:(i_VP+1)*N]
                distance = np.sqrt(np.sum(AA_**2, axis=0)) # NxN ; en metres
                distance = distance_struct  * (distance < distance_struct) + distance * (distance > distance_struct) # NxN ; en metres
                gravity_struct = np.sum( AA_  / (np.sqrt((AA_**2).sum(axis=0)) + self.p['eps']) /(distance.T **(n_s+2) + self.p['eps']), axis=1) # 3 x N; en metres
                force[0:3, i_VP*N:(i_VP+1)*N] += G_struct * gravity_struct
                BB_ = self.particles[3:6, i_VP*N:(i_VP+1)*N, np.newaxis]-self.particles[3:6, np.newaxis, i_VP*N:(i_VP+1)*N]
                #BB_ = self.particles[0:3, :][:, :, np.newaxis]-self.particles[3:6, :][:, :, np.newaxis]
                distance = np.sqrt(np.sum(BB_**2, axis=0)) # NxN ; en metres
                distance = distance_struct  * (distance < distance_struct) + distance * (distance > distance_struct) # NxN ; en metres
                #gravity_struct = - np.sum((distance < distance_struct) * BB_/(distance.T + self.p['eps'])**(n_s+2), axis=1) # 3 x N; en metres
                gravity_struct = np.sum(BB_/ (np.sqrt((BB_**2).sum(axis=0)) + self.p['eps']) /(distance.T **(n_s+2) + self.p['eps']), axis=1) # 3 x N; en metres
                force[3:6, i_VP*N:(i_VP+1)*N] += G_struct * gravity_struct

        ## forces individuelles pour chaque segment
        # ressort
        AB = self.particles[0:3, :]-self.particles[3:6, :] # 3 x N
        distance = np.sqrt(np.sum(AB**2, axis=0)) # en metres
        AB_0 = AB / (distance[np.newaxis, :] + self.p['eps'])
        # print 'DEBUG: longueur segments ', distance.mean(), distance.std()
        force[0:3, :] -= G_spring * (distance[np.newaxis, :] - self.l_seg) * AB_0
        force[3:6, :] += G_spring * (distance[np.newaxis, :] - self.l_seg) * AB_0


        # normalisation des forces pour éviter le chaos
        #if DEBUG: print  self.particles[0:3, :].mean(axis=1)
        #if DEBUG: print 'Force ', force.mean(axis=1), force.std(axis=1)
        #force *= self.l_seg.mean()/self.l_seg*speed_0
        force *= self.l_seg_normal.mean()/self.l_seg_normal*speed_0

        if self.p['scale'] < self.p['scale_max']: force = self.p['scale'] * np.tanh(force/self.p['scale'])

        # damping
        force -= damp * self.particles[6:12, :]/self.dt

        return force

    def do_scenario(self, positions=None, events=[0, 0, 0, 0, 0, 0, 0, 0], dt=None):
        d_x, d_y, d_z = self.volume
        if (dt==None):
            self.t_last = self.t
            self.t = time.time()
            self.dt = (self.t - self.t_last)
        else:
            self.dt = dt
        #print 'DEBUG modele dyn ', self.particles.shape

        if self.scenario == 'croix':
            longueur_segments_z = .8
            longueur_segments_y = 0.8
            for i_VP, OV in enumerate(self.vps[:]):
                # ligne horizontale
                self.particles[0, i_VP*self.N:(i_VP*self.N+self.N/2)] = self.croix[0] # on the reference plane
                self.particles[1, i_VP*self.N:(i_VP*self.N+self.N/2)] = self.croix[1]
                self.particles[2, i_VP*self.N:(i_VP*self.N+self.N/2)] = self.croix[2] - longueur_segments_z/2.
                self.particles[3, i_VP*self.N:(i_VP*self.N+self.N/2)] = self.croix[0] # on the reference plane
                self.particles[4, i_VP*self.N:(i_VP*self.N+self.N/2)] = self.croix[1]
                self.particles[5, i_VP*self.N:(i_VP*self.N+self.N/2)] = self.croix[2] + longueur_segments_z/2.
                # ligne verticale
                self.particles[0, (i_VP*self.N+self.N/2):(i_VP*self.N+self.N)] = self.croix[0] # on the reference plane
                self.particles[1, (i_VP*self.N+self.N/2):(i_VP*self.N+self.N)] = self.croix[1] - longueur_segments_y/2.
                self.particles[2, (i_VP*self.N+self.N/2):(i_VP*self.N+self.N)] = self.croix[2]
                self.particles[3, (i_VP*self.N+self.N/2):(i_VP*self.N+self.N)] = self.croix[0] # on the reference plane
                self.particles[4, (i_VP*self.N+self.N/2):(i_VP*self.N+self.N)] = self.croix[1] + longueur_segments_y/2.
                self.particles[5, (i_VP*self.N+self.N/2):(i_VP*self.N+self.N)] = self.croix[2]

        elif self.scenario == 'damier':
            """
            crée un damier sur le sol (z=0)

            le damier est carré, avec N/2 lignes dans l'axe x, N/2 dans l'axe y
            il est centré au pied de la croix

            """
            longueur_damier = 7.
            grid_spacing = np.linspace(-longueur_damier/2, longueur_damier/2, self.N/2)
            for i_VP, OV in enumerate(self.vps[:]):
                # lignes dans l'axe de la salle
                self.particles[0, i_VP*self.N:(i_VP*self.N+self.N/2)] = self.croix[0] - longueur_damier/2.
                self.particles[1, i_VP*self.N:(i_VP*self.N+self.N/2)] = self.croix[1] + grid_spacing
                self.particles[2, i_VP*self.N:(i_VP*self.N+self.N/2)] = 0
                self.particles[3, i_VP*self.N:(i_VP*self.N+self.N/2)] = self.croix[0] + longueur_damier/2.
                self.particles[4, i_VP*self.N:(i_VP*self.N+self.N/2)] = self.croix[1] + grid_spacing
                self.particles[5, i_VP*self.N:(i_VP*self.N+self.N/2)] = 0
                # lignes perpendiculaires à l'axe de la salle
                self.particles[0, (i_VP*self.N+self.N/2):(i_VP*self.N+self.N)] = self.croix[0] + grid_spacing
                self.particles[1, (i_VP*self.N+self.N/2):(i_VP*self.N+self.N)] = self.croix[1] - longueur_damier/2.
                self.particles[2, (i_VP*self.N+self.N/2):(i_VP*self.N+self.N)] = 0.
                self.particles[3, (i_VP*self.N+self.N/2):(i_VP*self.N+self.N)] = self.croix[0] + grid_spacing
                self.particles[4, (i_VP*self.N+self.N/2):(i_VP*self.N+self.N)] = self.croix[1] + longueur_damier/2.
                self.particles[5, (i_VP*self.N+self.N/2):(i_VP*self.N+self.N)] = 0.


        elif self.scenario == 'damierv':
            """
            crée un damier vertical

            le damier est carré, avec N/2 lignes dans l'axe z, N/2 dans l'axe y
            il est centré autour de la croix

            """
            longueur_damier = 1.
            grid_spacing = np.linspace(-longueur_damier/2, longueur_damier/2, self.N/2)
            for i_VP, OV in enumerate(self.vps[:]):
                # lignes dans l'axe de la salle
                self.particles[0, i_VP*self.N:(i_VP*self.N+self.N/2)] = self.croix[0]
                self.particles[1, i_VP*self.N:(i_VP*self.N+self.N/2)] = self.croix[1] + grid_spacing
                self.particles[2, i_VP*self.N:(i_VP*self.N+self.N/2)] = self.croix[2] - longueur_damier/2.
                self.particles[3, i_VP*self.N:(i_VP*self.N+self.N/2)] = self.croix[0]
                self.particles[4, i_VP*self.N:(i_VP*self.N+self.N/2)] = self.croix[1] + grid_spacing
                self.particles[5, i_VP*self.N:(i_VP*self.N+self.N/2)] = self.croix[2] + longueur_damier/2.
                # lignes perpendiculaires à l'axe de la salle
                self.particles[0, (i_VP*self.N+self.N/2):(i_VP*self.N+self.N)] = self.croix[0]
                self.particles[1, (i_VP*self.N+self.N/2):(i_VP*self.N+self.N)] = self.croix[1] - longueur_damier/2.
                self.particles[2, (i_VP*self.N+self.N/2):(i_VP*self.N+self.N)] = self.croix[2] + grid_spacing
                self.particles[3, (i_VP*self.N+self.N/2):(i_VP*self.N+self.N)] = self.croix[0]
                self.particles[4, (i_VP*self.N+self.N/2):(i_VP*self.N+self.N)] = self.croix[1] + longueur_damier/2.
                self.particles[5, (i_VP*self.N+self.N/2):(i_VP*self.N+self.N)] = self.croix[2] + grid_spacing

        elif self.scenario == 'assiette':
            """
            crée une ligne horizontale sur le sol

            """
            longueur_ligne = 10.
            for i_VP, OV in enumerate(self.vps[:]):
                # lignes dans l'axe de la salle
                self.particles[0, i_VP*self.N:(i_VP*self.N+self.N/2)] = self.croix[0]
                self.particles[1, i_VP*self.N:(i_VP*self.N+self.N/2)] = self.croix[1] - longueur_ligne/2.
                self.particles[2, i_VP*self.N:(i_VP*self.N+self.N/2)] = 0.
                self.particles[3, i_VP*self.N:(i_VP*self.N+self.N/2)] = self.croix[0]
                self.particles[4, i_VP*self.N:(i_VP*self.N+self.N/2)] = self.croix[1] + longueur_ligne/2.
                self.particles[5, i_VP*self.N:(i_VP*self.N+self.N/2)] = 0.
                # lignes perpendiculaires à l'axe de la salle
                self.particles[0, (i_VP*self.N+self.N/2):(i_VP*self.N+self.N)] = self.croix[0]
                self.particles[1, (i_VP*self.N+self.N/2):(i_VP*self.N+self.N)] = self.croix[1] - longueur_ligne/2.
                self.particles[2, (i_VP*self.N+self.N/2):(i_VP*self.N+self.N)] = 0.
                self.particles[3, (i_VP*self.N+self.N/2):(i_VP*self.N+self.N)] = self.croix[0]
                self.particles[4, (i_VP*self.N+self.N/2):(i_VP*self.N+self.N)] = self.croix[1] + longueur_ligne/2.
                self.particles[5, (i_VP*self.N+self.N/2):(i_VP*self.N+self.N)] = 0.

        elif self.scenario == 'calib':
            longueur_segments, undershoot_z = .05, .0

            for i_VP, OV in enumerate(self.vps[:]):
                # ligne horizontale
                self.particles[0, i_VP*self.N:(i_VP*self.N+self.N/2)] = self.croix[0] # on the reference plane
                self.particles[1, i_VP*self.N:(i_VP*self.N+self.N/2)] = np.linspace(0, d_y, self.N/2)
                self.particles[2, i_VP*self.N:(i_VP*self.N+self.N/2)] = self.croix[2] - longueur_segments/2. - undershoot_z
                self.particles[3, i_VP*self.N:(i_VP*self.N+self.N/2)] = self.croix[0] # on the reference plane
                self.particles[4, i_VP*self.N:(i_VP*self.N+self.N/2)] = np.linspace(0, d_y, self.N/2)
                self.particles[5, i_VP*self.N:(i_VP*self.N+self.N/2)] = self.croix[2] + longueur_segments/2. - undershoot_z
                # ligne verticale
                self.particles[0, (i_VP*self.N+self.N/2):(i_VP*self.N+self.N)] = self.croix[0] # on the reference plane
                self.particles[1, (i_VP*self.N+self.N/2):(i_VP*self.N+self.N)] = self.croix[1] - longueur_segments/2.
                self.particles[2, (i_VP*self.N+self.N/2):(i_VP*self.N+self.N)] = np.linspace(0, d_z, self.N/2) - undershoot_z
                self.particles[3, (i_VP*self.N+self.N/2):(i_VP*self.N+self.N)] = self.croix[0] # on the reference plane
                self.particles[4, (i_VP*self.N+self.N/2):(i_VP*self.N+self.N)] = self.croix[1] + longueur_segments/2.
                self.particles[5, (i_VP*self.N+self.N/2):(i_VP*self.N+self.N)] = np.linspace(0, d_z, self.N/2) - undershoot_z
            #            print self.particles.mean(axis=1)

        elif self.scenario == 'cristal':
            #             self.particles = 1000*np.ones((6, self.N)) # segments outside
            frequency_plane = .005 # how fast the whole disk moves in Hz
            length = .5 # length of each AB
            mean_elevation, frequency_elevation, std_elevation = 90. * np.pi / 180., 0.3, 45. * np.pi / 180.  # elevation (in radians) of viual angle for the points of convergence defiing the cristal's circle
            radius = 1.5 # convergence happens on a circle defined as the section of sphere of radius=radius and the elevation

            N_dots = self.N # np.min(16, self.N) # number of segments
            angle = 2 * np.pi * frequency_plane * self.t + np.linspace(0, 2 * np.pi, N_dots)
            elevation = mean_elevation + std_elevation * np.sin(2 * np.pi * frequency_elevation * self.t)

            self.particles[0, :N_dots] = radius*np.cos(elevation)
            self.particles[1, :N_dots] = radius*np.sin(elevation) * np.sin(angle)
            self.particles[2, :N_dots] = radius*np.sin(elevation) * np.cos(angle)
            self.particles[3, :N_dots] = (radius+length)*np.cos(elevation)
            self.particles[4, :N_dots] = (radius+length)*np.sin(elevation) * np.sin(angle)
            self.particles[5, :N_dots] = (radius+length)*np.sin(elevation) * np.cos(angle)
            self.particles[0:3, :] += np.array(positions[0])[:, np.newaxis]
            self.particles[3:6, :] += np.array(positions[0])[:, np.newaxis]


        elif self.scenario == 'fan':
            self.particles = np.zeros(self.particles.shape)
            frequency_plane = .005 # how fast the disk moves in Hz
            radius_min, radius_max = 2.0, 5.0
            radius, length_ratio = .2 * d_z, 1.4
            N_dots = 16 #np.min(16, self.N)

#            N_dots = 50
#            radius, length_ratio = .1 * d_z, 2
            angle = 2 * np.pi * frequency_plane * self.t + np.linspace(0, 2 * np.pi, N_dots)

            # a circle drawn on a rotating plane
            self.particles[0, :N_dots] = self.center[0] #+ radius #* np.sin(angle) #* np.sin(2*np.pi*frequency_rot*self.t)
            self.particles[1, :N_dots] = self.center[1] + radius * np.sin(angle) #* np.cos(2*np.pi*frequency_rot*self.t)
            self.particles[2, :N_dots] = self.center[2] + radius * np.cos(angle)
            self.particles[3, :N_dots] = self.center[0] #+ radius * length_ratio  #* np.sin(angle) #* np.sin(2*np.pi*frequency_rot*self.t)
            self.particles[4, :N_dots] = self.center[1] + radius * length_ratio * np.sin(angle) #* np.cos(2*np.pi*frequency_rot*self.t)
            self.particles[5, :N_dots] = self.center[2] + radius * length_ratio * np.cos(angle)
#            self.particles[0:3, N_dots:] = self.origin[:, np.newaxis] # un rayon vers l'origine
#            self.particles[3:6, N_dots:] = self.origin[:, np.newaxis] + .0001 # très fin

        elif self.scenario == '2fan':
            self.particles = np.zeros(self.particles.shape)
            frequency_rot, frequency_plane = .1, .05 # how fast the whole disk moves in Hz
            radius, length_ratio = .2 * d_z, 1.4
            N_dots = np.min(16, self.N)
            angle = 2 * np.pi * frequency_plane * self.t + np.linspace(0, 2 * np.pi, N_dots)
            # a circle on the reference plane
            self.particles[0, :N_dots] = self.center[0] # on the refrerence plane
            self.particles[1, :N_dots] = self.center[1] + radius * np.sin(angle)
            self.particles[2, :N_dots] = self.center[2] + radius * np.cos(angle)
            self.particles[3, :N_dots] = self.center[0] # on the refrerence plane
            self.particles[4, :N_dots] = self.center[1] + radius * length_ratio * np.sin(angle)
            self.particles[5, :N_dots] = self.center[2] + radius * length_ratio * np.cos(angle)
            # a circle of same radius but in front going opposite sign
            self.particles[0, N_dots:2*N_dots] = self.center[0] + 1. # on the reference plane
            self.particles[1, N_dots:2*N_dots] = self.center[1] + radius * np.sin(-angle)
            self.particles[2, N_dots:2*N_dots] = self.center[2] + radius * np.cos(-angle)
            self.particles[3, N_dots:2*N_dots] = self.center[0] + 1. # on the reference plane
            self.particles[4, N_dots:2*N_dots] = self.center[1] + radius * length_ratio * np.sin(-angle)
            self.particles[5, N_dots:2*N_dots] = self.center[2] + radius * length_ratio * np.cos(-angle)
            self.particles[0:3, N_dots:] = self.center[:, np.newaxis] # un rayon vers l'origine
            self.particles[3:6, N_dots:] = self.center[:, np.newaxis] + .0001 # très fin

        elif self.scenario == 'rotating-circle':
            #             self.particles = np.zeros((6, self.N))
            frequency_rot, frequency_plane = .1, .05 # how fast the whole disk moves in Hz
            N_dots = np.min(16, self.N)
            radius, length_ratio = .3 * d_z, 2.5
            angle = 2 * np.pi * frequency_plane * self.t + np.linspace(0, 2 * np.pi, N_dots)

            # a circle drawn on a rotating plane
            self.particles[0, :N_dots] = self.center[0] + radius * np.sin(angle) * np.sin(2*np.pi*frequency_rot*self.t)
            self.particles[1, :N_dots] = self.center[1] + radius * np.sin(angle) * np.cos(2*np.pi*frequency_rot*self.t)
            self.particles[2, :N_dots] = self.center[2] + radius * np.cos(angle)
            self.particles[3, :N_dots] = self.center[0] + radius * length_ratio * np.sin(angle) * np.sin(2*np.pi*frequency_rot*self.t)
            self.particles[4, :N_dots] = self.center[1] + radius * length_ratio * np.sin(angle) * np.cos(2*np.pi*frequency_rot*self.t)
            self.particles[5, :N_dots] = self.center[2] + radius * length_ratio * np.cos(angle)
            self.particles[0:3, N_dots:] = self.center[:, np.newaxis] # un rayon vers l'origine
            self.particles[3:6, N_dots:] = self.center[:, np.newaxis] + .0001 # très fin

        elif self.scenario == 'odyssey':
            np.random.seed(12345)
            up_down = np.sign(np.random.randn(self.N)-.5)
            speed_flow, frequency_flow_trans = 20., .01 # how fast the whole disk moves in Hz
            frequency_plane_1, frequency_plane_2 = .0101, .01 # how fast the whole disk moves in Hz
#            frequency_rot_1, frequency_rot_2 = .02, .01 # how fast the whole disk moves in Hz
            radius, length, width = .3 * d_z, .8, d_y*4

            angle = 2 * np.pi * frequency_plane_1 * self.t
#            angle_rot_1 = 2 * np.pi * (frequency_rot_1 * self.t)+ np.ones(self.N)# + np.random.rand(self.N))
#            angle_rot_2 = 2 * np.pi * (frequency_rot_2 * self.t)+ np.ones(self.N)# + np.random.rand(self.N))

            # coordinates before the plane rotation
            #            x = d_x * np.sin(2*np.pi*frequency_flow*self.t)
#            x = np.mod(np.random.rand(self.N)*d_x + speed_flow*self.t, d_x)
            dx = -2*np.abs((np.linspace(0, d_x, self.N)-d_x/2))
            x = np.mod(dx + speed_flow*self.t, 3*d_x)
            y = np.linspace(-width/2, width/2, self.N) # np.random.rand(self.N)*width - width/2 #
            z = d_z / 5. * up_down
            l = np.sqrt(y**2 + z**2)
#            vector = np.array([np.zeros(self.N), y/l, z/l ])
#
#            self.particles[0, :] = x
#            self.particles[1, :] = y
#            self.particles[2, :] = z
#
#            self.particles[0:3, :] -= length/2. * vector
#            self.particles[3:6, :] += length/2. * vector
#
            # a circle drawn on a rotating plane
            self.particles[0, :] = x
            self.particles[1, :] = (y - y*length/l) * np.sin(angle) + (z - z*length/l) * np.cos(angle)
            self.particles[2, :] = (y - y*length/l) * np.cos(angle) - (z - z*length/l) * np.sin(angle)
#            self.particles[3:6, :] = self.particles[0:3, :]
            self.particles[3, :] = x
            self.particles[4, :] = (y + y*length/l) * np.sin(angle) + (z + z*length/l) * np.cos(angle)
            self.particles[5, :] = (y + y*length/l) * np.cos(angle) - (z + z*length/l) * np.sin(angle)

#            vector = np.array([np.cos(angle_rot_2)*np.cos(angle_rot_1), np.cos(angle_rot_2)*np.sin(angle_rot_1), np.sin(angle_rot_2) ])
#            vector = np.array([np.sin(angle_rot_2), np.cos(angle_rot_2)*np.cos(angle_rot_1), np.cos(angle_rot_2)*np.sin(angle_rot_1) ])
#            l = np.sqrt(self.particles[1, :]**2 + self.particles[2, :]**2)
#            vector = np.array([np.zeros(self.N), self.particles[2, :]/l, self.particles[1, :]/l ])

#            self.particles[0:3, :] -= length/2. * vector
#            self.particles[3:6, :] += length/2. * vector

            self.particles[0:3, :] += self.center[:, np.newaxis]
            self.particles[3:6, :] += self.center[:, np.newaxis]

        elif self.scenario == 'snake':
            np.random.seed(12345)
            up_down = np.sign(np.random.randn(self.N)-.5)
            speed_flow, frequency_flow_trans = 1., .01 # how fast the whole disk moves in Hz
            frequency_plane_1, frequency_plane_2 = .0101, .01 # how fast the whole disk moves in Hz
            #            frequency_rot_1, frequency_rot_2 = .02, .01 # how fast the whole disk moves in Hz
            radius, length, width = .3 * d_z, .8, d_y/4

            angle = 2 * np.pi * frequency_plane_1 * self.t
            #            angle_rot_1 = 2 * np.pi * (frequency_rot_1 * self.t)+ np.ones(self.N)# + np.random.rand(self.N))
            #            angle_rot_2 = 2 * np.pi * (frequency_rot_2 * self.t)+ np.ones(self.N)# + np.random.rand(self.N))

            # coordinates before the plane rotation
            #            x = d_x * np.sin(2*np.pi*frequency_flow*self.t)
            #            x = np.mod(np.random.rand(self.N)*d_x + speed_flow*self.t, d_x)
            #dx = -2*np.abs((np.linspace(0, d_x, self.N)-d_x/2))
#            x = np.mod(np.linspace(0, d_x, self.N) + speed_flow*self.t, d_x)
            x = d_x/2+np.sin(np.linspace(0, 2*np.pi, self.N)/3 + speed_flow*self.t) * d_x/2
            y = np.sin(np.linspace(0, 2*np.pi, self.N)/4) * width # np.random.rand(self.N)*width - width/2 #

            z = d_z / 5. #* up_down
            l = np.sqrt(y**2 + z**2)
            self.particles[0, :] = x
            self.particles[1, :] = y * np.sin(angle) + z * np.cos(angle)
            self.particles[2, :] = y * np.cos(angle) - z * np.sin(angle)

            self.particles[3:6, :] = np.roll(self.particles[0:3, :], 1, axis=1)
            self.particles[3:6, 0] = self.particles[0:3, 0]

            self.particles[0:3, :] += self.center[:, np.newaxis]
            self.particles[3:6, :] += self.center[:, np.newaxis]

        elif self.scenario == 'leapfrog':
            self.particles[:6, :] += self.particles[6:12, :] * self.dt/2
            force = self.champ(positions=positions, events=events)
            #self.particles[6:12, :] *= (1. - self.p['damp'])
            self.particles[6:12, :] += force * self.dt
            # TODO utiliser mla force comme la vitesse désirée?
            #self.particles[6:12, :] = force
            # application de l'acceleration calculée sur les positions
            self.particles[:6, :] += self.particles[6:12, :] * self.dt/2
#             if DEBUG: print 'DEBUG modele dynamique , mean A, mean B, std A, std B ', self.particles[0:3, :].mean(axis=1), self.particles[3:6, :].mean(axis=1), self.particles[0:3, :].std(axis=1), self.particles[3:6, :].std(axis=1)
            if np.isnan(self.particles[:6, :]).any():
                #print self.p
                #raise ValueError("some values like NaN breads")
                self.init()
                print("some values like NaN breads")

            pos_barrier, vmax = 1, 100.
            index_out =  self.particles[:, :] < np.array([-pos_barrier, -pos_barrier, -pos_barrier,
                                                          -pos_barrier, -pos_barrier, -pos_barrier,
                                                          -vmax, -vmax, -vmax, -vmax, -vmax, -vmax])[:, np.newaxis]
            index_out += self.particles[:, :] > np.array([d_x+pos_barrier, d_y+pos_barrier, d_z+pos_barrier,
                                                          d_x+pos_barrier, d_y+pos_barrier, d_z+pos_barrier,
                                                          vmax, vmax, vmax, vmax, vmax, vmax])[:, np.newaxis]
#             print self.particles.shape, self.particles_init.shape, index_out.shape
#             index_out_any = index_out
            if DEBUG: print 'DEBUG modele dynamique # of corrected particles coordinates ',  index_out.sum()
            self.particles[index_out] = self.particles_init[index_out]
            if DEBUG: print 'DEBUG modele dynamique POS, mean , min, std, max ', self.particles[:6, :].mean(), self.particles[:6, :].min(), self.particles[:6, :].std(), self.particles[:6, :].max(),
            if DEBUG: print 'DEBUG modele dynamique VEL, mean , min, std, max ', self.particles[6:, :].mean(), self.particles[6:, :].min(), self.particles[6:, :].std(), self.particles[6:, :].max()

        elif self.scenario == 'euler':
            force = self.champ(positions=positions, events=events)
            self.particles[6:12, :] += force * self.dt
            # application de l'acceleration calculée sur les positions
            self.particles[:6, :] += self.particles[6:12, :] * self.dt

        # pour les scenarios de controle du suivi, on centre autour de la position du premier player
        if not(positions == None) and not(positions == []) and (self.scenario in ['croix', 'fan', '2fan', 'rotating-circle', 'calibration']):
            # pour la calibration on centre le pattern autour de la premiere personne captée
            self.particles[0:3, :] -= self.croix[:, np.newaxis]
            self.particles[3:6, :] -= self.croix[:, np.newaxis]
            self.particles[0:3, :] += np.array(positions[0])[:, np.newaxis]
            self.particles[3:6, :] += np.array(positions[0])[:, np.newaxis]

if __name__ == "__main__":
    import display_modele_dynamique