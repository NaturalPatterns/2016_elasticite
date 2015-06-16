import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os

class EdgeGrid():
    def __init__(self):

        self.figsize = 13
        self.line_width = 4.

        self.N_lame = 72
        self.N_lame_X = np.int(np.sqrt(self.N_lame))#*np.sqrt(3) / 2)

        self.lames = np.zeros((3, self.N_lame))
        self.lames[0, :] = np.mod(np.arange(self.N_lame), self.N_lame_X)
        self.lames[0, :] += np.mod(np.floor(np.arange(self.N_lame)/self.N_lame_X), 2)/2
        self.lames[1, :] = np.floor(np.arange(self.N_lame)/self.N_lame_X)
        self.lames[1, :] *= np.sqrt(3) / 2
        self.lames[0, :] /= self.N_lame_X
        self.lames[1, :] /= self.N_lame_X

        self.lames_minmax = np.array([self.lames[0, :].min(), self.lames[0, :].max(), self.lames[1, :].min(), self.lames[1, :].max()])
        self.lame_length = .45/self.N_lame_X
        self.lines = self.set_lines()

        self.N_theta = 12
        self.thetas = np.linspace(0, np.pi, self.N_theta)
        self.sf_0 = .3
        self.B_sf = .3

        self.vext = '.webm'
        self.figpath = '../files/figures/elasticite/'
        self.fps = 25

    def show_edges(self, fig=None, a=None):
        """
        Shows the quiver plot of a set of edges, optionally associated to an image.

        """
        import pylab
        import matplotlib.cm as cm
        if fig==None:
            fig = pylab.figure(figsize=(self.figsize, self.figsize))
        if a==None:
            border = 0.0
            a = fig.add_axes((border, border, 1.-2*border, 1.-2*border), axisbg='w')
        else:
            self.update_lines()
        marge = self.lame_length*3.
        a.axis(self.lames_minmax + np.array([-marge, +marge, -marge, +marge]))
        a.add_collection(self.lines)
        a.axis(c='b', lw=0)
        pylab.setp(a, xticks=[])
        pylab.setp(a, yticks=[])
        pylab.draw()
        return fig, a

    def set_lines(self):
        from matplotlib.collections import LineCollection
        import matplotlib.patches as patches
        # draw the segments
        segments, colors, linewidths = list(), list(), list()

        X, Y, Theta = self.lames[0, :], self.lames[1, :].real, self.lames[2, :]
        for x, y, theta in zip(X, Y, Theta):
            u_, v_ = np.cos(theta)*self.lame_length, np.sin(theta)*self.lame_length
            segments.append([(x - u_, y - v_), (x + u_, y + v_)])
            colors.append((0, 0, 0, 1))# black
            linewidths.append(self.line_width)
        return LineCollection(segments, linewidths=linewidths, colors=colors, linestyles='solid')

    def update_lines(self):
        from matplotlib.collections import LineCollection
        import matplotlib.patches as patches
        X, Y, Theta = self.lames[0, :], self.lames[1, :], self.lames[2, :]
        segments = list()

        for i, (x, y, theta) in enumerate(zip(X, Y, Theta)):
            u_, v_ = np.cos(theta)*self.lame_length, np.sin(theta)*self.lame_length
            segments.append([(x - u_, y - v_), (x + u_, y + v_)])
        self.lines.set_segments(segments)

    def theta_E(self, im, X_, Y_, w):
        try:
            assert(self.slip.N_X==im.shape[1])
        except:
            from NeuroTools.parameters import ParameterSet
            from SLIP import Image
            from LogGabor import LogGabor
            self.slip = Image(ParameterSet({'N_X':im.shape[1], 'N_Y':im.shape[0]}))
            self.lg = LogGabor(self.slip)
        im_ = im.sum(axis=-1)
        im_ = im_ * np.exp(-.5*((.5 + .5*self.slip.x-Y_)**2+(.5 + .5*self.slip.y-X_)**2)/w**2)
        E = np.zeros((self.N_theta,))
        for i_theta, theta in enumerate(self.thetas):
            params= {'sf_0':self.sf_0, 'B_sf':self.B_sf, 'theta':theta, 'B_theta': np.pi/self.N_theta}
            FT_lg = self.lg.loggabor(0, 0, **params)
            E[i_theta] = np.sum(np.absolute(self.slip.FTfilter(np.rot90(im_, -1), FT_lg, full=True))**2)
        return E

    def theta_max(self, im, X_, Y_, w):
        E = self.theta_E(im, X_, Y_, w)
        return self.thetas[np.argmax(E)] - np.pi/2

    
    def theta_sobel(self, im, N_blur):
        im_ = im.copy()
        sobel = np.array([[1,   2,  1,],
                          [0,   0,  0,],
                          [-1, -2, -1,]])
        if im_.ndim==3: im_ = im_.sum(axis=-1)
        from scipy.signal import convolve2d
        im_X = convolve2d(im_, sobel, 'same')
        im_Y = convolve2d(im_, sobel.T, 'same')
        
        #N_X, N_Y = im_.shape
        #x, y = np.mgrid[0:1:1j*N_X, 0:1:1j*N_Y]
        #mask = np.exp(-.5*((x-.5)**2+(y-.5)**2)/w**2)  
        #im_X = convolve2d(im_X, mask, 'same')
        #im_Y = convolve2d(im_Y, mask, 'same')
        blur = np.array([[1, 2, 1],
                         [1, 8, 2],
                         [1, 2, 1]])
        for i in range(N_blur):
            im_X = convolve2d(im_X, blur, 'same')
            im_Y = convolve2d(im_Y, blur, 'same')
            
        angle = np.arctan2(im_Y, im_X)
        
        bord = .1
        angles = np.empty(self.N_lame)
        N_X, N_Y = im_.shape
        for i in range(self.N_lame):
            angles[i] = angle[int((bord+self.lames[0, i]*(1-2*bord))*N_X), 
                              int((bord+self.lames[1, i]*(1-2*bord))*N_Y)]
        return angles - np.pi/2
            
    def fname(self, name):
        return os.path.join(self.figpath, name + self.vext)
    
    def make_anim(self, name, make_lames, duration=3., redo=False):
        if redo or not os.path.isfile(self.fname(name)):

            import matplotlib.pyplot as plt
            from moviepy.video.io.bindings import mplfig_to_npimage
            import moviepy.editor as mpy

            fig_mpl, ax = plt.subplots(1, figsize=(self.figsize, self.figsize), facecolor='white')

            def make_frame_mpl(t):
                # on ne peut changer que l'orientation des lames:
                self.t = t
                self.lames[2, :] = make_lames(self)
                self.update_lines()
                fig_mpl, ax = self.show_edges()#fig_mpl, ax)
                self.t_old = t
                return mplfig_to_npimage(fig_mpl) # RGB image of the figure

            animation = mpy.VideoClip(make_frame_mpl, duration=duration)
            animation.write_videofile(self.fname(name), fps=self.fps)

    def ipython_display(self, name, loop=True, autoplay=True, controls=True):
        """
        showing the grid in the notebook by pointing at the file stored in the proper folder

        """
        import os
        from IPython.core.display import display, Image, HTML
        opts = ' '
        if loop: opts += 'loop="1" '
        if autoplay: opts += 'autoplay="1" '
        if controls: opts += 'controls '
        s = """
            <center><table border=none width=100% height=100%>
            <tr> <td width=100%><center><video {0} src="{2}" type="video/{1}"  width=100%>
            </td></tr></table></center>""".format(opts, self.vext[1:], self.fname(name))
        return display(HTML(s))