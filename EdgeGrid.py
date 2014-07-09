import numpy as np
import matplotlib.pyplot as plt

class EdgeGrid():
    def __init__(self):

        self.figsize = 16
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
            self.lines = self.set_lines()
            a.add_collection(self.lines)
            a.axis(c='b', lw=0)

            pylab.setp(a, xticks=[])
            pylab.setp(a, yticks=[])

            marge = self.lame_length*3.
            a.axis(self.lames_minmax + np.array([-marge, +marge, -marge, +marge]))
        else:
            self.update_lines()
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
  