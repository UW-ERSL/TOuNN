import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

class Plotter:
    def plotConvergence(self, convg):
        plt.figure();
        for key in convg:
            y = np.array(convg[key]);
            plt.semilogy(y, label = str(key));
            plt.xlabel('Iterations');
            plt.ylabel(str(key));
            plt.grid('True')
            plt.figure();

    def plotDensity(self, xy, density, titleStr):
        fig, ax = plt.subplots();
        plt.subplot(1,1,1);
        plt.imshow(-np.flipud(density.T), cmap='gray',\
                    interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
        plt.axis('Equal')
        plt.title(titleStr)
        plt.grid(False)
        fig.canvas.draw();
        plt.pause(0.01)
