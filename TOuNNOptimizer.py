# TOuNN: Topology Optimization using Neural Networks
# Authors : Aaditya Chandrasekhar, Krishnan Suresh
# Affliation : University of Wisconsin - Madison
# Corresponding Author : ksuresh@wisc.edu , achandrasek3@wisc.edu
# Submitted to Structural and Multidisciplinary Optimization
# For academic purposes only

#Versions
#Numpy 1.18.1
#Pytorch 1.5.0
#scipy 1.4.1
#cvxopt 1.2.0

#%% imports
import numpy as np
import torch
import torch.optim as optim
from os import path
from FE import FE
from plotUtil import Plotter
import matplotlib.pyplot as plt
from network import TopNet
from torch.autograd import grad

#%% main TO functionalities
class TopologyOptimizer:
    #-----------------------------#
    def __init__(self, mesh, matProp, bc, nnSettings, \
                  desiredVolumeFraction, densityProjection, overrideGPU = True):

        self.exampleName = bc['exampleName'];
        self.device = self.setDevice(overrideGPU);
        self.boundaryResolution  = 3; # default value for plotting and interpreting
        self.FE = FE(mesh, matProp, bc);
        self.xy = torch.tensor(self.FE.elemCenters, requires_grad = True).\
                                        float().view(-1,2).to(self.device);
        self.xyPlot = torch.tensor(self.FE.generatePoints(self.boundaryResolution, True),\
                        requires_grad = True).float().view(-1,2).to(self.device);
        self.Pltr = Plotter();

        self.desiredVolumeFraction = desiredVolumeFraction;
        self.density = self.desiredVolumeFraction*np.ones((self.FE.numElems));
        self.symXAxis = bc['symXAxis'];
        self.symYAxis = bc['symYAxis'];

        self.densityProjection = densityProjection;

        inputDim = 2; # x and y coordn
        self.topNet = TopNet(nnSettings, inputDim).to(self.device);
        self.objective = 0.;
    #-----------------------------#
    def setDevice(self, overrideGPU):
        if(torch.cuda.is_available() and (overrideGPU == False) ):
            device = torch.device("cuda:0");
            print("GPU enabled")
        else:
            device = torch.device("cpu")
            print("Running on CPU")
        return device;

    #-----------------------------#
    def applySymmetry(self, x):
        if(self.symYAxis['isOn']):
            xv =( self.symYAxis['midPt'] + torch.abs( x[:,0] - self.symYAxis['midPt']));
        else:
            xv = x[:,0];
        if(self.symXAxis['isOn']):
            yv = (self.symXAxis['midPt'] + torch.abs( x[:,1] - self.symXAxis['midPt'])) ;
        else:
            yv = x[:,1];
        x = torch.transpose(torch.stack((xv,yv)),0,1);
        return x;

    #-----------------------------#
    def projectDensity(self, x):
        if(self.densityProjection['isOn']):
            b = self.densityProjection['sharpness']
            nmr = np.tanh(0.5*b) + torch.tanh(b*(x-0.5));
            x = 0.5*nmr/np.tanh(0.5*b);
        return x;
    #-----------------------------#
    def optimizeDesign(self,maxEpochs, minEpochs):
        self.convergenceHistory = {'compliance':[], 'vol':[], 'grayElems':[]};
        learningRate = 0.01;
        alphaMax = 100*self.desiredVolumeFraction;
        alphaIncrement= 0.05;
        alpha = alphaIncrement; # start
        nrmThreshold = 0.01; # for gradient clipping
        self.optimizer = optim.Adam(self.topNet.parameters(), amsgrad=True,lr=learningRate);

        for epoch in range(maxEpochs):

            self.optimizer.zero_grad();
            x = self.applySymmetry(self.xy);
            nn_rho = torch.flatten(self.topNet(x)).to(self.device);
            nn_rho = self.projectDensity(nn_rho);
            rho_np = nn_rho.cpu().detach().numpy(); # move tensor to numpy array
            self.density = rho_np;
            u, Jelem = self.FE.solve(rho_np); # Call FE 88 line code [Niels Aage 2013]

            if(epoch == 0):
                self.obj0 = ( self.FE.Emax*(rho_np**self.FE.penal)*Jelem).sum()
            # For sensitivity analysis, exponentiate by 2p here and divide by p in the loss func hence getting -ve sign

            Jelem = np.array(self.FE.Emax*(rho_np**(2*self.FE.penal))*Jelem).reshape(-1);
            Jelem = torch.tensor(Jelem).view(-1).float().to(self.device) ;
            objective = torch.sum(torch.div(Jelem,nn_rho**self.FE.penal))/self.obj0; # compliance

            volConstraint =((torch.mean(nn_rho)/self.desiredVolumeFraction) - 1.0); # global vol constraint
            currentVolumeFraction = np.average(rho_np);
            self.objective = objective;
            loss =   self.objective+ alpha*torch.pow(volConstraint,2);

            alpha = min(alphaMax, alpha + alphaIncrement);
            loss.backward(retain_graph=True);
            torch.nn.utils.clip_grad_norm_(self.topNet.parameters(),nrmThreshold)
            self.optimizer.step();

            greyElements= sum(1 for rho in rho_np if ((rho > 0.2) & (rho < 0.8)));
            relGreyElements = self.desiredVolumeFraction*greyElements/len(rho_np);
            self.convergenceHistory['compliance'].append(self.objective.item());
            self.convergenceHistory['vol'].append(currentVolumeFraction);
            self.convergenceHistory['grayElems'].append(relGreyElements);
            self.FE.penal = min(8.0,self.FE.penal + 0.02); # continuation scheme

            if(epoch % 20 == 0):
                titleStr = "Iter {:d} , Obj {:.2F} , vol {:.2F}".format(epoch, self.objective.item()*self.obj0, currentVolumeFraction);
                self.Pltr.plotDensity(self.xy.detach().cpu().numpy(), rho_np.reshape((self.FE.nelx, self.FE.nely)), titleStr);
                print(titleStr);
            if ((epoch > minEpochs ) & (relGreyElements < 0.025)):
                break;
        self.Pltr.plotDensity(self.xy.detach().cpu().numpy(), rho_np.reshape((self.FE.nelx, self.FE.nely)), titleStr);
        print("{:3d} J: {:.2F}; Vf: {:.3F}; loss: {:.3F}; relGreyElems: {:.3F} "\
             .format(epoch, self.objective.item()*self.obj0 ,currentVolumeFraction,loss.item(),relGreyElements));

        print("Final J : {:.3f}".format(self.objective.item()*self.obj0));
        self.Pltr.plotConvergence(self.convergenceHistory);

        x = self.applySymmetry(self.xyPlot);
        rho = torch.flatten(self.projectDensity(self.topNet(x)));
        rho_np = rho.cpu().detach().numpy();

        titleStr = "Iter {:d} , Obj {:.2F} , vol {:.2F}".format(epoch, self.objective.item()*self.obj0, currentVolumeFraction);
        self.Pltr.plotDensity(self.xyPlot.detach().cpu().numpy(), rho_np.reshape((self.FE.nelx*self.boundaryResolution, self.FE.nely*self.boundaryResolution)), titleStr);
        return self.convergenceHistory;
