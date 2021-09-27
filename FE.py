import numpy as np
from scipy.sparse import coo_matrix
import numpy.matlib
import cvxopt
import cvxopt.cholmod
#-----------------------#

class FE:
    PHYSICS_OPTIONS = {'Structural':0, 'Thermal':1}

    def getDMatrix(self, materialProperty):
        if(self.physics == self.PHYSICS_OPTIONS['Structural']):
            E= 1
            nu= materialProperty['nu'];
            k=np.array([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,-1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8])
            KE = E/(1-nu**2)*np.array([ [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
            [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
            [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
            [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
            [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
            [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
            [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
            [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]] ]);
        else:
            KE = np.array([ 2/3, -1/6, -1/3, -1/6,\
              -1/6,  2/3, -1/6, -1/3,\
              -1/3, -1/6,  2/3, -1/6,\
              -1/6, -1/3, -1/6,  2/3]);
        return (KE); #
    #-----------------------#
    def __init__(self, mesh, matProp, bc):
        self.physics = self.PHYSICS_OPTIONS[bc['physics']];
        self.mesh = mesh;
        self.matProp = matProp;
        self.bc = bc;
        self.penal = matProp['penal'];

        if(self.physics == self.PHYSICS_OPTIONS['Structural']):
            self.numDOFPerNode = 2;
            self.structMaterial = matProp;
            self.Emax = matProp['E'];
        else:
            self.numDOFPerNode = 1;
            self.thermalMaterial = matProp;
            self.Emax = matProp['K'];

        self.KE = self.getDMatrix(matProp);
        self.initializeRectangularGeometry(mesh, bc);

    #-----------------------#
    def initializeRectangularGeometry(self, mesh, bc):
        self.nelx = mesh['nelx'];
        self.nely = mesh['nely'];
        self.elemSize = mesh['elemSize'];
        self.numElems = self.nelx*self.nely;
        self.ndof = self.numDOFPerNode*(self.nelx+1)*(self.nely+1);
        self.fixed = bc['fixed'];
        self.free = np.setdiff1d(np.arange(self.ndof),self.fixed);
        self.f = bc['force'];

        self.elemNodes = np.zeros((self.numElems, 4));
        for elx in range(self.nelx):
            for ely in range(self.nely):
                el = ely+elx*self.nely;
                n1=(self.nely+1)*elx+ely
                n2=(self.nely+1)*(elx+1)+ely
                self.elemNodes[el,:] = np.array([n1+1, n2+1, n2, n1]);
        self.elemNodes = self.elemNodes.astype(int)
        self.edofMat=np.zeros((self.nelx*self.nely,4*self.numDOFPerNode),dtype=int);
        if(self.physics == self.PHYSICS_OPTIONS['Structural']):
            for elx in range(self.nelx):
                for ely in range(self.nely):
                    el = ely+elx*self.nely
                    n1=(self.nely+1)*elx+ely
                    n2=(self.nely+1)*(elx+1)+ely
                    self.edofMat[el,:]=np.array([2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3,2*n2, 2*n2+1, 2*n1, 2*n1+1])
        else:
            nodenrs = np.reshape(np.arange(0, self.ndof), (1+self.nelx, 1+self.nely)).T;
            edofVec = np.reshape(nodenrs[0:-1,0:-1]+1,self.nelx*self.nely,'F');
            self.edofMat = np.matlib.repmat(edofVec,4,1).T + np.matlib.repmat(np.array([0, self.nely+1, self.nely, -1]),self.nelx*self.nely,1);
        self.edofMat = self.edofMat.astype(int)

        self.nodeXY = np.zeros((int(self.ndof/self.numDOFPerNode),2));
        ctr = 0;
        for i in range(self.nelx+1):
            for j in range(self.nely+1):
                self.nodeXY[ctr,0] = self.elemSize[0]*i;
                self.nodeXY[ctr,1] = self.elemSize[1]*j;
                ctr += 1;

        self.elemCenters = self.generatePointsRectangularDomain();

        self.iK = np.kron(self.edofMat,np.ones((4*self.numDOFPerNode,1))).flatten()
        self.jK = np.kron(self.edofMat,np.ones((1,4*self.numDOFPerNode))).flatten()

        self.bb_xmin,self.bb_xmax,self.bb_ymin,self.bb_ymax = 0.,self.nelx*self.elemSize[0],0., self.nely*self.elemSize[1];

    #-----------------------#
    def generatePoints(self, res, includeEndPts = False):
        xy = self.generatePointsRectangularDomain(res);
        return xy;
   #-----------------------#
    def generatePointsRectangularDomain(self,  resolution = 1): # generate points in elements
        ctr = 0;
        xy = np.zeros((resolution*self.nelx*resolution*self.nely,2));

        for i in range(resolution*self.nelx):
            for j in range(resolution*self.nely):
                xy[ctr,0] = self.elemSize[0]*(i + 0.5)/resolution;
                xy[ctr,1] = self.elemSize[1]*(j + 0.5)/resolution;
                ctr += 1;

        return xy;
   #-----------------------#
    def generatePointsInMesh(self, res = 1, includeEndPts = False):
        if(includeEndPts):
            endPts = 2;
            resMin, resMax = 0, res+2;
        else:
            endPts = 0;
            resMin, resMax = 1, res+1;
        points = np.zeros((self.numElems*(res+endPts)**2,2));
        ctr = 0;
        for elm in range(self.numElems):
            nodes = self.elemNodes[elm,:];
            xmin, xmax = np.min(self.nodeXY[nodes,0]), np.max(self.nodeXY[nodes,0]);
            ymin, ymax = np.min(self.nodeXY[nodes,1]), np.max(self.nodeXY[nodes,1]);
            delX = (xmax-xmin)/(res+1.);
            delY = (ymax-ymin)/(res+1.);
            for rx in range(resMin,resMax):
                xv = xmin + rx*(delX);
                for ry in range(resMin, resMax):
                    points[ctr,0] = xv;
                    points[ctr,1] = ymin + ry*(delY);
                    ctr+= 1;
        return points;
    #-----------------------#
    def solve(self, density):
        self.densityField = density;
        self.u=np.zeros((self.ndof,1));
        sK=((self.KE.flatten()[np.newaxis]).T*((0.01 + density)**self.penal*(self.Emax))).flatten(order='F')
        K = coo_matrix((sK,(self.iK,self.jK)),shape=(self.ndof,self.ndof)).tocsc()
        K = self.deleterowcol(K,self.fixed,self.fixed).tocoo()
        K = cvxopt.spmatrix(K.data,K.row.astype(np.int),K.col.astype(np.int))
        B = cvxopt.matrix(self.f[self.free,0])
        cvxopt.cholmod.linsolve(K,B)
        self.u[self.free,0]=np.array(B)[:,0];
        n = 4*self.numDOFPerNode;
        self.Jelem = (np.dot(self.u[self.edofMat].reshape(self.numElems,n),\
                    self.KE.reshape(n,n)) * \
                    self.u[self.edofMat].reshape(self.numElems,n) ).sum(1);
        return self.u, self.Jelem;
    #-----------------------#
    def deleterowcol(self, A, delrow, delcol):
        #Assumes that matrix is in symmetric csc form !
        m = A.shape[0]
        keep = np.delete (np.arange(0, m), delrow)
        A = A[keep, :]
        keep = np.delete (np.arange(0, m), delcol)
        A = A[:, keep]
        return A
    #-----------------------#
