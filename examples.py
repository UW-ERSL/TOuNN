import numpy as np

example = 1
nelx, nely = 60, 30
#  ~~~~~~~~~~~~ Examples ~~~~~~~~~~~~~#
if(example == 1): # tip cantilever
    exampleName = 'TipCantilever'
    physics = 'Structural'
    meshType = 'rectGeom';
    matProp = {'E':1.0, 'nu':0.3};
    desiredVolumeFraction = 0.6; # between 0.1 and 0.9 
    ndof = 2*(nelx+1)*(nely+1);
    force = np.zeros((ndof,1))
    dofs=np.arange(ndof);
    fixed = dofs[0:2*(nely+1):1];
    force[2*(nelx+1)*(nely+1)-2*nely+1, 0 ] = -1;
    nonDesignRegion = None;
    symXAxis = {'isOn':False, 'midPt':0.5*nely};
    symYAxis = {'isOn':False, 'midPt':0.5*nelx};

elif(example == 2): # mid cantilever
    exampleName = 'MidCantilever'
    physics = 'Structural';
    meshType = 'rectGeom';
    matProp = {'E':1.0, 'nu':0.3};
    desiredVolumeFraction = 0.5; # between 0.1 and 0.9 
    ndof = 2*(nelx+1)*(nely+1);
    force = np.zeros((ndof,1))
    dofs=np.arange(ndof);
    fixed = dofs[0:2*(nely+1):1];
    force[2*(nelx+1)*(nely+1)- (nely+1), 0 ] = -1;
    nonDesignRegion = None;
    symXAxis = {'isOn':True, 'midPt':0.5*nely};
    symYAxis = {'isOn':False, 'midPt':0.5*nelx};

elif(example == 3): #  MBBBeam
    desiredVolumeFraction = 0.5; # between 0.1 and 0.9 
    exampleName = 'MBBBeam'
    physics = 'Structural'
    meshType = 'rectGeom';
    matProp = {'E':1.0, 'nu':0.3};
    ndof = 2*(nelx+1)*(nely+1);
    force = np.zeros((ndof,1))
    dofs=np.arange(ndof);
    fixed= np.union1d(np.arange(0,2*(nely+1),2), 2*(nelx+1)*(nely+1)-2*(nely+1)+1);
    force[2*(nely+1)+1 ,0]=-1;
    nonDesignRegion = None;
    symXAxis = {'isOn':False, 'midPt':0.5*nely};
    symYAxis = {'isOn':False, 'midPt':0.5*nelx};

elif(example == 4): #  Michell
    desiredVolumeFraction = 0.4; # between 0.1 and 0.9 
    exampleName = 'Michell'
    physics = 'Structural'
    meshType = 'rectGeom';
    matProp = {'E':1.0, 'nu':0.3};
    ndof = 2*(nelx+1)*(nely+1);
    force = np.zeros((ndof,1))
    dofs=np.arange(ndof);
    fixed=np.array([ 0,1,2*(nelx+1)*(nely+1)-2*nely] ); # ,2*(nelx+1)*(nely+1)-2*nely+1,
    force[nelx*(nely+1)+2 ,0]=-1;
    nonDesignRegion = None;
    symXAxis = {'isOn':False, 'midPt':0.5*nely};
    symYAxis = {'isOn':True, 'midPt':0.5*nelx};

elif(example == 5): #  DistributedMBB
    exampleName = 'Bridge'
    physics = 'Structural'
    meshType = 'rectGeom';
    desiredVolumeFraction = 0.45; # between 0.1 and 0.9 
    matProp = {'E':1.0, 'nu':0.3};
    ndof = 2*(nelx+1)*(nely+1);
    force = np.zeros((ndof,1))
    dofs=np.arange(ndof);
    fixed= np.array([ 0,1,2*(nelx+1)*(nely+1)-2*nely+1,2*(nelx+1)*(nely+1)-2*nely] );
    force[2*nely+1:2*(nelx+1)*(nely+1):8*(nely+1),0]=-1/(nelx+1);
    nonDesignRegion = None # s{'x>':0, 'x<':nelx,'y>':nely-1,'y<':nely}; # None #
    symXAxis = {'isOn':False, 'midPt':0.5*nely};
    symYAxis = {'isOn':True, 'midPt':0.5*nelx};