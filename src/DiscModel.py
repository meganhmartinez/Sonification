import numpy as np
import time
import h5py

# for the Laguerre polynomials
from scipy.special import eval_genlaguerre

# for interpolation
from scipy import interpolate

# for resampling technology
#from lintsampler import LintSampler
#import lintsampler

# if you leave LaguerreAmplitudes in a different file
from src.FLEXbase import LaguerreAmplitudes



class DiscGalaxy(object):


    def __init__(self,N=None,phasespace=None,a=1.,M=1.,vcirc=200.,rmax=100.):

        self.a = a # scale length of the disc
        self.M = M # total mass of the disc
        self.vcirc = vcirc # circular velocity of the disc
        self.rmax = rmax*a   # maximum radius of the disc

        if N is not None:
            self.N = N # number of particles in the disc
            self.x,self.y,self.z,self.u,self.v,self.w = self._generate_basic_disc_points()

        else:
            self.x,self.y,self.z,self.u,self.v,self.w = phasespace
            self.N = len(self.x)
            
        self.mass=np.ones_like(self.x) * (self.M/self.N)  # assume equal mass for each particle
    def _generate_basic_disc_points(self):
        """generate a flat exponential disc, just for demo purposes"""
        
        x = np.linspace(0.,self.rmax,10000)

        # define the mass enclosed for an exponential disc
        def menclosed(r,a=self.a):
            return (1.0 - np.exp(-r/a)*(1.0+r/a))
        
        f = interpolate.interp1d(menclosed(x),x)

        # pull a bunch of points: pick a random radius in the disc
        np.random.seed(42)  # for reproducibility: this might need to go somewhere else?
        m = np.random.rand(self.N)
        r = f(m)

        # pick a random azimuthal angle
        p = 2.*np.pi*np.random.rand(self.N)

        x = r*np.cos(p)
        y = r*np.sin(p)
        z = r*0.0 # perfectly flat!
        # give them a perfect fixed circular velocity
        # this is a place we could upgrade, e.g. np.tanh(r/scale) instead of np.ones(r.size)
        # plus adding bar velocities or something (but then we'd want to add bar density, probably)
        u = self.vcirc*np.sin(p)*np.ones(r.size)
        v = self.vcirc*np.cos(p)*np.ones(r.size)
        w = r*0.0
        
        
        return x,y,z,u,v,w

    @staticmethod
    def make_rotation_matrix(xrotation,yrotation,zrotation,euler):
        
        radfac = np.pi/180.

        # set rotation in radians
        a = xrotation*radfac#np.pi/2.2  # xrotation (the tip into/out of page)
        b = yrotation*radfac#np.pi/3.   # yrotation
        c = zrotation*radfac#np.pi      # zrotation

        # construct the rotation matrix TAIT-BRYAN method (x-y-z,
        # extrinsic rotations)
        Rx = np.array([[1.,0.,0.],[0.,np.cos(a),np.sin(a)],[0.,-np.sin(a),np.cos(a)]])
        Ry = np.array([[np.cos(b),0.,-np.sin(b)],[0.,1.,0.],[np.sin(b),0.,np.cos(b)]])
        Rz = np.array([[np.cos(c),np.sin(c),0.,],[-np.sin(c),np.cos(c),0.],[0.,0.,1.]])
        Rmatrix = np.dot(Rx,np.dot(Ry,Rz))

        # construct the rotation matrix EULER ANGLES (z-x-z) (phi, theta,
        # psi)
        # follow the Wolfram Euler angle conventions
        if euler:
            phi = a
            theta = b
            psi = c
            D = np.array([[np.cos(phi),np.sin(phi),0.,],[-np.sin(phi),np.cos(phi),0.],[0.,0.,1.]])
            C = np.array([[1.,0.,0.],[0.,np.cos(theta),np.sin(theta)],[0.,-np.sin(theta),np.cos(theta)]])
            B = np.array([[np.cos(psi),np.sin(psi),0.,],[-np.sin(psi),np.cos(psi),0.],[0.,0.,1.]])
            Rmatrix = np.dot(B,np.dot(C,D))
            
        return Rmatrix


        

    def rotate_disc(self,xrotation=0.,yrotation=0.,zrotation=0.,euler=False):
        '''
        rotate_point_vector
            take a collection of 3d points and return the positions rotated by a specified set of angles
        inputs
        ------------------
        A           : input set of points
        xrotation   : rotation into/out of page around x axis, in degrees (inclination)
        yrotation   : rotation into/out of page around y axis, in degrees
        zrotation   : rotation in the plane of the page (z axis), in degrees
        euler       : boolean
            if True, transform as ZXZ' convention
        returns
        ------------------
        B           : the rotated phase-space output
        '''

        x,y,z = self.x,self.y,self.z
        u,v,w = self.u,self.v,self.w

        Rmatrix = self.make_rotation_matrix(xrotation,yrotation,zrotation,euler=euler)

        #
        # do the transformation in position
        tmp = np.dot(np.array([x,y,z]).T,Rmatrix)
        
        try:
            xout = tmp[:,0]
            yout = tmp[:,1]
            zout = tmp[:,2]
        except:
            xout = tmp[0]
            yout = tmp[1]
            zout = tmp[2]

        # and in velocity
        tmpv = np.dot(np.array([u,v,w]).T,Rmatrix)

        try:
            uout = tmpv[:,0]
            vout = tmpv[:,1]
            wout = tmpv[:,2]
        except:
            uout = tmpv[0]
            vout = tmpv[1]
            wout = tmp[v2]        

        self.x = xout
        self.y = yout
        self.z = zout
        self.u = uout
        self.v = vout
        self.w = wout

    @staticmethod
    def _angle_from_faceon(xrotation,yrotation,zrotation):
        """compute the total inclination, relative to face on.
        
        we're doing it this way because inclination is degenerate with the two dimensions into the page, 
        so we just want a rough idea of how to correct.
        """
        x = np.array([0,0,1.0])
        Rmatrix = make_rotation_matrix(xrotation,yrotation,zrotation)
        y = rotate_point_vector([0,0,1],Rmatrix)
        print('Angle from faceon: ',(180./np.pi)*np.arccos(np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))))


    def generate_image(self,rmax,nbins,noiselevel=-1.0):

        x_range = (-rmax, rmax)  # range for the x-axis
        y_range = (-rmax, rmax)  # range for the y-axis

        # Compute the 2D histogram
        img, self.x_edges, self.y_edges = np.histogram2d(self.x, self.y, bins=[nbins, nbins],range=[x_range, y_range])

        self.img = img.T

        if noiselevel > -1.0:
            self.noisyimage = self.img + np.random.normal(0,noiselevel,self.img.shape)

        # Calculate bin centers for the x-axis
        self.x_centers = (self.x_edges[:-1] + self.x_edges[1:]) / 2
        self.y_centers = (self.y_edges[:-1] + self.y_edges[1:]) / 2

    
    def make_expansion(self,mmax,nmax,rscl,xmax=10000.,noisy=False): #expands the galaxy image into Laguerre 
        try:
            snapshot = self.img
        except:
            print('No image data to expand... run generate_image first.')
            return
        
        if noisy:
            snapshot = self.noisyimage

        dx = self.x_edges[1]-self.x_edges[0]
        xpix,ypix = np.meshgrid(self.x_centers,self.y_centers,indexing='ij')
        rr,pp = np.sqrt(xpix**2+ypix**2),np.arctan2(ypix,xpix)

        rval = np.sqrt(xpix**2+ypix**2).reshape(-1,)
        phi  = np.arctan2(ypix,xpix).reshape(-1,)
        """ snapshotflat = snapshot.reshape(-1,) * (dx*dx)

        # create a mask for pixels outside the maximum radius
        gvals = np.where(rval>xmax)

        #rval[gvals]         = np.nan
        #phi[gvals]          = np.nan
        snapshotflat[gvals] = np.nan

        laguerre = LaguerreAmplitudes(rscl,mmax,nmax,rval,phi,snapshotflat) """
        snapshotflat = snapshot.reshape(-1,)
        # mask out pixels beyond xmax
        gvals = rval > xmax
        snapshotflat[gvals] = np.nan

        # — NEW: only keep the unmasked points for R, φ, and “mass” —
        mask       = ~np.isnan(snapshotflat)
        rval_mask  = rval [mask]
        phi_mask   = phi  [mask]
        mass_mask  = snapshotflat[mask]

        # call LaguerreAmplitudes on the masked arrays so all lengths agree
        laguerre = LaguerreAmplitudes(
            rscl, mmax, nmax,
            rval_mask,      # 1D array of radii
            phi_mask,       # 1D array of angles
            mass=mass_mask  # 1D array of pixel masses
        )
        self.r = rr
        self.p = pp
        return laguerre
    
    def make_pointexpansion(self, mmax, nmax, rscl,noisy=False): #Expands the galaxy points 
        if self.x is None or self.y is None:
            raise ValueError("Particle positions not initialized. Cannot compute expansion.")

        rr = np.sqrt(self.x**2 + self.y**2)
        pp = np.arctan2(self.y, self.x)

        mass = np.ones_like(rr) * (self.M / self.N)  # assume equal mass
        laguerre = LaguerreAmplitudes(rscl, mmax, nmax, rr, pp, mass,)

        # Save R and phi for possible reconstruction later
        self.r = rr
        self.p = pp
        return laguerre



    def resample_expansion(self,E):
        def rndmpdf(X): return np.random.uniform()
        g = lintsampler.DensityGrid((self.x_centers,self.x_centers), rndmpdf)

        E.laguerre_reconstruction(self.r, self.p)
        g.vertex_densities = E.reconstruction.T/(2.*np.pi)
            
        g.masses = g._calculate_faverages() * g._calculate_volumes()
        g._total_mass = np.sum(g.masses)
        pos = LintSampler(g).sample(self.N)
        return pos
        

    def compute_a1(self,E):
        A1 = np.linalg.norm(np.linalg.norm([E.coscoefs,E.sincoefs],axis=2)[:,1])
        A0 = np.linalg.norm(np.linalg.norm([E.coscoefs,E.sincoefs],axis=2)[:,0])
        return A1/A0


def SaveCoeff(galaxy_id, fits_files, filename):
    """
    Process a galaxy by extracting data, computing Laguerre amplitudes, and generating plots.

    Parameters:
    galaxy_id (int): The ID of the galaxy to be processed.
    fits_files (list of str): List of paths to the FITS files.
    filename (str): Path to the mass catalog CSV file.

    Returns:
    - Cos and Sine arrays with size (new_mmax * new_nmax * num_realizations)
    - Stored on a HDF5 file with titled formatted as '{galaxy_id:05d}_error.hdf5'
    """
    
    # Define HDF5 filename where centre and scale length values are stored
    filepath = f"EGS_{galaxy_id:05d}.hdf5"  
    
    # Loop over each filter
    for filter_name in filters:
        
        # Create arrays to hold the coefficients for all realizations
        coscoefs_array = np.zeros((new_mmax, new_nmax, num_realizations))
        sincoefs_array = np.zeros((new_mmax, new_nmax, num_realizations))

        # Loop over the number of realizations needed for the task 
        for realization in range(num_realizations):
            
            # Extract image pixel values from FITS file for the current filter
            extractor = GalaxyDataExtractor(fits_files, filename, [galaxy_id])
            extractor.process_galaxies()
                
            # Open the HDF5 file where the numerous realizations of the galaxy are stored. 
            with h5py.File(f"EGS(error)_{galaxy_id:05d}.hdf5", "a") as f:
                
                # Create the group name for the current filter
                filter_group = f.require_group(filter_name)

                # Create an instance of LaguerreAmplitudes and read snapshot data
                L = LaguerreAmplitudes(rscl_initial, mmax_initial, nmax_initial)
                
                try:
                    rr, pp, xpix, ypix, fixed_image, xdim, ydim = L.readsnapshot(f"EGS(error)_{galaxy_id:05d}.hdf5", filter_name)
                    
                except KeyError as e:
                    print(f"Error processing filter {filter_name} for galaxy ID {galaxy_id}: {e}")
                    continue  # Skip to the next realization
                
                # Calculate the Laguerre amplitudes
                L.laguerre_amplitudes()

                # Update center and scale length
                L.read_center_values(filepath, f"{galaxy_id}")
                
                # Read in HDF5 file for scale parameter value and set the value
                with h5py.File(filepath, "r") as c:
                    best_rscl = c['f444w'][f'{galaxy_id}']['expansion'].attrs['scale_length']
                    
                L.rscl = best_rscl
                
                # Update orders and recalculate the Laguerre amplitudes
                L.update_orders(new_mmax, new_nmax)
                L.laguerre_amplitudes()

                # Store the coefficients for this realization
                coscoefs_array[:, :, realization] = L.coscoefs
                sincoefs_array[:, :, realization] = L.sincoefs

        # Save the coefficients and other relevant data to the HDF5 file
        with h5py.File(f"{galaxy_id:05d}_error.hdf5", "a") as a:
            
            # Create the group name for the current filter
            filter_group = a.require_group(filter_name)

            # Create datasets for the coefficients
            dset_cos = filter_group.create_dataset(f"{galaxy_id}/expansion/coscoefs", data=coscoefs_array)
            dset_sin = filter_group.create_dataset(f"{galaxy_id}/expansion/sincoefs", data=sincoefs_array)

            '''
            
            Verify that the dataset contents are unique for each realisation by printing these statements
            
            print(f"Dataset 'coscoefs' contents for filter {filter_name}:", dset_cos[:])
            print(f"Dataset 'sincoefs' contents for filter {filter_name}:", dset_sin[:])
            
            '''
        print(f"{galaxy_id} coefficients for filter {filter_name} saved successfully.")
        

# Constants
rscl_initial = 10
mmax_initial = 2
nmax_initial = 10
rscl_values = np.linspace(1, 20, 100)
new_mmax = 2
new_nmax = 24
num_realizations = 100
filters = ['f444w', 'f356w', 'f277w', 'f200w', 'f115w', 'f410m', 'f125w', 'f160w', 'f606w', 'f814w']
        