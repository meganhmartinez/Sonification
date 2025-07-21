import numpy as np
from scipy.special import eval_genlaguerre
from scipy import interpolate
from scipy.optimize import curve_fit
from PIL import Image

# your existing Laguerre engine
from src.FLEXbase import LaguerreAmplitudes

class DiscGalaxy(object):
    """
    DiscGalaxy: either generate a toy exponential disc *or* ingest a real image,
    automatically fit its radius via a Sersic profile, mask everything outside,
    then expand into a Fourier–Laguerre basis.
    """

    def __init__(self, N=None, phasespace=None, a=1.0, M=1.0, vcirc=200.0, rmax=5.0):
        """
        N          : number of particles for toy disc (None if using real image)
        phasespace : tuple (x,y,z,u,v,w) if you already have points
        a          : scale length (kpc, arcsec, etc)
        M          : total mass (or flux)
        vcirc      : circular velocity for toy disc
        rmax       : maximum radius IN UNITS OF a for toy disc *and* mask
        """
        self.a           = a
        self.M           = M
        self.vcirc       = vcirc
        # this physical radius (a × rmax) is used to mask *both* toy discs and real galaxies
        self.rmax        = rmax * a
        self.galaxy_radius = None

        if N is not None:
            self.N = N
            self.x, self.y, self.z, self.u, self.v, self.w = self._generate_basic_disc_points()
        elif phasespace is not None:
            self.x, self.y, self.z, self.u, self.v, self.w = phasespace
            self.N = len(self.x)

    def _generate_basic_disc_points(self):
        """Toy exponential disc in the plane z=0, fixed circular speed."""
        rgrid = np.linspace(0, self.rmax, 10000)
        def mencl(r): return self.M * (1.0 - np.exp(-r/self.a)*(1.0 + r/self.a))
        m_enc = mencl(rgrid)
        inv_cdf = interpolate.interp1d(
            m_enc, rgrid,
            bounds_error=False,
            fill_value=(0.0, self.rmax)
        )
        # draw random radii
        np.random.seed(42)
        u_rand = np.random.rand(self.N) * m_enc[-1]
        r = inv_cdf(u_rand)
        phi = 2*np.pi*np.random.rand(self.N)

        x = r*np.cos(phi)
        y = r*np.sin(phi)
        z = np.zeros_like(r)
        u_vel = self.vcirc*np.sin(phi)
        v_vel = self.vcirc*np.cos(phi)
        w_vel = np.zeros_like(r)
        return x, y, z, u_vel, v_vel, w_vel

    def sersic_profile(self, r, I0, Reff, n):
        """Sersic profile: I(r) = I0 exp[-(r/Reff)^(1/n)]."""
        return I0 * np.exp(- (r/Reff)**(1.0/n))

    def determine_galaxy_radius(self, image_data, n_guess=1.0):
        """
        Fit a Sersic profile to the 2D image’s radial profile and return
        6× the effective radius (in PIXELS) as the mask radius.
        """
        ny, nx = image_data.shape
        y_idx, x_idx = np.indices(image_data.shape)
        cx, cy = nx//2, ny//2

        # radial distances in pixel units
        r = np.sqrt((x_idx - cx)**2 + (y_idx - cy)**2).ravel()
        I = image_data.ravel()

        # drop NaNs & zeros
        mask = np.isfinite(I) & (I > 0)
        r, I = r[mask], I[mask]

        # sort by radius
        idx = np.argsort(r)
        r_sorted, I_sorted = r[idx], I[idx]

        if len(r_sorted) < 10:
            raise ValueError("Not enough valid pixels for Sersic fit")

        # initial guesses: I0 ~ max(I), Reff ~ half image, n ~ n_guess
        p0 = (I_sorted.max(), nx/4, n_guess)
        bounds = ([0, 0, 0.1], [np.inf, max(nx,ny), 10])

        popt, _ = curve_fit(
            self.sersic_profile,
            r_sorted, I_sorted,
            p0=p0, bounds=bounds, maxfev=2000
        )
        I0, Reff_pix, n = popt
        return 6.0 * Reff_pix

    def ingest_image(self, filename, extent, nbins):
        """
        Load a real image, fit its radius, mask outside, and set up
        self.img, self.x_edges, self.y_edges, self.x_centers, self.y_centers.
        extent : physical size from center to edge (same units as a)
        nbins  : number of pixels/bins per axis
        """
        # load grayscale image
        arr = np.asarray(Image.open(filename).convert("L"), dtype=float)
        # we want shape = (nx,ny) to match meshgrid(indexing='ij'), so transpose
        self.img = arr.T

        # build edges from -extent→+extent
        self.x_edges = np.linspace(-extent, extent, self.img.shape[0]+1)
        self.y_edges = np.linspace(-extent, extent, self.img.shape[1]+1)
        # centers
        self.x_centers = 0.5*(self.x_edges[:-1] + self.x_edges[1:])
        self.y_centers = 0.5*(self.y_edges[:-1] + self.y_edges[1:])

        # Fit galaxy radius in PIXELS, then convert to physical by multiplying dx
        pix_rad = self.determine_galaxy_radius(self.img)
        dx = self.x_centers[1] - self.x_centers[0]
        self.galaxy_radius = pix_rad * dx

        # build circular mask & apply
        X, Y = np.meshgrid(self.x_centers, self.y_centers, indexing='ij')
        R = np.sqrt(X**2 + Y**2)
        self.img[R > self.galaxy_radius] = np.nan

    def make_expansion(self, mmax, nmax, rscl, noisy=False):
        """
        Expand the (possibly masked) 2D image into Laguerre amplitudes.
        """
        snapshot = self.img.copy()
        dx = self.x_edges[1] - self.x_edges[0]

        # flatten coords & pixel masses
        X, Y = np.meshgrid(self.x_centers, self.y_centers, indexing='ij')
        rr = np.sqrt(X**2 + Y**2).ravel()
        phi = np.arctan2(Y, X).ravel()
        masses = (snapshot * dx*dx).ravel()

        # only keep finite, positive pixels
        ok = np.isfinite(masses) & (masses > 0)
        rr_ok, phi_ok, m_ok = rr[ok], phi[ok], masses[ok]

        L = LaguerreAmplitudes(rscl, mmax, nmax, rr_ok, phi_ok, mass=m_ok)
        self.r, self.p = rr_ok, phi_ok
        return L

    # … include your make_pointexpansion, rotate_disc, compute_a1, etc. unchanged …
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
        