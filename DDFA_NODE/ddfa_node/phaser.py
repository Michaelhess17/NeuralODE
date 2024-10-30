from jax import numpy as jnp
from jax import lax
from .shai_util import *
from .shai_util import hilbert
from scipy import signal
import warnings

"""
The phaser module provides an implementation of the phase estimation algorithm
of "Estimating the phase of synchronized oscillators";
   S. Revzen & J. M. Guckenheimer; Phys. Rev. E; 2008, v. 78, pp. 051907
   doi: 10.1103/PhysRevE.78.051907

Phaser takes in multidimensional data from multiple experiments and fits the
parameters of the phase estimator, which may then be used on new data or the
training data. The output of Phaser is a phase estimate for each time sample
in the data. This phase estimate has several desirable properties, such as:
(1) d/dt Phase is approximately constant
(2) Phase estimates are robust to measurement errors in any one variable
(3) Phase estimates are robust to systematic changes in the measurement error

The top-level class of this module is Phaser.
An example is found in test_sincos(); it requires matplotlib
"""
# print("Running phaser")
class ZScore( object ):
  """
  Class for finding z scores of given measurements with given or computed
  covarance matrix.

  This class implements equation (7) of [Revzen08]

  Properties:
    y0 -- Dx1 -- measurement mean
    M -- DxD -- measurement covariance matrix
    S -- DxD -- scoring matrix
  """

  def __init__( self, y = None, M = None ):
    """Computes the mean and scoring matrix of measurements
    INPUT:
      y -- DxN -- N measurements of a time series in D dimensions
      M -- DxD (optional) -- measurement error covariance for y
        -- If M is missing, it is assumed to be diagonal with variances
    -- given by 1/2 variance of the second order differences of y
    """

    # if M given --> use fromCovAndMean
    # elif we got y --> use fromData
    # else --> create empty object with None in members
    if M is not None:
      self.fromCovAndMean( jnp.mean(y, 1), M)
    elif y is not None:
      self.fromData( y )
    else:
      self.y0 = None
      self.M = None
      self.S = None
    

  def fromCovAndMean( self, y0, M ):
    """
    Compute scoring matrix based on square root of M through svd
    INPUT:
      y0 -- Dx1 -- mean of data
      M -- DxD -- measurement error covariance of data
    """
    self.y0 = y0
    self.M = M
    (D, V) = jnp.linalg.eig( M )
    self.S = jnp.dot( V.transpose(), jnp.diag( 1/jnp.sqrt( D ) ) )

  def fromData( self, y ):
    """
    Compute scoring matrix based on estimated covariance matrix of y
    Estimated covariance matrix is geiven by 1/2 variance of the second order
    differences of y
    INPUT:
      y -- DxN -- N measurements of a time series in D dimensions
    """
    self.y0 = jnp.mean( y, 1 )
    self.M = jnp.diag( jnp.std( jnp.diff( y, n=2, axis=1 ), axis=1 ) )
    self.S = jnp.diag( 1/jnp.sqrt( jnp.diag( self.M ) ) )

  def __call__( self, y ):
    """
    Callable wrapper for the class
    Calls self.zScore internally
    """
    return self.zScore( y )

  def zScore( self, y ):
    """Computes the z score of measurement y using stored mean and scoring
    matrix
    INPUT:
      y -- DxN -- N measurements of a time series in D dimensions
    OUTPUT:
      zscores for y -- DxN
    """
    return jnp.dot( self.S, y - self.y0.reshape( len( self.y0 ), 1 ) )


def _default_psf(x):
  """Default Poincare section function
     by rights, this should be inside the Phaser class, but pickle
     would barf on Phaser objects if they contained functions that
     aren't defined in the module top-level.
  """
  return signal.lfilter(
    jnp.array([0.02008336556421, 0.04016673112842,0.02008336556421] ),
    jnp.array([1.00000000000000,-1.56101807580072,0.64135153805756] ),
  x[0,:] )

class Phaser( object ):
  """
  Concrete class implementing a Phaser phase estimator

  Instance attributes:
    sc -- ZScore object for converting y to z-scores
    P_k -- list of D FourierSeries objects -- series correction for correcting proto-phases
    prj -- D x 1 complex -- projector on combined proto-phase
    P -- FourierSeries object -- series correction for combined phase
    psf -- callable -- callback to psecfun
  """

  def __init__( self, y = None, C = None, ordP = None, psecfunc = None ):
    """
    Initilizing/training a phaser object
    INPUT:
      y -- DxN or [ DxN_1, DxN_2, DxN_3, ... ] -- Measurements used for training
      C -- DxD (optional) -- Covariance matrix of measurements
      ordP -- 1x1 (optional) -- Orders of series to use in series correction
      psecfunc -- 1x1 (optional) -- Poincare section function
    """

    # if psecfunc given -> use given
    if psecfunc is not None:
      self.psf = psecfunc
    else:
      self.psf = _default_psf

    # if y given -> calls self.phaserTrain
    if y is not None:
      self.phaserTrain( y, C, ordP )

  def __call__( self, dat ):
    """
    Callable wrapper for the class. Calls phaserEval internally
    """
    return self.phaserEval( dat )


  def phaserEval( self, dat ):
    """
    Computes the phase of testing data
    INPUT:
      dat -- DxN -- Testing data whose phase is to be determined
    OUTPUT:
      Returns the complex phase of input data
    """

    # compute z score
    z = self.sc.zScore( dat )

    # compute Poincare section
    p0 = self.psf( dat )

    # compute protophase using Hilbert transform
    zeta = self.mangle * hilbert( z )
    z0, ido0 = Phaser.sliceN( zeta, p0 )

    # Compute phase offsets for proto-phases
    ofs = jnp.exp(-1j * jnp.angle(jnp.mean(z0, axis = 1)).T)

    # series correction for each dimision using self.P_k
    th = Phaser.angleUp( zeta * ofs[:,jnp.newaxis] )

    # evaluable weights based on sample length
    p = 1j * jnp.zeros( th.shape )
    for k in range( th.shape[0] ):
      val = self.P_k[k].val( th[k,:] ).T + th[k,:]
      p = p.at[k,:].set(val.flatten())

    rho = jnp.mean( jnp.abs( zeta ), 1 ).reshape(( zeta.shape[0], 1 ))
    # compute phase projected onto first principal components using self.prj
    ph = Phaser.angleUp( jnp.dot( self.prj.T, jnp.vstack( [jnp.cos( p ) * rho, jnp.sin( p ) * rho] ) ))

    # return series correction of combined phase using self.P
    phi = jnp.real( ph + self.P.val( ph ).T )
    pOfs2 = (p0[ido0+1] * jnp.exp(1j * phi.T[ido0+1]) - p0[ido0] * jnp.exp(1j * phi.T[ido0] )) / (p0[ido0+1] - p0[ido0])
    return phi - jnp.angle(jnp.sum(pOfs2))

  def phaserTrain( self, y, C = None, ordP = None ):
    """
    Trains the phaser object with given data.
    INPUT:
      y -- DxN or [ DxN_1, DxN_2, DxN_3, ... ] -- Measurements used for training
      C -- DxD (optional) -- Covariance matrix of measurements
    """

    # if given one sample -> treat it as an ensemble with one element
    if y.__class__ is jnp.ndarray:
      y = [y]
    # Copy the list container
    y = [yi for yi in y]
    # check dimension agreement in ensemble
    if len( set( [ ele.shape[0] for ele in y ] ) ) != 1:
      raise( Exception( 'newPhaser:dims','All datasets in the ensemble must have the same dimension' ) )
    D = y[0].shape[0]

    # train ZScore object based on the entire ensemble
    self.sc = ZScore( jnp.hstack( y ), C )

    # initializing proto phase variable
    zetas = []
    cycl = jnp.zeros( len( y ))
    svm = 1j*jnp.zeros( (D, len( y )) )
    svv = jnp.zeros( (D, len( y )) )

    # compute protophases for each sample in the ensemble
    for k in range( len( y ) ):
      # hilbert transform the sample's z score
      
      zetas.append( hilbert( self.sc.zScore( y[k] ) ) )
      # print(f"Zeta 1: {zetas[k]}")

      # trim beginning and end cycles, and check for cycle freq and quantity
      cycl = cycl.at[k].set(Phaser.trimCycle( zetas[k], y[k] )[0])
      zetas[k], y[k] = Phaser.trimCycle( zetas[k], y[k] )[1:]
      # Computing the Poincare section
      sk = self.psf( y[k] )
      # print(f"sk: {sk}")
      (sv, idx) = Phaser.sliceN( zetas[k], sk )
      if idx.shape[-1] == 0:
        raise Exception( 'newPhaser:emptySection', 'Poincare section is empty -- bailing out' )

      svm = svm.at[:,k].set(jnp.mean( sv, 1 ))
      svv = svv.at[:,k].set(jnp.var( sv, 1 ) * sv.shape[1] / (sv.shape[1] - 1))
      # print(sv.shape)
    # print(sum(jnp.isnan(svm)), sum(jnp.isnan(svv)))

    # computing phase offset based on psecfunc
    self.mangle, ofs = Phaser.computeOffset( svm, svv )

    # correcting phase offset for proto phase and compute weights
    wgt = jnp.zeros( len( y ) )
    rho_i = jnp.zeros(( len( y ), y[0].shape[0] ))
    for k in range( len( y ) ):
      zetas[k] = self.mangle * jnp.exp( -1j * ofs[k] ) * zetas[k]
      wgt = wgt.at[k].set(zetas[k].shape[0])
      rho_i = rho_i.at[k,:].set(jnp.mean( jnp.abs( zetas[k] ), 1 ))

    # compute normalized weight for each dimension using weights from all samples
    wgt = wgt.reshape(( 1, len( y )))
    rho = ( jnp.dot( wgt, rho_i ) / jnp.sum( wgt ) ).T
    # if ordP is None -> use high enough order to reach Nyquist/2
    if ordP is None:
      ordP = jnp.ceil( jnp.max( cycl ) / 4 )

    # correct protophase using seriesCorrection
    self.P_k = Phaser.seriesCorrection( zetas, ordP )


    # loop over all samples of the ensemble
    q = []
    for k in range( len( zetas ) ):
      # compute protophase angle
      th = Phaser.angleUp( zetas[k] )

      phi_k = 1j * jnp.ones( th.shape )

      # loop over all dimensions
      for ki in range( th.shape[0] ):
        # compute corrected phase based on protophase
        val = self.P_k[ki].val( th[ki,:] ).T + th[ki,:]
        
        phi_k = phi_k.at[ki,:].set(val.flatten())

      # computer vectorized phase
      q.append( jnp.vstack( [jnp.cos( phi_k ) * rho, jnp.sin( phi_k ) * rho] ) )

    # project phase vectors using first two principal components
    W = jnp.hstack( q[:] )
    W = W - jnp.mean( W, 1 )[:,jnp.newaxis]
    pc = jnp.linalg.svd( W, full_matrices=False )[0]
    self.prj = jnp.reshape( pc[:,0] + 1j * pc[:,1], ( pc.shape[0], 1 ) )

    # Series correction of combined phase
    qz = []
    for k in range( len( q ) ):
      qz.append( jnp.dot( self.prj.T, q[k] ) )

    # store object members for the phase estimator
    self.P = Phaser.seriesCorrection( qz, ordP )[0]

  def computeOffset( svm, svv ):
    """
    """
    # convert variances into weights
    svv = svv / jnp.sum( svv, 1 ).reshape( svv.shape[0], 1 )

    # compute variance weighted average of phasors on cross section to give the phase offset of each protophase
    mangle = jnp.sum( svm * svv, 1)
    if jnp.any( jnp.abs( mangle ) < .1 ):
      b = jnp.where( jnp.abs( mangle ) < .1 )
      raise Exception( 'computeOffset:badmeasureOfs', len( b ) + ' measuremt(s), including ' + b[0] + ' are too noisy on Poincare section' )

    # compute phase offsets for trials
    mangle = jnp.conj( mangle ) / jnp.abs( mangle )
    mangle = mangle.reshape(( len( mangle ), 1))
    svm = mangle * svm
    ofs = jnp.mean( svm, 0 )
    if jnp.any( jnp.abs( ofs ) < .1 ):
      b = jnp.where( jnp.abs( ofs ) < .1 )
      raise Exception( f'computeOffset:badTrialOfs {len( b )} trial(s), including {b[0]} are too noisy on Poincare section' )

    return mangle, jnp.angle( ofs )

  computeOffset = staticmethod( computeOffset )

  def sliceN( x, s, h = None ):
    """
    Slices a D-dimensional time series at a surface
    INPUT:
      x -- DxN -- data with colums being points in the time series
      s -- N, array -- values of function that is zero and increasing on surface
      h -- 1x1 (optional) -- threshold for transitions, transitions>h are ignored
    OUPUT:
      slc -- DxM -- positions at estimated value of s==0
      idx -- M -- indices into columns of x indicating the last point before crossing the surface
    """

    # checking for dimension agreement
    if x.shape[1] != s.shape[0]:
      raise Exception( 'sliceN:mismatch', 'Slice series must have matching columns with data' )
    #print(s)
    #print(s.shape)
    idx = jnp.where(( s[1:] > 0 ) & ( s[0:-1] <= 0 )) #changed here

    indices = jnp.where(idx[0] < x.shape[1])[0]

    idx = jnp.take(idx[0], indices)
    #print(idx)


# array([4, 3, 6])
    

    if h != None:
      idx = idx( jnp.abs( s[idx] ) < h & jnp.abs( s[idx+1] ) < h );

    N = x.shape[0]

    if len( idx ) == 0:
      return jnp.zeros(( N, 0 )), idx

    wBfr = jnp.abs( s[idx] )
    wBfr = wBfr.reshape((1, len( wBfr )))
    wAfr = jnp.abs( s[idx+1] )
    wAfr = wAfr.reshape((1, len( wAfr )))
    slc = ( x[:,idx]*wAfr + x[:,idx+1]*wBfr ) / ( wBfr + wAfr )

    return slc, idx

  sliceN = staticmethod( sliceN )

  def angleUp( zeta ):
    """
    Convert complex data to increasing phase angles
    INPUT:
      zeta -- DxN complex
    OUPUT:
      returns DxN phase angle of zeta
    """
    # unwind angles
    th = jnp.unwrap( jnp.angle ( zeta ) )
    # print(f"Zeta (angle up): {zeta}, \n th:{th}")
    # reverse decreasing sequences
    bad = th[:,0] > th[:,-1]
    th = jnp.where(bad[:,None], -th, th)
    return th

  angleUp = staticmethod( angleUp )

  def trimCycle( zeta, y ):
    """
    """
    # compute wrapped angle for hilbert transform
    ph = Phaser.angleUp( zeta )
    # print(f"\n ph:{ph}")

    # estimate nCyc in each dimension
    
    nCyc = jnp.abs( ph[:,-1] - ph[:,0] ) / 2 / jnp.pi
    # print(f"nCycles: {nCyc}")
    cycl = jnp.ceil( zeta.shape[1] / jnp.max( nCyc ) ).astype(int)

    # if nCyc < 7 -> warning
    # elif range(nCyc) > 2 -> warning
    # else truncate beginning and ending cycles
    # # if jnp.any( nCyc < 7 ):
    # #   warnings.warn( "PhaserForSample:tooShort" )
    # # elif jnp.max( nCyc ) - jnp.min( nCyc ) > 2:
    # #   warnings.warn( "PhaserForSample:nCycMismatch" )
    # else:
    zeta = zeta[:,cycl:-cycl]
    y = y[:,cycl:-cycl]

    return cycl, zeta, y

  trimCycle = staticmethod( trimCycle )

  def seriesCorrection( zetas, ordP ):
    """
    Fourier series correction for data zetas up to order ordP
    INPUT:
      zetas -- [DxN_1, DxN_2, ...] -- list of D dimensional data to be corrected using Fourier series
      ordP -- 1x1 -- Number of Fourier modes to be used
    OUPUT:
      Returns a list of FourierSeries object fitted to zetas
    """

    # initialize proto phase series 2D list
    proto = []

    # loop over all samples of the ensemble
    wgt = jnp.zeros( len( zetas ) )
    for k in range( len( zetas ) ):
      proto.append([])
      # compute protophase angle (theta)
      zeta = zetas[k]
      N = zeta.shape[1]
      theta = Phaser.angleUp( zeta )

      # generate time variable
      t = jnp.linspace( 0, 1, N )
      # compute d_theta
      dTheta = jnp.diff( theta, 1 )
      # compute d_t
      dt = jnp.diff( t )
      # mid-sampling of protophase angle
      th = ( theta[:,1:] + theta[:,:-1] ) / 2.0

      # loop over all dimensions
      for ki in range( zeta.shape[0] ):
        # evaluate Fourier series for (d_theta/d_t)(theta)
        # normalize Fourier coefficients to a mean of 1
        fdThdt = FourierSeries().fit( ordP * 2, th[ki,:].reshape(( 1, th.shape[1])), dTheta[ki,:].reshape(( 1, dTheta.shape[1])) / dt )
        fdThdt.coef = fdThdt.coef / fdThdt.m
        fdThdt.m = jnp.array([1])

        # evaluate Fourier series for (d_t/d_theta)(theta) based on Fourier
        # approx of (d_theta/d_t)
        # normalize Fourier coefficients to a mean of 1
        fdtdTh = FourierSeries().fit( ordP, th[ki,:].reshape(( 1, th.shape[1])), 1 / fdThdt.val( th[ki,:].reshape(( 1, th.shape[1] )) ).T )
        fdtdTh.coef = fdtdTh.coef / fdtdTh.m
        fdtdTh.m = jnp.array([1])

        # evaluate corrected phsae phi(theta) series as symbolic integration of
        # (d_t/d_theta), this is off by a constant
        proto[k].append(fdtdTh.integrate())

      # compute sample weight based on sample length
      wgt = wgt.at[k].set(zeta.shape[0])

    wgt = wgt / jnp.sum( wgt )

    # return phase estimation as weighted average of phase estimation of all samples
    proto_k = []
    for ki in range( zetas[0].shape[0] ):
      proto_k.append( FourierSeries.bigSum( [p[ki] for p in proto], wgt ))

    return proto_k

  seriesCorrection = staticmethod( seriesCorrection )

def test_sincos():
  """
  Simple test/demo of Phaser, recovering a sine and cosine

  Demo courtesy of Jimmy Sastra, U. Penn 2011
  """
  from jax.numpy import sin,cos,pi,array,linspace,cumsum,asarray,dot,ones
  from pylab import plot, legend, axis, show, randint, randn, std,lstsq
  # create separate trials and store times and data
  dats=[]
  t0 = []
  period = 55 # i
  phaseNoise = 0.5/jnp.sqrt(period)
  snr = 20
  N = 10
  print(N,"trials with:")
  print("\tperiod %.2g"%period,"(samples)\n\tSNR %.2g"%snr,"\n\tphase noise %.2g"%phaseNoise,"(radian/cycle)")
  print("\tlength = [",)
  for li in range(N):
    l = randint(400,2000) # length of trial
    dt = pi*2.0/period + randn(l)*0.07 # create noisy time steps
    t = cumsum(dt)+randn()*2*pi # starting phase is random
    raw = asarray([sin(t),cos(t)]) # signal
    raw = raw + randn(*raw.shape)/snr # SNR=20 noise
    t0.append(t)
    dats.append( raw - raw.mean(axis=1)[:,jnp.newaxis] )
    print(l,)
  print("]")
  # use points where sin=cos as poincare section
  phr = Phaser( dats, psecfunc = lambda x : dot([1,-1],x) )
  phi = [ phr.phaserEval( d ) for d in dats ] # extract phase
  reg = array([linspace(0,1,t0[0].size),ones(t0[0].size)]).T
  tt = dot(reg, lstsq(reg,t0[0], rcond=None)[0])
  plot(((tt-pi/4) % (2*pi))/pi-1, dats[0].T,'x')
  plot( (phi[0].T % (2*pi))/pi-1, dats[0].T,'.')#plot data versus phase
  legend(['sin(t)','cos(t)','sin(phi)','cos(phi)'])
  axis([-1,1,-1.2,1.2])
  show()

if __name__=="__main__":
  test_sincos()
