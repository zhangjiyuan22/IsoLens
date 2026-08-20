#import cupy as cp
import numpy as np
import os
import galmod as gm
# np.seterr(divide='ignore')
import time
import os
# from getconfig import *
from multiprocessing import get_context
import json
import pandas as pd
import h5py


def mkdir(path):
    Exist = os.path.exists(path)
    if Exist:
        print(path, '  already exists!')
    else:
        os.makedirs(path)
        print(path,'  created!')

# def save_path(savedir,strs,comp,mdname,tEmean,n,npz=False):
#     path = savedir + '%s-%s-tE%d%s-%d.npy'%(strs,mdname,tEmean,comp,n)
#     if npz:
#         path = path[:-1]+'z'
#     return path

def vRvT2vl(vR,vT,R_vec,dir_vec,sign_theta):
    '''
    convert 2D velocity in plane (vR,vT) to vl，i.e. the velocity component in plane that is prependicular to line-of-sight
    vR: radial velocity; 
    vT: Tangential velocity (direction of rotation); 
    R_vec: The vector of the galactic center pointing to the position of the star; 
    dir_vec: Unit vector of line of sight direction;
    sign_theta: Specifies the sign of sin (when l>0/l<0, sin>0/sin<0)
    '''
    cos_theta = -np.dot(R_vec,dir_vec)/np.linalg.norm(R_vec,axis=1)
    sin_theta = sign_theta * np.sqrt(1-cos_theta**2)
    vl = vT * cos_theta + vR * sin_theta
    return vl

def RandomSampleFromCDF(x,CDF,n_sample,smooth=True,return_index=False):
    '''
    smooth: if True, smooth out discreate random values (by adding a random number in [-dx/2,dx/2])
    '''
    x_gen = np.random.uniform(0.,CDF[-1],n_sample)
    index = np.searchsorted(CDF,x_gen)
    if smooth:
        dx = x[1]-x[0]
        x_result = x[index] + (np.random.rand(n_sample)-0.5)*dx
    else:
        x_result = x[index]
    
    if return_index:
        return x_result,index
    else:
        return x_result
    
def RandomSampleFromCDF2(xs,CDFs,ratios,n_sample,smooth=True):
    '''
    support multi-CDF
    smooth: if True, smooth out discreate random values (by adding a random number in [-dx/2,dx/2])
    '''
    n_component = int(len(ratios))
    ratios = np.array(ratios)/np.sum(ratios)
    cum_ratios = np.cumsum(ratios)
    classify = np.random.rand(n_sample)
    
    classification = np.searchsorted(cum_ratios,classify)
    x_result = np.zeros(n_sample)
    
    for c in range(n_component):
        classfilter = (classification==c)
        n_class = int(np.sum(classfilter))
        x_gen = np.random.uniform(0.,CDFs[c][-1],n_class)
        index = np.searchsorted(CDFs[c],x_gen)
        
        if smooth:
            dx = xs[c][1]-xs[c][0]
            x_result[classfilter] = xs[c][index] + (np.random.rand(n_class)-0.5)*dx
        else:
            x_result[classfilter] = xs[c][index]
    
    return x_result,classification
    
    
def RandomSampleFromPDF(x,PDF,n_sample,smooth=True,return_index=False):
    CDF = np.cumsum(PDF)
    return RandomSampleFromCDF(x,CDF,n_sample,smooth,return_index)
"""
def SampleFromTruncatedGaussian1D(mean,std,nsample,truncation=[-np.inf,np.inf],precision=100,cupy=False):
    '''
    mean: peak location (not necessary to be the mean value) of Gaussian distribution
    std: the standard deviation
    nsample: number of output sampled points
    truncation: list-like, lower and upper truncation value
    precision: sample rate of PDF in 1-sigma
    '''
    dx = std/precision
    x_min = max(mean-10*std,truncation[0]+dx/2)
    x_max = min(mean+10*std,truncation[1]+dx/2)
    if cupy==False:
        x = np.arange(x_min,x_max, dx)
        PDF = np.exp(-0.5* ((x-mean)/std)**2 )
        CDF = np.cumsum(PDF)
        sample_cdf = np.random.rand(nsample)*CDF[-1]
        sample_arg = np.searchsorted(CDF,sample_cdf)
        sample_x = x[sample_arg]
        sample_x = sample_x + (np.random.rand(nsample)-0.5)*dx
    else:
        x = cp.arange(x_min,x_max, dx)
        PDF = cp.exp(-0.5* ((x-mean)/std)**2 )
        CDF = cp.cumsum(PDF)
        sample_cdf = cp.random.rand(nsample)*CDF[-1]
        sample_arg = cp.searchsorted(CDF,sample_cdf)
        sample_x = x[sample_arg]
        sample_x = sample_x + (cp.random.rand(nsample)-0.5)*dx
    return sample_x

def SampleFromOneSideGaussian1D(mean,std,nsample,flat='below',truncation=[-np.inf,np.inf],precision=100,cupy=False):
    '''
    mean: peak location (not necessary to be the mean value) of Gaussian distribution
    std: the standard deviation
    nsample: number of output sampled points
    flat: 'below' or 'above'
          'below': PDF below mean is flat
          'above': PDF above mean is flat
    truncation: list-like, lower and upper truncation value
    precision: sample rate of PDF in 1-sigma
    '''
    dx = std/precision
    x_min = max(mean-10*std,truncation[0]+dx/2)
    x_max = min(mean+10*std,truncation[1]+dx/2)
    if cupy==False:
        x = np.arange(x_min,x_max, dx)
        PDF = np.exp(-0.5* ((x-mean)/std)**2 )
        if flat=='below':
            PDF[x<=mean] = 1.
        elif flat=='above':
            PDF[x>=mean] = 1.
        CDF = np.cumsum(PDF)
        sample_cdf = np.random.rand(nsample)*CDF[-1]
        sample_arg = np.searchsorted(CDF,sample_cdf)
        sample_x = x[sample_arg]
        sample_x = sample_x + (np.random.rand(nsample)-0.5)*dx
    else:
        x = cp.arange(x_min,x_max, dx)
        PDF = cp.exp(-0.5* ((x-mean)/std)**2 )
        if flat=='below':
            PDF[x<=mean] = 1.
        elif flat=='above':
            PDF[x>=mean] = 1.
        CDF = cp.cumsum(PDF)
        sample_cdf = cp.random.rand(nsample)*CDF[-1]
        sample_arg = cp.searchsorted(CDF,sample_cdf)
        sample_x = x[sample_arg]
        sample_x = sample_x + (cp.random.rand(nsample)-0.5)*dx
    return sample_x

from math import erf
def erf_array(x):
    if type(x) in [float,np.float64]:
        return erf(x)
    result = np.fromiter(map(erf,x),dtype='float')
    return result

def RatioOf2GaussianDistribution(x,mean1,mean2,std1,std2):
    '''
    Z = X1/X2
    X1 ~ N(mean1,std1)
    X2 ~ N(mean2,std2)
    '''
    w1 = 1/(std1)**2
    w2 = 1/(std2)**2
    
    at = np.sqrt(x**2*w1 + w2)
    bt = x*mean1*w1 + mean2*w2
    rt2 = (bt/at)**2
    ct = mean1**2*w1 + mean2**2*w2
    dt = np.exp(0.5*(rt2 - ct))
    
    A1 = bt*dt/(at**3)/(np.sqrt(2*np.pi)*std1*std2)*( erf_array(bt/at) )
    A2 = 1/(at**2*np.pi*std1*std2)*np.exp(-0.5*ct)
    
    return A1 + A2
"""


def genVelocityDistribution_Shu(sigma_r0=38., Rd=2.5, vc=220.
                                ,dV=1.0/200.0, Rmin=0.05,Rmax=8.35,Rstep=0.1
                                ,savepath = 'velocity_data'):
    Shuinfo = np.round(np.array([sigma_r0,Rd,vc,dV,Rmin,Rmax,Rstep]),5)
    if os.path.exists(savepath+'/Shuconfig.npy'):
        Shuinfo_old = np.load(savepath+'/Shuconfig.npy')
        if (Shuinfo==Shuinfo_old).all():
            print('Shu velocity distribution files already exist.')
            return
        
    from galpy.df import shudf
    print('Generating Shu velocity distribution...',flush=True)
    for rs in np.arange(Rmin,Rmax,Rstep): # (Rmin,Rmax,Rstep), in kpc
        ShuDistFilenamevT = savepath+'/Shu-vT-CDF-R%.2f-dV5e-3.npy'%(rs)
        ShuDistFilenamevR = savepath+'/Shu-vR-CDF-R%.2f-dV5e-3.npy'%(rs)

        Vt_gen0 = np.arange(0, 3, dV)
        Vr_gen0 = np.arange(-2, 2, dV)
        ### calculate Shu DistributionFunction's (vT,vR) CDF ###
        print('  Generating Shu DF... (R=%.2f)'%rs,flush=True)
        df = shudf(profileParams=(Rd/8.3,(4*Rd)/8.3,sigma_r0/vc),beta=0.)

        vts = np.linspace(0.0,3.0,151)
        vrs = np.linspace(-2.0,2.0,201)
#                 Vt_gen0 = np.arange(0, 3, dV)
        pdf_Vt_Disk = ( lambda x: np.sum([df(np.array([rs/8.3,vr,x])) for vr in vrs]) )
        Vt_pdf = np.array([pdf_Vt_Disk(xi) for xi in Vt_gen0])
        Vt_cdf = np.cumsum(Vt_pdf)
        np.save(ShuDistFilenamevT,np.array(Vt_cdf))
#                 Vr_gen0 = np.arange(-2, 2, dV)
        pdf_Vr_Disk = ( lambda x: np.sum([df(np.array([rs/8.3,x,vt])) for vt in vts]) )
        Vr_pdf = np.array([pdf_Vr_Disk(xi) for xi in Vr_gen0])
        Vr_cdf = np.cumsum(Vr_pdf)
        np.save(ShuDistFilenamevR,np.array(Vr_cdf))
    np.save(savepath+'/Shuconfig.npy',Shuinfo)
    print('Done',flush=True)
    return

def Gaussian1D(x,mean,sigma):
    A = 1/np.sqrt(2*np.pi)/sigma
    cx = (x-mean)/sigma
    #chi = cx**2
    return A*np.exp(-0.5*cx**2)

def Gaussian2D(x1,x2,mean1,mean2,sigma1,sigma2,rho12):
    A = 1/(2*np.pi*sigma1*sigma2*np.sqrt(1-rho12**2))
    cx1 = (x1-mean1)/sigma1
    cx2 = (x2-mean2)/sigma2
    chi = 1/(1-rho12**2)*( cx1**2 -2*rho12*cx1*cx2 + cx2**2 )
    return A*np.exp(-0.5*chi)




######################################################################
# Vectorized non-parametric disk lens age sampler using:
#   - age_concat_map_40by40_gaussN1000_resampled_fp32.npy
#   - age_index_map_40by40_gaussN1000_resampled.csv
#   - age_sampler_meta_40by40.json

# Method:
# 1) (R,z) -> cell indices (iR,iZ)
# 2) cell_id -> (start,length) via index map
# 3) random integer in [0, length)
# 4) sampled age = ages[start + rand]
######################################################################
class DiskLensAgeSampler:
    def __init__(self,
                 ages_npy: str = "age_concat_map_40by40_gaussN1000_resampled_fp32.npy",
                 index_csv: str = "age_index_map_40by40_gaussN1000_resampled.csv",
                 meta_json: str = "age_sampler_meta_40by40.json"):
        
        # Load meta (edges define bin boundaries)
        with open(meta_json, "r") as f:
            meta = json.load(f)
        self.R_edges = np.asarray(meta["R_edges"], dtype=np.float64)
        self.Z_edges = np.asarray(meta["Z_edges"], dtype=np.float64)
        # print(self.R_edges)
        # print(self.Z_edges)
        
        self.nR = int(meta["nR"])
        self.nZ = int(meta["nZ"])
        # n_cells = self.nR * self.nZ
        # print(n_cells)
        
        
        # Load ages (contiguous pool)
        # Using float32 to halve memory, enough for ages in Gyr
        self.ages = np.load(ages_npy)
        # print(self.ages)
        # print('max/min stellar age in all ~160000 stars', max(self.ages), min(self.ages))
        # print(len(self.ages))
        # print(self.ages.dtype)
        
        
        # Load the per-cell (start,length) lookup (sorted by cell_id)
        idx_df = pd.read_csv(index_csv, usecols=["start","length"])
#         idx_df.sort_values("cell_id", inplace=True)
#         if idx_df.shape[0] != n_cells:
#             raise ValueError(f"Index map rows ({idx_df.shape[0]}) != nR*nZ ({n_cells})")

        self.starts  = idx_df["start"].to_numpy(dtype=np.int64,  copy=True)
        self.lengths = idx_df["length"].to_numpy(dtype=np.int32, copy=True)
        # print(self.starts)
        # print(self.lengths)
        
        
        
    def sample(self, R, Z, out_of_bounds="nan"):
        """
        Sample ages for arrays R, Z (same shape).
        out_of_bounds: 'nan' (default) => points outside edges get NaN
                       'clip'           => clip to nearest valid cell
        Returns ages with same shape as inputs.
        """
        R = np.asarray(R, dtype=np.float64)
        Z = np.asarray(Z, dtype=np.float64)
        sampled_age_array = np.zeros(R.shape, dtype=self.ages.dtype) # float32

        # Bin to cell indices (left-closed, right-open), 
        # thus can be applied to Z=[-3, 3), which results in iZ=[0,39]
        # Z<-3 leads to iZ=-1, Z>=3 leads to iZ=40
        iR = np.digitize(R, self.R_edges, right=False) - 1
        iZ = np.digitize(Z, self.Z_edges, right=False) - 1
        # print(iR,iZ)
        
        
        if out_of_bounds == "clip":
            np.clip(iR, 0, self.nR-1, out=iR)
            np.clip(iZ, 0, self.nZ-1, out=iZ)
            valid = np.ones(iR.shape, dtype=bool)
        else:
            valid = (iR >= 0) & (iR < self.nR) & (iZ >= 0) & (iZ < self.nZ)
            # print(valid)
            # print(sum(~valid))
            sampled_age_array[~valid] = np.nan
            # print(sampled_age_array)
            
            sum_valid = sum(valid)
            if sum(~valid) != 0 : 
                print("Disk Lens (R,z) out of age map !")
            

        if sum_valid == 0 :
            print('ALL Disk Lens (R,z) out of age map ! Return NaN array')
            return sampled_age_array
        
        
        # Cell id, then map to (start, length)
        cell_id = (iR * self.nZ + iZ).astype(np.int32, copy=False)
        # print(cell_id)
        # print(cell_id[valid])
        start_index  =  self.starts[  cell_id[valid]  ]
        cell_length  = self.lengths[  cell_id[valid]  ]
        # print(start_index, cell_length)

        
        ## uniform distribution: [0,1)
        ## then [0,cell_length)
        ## then truncate toward 0: [0,cell_length-1]
        random_index = (np.random.random(size=sum_valid) * cell_length).astype(np.int32)
        # print(random_index)
        
        
        # Sampling
        sampled_age_array[valid] = self.ages[start_index + random_index]
        # print(sampled_age_array)
        
        return sampled_age_array




#######################################################################
# disk radial [Fe/H] profile (age dependence)
# applized for -1 < z < 1 kpc
# DISK_AGE_BINS: (age_min, age_max, slope m, intercept b, sigma, special)
#######################################################################
DISK_AGE_BINS = [
    (None, 2.0,  -0.07 ,  0.60,  0.10,  "cap_at_0p5_flat_lowR"), #<2 Gyr
    (2.0,  4.0,  -0.068,  0.50,  0.15,  None), # 2–4 Gyr
    (4.0,  6.0,  -0.068,  0.48,  0.18,  None), # 4–6 Gyr
    (6.0,  8.0,  -0.05 ,  0.38,  0.20,  None), # 6–8 Gyr
    (8.0,  9.0,  -0.038,  0.19,  0.225, None), # 8–9 Gyr
    (9.0,  10.0, -0.015, -0.05,  0.225, None), # 9–10 Gyr
    (10.0, None,  0.0  , -0.35,  0.20,  None) # >10 Gyr
]

def _mean_feh_of_disk_age_bin(R_array_mask, m, b, special=None):
    """
    Mean [Fe/H] for array R_array_mask with bin-specific rule(s).
    For the <2 Gyr bin: cap mean at +0.5 and flatten at low R where the line would exceed 0.5.
    """
    y = m * R_array_mask + b
    if special == "cap_at_0p5_flat_lowR":
        y = np.minimum(0.5, y)
    return y

def _disk_feh_mu_sigma_from_R_age(R_array, age_array):
    """
    Map (R, age) -> (mu, sigma) using the piecewise age bins.
    R, age must be same-shape arrays. Returns arrays of same shape.
    """
    
    mu_array    = np.zeros_like(R_array)
    sigma_array = np.zeros_like(R_array)

    
    for (age_min, age_max, m, b, s, special) in DISK_AGE_BINS: 
        
        if age_min is None:
            mask = age_array < age_max
        elif age_max is None:
            mask = age_array >= age_min
        else:
            mask = (age_array >= age_min) & (age_array < age_max)
            
        if np.any(mask):
            mu_array[mask]    = _mean_feh_of_disk_age_bin(R_array[mask], m, b, special)
            sigma_array[mask] = s
            
    return mu_array, sigma_array

def _sample_truncnorm_plusminus3(size):
    """
    Draw z ~ N(0,1) truncated to [-3, 3] using np.random (global RNG).
    Returns values strictly within ±3 (no clipping).
    Assumes np.random.seed(...) has been set outside if reproducibility is desired.
    """
    z_array = np.random.normal(size=size)
    mask = (z_array < -3.0) | (z_array > 3.0)
    
    # Very few rejections (~0.27%); loop until all are inside
    while np.any(mask):
        z_array[mask] = np.random.normal(size=mask.sum())
        mask = (z_array < -3.0) | (z_array > 3.0)
        
    return z_array

def sample_disk_feh(R_array, age_array):
    """
    Sample disk [Fe/H] directly inside ±3 sigma for each (R, age) point.

    Args:
        R_array (array-like): Galactocentric radius (kpc)
        age_array (array-like): Age (Gyr), same shape as R

    Returns:
        feh_array  (np.ndarray)          
    """

    # Compute means/σ 
    mu_array, sigma_array = _disk_feh_mu_sigma_from_R_age(R_array, age_array)
    
    # sample from N(0,1) within 3 sigma
    z_array = _sample_truncnorm_plusminus3(R_array.size)
    
    # convert as sampled feh = mu + sigma * N(0,1)
    feh_array = mu_array +  sigma_array * z_array

    return feh_array


####################################
## disk [alpha/Fe]|Age sampler #####
####################################
def sample_disk_alpha_fe(age_array,
                         mu_lo=0.03, mu_hi=0.23,
                         sigma_lo=0.03, sigma_hi=0.04):
    """
    Sample [alpha/Fe] given Age with a piecewise model:
      - Age <= 8.5 Gyr:   mu=mu_lo,  sigma=sigma_lo
      - 8.5 < Age < 10 Gyr: mu and sigma vary linearly between the endpoints
      - Age >= 10 Gyr:    mu=mu_hi,  sigma=sigma_hi

    Sampling uses a truncated standard normal (±3 sigma) via rejection.
    Uses np.random.* so external np.random.seed(...) controls reproducibility.

    Args:
      age_array (array-like): Ages in Gyr
      mu_lo, mu_hi (float):   means at <=8.5 and >=10 Gyr
      sigma_lo, sigma_hi (float): stddevs at <=8.5 and >=10 Gyr

    Returns:
      alpha_fe_array (ndarray) 
    """

    mu_array  = np.zeros_like(age_array)
    sig_array = np.zeros_like(age_array)

    # Masks
    mask_lo  = age_array <= 8.5
    mask_hi  = age_array >= 10.0
    mask_mid = (~mask_lo) & (~mask_hi)  # 8.5 < age < 10

    # End segments
    mu_array[mask_lo],  sig_array[mask_lo]  = mu_lo, sigma_lo
    mu_array[mask_hi],  sig_array[mask_hi]  = mu_hi, sigma_hi

    # Linear transition 8.5–10 Gyr
    if np.any(mask_mid):
        t = (age_array[mask_mid] - 8.5) / 1.5  # 0..1 across 8.5->10
        mu_array[mask_mid]  = (1.0 - t) * mu_lo + t * mu_hi
        sig_array[mask_mid] = (1.0 - t) * sigma_lo + t * sigma_hi

    # Draw truncated standard normal z ~ N(0,1) in [-3,3]
    z = np.random.normal(size=age_array.size)
    mask = (z < -3.0) | (z > 3.0)
    while np.any(mask):
        z[mask] = np.random.normal(size=mask.sum())
        mask = (z < -3.0) | (z > 3.0)

    alpha_fe_array = mu_array + sig_array * z
    
    return alpha_fe_array



def mh_from_feh_alphafe(feh, alpha_fe):
    # Salaris+93
    return feh + np.log10(0.638 * 10.0**alpha_fe + 0.362)



########################################
## bulge [alpha/Fe]|[Fe/H] sampler #####
########################################
def bulge_alpha_fe_mu_sigmoid(feh, alpha_low, alpha_high, feh0, w):
    """
    Sigmoid mean μ([Fe/H]) = α_low + (α_high - α_low) / (1 + exp((FeH - FeH0)/w))
    feh: array-like of [Fe/H]
    """
    return alpha_low + (alpha_high - alpha_low) / (1.0 + np.exp((feh - feh0)/w))

def sample_bulge_alpha_fe(feh_array, fit_alpha_low=0.038, fit_alpha_high=0.298, fit_z0=-0.223, fit_w=0.122, sigma=0.038):
    """
    Vectorized sampling of bulge [alpha/Fe] | [Fe/H].

    Args
    ----
    feh : array-like
        Array of [Fe/H] values (shape: N or any shape).
    sigma : float, default 0.038
        Gaussian sigma (dex). May be a scalar or broadcastable to feh's shape.
        (Use 0.026 for intrinsic-only, ~0.046 for observed-like, or 0.038 compromise.)
    """
    
    mu_array  = bulge_alpha_fe_mu_sigmoid(feh_array, fit_alpha_low, fit_alpha_high, fit_z0, fit_w)

    # Draw truncated standard normal z ~ N(0,1) in [-3,3]
    z = np.random.normal(size=feh_array.size)
    mask = (z < -3.0) | (z > 3.0)
    while np.any(mask):
        z[mask] = np.random.normal(size=mask.sum())
        mask = (z < -3.0) | (z > 3.0)

    alpha_fe_array = mu_array + sigma * z
    
    return alpha_fe_array



####################################
## Look up for lens mass range #####
####################################
def _nearest_index_uniform(x, x0, dx, n):
    #"""Nearest integer index on a uniform axis; robust to tiny float jitter."""
    idx = np.rint((x - x0) / dx).astype(np.int64)
    return np.clip(idx, 0, n - 1)

def mass_bounds_for_samples(npz_path, Age_lens, MH_lens):
    #     """
    #     Parameters
    #     ----------
    #     npz_path : str
    #         Path to your packed PARSEC NPZ (MH-outer, Age-inner).
    #     Age_lens, MH_lens : array-like (same shape)
    #         Sampled ages [Gyr] and metallicities [M/H].

    #     Returns
    #     -------
    #     mass_max_arr, mass_min_arr : ndarray
    #         Arrays of max/min allowable mass for each sample, same shape as inputs.
    #     """
    DB = np.load(npz_path, allow_pickle=False)

    # axes / spacing
    mh_min, dmh, n_mh   = float(DB["mh_min"]),  float(DB["dmh"]),  int(DB["n_mh"])
    age_min, dage, n_age = float(DB["age_min"]), float(DB["dage"]), int(DB["n_age"])

    # nearest grid indices (MH outer, Age inner)
    i_mh  = _nearest_index_uniform(MH_lens,  mh_min,  dmh,  n_mh)
    j_age = _nearest_index_uniform(Age_lens, age_min, dage, n_age)
    iso_idx = i_mh * n_age + j_age  # linear index in your NPZ packing

    # lookup per-iso bounds and gather
    mass_min_tbl = np.asarray(DB["mass_min"], float)  # shape (n_mh*n_age,)
    mass_max_tbl = np.asarray(DB["mass_max"], float)
    
    mass_min_arr = mass_min_tbl[iso_idx]
    mass_max_arr = mass_max_tbl[iso_idx]

    return mass_min_arr, mass_max_arr










def simulate_one_run(args):
    """
    Worker: perform one 'ni' run and save one output file.
    We pass a single tuple (ni, ctx) because Pool.map only maps one arg.
    """
    
    ni, ctx = args

    # unpack context (everything computed once in SimulateEvents)
    (eventname, ndots, direction, Earth,
     Dss, cdfDs, ratioDs, Dls, cdfDl, ratioDl,
     Mls, cdfMs, ratioMs, 
     DiskModel, Rd, vc, sigma_t0, sigma_b0, sigma_r0,
     SourceFromPM, mus_mean, mus_cov,
     SolarMotion, EarthMotion, Solar_Vl, Solar_Vb, Earth_Vl, Earth_Vb,
     sign_theta, Rmin, Rmax, Rstep, dV, k_trans, Year2Day, 
     Lens_Age_Metallicity_Prior) = ctx


    # Per-process seed for reproducibility and independence
    np.random.seed( 1234567 + int(ni) )


    time_begin = time.time()
    print(f'{ni}th run...', flush=True)
    

    # -------------- body of code starts --------------
    print('  Generating Ds, Dl for process %s'%ni, flush=True)
    # generate source position. sloc=(0,1): (bulge,disk)
    Ds_gen, sloc = RandomSampleFromCDF2(xs=Dss, CDFs=cdfDs, ratios=ratioDs, n_sample=ndots)
    # Rs_gen_vec = (Ds_gen * direction[:2, np.newaxis] + Earth[:2, np.newaxis]).T
    # Rs_gen = np.linalg.norm(Rs_gen_vec, axis=1)
    Rs_gen_vec = (Ds_gen * direction[:3, np.newaxis] + Earth[:3, np.newaxis]).T
    Rs_gen = np.hypot(Rs_gen_vec[:,0], Rs_gen_vec[:,1])
    Zs_gen = Rs_gen_vec[:,2]

    # generate lens position. lloc=(0,1): (bulge,disk)
    Dl_gen, lloc = RandomSampleFromCDF2(xs=Dls, CDFs=cdfDl, ratios=ratioDl, n_sample=ndots)
    # Rl_gen_vec = (Dl_gen * direction[:2, np.newaxis] + Earth[:2, np.newaxis]).T
    # Rl_gen = np.linalg.norm(Rl_gen_vec, axis=1)
    Rl_gen_vec = (Dl_gen * direction[:3, np.newaxis] + Earth[:3, np.newaxis]).T
    Rl_gen = np.hypot(Rl_gen_vec[:,0], Rl_gen_vec[:,1])
    Zl_gen = Rl_gen_vec[:,2]



    print('  Drop unphysical source-lens pairs for process %s'%ni, flush=True)
    # Drop Dl>Ds pairs
    FineQ = (Dl_gen < Ds_gen)
    Ds, Dl = Ds_gen[FineQ], Dl_gen[FineQ]
    Source_loc, Lens_loc = sloc[FineQ], lloc[FineQ]
    Rs, Rl = Rs_gen[FineQ], Rl_gen[FineQ]
    ##### new #####
    Zs, Zl = Zs_gen[FineQ], Zl_gen[FineQ]
    ##### new #####
    Rs_vec, Rl_vec = Rs_gen_vec[FineQ], Rl_gen_vec[FineQ]
    nFine = int(np.sum(FineQ))

    sBulgQ = (Source_loc == 0)
    sDiskQ = (Source_loc == 1)
    lBulgQ = (Lens_loc == 0)
    lDiskQ = (Lens_loc == 1)

    ns_Bulg = int(np.sum(sBulgQ))
    ns_Disk = int(np.sum(sDiskQ))
    nl_Bulg = int(np.sum(lBulgQ))
    nl_Disk = int(np.sum(lDiskQ))

    del Ds_gen, Dl_gen, sloc, lloc, Rs_gen_vec, Rl_gen_vec, Rs_gen, Rl_gen, Zs_gen, Zl_gen, FineQ
    ### large arrays remained: 
    ###     Ds, Dl, Source_loc, Lens_loc, Rs, Rl, Zs, Zl, Rs_vec, Rl_vec, sBulgQ, sDiskQ, lBulgQ, lDiskQ 




    if Lens_Age_Metallicity_Prior == 'informative' : 

        print('Informative disk/bulge lens age, [Fe/H], and [alpha/Fe] prior adopted', flush=True)

        print('  Generating (disk) lens age from P_disk(age|(R,z)) for process %s'%ni, flush=True)
        # 1) Initialize sampler (loads CSV/JSON once)
        disk_lens_age_sampler = DiskLensAgeSampler(
            ages_npy ="age_distribution_disk/release_package/age_concat_map_40by40_gaussN1000_resampled_fp32.npy",
            index_csv="age_distribution_disk/release_package/age_index_map_40by40_gaussN1000_resampled.csv",
            meta_json="age_distribution_disk/release_package/age_sampler_meta_40by40.json"
        )
        
        # 2) Sample ages
        Rl_Disk = Rl[lDiskQ]
        Zl_Disk = Zl[lDiskQ]
        Age_lens_Disk = disk_lens_age_sampler.sample(Rl_Disk, Zl_Disk, out_of_bounds="nan")

        print('  Generating (disk) lens [Fe/H] from P_disk([Fe/H]|(R,age)) for process %s'%ni, flush=True)
        FeH_lens_Disk = sample_disk_feh(Rl_Disk, Age_lens_Disk)

        print('  Generating (disk) lens [alpha/Fe] from P_disk([alpha/Fe]|age) for process %s'%ni, flush=True)
        AlphaFe_lens_Disk = sample_disk_alpha_fe(Age_lens_Disk)

        print('  Converting to (disk) lens [M/H] for process %s'%ni, flush=True)
        MH_lens_Disk = mh_from_feh_alphafe(FeH_lens_Disk, AlphaFe_lens_Disk)

        #### here need to resample the samples that are outside PARSEC [M/H] range: -2 < [M/H] <= 0.65
        #### we resample instead of cliping to the edge, as the latter will cause pile up at the edge
        #### the 0.69525 is a hard clip for PARSEC, thus we download isochrone with [M/H] up to 0.65; 
        #### while -2 is choosen by us, as there are basically no such stars in bulge and disk
        index_out_of_MH_range_lens_Disk = (MH_lens_Disk > 0.65) | (MH_lens_Disk < -2.0)
        count_out_of_MH_range_lens_Disk = np.sum(index_out_of_MH_range_lens_Disk)

        if count_out_of_MH_range_lens_Disk > 0 : 
            print('  Resampling fraction = %.3e (disk) lens out of PARSEC [M/H] range for process %s'%(count_out_of_MH_range_lens_Disk/nl_Disk, ni), flush=True)
        
            while np.any(index_out_of_MH_range_lens_Disk) : 
            
                Age_lens_Disk[index_out_of_MH_range_lens_Disk] = disk_lens_age_sampler.sample(Rl_Disk[index_out_of_MH_range_lens_Disk], Zl_Disk[index_out_of_MH_range_lens_Disk], out_of_bounds="nan")
                FeH_lens_Disk[index_out_of_MH_range_lens_Disk] = sample_disk_feh(Rl_Disk[index_out_of_MH_range_lens_Disk], Age_lens_Disk[index_out_of_MH_range_lens_Disk])
                AlphaFe_lens_Disk[index_out_of_MH_range_lens_Disk] = sample_disk_alpha_fe(Age_lens_Disk[index_out_of_MH_range_lens_Disk])
                MH_lens_Disk[index_out_of_MH_range_lens_Disk] = mh_from_feh_alphafe(FeH_lens_Disk[index_out_of_MH_range_lens_Disk], AlphaFe_lens_Disk[index_out_of_MH_range_lens_Disk])

                index_out_of_MH_range_lens_Disk = (MH_lens_Disk > 0.65) | (MH_lens_Disk < -2.0)

        


        
        print('  Generating (bulge) lens [Fe/H] from P_bulge([Fe/H]) for process %s'%ni, flush=True)
        bulge_feh_cdf_file = np.load("Bensby_catalog_bulge/bensby_bulge_feh_cdf.npz")
        bulge_feh_edge, bulge_feh_cdf = bulge_feh_cdf_file["x"], bulge_feh_cdf_file["CDF"]

        ### sampled range [-2, 0.7], as PARSEC is valid till [M/H]~0.7; later a resample is needed for samples with [M/H]>0.7
        FeH_lens_Bulge = RandomSampleFromCDF(bulge_feh_edge, bulge_feh_cdf, nl_Bulg, smooth=True)



        print('  Generating (bulge) lens age from P_bulge(age|[Fe/H]) for process %s'%ni, flush=True)
        bulge_age_cdf_by_feh_file = np.load("Bensby_catalog_bulge/bensby_bulge_age_cdfs_by_feh.npz")
        bulge_age_center       = bulge_age_cdf_by_feh_file["age"]      # common age grid (0 to 13.8 Gyr with delta age = 0.1 Gyr)
        bulge_age_cdf_low_feh  = bulge_age_cdf_by_feh_file["CDF_low"]  # [Fe/H] < -0.5
        bulge_age_cdf_mid_feh  = bulge_age_cdf_by_feh_file["CDF_mid"]  #          -0.5 <= [Fe/H] < 0.0
        bulge_age_cdf_high_feh = bulge_age_cdf_by_feh_file["CDF_high"] #                           0.0 <= [Fe/H]

        index_bulge_low_feh  = (FeH_lens_Bulge < -0.5)
        index_bulge_mid_feh  = (FeH_lens_Bulge >= -0.5) & (FeH_lens_Bulge < 0.0)
        index_bulge_high_feh = (FeH_lens_Bulge >= 0.0)

        count_bulge_low_feh  = np.sum(index_bulge_low_feh)
        count_bulge_mid_feh  = np.sum(index_bulge_mid_feh)
        count_bulge_high_feh = np.sum(index_bulge_high_feh)

        Age_lens_Bulge = np.zeros_like(FeH_lens_Bulge)

        ### sampled range [0, 13.8 Gyr]
        if count_bulge_low_feh > 0: 
            Age_lens_Bulge[index_bulge_low_feh]  = RandomSampleFromCDF(bulge_age_center, bulge_age_cdf_low_feh,  count_bulge_low_feh, smooth=True)
        if count_bulge_mid_feh > 0: 
            Age_lens_Bulge[index_bulge_mid_feh]  = RandomSampleFromCDF(bulge_age_center, bulge_age_cdf_mid_feh,  count_bulge_mid_feh, smooth=True)
        if count_bulge_high_feh > 0: 
            Age_lens_Bulge[index_bulge_high_feh] = RandomSampleFromCDF(bulge_age_center, bulge_age_cdf_high_feh, count_bulge_high_feh, smooth=True)



        print('  Generating (bulge) lens [alpha/Fe] from P_bulge([alpha/Fe]|[Fe/H]) for process %s'%ni, flush=True)
        AlphaFe_lens_Bulge = sample_bulge_alpha_fe(FeH_lens_Bulge)



        print('  Converting to (bulge) lens [M/H] for process %s'%ni, flush=True)
        MH_lens_Bulge = mh_from_feh_alphafe(FeH_lens_Bulge, AlphaFe_lens_Bulge)



        #### here need to resample the samples that are outside PARSEC [M/H] range: -2 < [M/H] <= 0.65
        #### we resample instead of cliping to the edge, as the latter will cause pile up at the edge
        #### the 0.69525 is a hard clip for PARSEC, thus we download isochrone with [M/H] up to 0.65;
        #### while -2 is choosen by us, as there are basically no such stars in bulge and disk
        index_out_of_MH_range_lens_Bulge = (MH_lens_Bulge > 0.65) | (MH_lens_Bulge < -2.0)
        count_out_of_MH_range_lens_Bulge = np.sum(index_out_of_MH_range_lens_Bulge)

        if count_out_of_MH_range_lens_Bulge > 0 : 
            print('  Resampling fraction = %.3e (bulge) lens out of PARSEC [M/H] range for process %s'%(count_out_of_MH_range_lens_Bulge/nl_Bulg, ni), flush=True)
        
            while np.any(index_out_of_MH_range_lens_Bulge) : 

                ### sampled range [-2, 0.7]
                FeH_lens_Bulge_temporary = RandomSampleFromCDF(bulge_feh_edge, bulge_feh_cdf, np.sum(index_out_of_MH_range_lens_Bulge), smooth=True)
                FeH_lens_Bulge[index_out_of_MH_range_lens_Bulge] = FeH_lens_Bulge_temporary
                

                ### sampled range [0, 13.8 Gyr]
                Age_lens_Bulge_temporary = np.zeros_like(FeH_lens_Bulge_temporary)

                index_bulge_low_feh_temporary  = (FeH_lens_Bulge_temporary < -0.5)
                index_bulge_mid_feh_temporary  = (FeH_lens_Bulge_temporary >= -0.5) & (FeH_lens_Bulge_temporary < 0.0)
                index_bulge_high_feh_temporary = (FeH_lens_Bulge_temporary >= 0.0)

                count_bulge_low_feh_temporary  = np.sum(index_bulge_low_feh_temporary)
                count_bulge_mid_feh_temporary  = np.sum(index_bulge_mid_feh_temporary)
                count_bulge_high_feh_temporary = np.sum(index_bulge_high_feh_temporary)

                if count_bulge_low_feh_temporary > 0: 
                    Age_lens_Bulge_temporary[index_bulge_low_feh_temporary]  = RandomSampleFromCDF(bulge_age_center, bulge_age_cdf_low_feh,  count_bulge_low_feh_temporary, smooth=True)
                if count_bulge_mid_feh_temporary > 0: 
                    Age_lens_Bulge_temporary[index_bulge_mid_feh_temporary]  = RandomSampleFromCDF(bulge_age_center, bulge_age_cdf_mid_feh,  count_bulge_mid_feh_temporary, smooth=True)
                if count_bulge_high_feh_temporary > 0: 
                    Age_lens_Bulge_temporary[index_bulge_high_feh_temporary] = RandomSampleFromCDF(bulge_age_center, bulge_age_cdf_high_feh, count_bulge_high_feh_temporary, smooth=True)

                Age_lens_Bulge[index_out_of_MH_range_lens_Bulge] = Age_lens_Bulge_temporary

                ### 
                AlphaFe_lens_Bulge[index_out_of_MH_range_lens_Bulge] = sample_bulge_alpha_fe(FeH_lens_Bulge[index_out_of_MH_range_lens_Bulge])

                ### 
                MH_lens_Bulge[index_out_of_MH_range_lens_Bulge] = mh_from_feh_alphafe(FeH_lens_Bulge[index_out_of_MH_range_lens_Bulge], AlphaFe_lens_Bulge[index_out_of_MH_range_lens_Bulge])

                index_out_of_MH_range_lens_Bulge = (MH_lens_Bulge > 0.65) | (MH_lens_Bulge < -2.0)

            del FeH_lens_Bulge_temporary, Age_lens_Bulge_temporary
            del index_bulge_low_feh_temporary, index_bulge_mid_feh_temporary, index_bulge_high_feh_temporary
            
                



        # combine bulge/disk lens age, [M/H]
        Age_lens = np.zeros(nFine)
        Age_lens[lBulgQ] = Age_lens_Bulge
        Age_lens[lDiskQ] = Age_lens_Disk

        MH_lens = np.zeros(nFine)
        MH_lens[lBulgQ] = MH_lens_Bulge
        MH_lens[lDiskQ] = MH_lens_Disk

        del disk_lens_age_sampler
        del index_out_of_MH_range_lens_Disk
        del Age_lens_Disk,  FeH_lens_Disk,  AlphaFe_lens_Disk,  MH_lens_Disk, Rl_Disk, Zl_Disk

        del bulge_feh_cdf_file, bulge_feh_edge, bulge_feh_cdf
        del index_bulge_low_feh, index_bulge_mid_feh, index_bulge_high_feh
        del index_out_of_MH_range_lens_Bulge
        del Age_lens_Bulge, FeH_lens_Bulge, AlphaFe_lens_Bulge, MH_lens_Bulge
        ### large arrays remained: 
        ###     Ds, Dl, Source_loc, Lens_loc, Rs, Rl, Zs, Zl, Rs_vec, Rl_vec, sBulgQ, sDiskQ, lBulgQ, lDiskQ, Age_lens, MH_lens



    elif Lens_Age_Metallicity_Prior == 'uniform' : 

        print('Uniform disk/bulge lens age and [M/H] prior adopted', flush=True)

        # Uniform age in [0.25, 13.75] Gyr
        Age_lens = np.random.uniform(0.25, 13.75, size=nFine)

        # Uniform [M/H] in [-2.0, 0.65]
        MH_lens  = np.random.uniform(-2.0, 0.65, size=nFine)

    else:
        raise ValueError(
            f"Unknown Lens_Age_Metallicity_Prior = {Lens_Age_Metallicity_Prior!r}. "
            "Must be 'informative' or 'uniform'."
        )





    print('Look up for lens mass range for each sampled disk/bulge lens (age, [M/H]) pair', flush=True)
    mass_min_arr, mass_max_arr = mass_bounds_for_samples("PARSEC_isochrone/final_isochrone/final_isochrone_label012_vista_roman_euclid_ogle2_csst_merged_no_repeating_mass.npz", Age_lens, MH_lens)





    print('  Generating lens masses for process %s'%ni, flush=True)
    # Generate Lens mass 
    Ml, Lens_type = RandomSampleFromCDF2(xs=Mls, CDFs=cdfMs, ratios=ratioMs,
                                        n_sample=nFine, smooth=True)
    
    #### here need to resample the samples that are outside the mass range of EACH isochrone(Age_lens, MH_lens)
    index_out_of_mass_range = (Ml > mass_max_arr) | (Ml < mass_min_arr)
    count_out_of_mass_range = np.sum(index_out_of_mass_range)

    if count_out_of_mass_range > 0 : 
        print('  Resampling fraction = %.3e lens out of the mass range of EACH isochrone(Age_lens, MH_lens) for process %s'%(count_out_of_mass_range/nFine, ni), flush=True)
    
        while np.any(index_out_of_mass_range) : 
            
            Ml[index_out_of_mass_range], Lens_type[index_out_of_mass_range] = RandomSampleFromCDF2(xs=Mls, CDFs=cdfMs, ratios=ratioMs,
                                                                                                    n_sample=np.sum(index_out_of_mass_range), smooth=True)

            index_out_of_mass_range = (Ml > mass_max_arr) | (Ml < mass_min_arr)


    del mass_min_arr, mass_max_arr
    ### large arrays remained: 
    ###     Ds, Dl, Source_loc, Lens_loc, Rs, Rl, Zs, Zl, Rs_vec, Rl_vec, sBulgQ, sDiskQ, lBulgQ, lDiskQ, Age_lens, MH_lens
    ###     Ml, Lens_type





    print('  Generating source & lens velocities (bulge) for process %s'%ni, flush=True)
    # Generate bulge velocities
    if SourceFromPM:
        musl_Bulg, musb_Bulg = np.random.multivariate_normal(mus_mean, mus_cov, size=ns_Bulg).T
        Vsl_Bulg = musl_Bulg / Year2Day / k_trans * Ds[sBulgQ] + Solar_Vl
        Vsb_Bulg = musb_Bulg / Year2Day / k_trans * Ds[sBulgQ] + Solar_Vb

        del musl_Bulg, musb_Bulg

    else:
        Vsl_Bulg = np.random.normal(0., 120., ns_Bulg)
        Vsb_Bulg = np.random.normal(0., 120., ns_Bulg)

    Vll_Bulg = np.random.normal(0., 120., nl_Bulg)
    Vlb_Bulg = np.random.normal(0., 120., nl_Bulg)
    ### large arrays remained: 
    ###     Ds, Dl, Source_loc, Lens_loc, Rs, Rl, Zs, Zl, Rs_vec, Rl_vec, sBulgQ, sDiskQ, lBulgQ, lDiskQ, Age_lens, MH_lens
    ###     Ml, Lens_type
    ###     Vsl_Bulg, Vsb_Bulg, Vll_Bulg, Vlb_Bulg



    print('  Generating source & lens velocities (disk) for process %s, Model: %s (Rd=%.1f)...'%(ni,DiskModel,Rd), flush=True)
    # Generate disk velocities
    Rs_Disk = Rs[sDiskQ]
    Rl_Disk = Rl[lDiskQ]
    Rs_vec_Disk = Rs_vec[sDiskQ] # (N,3)
    Rl_vec_Disk = Rl_vec[lDiskQ] # (N,3)

    del Rs, Rl, Rs_vec, Rl_vec, Zs, Zl


    if DiskModel == 'Shu':
        Vsl_Disk = np.zeros(ns_Disk)
        Vsb_Disk = np.zeros(ns_Disk)
        Vll_Disk = np.zeros(nl_Disk)
        Vlb_Disk = np.zeros(nl_Disk)
        for rs in np.arange(Rmin, Rmax, Rstep):
            sigma_b = sigma_b0*np.exp((8.3 - rs)/(4*Rd))
            Rs_select = ((Rs_Disk > rs-0.05) & (Rs_Disk <= rs+0.05))
            Rl_select = ((Rl_Disk > rs-0.05) & (Rl_Disk <= rs+0.05))
            nRs_select = int(np.sum(Rs_select))
            nRl_select = int(np.sum(Rl_select))
            if (nRs_select + nRl_select) == 0:
                continue
            Vt_gen0 = np.arange(0, 3, dV)
            Vr_gen0 = np.arange(-2, 2, dV)
            Vt_cdf = np.load(f'velocity_data/Shu-vT-CDF-R{rs:.2f}-dV5e-3.npy')
            Vr_cdf = np.load(f'velocity_data/Shu-vR-CDF-R{rs:.2f}-dV5e-3.npy')

            Vt_Disk_1 = RandomSampleFromCDF(Vt_gen0, Vt_cdf, nRs_select+nRl_select)
            Vr_Disk_1 = RandomSampleFromCDF(Vr_gen0, Vr_cdf, nRs_select+nRl_select)

            r_vec = np.vstack([Rs_vec_Disk[Rs_select], Rl_vec_Disk[Rl_select]]) # (N,3)
            Vl_Disk_1 = vRvT2vl(Vr_Disk_1, Vt_Disk_1, r_vec[:, :2], direction[:2], sign_theta) # (N,2)

            Vsl_Disk[Rs_select] = Vl_Disk_1[:nRs_select]
            Vll_Disk[Rl_select] = Vl_Disk_1[nRs_select:]

            Vsb_Disk[Rs_select] = np.random.normal(0., sigma_b, size=nRs_select)
            Vlb_Disk[Rl_select] = np.random.normal(0., sigma_b, size=nRl_select)

        Vsl_Disk *= vc
        Vll_Disk *= vc

        del Rs_select, Rl_select, Vt_gen0, Vr_gen0, Vt_cdf, Vr_cdf, Vt_Disk_1, Vr_Disk_1, r_vec, Vl_Disk_1

    elif DiskModel == 'Gaussian':
        Vst_Disk = np.random.normal(vc, sigma_t0, size=ns_Disk)
        Vsr_Disk = np.random.normal(0,  sigma_r0, size=ns_Disk)
        Vlt_Disk = np.random.normal(vc, sigma_t0, size=nl_Disk)
        Vlr_Disk = np.random.normal(0,  sigma_r0, size=nl_Disk)
        Vsl_Disk = vRvT2vl(Vsr_Disk, Vst_Disk, Rs_vec_Disk[:, :2], direction[:2], sign_theta)
        Vll_Disk = vRvT2vl(Vlr_Disk, Vlt_Disk, Rl_vec_Disk[:, :2], direction[:2], sign_theta)
        Vsb_Disk = np.random.normal(0., sigma_b0, ns_Disk)
        Vlb_Disk = np.random.normal(0., sigma_b0, nl_Disk)

        del Vst_Disk, Vsr_Disk, Vlt_Disk, Vlr_Disk

    elif DiskModel == 'GaussianM':
        Vsl_Disk = np.zeros(ns_Disk); Vsb_Disk = np.zeros(ns_Disk)
        Vll_Disk = np.zeros(nl_Disk); Vlb_Disk = np.zeros(nl_Disk)
        for rs in np.arange(0.05, 8.35, 0.1):
            sigma_t = sigma_t0*np.exp((8.3 - rs)/(4*Rd))
            sigma_r = sigma_r0*np.exp((8.3 - rs)/(4*Rd))
            sigma_b = sigma_b0*np.exp((8.3 - rs)/(4*Rd))
            Rs_select = ((Rs_Disk >= rs-0.05) & (Rs_Disk < rs+0.05))
            Rl_select = ((Rl_Disk >= rs-0.05) & (Rl_Disk < rs+0.05))
            nRs_select = int(np.sum(Rs_select))
            nRl_select = int(np.sum(Rl_select))
            if (nRs_select + nRl_select) == 0:
                continue
            Vt_Disk_1 = np.random.normal(vc, sigma_t, size=nRs_select+nRl_select)
            Vr_Disk_1 = np.random.normal(0,  sigma_r, size=nRs_select+nRl_select)
            r_vec = np.vstack([Rs_vec_Disk[Rs_select], Rl_vec_Disk[Rl_select]]) # (N,3)
            Vl_Disk_1 = vRvT2vl(Vr_Disk_1, Vt_Disk_1, r_vec[:, :2], direction[:2], sign_theta) # (N,2)
            Vsl_Disk[Rs_select] = Vl_Disk_1[:nRs_select]
            Vll_Disk[Rl_select] = Vl_Disk_1[nRs_select:]
            Vsb_Disk[Rs_select] = np.random.normal(0., sigma_b, size=nRs_select)
            Vlb_Disk[Rl_select] = np.random.normal(0., sigma_b, size=nRl_select)
        
        del Rs_select, Rl_select, Vt_Disk_1, Vr_Disk_1, r_vec, Vl_Disk_1

    del Rs_Disk, Rl_Disk, Rs_vec_Disk, Rl_vec_Disk
    ### large arrays remained: 
    ###     Ds, Dl, Source_loc, Lens_loc, sBulgQ, sDiskQ, lBulgQ, lDiskQ, Age_lens, MH_lens
    ###     Ml, Lens_type
    ###     Vsl_Bulg, Vsb_Bulg, Vll_Bulg, Vlb_Bulg
    ###     Vsl_Disk, Vsb_Disk, Vll_Disk, Vlb_Disk

    # override disk source kinematics if SourceFromPM
    if SourceFromPM:
        musl_Disk, musb_Disk = np.random.multivariate_normal(mus_mean, mus_cov, size=ns_Disk).T
        Vsl_Disk = musl_Disk/Year2Day/k_trans*Ds[sDiskQ] + Solar_Vl
        Vsb_Disk = musb_Disk/Year2Day/k_trans*Ds[sDiskQ] + Solar_Vb

        del musl_Disk, musb_Disk

    # combine bulge/disk
    Vsl = np.zeros(nFine); Vsb = np.zeros(nFine)
    Vll = np.zeros(nFine); Vlb = np.zeros(nFine)
    Vsl[sBulgQ] = Vsl_Bulg; Vsb[sBulgQ] = Vsb_Bulg
    Vsl[sDiskQ] = Vsl_Disk; Vsb[sDiskQ] = Vsb_Disk
    Vll[lBulgQ] = Vll_Bulg; Vlb[lBulgQ] = Vlb_Bulg
    Vll[lDiskQ] = Vll_Disk; Vlb[lDiskQ] = Vlb_Disk

    del Vsl_Bulg, Vsb_Bulg, Vll_Bulg, Vlb_Bulg, Vsl_Disk, Vsb_Disk, Vll_Disk, Vlb_Disk
    del sBulgQ, sDiskQ, lBulgQ, lDiskQ 
    ### large arrays remained: 
    ###     Ds, Dl, Source_loc, Lens_loc, Age_lens, MH_lens
    ###     Ml, Lens_type
    ###     Vsl, Vsb, Vll, Vlb





    # Save
    savename = f'output/{eventname}/MockEvent_{eventname}_batch_{ni}.h5'
    print('  Saving data to %s ...'%(savename), flush=True)

    if SolarMotion:
        Vsl -= Solar_Vl; Vsb -= Solar_Vb; Vll -= Solar_Vl; Vlb -= Solar_Vb
    if EarthMotion:
        Vsl -= Earth_Vl; Vsb -= Earth_Vb; Vll -= Earth_Vl; Vlb -= Earth_Vb


    with h5py.File(savename, "w") as f:
        g = f.create_group("events")

        # Use "compression=None" to disable compression. "lzf" = fast, low-CPU.
        comp = "lzf"   # or None, or "gzip"

        g.create_dataset("Ds",           data=np.asarray(Ds,           np.float32), compression=comp)
        g.create_dataset("Dl",           data=np.asarray(Dl,           np.float32), compression=comp)
        g.create_dataset("Vsl",          data=np.asarray(Vsl,          np.float32), compression=comp)
        g.create_dataset("Vsb",          data=np.asarray(Vsb,          np.float32), compression=comp)
        g.create_dataset("Vll",          data=np.asarray(Vll,          np.float32), compression=comp)
        g.create_dataset("Vlb",          data=np.asarray(Vlb,          np.float32), compression=comp)
        g.create_dataset("Ml",           data=np.asarray(Ml,           np.float32), compression=comp)
        g.create_dataset("Age_lens",     data=np.asarray(Age_lens,     np.float32), compression=comp)
        g.create_dataset("MH_lens",      data=np.asarray(MH_lens,      np.float32), compression=comp)

        # LensType: 0-Main sequence; 1-White dwarf; 2-Neutron star; 3-Black hole
        # SourceLocation & LensLocation: 0-Bulge; 1-Disk
        g.create_dataset("Lens_type",    data=np.asarray(Lens_type,     np.int32), compression=comp)
        g.create_dataset("Source_loc",   data=np.asarray(Source_loc,    np.int32), compression=comp)
        g.create_dataset("Lens_loc",     data=np.asarray(Lens_loc,      np.int32), compression=comp)

    print(f'process {ni} done in {time.time()-time_begin:.2f}s', flush=True)



    # Ds_out = (np.round_(Ds, 3)).astype('str')
    # Dl_out = (np.round_(Dl, 3)).astype('str')
    # del Ds, Dl

    # if SolarMotion:
    #     Vsl -= Solar_Vl; Vsb -= Solar_Vb; Vll -= Solar_Vl; Vlb -= Solar_Vb
    # if EarthMotion:
    #     Vsl -= Earth_Vl; Vsb -= Earth_Vb; Vll -= Earth_Vl; Vlb -= Earth_Vb

    # Vsl_out = (np.round_(Vsl, 3)).astype('str')
    # Vsb_out = (np.round_(Vsb, 3)).astype('str')
    # Vll_out = (np.round_(Vll, 3)).astype('str')
    # Vlb_out = (np.round_(Vlb, 3)).astype('str')
    # Ml_out  = (np.round_(Ml,  3)).astype('str')
    # del Vsl, Vsb, Vll, Vlb, Ml

    # Lens_type_out  = ((Lens_type + 0.4).astype('int')).astype('str')
    # Source_loc_out = (Source_loc).astype('str')
    # Lens_loc_out   = (Lens_loc).astype('str')
    # del Lens_type, Source_loc, Lens_loc

    # # result = np.vstack((Ds_out, Dl_out, Vsl_out, Vsb_out, Vll_out, Vlb_out,
    # #                     Ml_out, Lens_type_out, Source_loc_out, Lens_loc_out)).T

    # header = 'Ds Dl Vsl Vsb Vll Vlb Ml LensType SourceLocation LensLocation'
    # header += '\nLensType: 0-Main sequence; 1-White dwarf; 2-Neutron star; 3-Black hole'
    # header += '\nSourceLocation & LensLocation: 0-Bulge; 1-Disk\n'

    # print('  Saving data to %s ...'%(savename), flush=True)
    # np.savetxt(savename, np.vstack((Ds_out, Dl_out, Vsl_out, Vsb_out, Vll_out, Vlb_out, Ml_out, Lens_type_out, Source_loc_out, Lens_loc_out)).T, \
    #            fmt='%s', header=header)

    # # del Vsl, Vsb, Vll, Vlb, Ml, Lens_type, Source_loc, Lens_loc, result
    
    # -------------- body ends --------------
    return savename



def SimulateEvents(config):
    k_trans = 0.0005775483273639937   # km/s/kpc -> mas/day: 0.0005775483273639937
    Year2Day = 365.25

    ###################################
    ### read parameters from config ###
    eventname = config.name
    ndots = config.ndot
    nstart,nrun = config.nstart,config.nrun
    n_process = config.n_process

    Dl_profile = config.Dl_profile
    Ds_profile = config.Ds_profile
    mkdir('output/%s'%eventname)
    
    # alphas,deltas = [(17.+45./60.+37.224/3600.)*15],[-(28.+56./60.+10.23/3600.)]
    # alphaGC,deltaGC = (17.+45./60.+37.224/3600.)*15 , -(28.+56./60.+10.23/3600.)
    alphas,deltas = config.ra, config.dec
    Galactic_ls,Galactic_bs = gm.GetGalacticCoordinates(alphas,deltas)


    # Galactic model parameters
    gamma = config.gamma
    Rd = config.Rd
    vc = config.vc
    sigma_t0 = config.sigma_Vt
    sigma_b0 = config.sigma_Vz
    sigma_r0 = config.sigma_VR
    DiskModel = config.DiskModel
    # print('### Jiyuan ###')
    # print(DiskModel)
    # print('###  end   ###')
    

    # Galactic components
    bulge_source = config.bulge_source
    disk_source  = config.disk_source
    bulge_lens   = config.bulge_lens
    disk_lens    = config.disk_lens
    sDisk_min = config.sDisk_min
    sDisk_max = config.sDisk_max
    lDisk_min = config.lDisk_min
    lDisk_max = config.lDisk_max
    Bulg_min = config.Bulg_min
    Bulg_max = config.Bulg_max

    SourceFromPM = config.SourceFromPM
    if SourceFromPM == True:
        muslmean,muslerr = config.muslmean,config.muslerr
        musbmean,musberr = config.musbmean,config.musberr
        muscor = config.muscor

        mus_mean = [muslmean,musbmean]
        mus_cov  = [[muslerr**2,muscor*muslerr*musberr],[muscor*muslerr*musberr,musberr**2]]
    ############# changed here for parallized version #############
    else : 
        mus_mean = None
        mus_cov  = None
    ############# changed here for parallized version #############

    # Lens type
    main_sequence = config.main_sequence
    white_dwarf   = config.white_dwarf
    neutron_star  = config.neutron_star
    black_hole    = config.black_hole
    # max_main_sequence_mass_bulge = config.max_main_sequence_mass_bulge
    # max_main_sequence_mass_disk  = config.max_main_sequence_mass_disk

    # Lens age/metallicity prior type
    Lens_Age_Metallicity_Prior = config.Lens_Age_Metallicity_Prior

    # output option
    SolarMotion = config.SolarMotion
    EarthMotion = config.EarthMotion
    Earth_Vl = config.Earth_Vl
    Earth_Vb = config.Earth_Vb
    Solar_Vl = config.Solar_Vl
    Solar_Vb = config.Solar_Vb

    ### read parameters from config ###
    ###################################

    ### calculate sign ###
    sign_theta = 2*(Galactic_ls<=180)-1


    ### Get event direction ###
    ### set to Galactic center
    rgcl = gm.GetGalactocentricCoordinates(alphas, deltas, 8.3)
    Earth = gm.GetGalactocentricCoordinates(0.,0.,0.)
    direction = rgcl - Earth
    direction = direction/8.3  #normalized vector
    direction,Earth = np.array(direction), np.array(Earth)

    if DiskModel=='Shu':
        dV=1.0/200.0
        Rmin,Rmax,Rstep=0.05,8.35,0.1
        genVelocityDistribution_Shu(sigma_r0=sigma_r0, Rd=Rd, vc=vc
                                    ,dV=dV, Rmin=Rmin,Rmax=Rmax,Rstep=Rstep
                                    ,savepath = 'velocity_data')
    ############# changed here for parallized version #############
    else : 
        dV = None
        Rmin, Rmax, Rstep = None, None, None
    ############# changed here for parallized version #############


    ### Calculate cumulatives ###
    print('Calculating Ds, Dl probability distribution (PDF/CDF)...',end='    ',flush=True)
    dDs = 1.0/2000.0
    dDl = 1.0/2000.0

    Ds_Bulg = np.arange(Bulg_min,Bulg_max,dDs)   # source
    Ds_Disk = np.arange(sDisk_min,sDisk_max,dDs)   # source
    Dl_Bulg = np.arange(Bulg_min,Bulg_max,dDl) # lens in bulge
    Dl_Disk = np.arange(lDisk_min,lDisk_max,dDl) # lens in disk 
    # Dl_all = cp.array(Dl_Disk)

    # len_Ds_Bulg = len(Ds_Bulg)
    # len_Dl_Bulg = len(Dl_Bulg)

    # 计算在银心坐标(x,y,z)下恒星的位置
    posits_Bulg = Ds_Bulg * direction[:,np.newaxis] + Earth[:,np.newaxis]
    posits_Disk = Ds_Disk * direction[:,np.newaxis] + Earth[:,np.newaxis]
    positl_Bulg = Dl_Bulg * direction[:,np.newaxis] + Earth[:,np.newaxis]
    positl_Disk = Dl_Disk * direction[:,np.newaxis] + Earth[:,np.newaxis]

    pdfs_Bulg = gm.BulgeStellarDensity(posits_Bulg)*(Ds_Bulg**(2-gamma+Ds_profile))
    pdfs_Disk = gm.DiskStellarDensity(posits_Disk,Rd=Rd)*(Ds_Disk**(2-gamma+Ds_profile))
    cdfs_Bulg = np.cumsum(pdfs_Bulg)
    cdfs_Disk = np.cumsum(pdfs_Disk)
    if bulge_source ==False:
        cdfs_Bulg = np.zeros_like(Ds_Bulg)
    if disk_source  ==False:
        cdfs_Disk = np.zeros_like(Ds_Disk)
    Dss  = [  Ds_Bulg,  Ds_Disk]
    cdfDs = [cdfs_Bulg,cdfs_Disk]
    ratioDs = [float(cdfs_Bulg[-1]),float(cdfs_Disk[-1])]

    # sBulgRatio = np.sum(pdfs_Bulg)/(np.sum(pdfs_Bulg)+np.sum(pdfs_Disk))

    pdfl_Bulg = gm.BulgeStellarDensity(positl_Bulg)*(Dl_Bulg**Dl_profile)
    pdfl_Disk = gm.DiskStellarDensity(positl_Disk,Rd=Rd)*(Dl_Disk**Dl_profile)
    cdfl_Bulg = np.cumsum(pdfl_Bulg)
    cdfl_Disk = np.cumsum(pdfl_Disk)
    if bulge_lens ==False:
        cdfl_Bulg = np.zeros_like(Dl_Bulg)
    if disk_lens  ==False:
        cdfl_Disk = np.zeros_like(Dl_Disk)
    Dls  = [  Dl_Bulg,  Dl_Disk]
    cdfDl = [cdfl_Bulg,cdfl_Disk]
    ratioDl = [float(cdfl_Bulg[-1]),float(cdfl_Disk[-1])]

    print('done',flush=True)



    print('Calculating Ml probability distribution (PDF/CDF)...',end='    ',flush=True)
    dM = 1.0/2000.0
    Ml_MS = np.arange(0.090, 3.600, dM) # the mass range of all isochrone is [0.09, 3.596] M_sun
    Ml_WD = np.arange(0.15,1.5,dM)
    #Ml_WD = cp.arange(0.60,0.72,dM) # for kb193289
    Ml_NS = np.arange(0.60,3.2,dM)
    Ml_BH = np.arange(1.50,100,dM)
    cdfM_MS = np.cumsum(np.array( gm.MFMS(Ml_MS) ))
    cdfM_WD = np.cumsum(np.array( gm.MFWD(Ml_WD) ))
    #cdfM_WD = cp.cumsum(cp.array( np.ones_like(Ml_WD) )) # for kb193289
    cdfM_NS = np.cumsum(np.array( gm.MFNS(Ml_NS) ))
    cdfM_BH = np.cumsum(np.array( gm.MFBH(Ml_BH) ))
    if main_sequence == False:
        cdfM_MS = np.zeros_like(Ml_MS)
    if white_dwarf == False:
        cdfM_WD = np.zeros_like(Ml_WD)
    if neutron_star == False:
        cdfM_NS = np.zeros_like(Ml_NS)
    if black_hole  == False:
        cdfM_BH = np.zeros_like(Ml_BH)
    Mls   = [  Ml_MS,  Ml_WD,  Ml_NS,  Ml_BH]
    cdfMs = [cdfM_MS,cdfM_WD,cdfM_NS,cdfM_BH]
    ratioMs = [float(cdfM_MS[-1]),float(cdfM_WD[-1]),float(cdfM_NS[-1]),float(cdfM_BH[-1])]


    # ### jiyuan ###
    # # Ml_MS_Bulg = np.arange(0.01,1.1,dM)#here is the difference from above
    # Ml_MS_Bulg = np.arange(0.01, max_main_sequence_mass_bulge, dM)
    # Ml_WD_Bulg = np.arange(0.15,1.5,dM)
    # Ml_NS_Bulg = np.arange(0.60,3.2,dM)
    # Ml_BH_Bulg = np.arange(1.50,100,dM)
    # cdfM_MS_Bulg = np.cumsum(np.array( gm.MFMS(Ml_MS_Bulg) ))
    # cdfM_WD_Bulg = np.cumsum(np.array( gm.MFWD(Ml_WD_Bulg) ))
    # cdfM_NS_Bulg = np.cumsum(np.array( gm.MFNS(Ml_NS_Bulg) ))
    # cdfM_BH_Bulg = np.cumsum(np.array( gm.MFBH(Ml_BH_Bulg) ))
    # if main_sequence == False:
    #     cdfM_MS_Bulg = np.zeros_like(Ml_MS_Bulg)
    # if white_dwarf == False:
    #     cdfM_WD_Bulg = np.zeros_like(Ml_WD_Bulg)
    # if neutron_star == False:
    #     cdfM_NS_Bulg = np.zeros_like(Ml_NS_Bulg)
    # if black_hole  == False:
    #     cdfM_BH_Bulg = np.zeros_like(Ml_BH_Bulg)
    # Mls_Bulg   = [  Ml_MS_Bulg,  Ml_WD_Bulg,  Ml_NS_Bulg,  Ml_BH_Bulg]
    # cdfMs_Bulg = [cdfM_MS_Bulg,cdfM_WD_Bulg,cdfM_NS_Bulg,cdfM_BH_Bulg]
    # ratioMs_Bulg = [float(cdfM_MS_Bulg[-1]),float(cdfM_WD_Bulg[-1]),float(cdfM_NS_Bulg[-1]),float(cdfM_BH_Bulg[-1])]
    # ###   end  ###

    print('done',flush=True)

    # Build the read-only context we broadcast to workers:
    ctx = (eventname, ndots, direction, Earth,
           Dss, cdfDs, ratioDs, Dls, cdfDl, ratioDl,
           Mls, cdfMs, ratioMs, 
           DiskModel, Rd, vc, sigma_t0, sigma_b0, sigma_r0,
           SourceFromPM, mus_mean, mus_cov,
           SolarMotion, EarthMotion, Solar_Vl, Solar_Vb, Earth_Vl, Earth_Vb,
           sign_theta, Rmin, Rmax, Rstep, dV, k_trans, Year2Day, 
           Lens_Age_Metallicity_Prior)


    # -------- parallel map over ni --------
    ni_list = list(range(nstart, nstart+nrun))
    args_iter = [(ni, ctx) for ni in ni_list]

    # Use spawn for safety (works on Linux/Mac/Windows)
    with get_context("spawn").Pool(processes=n_process) as pool:
        # imap_unordered gives results as they finish
        for _ in pool.imap_unordered(simulate_one_run, args_iter, chunksize=1):
            pass
        