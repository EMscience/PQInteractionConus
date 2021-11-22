# Edom Moges 
# Environmental Systems Dynamics Laboratory (ESDL)
# University of California Berkeley
# This code inteds to compute different lagged interaction metrics. 
# Has the option to handle outliers and zeros with flags.


''' 
In computing probabilities, it accounts for the effect of zeros which are common in hydrology. For instancce, precipitation and streamflow data. The metrics include:


1. Entropies - $H(X)$ and $H(Y)$
2. Mutual information - $I(Y_{t};X_{t-lag})$
3. Transfer entropy - $I(Y_{t};X_{t-lag}|Y_{t-1})$
4. Correlation coeficients - $\rho (Y_{t},X_{t-lag})$
5. Total information based on X - $I(Y_{t}; X_{t},X_{t-1})$
6. Total information based on Y - $I(Y_{t}; X_{t},Y_{t-1})$

These metrics are not normalized. Normalization can be done in postprocessing.

'''
#get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import copy
import os
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy import stats

# Edom Moges @ ESDL
np.random.seed(50)

# Histogram - modified functions to handle outliers
def histogram(data, nbins,lower_bound ,upper_bound):
    if (lower_bound == None) & (upper_bound == None):
        return np.histogram(data,bins=nbins)

    elif lower_bound == None:
        upper_count = len(data[data>=upper_bound])
        counts, binEdges = np.histogram(data[data<upper_bound], bins = nbins)

        counts[-1] += upper_count # add to last bin

    elif upper_bound == None:

        lower_count =len(data[data<=lower_bound])
        counts, binEdges = np.histogram(data[data>lower_bound], bins = nbins)

        #add back in outliers
        counts[0] += lower_count

    else:
        lower_count = len(data[data<=lower_bound])
        upper_count =   len(data[data>=upper_bound])
        counts, binEdges = np.histogram(data[(data>lower_bound) & (data<upper_bound)], bins = nbins)

        #add back in outliers
        counts[0] += lower_count
        counts[-1] += upper_count

    return counts, binEdges

def digitize(data,binEdges, lower_bound, upper_bound):
    colcat = np.digitize(data, binEdges, right=False)

    if (lower_bound == None) & (upper_bound == None):
        #colcat[colcat == max(colcat)] = 0
        return colcat
    n = len(binEdges) - 1

    if lower_bound == None:
        colcat[data >= upper_bound] = n  #put upper outliers into the last bin

    elif upper_bound == None:
        colcat[data <= lower_bound] = 1 #put lower outliers into the first bin

    else:
        colcat[data <= lower_bound] = 1 #put lower outliers into the first bin
        colcat[data >= upper_bound] =n  #put upper outliers into the last bin


    #colcat[colcat == max(colcat)] = 0 # set NAN values to 0
    return colcat

def find_bounds(data, lower,upper):
    if (lower == None) & (upper == None):
        return None, None
    if lower == None:
        return None, np.nanpercentile(data, upper)
    if upper == None:
        return np.nanpercentile(data, lower), None
    return np.nanpercentile(data, lower), np.nanpercentile(data, upper)


def ZeroAdjustment(MM, nbins, ths, lower_bound, upper_bound): # develops a 1D histogram that is adjusted for zeros.
                                    # zero adjustment follows the method by Chapman 1986.
                                    # returns which bin (ID) is the data in M is located.
    # Identify the indexes of zero and nonzeros
    ZeroIndex = np.argwhere(MM==0)
    NonZeroIndex = np.argwhere(MM!=0)
    ZeroCat = np.nan*np.ones(MM.shape[0])

    M = MM[MM!=0]
    
    
    # Develop bins
    if type(nbins)==list:
        nbins = nbins[1]
    else:
        nbins = nbins
    counts, binEdges=histogram(M[np.isfinite(M)],nbins-1,lower_bound, upper_bound) # Sink Variable $$$ bin number less by one
    binEdges[0] = binEdges[0]-ths
    binEdges[len(binEdges)-1]=binEdges[len(binEdges)-1] + ths
    colcat = digitize(M, binEdges, lower_bound, upper_bound)  # which bin (ID) is the data located

    # Zero adjustment
    colcat = colcat + 1 # Shift the non-zero bin id by one.Thus, zero values occupy the first bin.

    ZeroCat[NonZeroIndex] = colcat.reshape(colcat.shape[0],1) # Bin index for non-zeros, starts at 2
    ZeroCat[ZeroIndex] = 1 # Bin index for zeros

#     print('col2cat = ', colcat)

    colcat = copy.deepcopy(ZeroCat.astype(int))
#     print('Final' ,colcat)
    
    return colcat # which bin (ID) is the data located

def computeEntropy(M,nbin,lower_bound, upper_bound, Zflag): # ZeroAdjusted 1D entropy
                            # Returns Entropy
    M = M[np.isfinite(M)] # Work on the non-NAN


    if Zflag == 1:
        ZeroIndex = np.argwhere(M==0)
        NonZeroIndex = np.argwhere(M!=0)
        NumZeros = ZeroIndex.shape[0]
        MM = M[M!=0]
        M = MM
        N, binEdges1d=histogram(M,nbin-1,lower_bound, upper_bound) #Which bin the data column is in
        N = np.r_[NumZeros,N]
        #print(N)
    else:
        N, binEdges1d=histogram(M[np.isfinite(M)],nbin,lower_bound, upper_bound) #Which bin the data column is in
    p2 = N/sum(N)
    
    # Shanon entropy
    p2gt0 = p2[p2>0] # py
    log2p2gt0 = np.log2(p2gt0)
    H = (-sum(p2gt0*log2p2gt0))
    return H

def computeBincountZeroAdjusted(M,nbin,lower_bound, upper_bound, Zflag): # ZeroAdjusted counts of data in each bin.
                                         # Bin 1 is the count of zeros.
    M = M[np.isfinite(M)] # work on the non-nan

    if Zflag == 1:
        ZeroIndex = np.argwhere(M==0)
        NonZeroIndex = np.argwhere(M!=0)
        NumZeros = ZeroIndex.shape[0]
        MM = M[M!=0]
        M = MM
        N, binEdges1d=histogram(M,nbin-1,lower_bound, upper_bound) #Which bin the data column is in
        N = np.r_[NumZeros,N]
        #print(N)
    else:
        N, binEdges1d=histogram(M[np.isfinite(M)],nbin,lower_bound, upper_bound) #Which bin the data column is in
    return N

def jointentropy3D(M2, nbins,lower_bound, upper_bound, Zdim1, Zdim2, Zdim3):
    # Calculates input N that is an input to the joint entropy for three variables H(x,y,z)
    # N - zero adjusted counts of data in each bin.
    # M2 is a three-column matrix that contains [Xlagged, Ytarget, Ylagged by 1] for TE
    # M2 is [Yt, Xt, Xt-1] - for I(Yt;Xt,Xt-1)
    # nvalidpoints is the number of rows (samples) used to calculate the joint entropy
    
    ths = 10e-4
    #print(M2.shape)
    M = M2[~np.isnan(M2).any(axis=1)] # clears the nans at both columns
    #print('J3D', M.shape, M[0:5,:])
    
    # zero adjustment
    if Zdim1 == 1:
        col1cat = ZeroAdjustment(M[:,0],nbins, ths,lower_bound[0], upper_bound[0])
    else:
        counts1, binEdges1= histogram(M[:,0][np.isfinite(M[:,0])],nbins,lower_bound[0], upper_bound[0]) 
        binEdges1[0] = binEdges1[0]-ths
        binEdges1[len(binEdges1)-1]=binEdges1[len(binEdges1)-1]+ths
        col1cat = digitize(M[:,0], binEdges1, lower_bound[0], upper_bound[0])
       
    
    if Zdim2 == 1:
        col2cat = ZeroAdjustment(M[:,1],nbins, ths,lower_bound[1], upper_bound[1])
    else:
        counts2, binEdges2=histogram(M[:,1][np.isfinite(M[:,1])],nbins,lower_bound[1], upper_bound[1]) 
        binEdges2[0] = binEdges2[0]-ths
        binEdges2[len(binEdges2)-1]=binEdges2[len(binEdges2)-1]+ths
        col2cat = digitize(M[:,1], binEdges2, lower_bound[1], upper_bound[1])  # which bin (ID) is the data located
    
    
    if Zdim3 == 1:
        col3cat = ZeroAdjustment(M[:,2],nbins, ths,lower_bound[2], upper_bound[2])
    else:
        counts3, binEdges3=histogram(M[:,2][np.isfinite(M[:,2])],nbins,lower_bound[2], upper_bound[2]) 
        binEdges3[0] = binEdges3[0]-ths
        binEdges3[len(binEdges3)-1]=binEdges3[len(binEdges3)-1]+ths
        col3cat = digitize(M[:,2], binEdges3, lower_bound[2], upper_bound[2])
    
    # This classifies the joint entropy bin into a number between 1 and nbins^2. 0 is assigned to rows with misisng data.
    jointentcat = (col1cat-1)*nbins**2 + (col2cat-1)*nbins + col3cat 
    
    #print(np.asarray((jointentcat,col1cat,col2cat, col3cat)).T)
    
    nbins_3 = nbins**3
    N = np.bincount(jointentcat)[1:] # Number of datapoints within each joint entropy bin.

    
    corrCoeff = np.corrcoef(M[:,0],M[:,1])[0,1]
    
    return N, corrCoeff



def ComputeAnyDimensionalEntropy(Mx):   # given a matrix Mx with bin counts in any dimension,
                        # Returns Entropy of any dimension.
    pp = Mx[Mx>0]/np.sum(Mx)
    
    return -1*np.sum(pp * np.log2(pp))



def ComputeInfoMetricsBasedOn3DBincounts(N,nbins):
    # Computes 1D entropy, Mutual enformation and Transfer entropy.
    # The number of bins should be uniform across the 3 dimensions.
    # N is the bin count the result of 3D joint entropy
    
    if N.shape[0] < np.sum(nbins**3):
        Nnew  = np.r_[N,np.zeros(np.sum(nbins**3) - N.shape[0] )]
    else:
        Nnew = N
        
    # From 1D to a 3D matrix
    N3 =  Nnew.reshape(nbins,nbins,nbins) # Dimensions Xt -> dim 0, yt -> dim 1 and yt_1 -> dim 2
    
    # Dimensions Xt -> 0, yt -> 1 and yt_1 -> 2
    
    # 1D bincounts - input to 1D entropy
    Mxt = np.sum(N3,(1,2)) # np.sum((m,n)) eliminates the indicated dimensions. eliminates dimensions m and n.
    Myt = np.sum(N3,(0,2))
    Myt_1 = np.sum(N3,(0,1))

    # 2D bincounts - input to 2D entropies
    Mxtyt = np.sum(N3,2)
    Mytyt_1 = np.sum(N3,0)
    Myt_1xt = np.sum(N3,1)
    
    # Entrop computation
    # 1D Entropies
    Hxt = ComputeAnyDimensionalEntropy(Mxt)
    Hyt = ComputeAnyDimensionalEntropy(Myt)
    Hyt_1 = ComputeAnyDimensionalEntropy(Myt_1)
    
    # 2D entropies
    Hxtyt = ComputeAnyDimensionalEntropy(Mxtyt)
    Hytyt_1 = ComputeAnyDimensionalEntropy(Mytyt_1)
    Hyt_1xt = ComputeAnyDimensionalEntropy(Myt_1xt)
    
    # 3D entropy H(X,Y, Z)
    Hxtytyt_1 = ComputeAnyDimensionalEntropy(N3)
    
    # Compute MI and TE
    MI = Hxt + Hyt - Hxtyt
    TE = Hytyt_1 + Hyt_1xt - Hyt_1 - Hxtytyt_1
    
    
    return Hxt, Hyt, MI, TE

def ComputeTotalEntropy(N,nbins):
    # Computes I(Qt;Xt,Xt-1) = H(Qt) + H(Xt,Xt-1) - H(Qt, Xt, Xt-1)
    # N is the bin count - the result of 3D joint entropy
    # In jointentropy3D - Input dimensions should be Qt-> column 0, Xt -> column 1, and Xt-1 -> column 2
    
    if N.shape[0] < np.sum(nbins**3):
        Nnew  = np.r_[N,np.zeros(np.sum(nbins**3) - N.shape[0] )]
    else:
        Nnew = N
        
    # From 1D to a 3D matrix
    N3 =  Nnew.reshape(nbins,nbins,nbins) # with Qt-> dim 0, Xt -> dim 1, and Xt-1 -> dim 2
    
    # 1D bincounts - input to 1D entropy
    MQt = np.sum(N3,(1,2)) # eliminates the indicated dimension
    MXt = np.sum(N3,(0,2))
    MXt_1 = np.sum(N3,(0,1))
    
    # 2D bincounts - input to 2D entropies
    
    MXtXt_1 = np.sum(N3,0)
    
    # Compute entropies
    # 1D Entropies
    HQt = ComputeAnyDimensionalEntropy(MQt)
    HXt = ComputeAnyDimensionalEntropy(MXt)
    HXt_1 = ComputeAnyDimensionalEntropy(MXt_1)
    
    # 2D Entropies
    HXtXt_1 = ComputeAnyDimensionalEntropy(MXtXt_1)
    
    # 3D Entropies H(Qt, Xt, Xt-1)
    HQtXtXt_1 = ComputeAnyDimensionalEntropy(N3)
    
    IQtXtXt_1 = HQt + HXtXt_1 - HQtXtXt_1
    
    
    return HQt, HXt, HXt_1, IQtXtXt_1

def LagData_new( M_unlagged, shift ):
    # LagData Shifts two time-series so that a lagged version is generated.
    # M_unlagged is a matrix [X Y..n], where X and Y are column vectors of the
    # variables to be compared. shift is a row vector that says how much each
    # variable in M_unlagged is to be shifted by.
    
    nR,nC = np.shape(M_unlagged)
    maxShift = max(shift)
    minShift = min(shift)
    newlength = nR - maxShift + minShift
    M_lagged = np.nan*np.ones([newlength, nC]) #[source_lagged(1:n-lag), sink_unlagged(lag:n), sink_lagged(1:n-lag)]
    

    for ii in range(np.shape(M_lagged)[1]):
        M_lagged[:,ii] = M_unlagged[(shift[ii]-minShift):(np.shape(M_unlagged)[0]-maxShift+shift[ii]), ii]
        
    #print('LagData_new', M_lagged.shape)
    
    return M_lagged


def shuffle( M ):
    # shuffles the entries of the matrix M in time while keeping NaNs (blank data values) NaNs.
    # M is the matrix where the columns are individual variables and the rows are entries in time
    
    
    #np.random.seed(50) #====<<<<<<<<<<<<This will Fix the shuffles>>>>>>>>>>>>>>>>>>>>========#
    
    Mss = np.ones(np.shape(M))*np.nan # Initialize
    
    for n in range(np.shape(M)[1]): # Columns are shuffled separately
        notnans = np.argwhere(~np.isnan(M[:,n]))
        R = np.random.rand(np.shape(notnans)[0],1) #np.random.rand(5,1)
        I = np.argsort(R,axis=0)

        Mss[notnans[:,0],n] = M[notnans[I[:],0],n].reshape(np.shape(M[notnans[I[:],0],n])[0],)
        
    return  Mss

def joint3DShuffle(M, nbins,lower_bound, upper_bound,Zdim1, Zdim2, Zdim3):
    
    # Shuffles the matrix and returns the shuffled matrix
    
    Mss = shuffle(M)
    MIss, corr = jointentropy3D(Mss, nbins,lower_bound, upper_bound,Zdim1, Zdim2, Zdim3)
    
    return MIss, corr 



def joint3D_critic(M2, nbins, numiter, ncores, lower_bound, upper_bound, Zdim1, Zdim2, Zdim3):
    #  Returns the N (bin count) for each shuffled M2. 
    # Number of shuffle is numiter
    
    MIss = Parallel(n_jobs=ncores)(delayed(joint3DShuffle)(M2, nbins,lower_bound, upper_bound, Zdim1, Zdim2, Zdim3) \
                                   for ii in range(numiter))
    #print('J3_critic',len(MIss),MIss)
    return MIss


def computeCritical_TE_MI_Corr(M, nbins, numiter, alpha, ncore,lower_bound, upper_bound,Zdim1, Zdim2, Zdim3):
    # computes critical MI, TE, and correlation coefficient.
    # Based on T-statistics
    
    E_Hxt = np.ones([numiter,1])*np.nan
    E_Hyt = np.ones([numiter,1])*np.nan
    E_MI = np.ones([numiter,1])*np.nan
    E_TE = np.ones([numiter,1])*np.nan
    CorrC = np.ones([numiter,1])*np.nan
    
    c = joint3D_critic(M, nbins, numiter, ncore,lower_bound, upper_bound, Zdim1, Zdim2, Zdim3) # bincounts for numiter times
    d = np.asarray(c) # tuple to array => dimension = numiter * two column, First column N, second column corrCoefft
    
    for i in np.arange(numiter):
        Matr = d[i,0]
        CorrC[i] = d[i,1]

        # compute Info-flow metrics for each numiter
        E_Hxt[i], E_Hyt[i], E_MI[i], E_TE[i] = ComputeInfoMetricsBasedOn3DBincounts(Matr,nbins)

#     MIss = np.sort(E_MI)
#     MIcrit = MIss[round((1-alpha)*numiter)] # develop a histogram and peak the 95% quantile 
    MIcrit = np.nanmean(E_MI) + stats.t.ppf(1-alpha, 100)*np.nanstd(E_MI) #T-statistics

#     MTEss = np.sort(E_TE)
#     MTEcrit = MTEss[round((1-alpha)*numiter)] # develop a histogram and peak the 95% quantile
    MTEcrit = np.nanmean(E_TE) + stats.t.ppf(1-alpha, 100)*np.nanstd(E_TE)

#     CoRss = np.sort(CorrC)
#     CorrCrit = CoRss[round((1-alpha)*numiter)] # develop a histogram and peak the 95% quantile
    CorrCrit = np.nanmean(CorrC) + stats.t.ppf(1-alpha, 100)*np.nanstd(CorrC)

    return MIcrit, MTEcrit, CorrCrit




def computeCritical_Total_MI(MtotalIn, nbins, numiter, alpha, ncore,lower_bound, upper_bound,Zdim1, Zdim2, Zdim3):
    # Computes critical total mutual information.
   
    #print('419', MtotalIn, nbins, numiter, alpha, ncore,lower_bound, upper_bound,Zdim1, Zdim2, Zdim3)
    
    HQt = np.ones([numiter,1])*np.nan
    HXt = np.ones([numiter,1])*np.nan
    HXt_1 = np.ones([numiter,1])*np.nan
    IQtXtXt_1 = np.ones([numiter,1])*np.nan
    
    cMI = joint3D_critic(MtotalIn, nbins, numiter, ncore,lower_bound, upper_bound,Zdim1, Zdim2, Zdim3) # shuffle for total MI
    dMI = np.asarray(cMI)
    
    for i in np.arange(numiter):
        Matr = dMI[i,0]

        # compute total entropy for each numiter
        HQt[i], HXt[i], HXt_1[i], IQtXtXt_1[i] = ComputeTotalEntropy(Matr,nbins)

#     MIss = np.sort(IQtXtXt_1)
#     MIcritTot = MIss[round((1-alpha)*numiter)] # develop a histogram and peak the 95% quantile 

    MIcritTot = np.nanmean(IQtXtXt_1) + stats.t.ppf(1-alpha, 100)*np.nanstd(IQtXtXt_1)
    
    return MIcritTot

def ComputeInofoTheoreticMetricsAndSignificance(Yt, Pt, nbins, numiter, alpha, shift, MaxLag, ncore,lowerSink, upperSink, lowerSource, upperSource,ZFlagSink, ZFlagSource):
    
    # Yt - streamflow/Sink
    # Pt - precipitation/Source
#     Qt = copy.deepcopy(Yt[1:])
#     Xt = copy.deepcopy(Pt[1:])
#     Xt_1 = copy.deepcopy(Pt[:-1])
#     Qt_1 = copy.deepcopy(Yt[:-1])

    
    ULSink = find_bounds(Yt[np.nonzero(Yt)],lowerSink,upperSink)
    ULSource = find_bounds(Pt[np.nonzero(Pt)],lowerSource,upperSource)
    #print('454', ULSink,ULSink)
    
    lower_bound1 = [ULSink[0],ULSource[0],ULSink[0]] # Yt,Pt,Yt
    upper_bound1 = [ULSink[1],ULSource[1],ULSink[1]]
    
    lower_bound2 = [ULSink[0],ULSource[0],ULSource[0]] # Yt,Pt,Pt
    upper_bound2 = [ULSink[1],ULSource[1],ULSource[1]]
    
    lower_bound3 = [ULSource[0],ULSink[0],ULSink[0]] # Pt,Yt,Yt
    upper_bound3 = [ULSource[1],ULSink[1],ULSink[1]]
    
    lower_bound4 = [ULSource[0],ULSink[0],ULSource[0]] # Pt,Yt,Pt
    upper_bound4 = [ULSource[1],ULSink[1],ULSource[1]]
    
    # Do the one time computations
    #===============================================
    # 1. Critical Total MI based on Qt-1
    #MTotQt_1 = np.column_stack((Qt,Xt,Qt_1))
        
    MTotQt_1 = LagData_new( M_unlagged=np.column_stack((Yt,Pt,Yt)), shift=[0,-1*MaxLag,-1])
   
    #print('473',MTotQt_1, nbins, numiter, alpha, ncore,lower_bound1, upper_bound1, ZFlagSink, ZFlagSource, ZFlagSink)
    criticalTotMI_Qt_1 = computeCritical_Total_MI(MTotQt_1, nbins, numiter, alpha, ncore,lower_bound1, upper_bound1, ZFlagSink, ZFlagSource, ZFlagSink)
          
        
    #2. Critic for totalMI based on Pt
    #MTot = np.column_stack((Qt, Xt, Xt_1))
    MTot = LagData_new( M_unlagged=np.column_stack((Yt,Pt,Pt)), shift=[0,0,-1*MaxLag])
    criticalTotMI = computeCritical_Total_MI(MTot, nbins, numiter, alpha, ncore,lower_bound2, upper_bound2,ZFlagSink, ZFlagSource,ZFlagSource)

       
    #3. Critic For TE, corr and MI -- conditioned on yesterdays streamflow
    #M = np.column_stack((Xt, Qt, Qt_1))
    M = LagData_new( M_unlagged=np.column_stack((Pt,Yt,Yt)), shift=[-1*MaxLag,shift[1],shift[2]])
    MIcrit, TEcrit, CorrCrit = computeCritical_TE_MI_Corr(M, nbins, numiter, alpha,ncore,lower_bound3, upper_bound3,ZFlagSource,ZFlagSink,ZFlagSink)
    #print('main TEcritic', M[~np.isnan(M).any(axis=1)][:5,:])
    
    #4. Critical TE, MI, Corr -- conditioned on yesterdays precipitation
    
    #MinpuPre = np.column_stack((Xt, Qt, Xt_1))
    MinpuPre = LagData_new( M_unlagged=np.column_stack((Pt,Yt,Pt)), shift=[-1*MaxLag,shift[1],shift[2]])
    MIcritP, TEcritP, CorrCritP = computeCritical_TE_MI_Corr(MinpuPre, nbins, numiter, alpha,ncore,lower_bound4, upper_bound4, ZFlagSource,ZFlagSink,ZFlagSource)


    #iterate across lags and run joint3D  InfoMetrics
    Hx = np.nan*np.ones([MaxLag])
    Hy = np.nan*np.ones([MaxLag])
    MI = np.nan*np.ones([MaxLag])
    TE = np.nan*np.ones([MaxLag])
    CorrCoef = np.nan*np.ones([MaxLag])
    IQtXtXt_1 = np.nan*np.ones([MaxLag])
    TEp = np.nan*np.ones([MaxLag])
    MIp = np.nan*np.ones([MaxLag])
    IQtXtQt_1  = np.nan*np.ones([MaxLag])
    
    for lag in np.arange(MaxLag):

        # MI and TE conditioned on Qt-1       
        Minput = LagData_new( M_unlagged=np.column_stack((Pt,Yt,Yt)), shift=[-lag,shift[1],shift[2]])
        #print('Main_lag TE', lag, Minput[~np.isnan(Minput).any(axis=1)][:5,:])
        N, CorrCoef[lag] = jointentropy3D(Minput, nbins,lower_bound3, upper_bound3,
                                         ZFlagSource,ZFlagSink,ZFlagSink)
        Hx[lag], Hy[lag], MI[lag], TE[lag] = ComputeInfoMetricsBasedOn3DBincounts(N,nbins)
        
        ### Total Info
            # based on Pt-1
        MinputTot = LagData_new( M_unlagged=np.column_stack((Yt,Pt,Pt)), shift=[0,0,-lag])
        NTot, cc = jointentropy3D(MinputTot, nbins,lower_bound2, upper_bound2,
                                 ZFlagSink, ZFlagSource,ZFlagSource)
        HQt, HXt, HXt_1, IQtXtXt_1[lag] = ComputeTotalEntropy(NTot,nbins)
            # based on Qt-1
        MTotQt_1 = LagData_new( M_unlagged=np.column_stack((Yt,Pt,Yt)), shift=[0,-lag,-1])
        NQt_1, CQt_1 = jointentropy3D(MTotQt_1, nbins,lower_bound1, upper_bound1,
                                     ZFlagSink, ZFlagSource,ZFlagSink)
        HQt_, HXt_, HXt_1_, IQtXtQt_1[lag] = ComputeTotalEntropy(NQt_1,nbins)
        
        # MI and TE conditioned on Yesterday's Precipitation than streamflow
        
        MinputP = LagData_new( M_unlagged=np.column_stack((Pt,Yt,Pt)), shift=[-lag,shift[1],shift[2]])
        Np, CorrCoefp = jointentropy3D(MinputP, nbins,lower_bound4, upper_bound4,
                                      ZFlagSource,ZFlagSink,ZFlagSource)
        Hxp, Hyp, MIp[lag], TEp[lag] = ComputeInfoMetricsBasedOn3DBincounts(Np,nbins) 

    return Hx, Hy, MI, TE, CorrCoef, IQtXtXt_1, IQtXtQt_1, MIp, TEp, MIcrit, TEcrit, CorrCrit, \
           criticalTotMI, MIcritP, TEcritP, criticalTotMI_Qt_1

