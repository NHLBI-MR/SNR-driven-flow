
import gadgetron
import ismrmrd as mrd

import numpy as np
import cupy as cp

import cmath
import math
import time

import cupyx.scipy

from scipy.fft import fft,ifft,fftshift,ifftshift
from cupyx.scipy.fft import fft as cufft
from cupyx.scipy.fft import ifft as cuifft
from cupyx.scipy.fft import fftshift as cufftshift
from cupyx.scipy.fft import ifftshift as cuifftshift

import kaiser_window
from BinningGadget import corr2,cufilterData,get_idx_to_send2,create_ismrmrd_image,binning, eprint, butter_highpass_filter
from scipy.signal import find_peaks
from itertools import groupby
from operator import itemgetter
import scipy
import ismrmrd as mrd
import calendar
def _parse_params(xml):
    return {p.get('name'): p.get('value') for p in xml.iter('property')}


def correctTrajectoryFluctuations(data_array,na):
    [nav_samples_times_channels,samples]=data_array.shape
    
    una = cp.unique(na)
    idx = cp.argsort(-1*na) #max to min
    interleaves = len(una)
    
    factor = (int(math.ceil(samples / interleaves)) % int(interleaves)) - round(samples/ interleaves)
    if (factor < 0):
        factor = 0
    
    nav_samplingTime = 0
    numNavsPerStack = int(na.shape[0]/ interleaves)
    nav_samplingTime=-float(cp.mean(cp.diff(na[idx[0:-1:numNavsPerStack]])))

    sorted_signal = data_array[:,idx]
    sorted_signal = cp.reshape(sorted_signal,(nav_samples_times_channels,int(samples/interleaves),interleaves+factor))
    

    filtered_signal = butter_highpass_filter(sorted_signal.squeeze(), 0.1, abs(1/(nav_samplingTime)), order=7) #5

    filtered_signalX = cp.reshape(filtered_signal,(data_array.shape[0],data_array.shape[1]))
    filtered_signal = cp.zeros((filtered_signalX.shape))
    filtered_signal[:,idx] =filtered_signalX
    
    return filtered_signal

def estimateCardiacGatingSignal_set(nav_data,nav_tstamp,navangles,kaiserBP=[0.8,0.85,2.0,2.1],angularCor=True):
    '''
    params: nav_data [channels,samples,nav_sample]

    '''

    [number_channels,samples,nav_sample]=nav_data.shape

    data_array = cp.abs(cufft(nav_data,axis=2))
    data_array= cp.transpose(data_array,(0,2,1)) 
    data_array= cp.reshape(data_array,(number_channels*nav_sample,samples))
    if angularCor:
        data_array = correctTrajectoryFluctuations(data_array,navangles)

    # Bandpass filterations 
    diff_nav_tstamp=cp.diff(nav_tstamp)
    mean_samplingTime=cp.mean(diff_nav_tstamp)
    max_samplingTime=cp.max(diff_nav_tstamp)
    
    samplingTime=2.5*mean_samplingTime
    bpfilter = kaiser_window.kaiser_window_generate(kaiserBP,[0.01,0.01,0.01],'bandpass',1/(samplingTime*1e-3),data_array.shape[1]) #DP

    filtered_signal = cufilterData(data_array,bpfilter)

    filtered_signal = cp.asarray(np.reshape(filtered_signal,(number_channels,nav_sample,samples)))
    compressed_signal = cp.zeros((filtered_signal.shape[0],filtered_signal.shape[2]),dtype=complex)


    temp = (filtered_signal.transpose((0,2,1))).astype(cp.csingle)
    [u,s,v] = cp.linalg.svd(temp,full_matrices=False)
    compressed_signal = u[:,:,0]


    C=cp.zeros((compressed_signal.shape[0],compressed_signal.shape[0]),dtype=complex)
    G=cp.zeros((compressed_signal.shape[0],compressed_signal.shape[0]),dtype=complex)
    

    threshold = 0.98
    for ii in range(compressed_signal.shape[0]):
        for jj in range(compressed_signal.shape[0]):
            C[ii,jj]= corr2(cp.real(compressed_signal[ii,:]).squeeze(),cp.real(compressed_signal[jj,:]).squeeze())
            G[ii,jj]= (cp.abs(C[ii,jj])>threshold)
    

    [ug,sg,vg]=cp.linalg.svd(G,full_matrices=False)
    ind_dom_motion = cp.argwhere(cp.abs(cp.sum(ug[:,cp.argwhere(cp.max(cp.diag(sg))==cp.diag(sg))],axis=1))>0.1)
    ind_dom_motion = cp.argwhere(cp.abs((ug[:,0]))>0.1)

    

    dominantM = C[ind_dom_motion,ind_dom_motion]

    negInd = ind_dom_motion[cp.argwhere(dominantM[:,0]<0)]
    negInd = cp.argwhere(dominantM[:,0]<0)
    yfilt1 = compressed_signal[ind_dom_motion,:]


   

    for ii in range(negInd.shape[0]):
        yfilt1[negInd(ii),:] = yfilt1[negInd(ii),:]*-1
    
    yfilt1 = cp.asnumpy(np.real(np.mean(yfilt1,axis=0)))
    
    return yfilt1, samplingTime/2.5

def estimated_initial_ecgfreq(selectedSig,samplingTime,freqwindow=0.025,smooth=False):
    """
    Detecting the initial ecg frequency based on the max of the power of the spectrum.

    Parameters
    ----------
    
    selectedSig : np.ndarray, 
        NAV/DC signal 

    samplingTime : float, 
        sampling time 

    freqwindow : float, 
        float Median filter window (default: 0.025 Hz)

    smooth : boolean, 
        Flag smooth the spectrum using a median filter with windows define by freqwindow (default: False)

    Returns
    ------- 

    ecg_freq_initial : np.float64,
        Initial ecg frequency estimated 
    
    """
    fs=1/float(2.5*samplingTime/1000)
    fft_waveform=np.abs(np.fft.fft(np.squeeze(selectedSig)))
    n_samples=cp.shape(selectedSig)[1]
    fr=np.arange(n_samples)*fs/int(n_samples)
    if smooth:
        window_s=int(np.ceil(freqwindow/(fs/n_samples)))
        fft_waveform_smooth=scipy.ndimage.median_filter(fft_waveform,window_s)
    else:
        fft_waveform_smooth=fft_waveform
    ecg_freq_initial=fr[np.argmax(fft_waveform_smooth[:int(n_samples/2)])]

    return ecg_freq_initial

def cardiacbinning(selectedSig,samplingTime,numBins,ecg_freq,evenbins=False,phantomflag=False,arrythmia_detection=False):
    """
    Calculating cardiac bins based on the navigator/DC signal. 

    Parameters
    ----------
    
    selectedSig : np.ndarray, 
        NAV/DC signal 

    samplingTime : float, 
        sampling time 

    numBins : int, 
        number of bins 

    ecg_freq : float, 
        initial ecg frequency estimated 

    evenbins : boolean, 
        Flag to obtain uniform bins (default: False)

    phantomflag : boolean, 
        Flag for phantom data (simplified binning: mod(sample index,numBins)) (default: False)

    arrythmia_detection : boolean, 
        Remove RR vectors which are too long (>1.5(ecg_freq_initial) (default: False)   

    Returns
    -------

    bins : List[List], List of binned indexes : 
        - Length of the list is equal to the number of bins    

    final_ecg_freq : np.float64,
        ecg frequency estimated 
    """
    # Selected the maximum in frequency for having a initial guess of cardiac frequency   
    fs=1/float(2.5*samplingTime/1000)
    n_samples=cp.shape(selectedSig)[1]

    #Minimum distance betwwen peak 
    min_distance=round(0.75* (fs/ecg_freq))
    peaks, _ = find_peaks(selectedSig.flatten(), distance=min_distance)
    #Binnings
    bins=[[] for n in range(numBins)]
    final_ecg_freq=ecg_freq
    if phantomflag:
        RR_vec=np.arange(n_samples)
        bin_data_label=RR_vec % (numBins)
        RR_seq=list(map(list, zip(RR_vec.tolist(), bin_data_label.tolist())))
        RR_seq.sort(key=itemgetter(1)) # Not necessary now 
        bins=[[x for x,y in data] for (key, data) in groupby(RR_seq, itemgetter(1))]

    else:
        if len(peaks)>=2:
            RR_ecg=list()
            peak_list=list(zip(peaks[:-1],peaks[1:],np.diff(peaks).tolist()))
            if arrythmia_detection:
                ln_p=len(peak_list)
                peak_list=[(peak_start,peak_stop,RR_vec) for peak_start,peak_stop,RR_vec in peak_list if RR_vec <1.5*(fs/ecg_freq)]
                eprint('Arrythmia reject (%)...')
                eprint(100*(ln_p-len(peak_list))/ln_p)
            for peak_start,peak_stop,RR_vec in peak_list:
                bin_data_label=np.arange(RR_vec)//(RR_vec/numBins)
                RR_seq=list(map(list, zip(np.arange(peak_start,peak_stop).tolist(), bin_data_label.tolist())))
                RR_seq.sort(key=itemgetter(1)) # Not necessary now 
                RR_peak_bins=[[x for x,y in data] for (key, data) in groupby(RR_seq, itemgetter(1))]
                bins=list(map(lambda x, y:x + y, bins, RR_peak_bins))
                RR_ecg.append(fs/RR_vec)
            final_ecg_freq=np.mean(np.array(RR_ecg))
    eprint('ECG FREQ INIT : {} ECG FREQ FINAL : {}'.format(ecg_freq,final_ecg_freq))
    if evenbins:
        len_bins=[len(bins[n]) for n in range(numBins)]
        min_len=min(len_bins)
        bins=[bins[n][:min_len] for n in range(numBins)]

    return bins,final_ecg_freq


 
def CardiacBinningGadget(connection):
    """
    Parameters
    ---------- 

    params : dict 
        Dictionnary that contains the following information :
                - numBins : int, 
                    number of bins (default: 25)
                - phantom : boolean, 
                    Flag for phantom data (simplified binning: mod(sample index,numBins)) (default: False)
                - respiRate : int,  
                    % of data kept after respiration gating (default: 100)
                - evenbins : boolean, 
                    Flag to obtain uniform bins (default: False)
                - version : string, 
                    Flag to decide on which signal should be calculated the bins ('v1': raw signal, 'v2': filtered signal around +-0.25Hz ecg initial frequency , default: 'v2')
                - arrythmia_detection : boolean, 
                    Remove RR vectors which are too long (>1.5(ecg_freq_initial) (default: True)
                - angularCorr : boolean,
                    Angular trajectories correction (default=False)  

    Returns
    -------

    idx_to_send : List[np.ndarray], List of binned indexes : 
        - Length of the list is equal to the number of bins * sets    
        - np.ndarray : 1rst element corresponds to the number of indexes of the bin
    
    """
    eprint("-------------Cardiac Binning-------------")

    connection.filter(mrd.Acquisition)
    
    params_init = _parse_params(connection.config)
    params={'numBins':25,
            'phantom':False,
            'respiRate':100,
            'evenbins': False,
            'version':'v2',
            'arrythmia_detection':True,
            'angularCorr':False
            }

    boolean_keys=['phantom','evenbins','arrythmia_detection','angularCorr']
    str_keys=['version']
    int_keys=['numBins','respiRate']
    
    for bkey in boolean_keys:
        if bkey in params_init:
            params[bkey]=params_init[bkey]=='True'
    for skey in str_keys:
         if skey in params_init:
            params[skey]=params_init[skey]
    for ikey in int_keys:
        if ikey in params_init:
            params[ikey]=int(params_init[ikey])
    
    count = 0
    nav_angles = []
    acq_tstamp = []
    nav_data    = []
    nav_tstamp  = []
    nav_indices = []
    data_indices = []
    kencode_step = []
    mrd_header = connection.header

    field_of_view = mrd_header.encoding[0].reconSpace.fieldOfView_mm

    encoding_limits = mrd_header.encoding[0].encodingLimits
    number_of_slices=encoding_limits.slice.maximum+1
    number_of_sets=encoding_limits.set.maximum+1
    eprint(encoding_limits.repetition.maximum+1)
    if number_of_slices == 1:
        eprint("2D acquistion" )
        reco_2D=True
        interleaves=encoding_limits.kspace_encoding_step_1.maximum+1
    else:
        reco_2D=False
    
    acquisition_0 = []

    time0=0
    for acq in connection:
        
        if reco_2D:
            nav_data.append(cp.array(acq.data)[:,0][:,cp.newaxis])
            nav_tstamp.append(acq.acquisition_time_stamp)
            acq_tstamp.append(acq.acquisition_time_stamp)
            nav_indices.append(count)
            kencode_step.append(acq.idx.kspace_encode_step_1)
            nav_angles.append(2*np.pi*(acq.idx.kspace_encode_step_1/interleaves))
            if(len(acquisition_0)<1):
                    acquisition_0.append(acq)
                    time0= acq.acquisition_time_stamp     
            connection.send(acq)
        else:

            if acq.isFlagSet(mrd.ACQ_IS_RTFEEDBACK_DATA or mrd.ACQ_IS_HPFEEDBACK_DATA or mrd.ACQ_IS_NAVIGATION_DATA):
                nav_data.append(cp.array(acq.data))
                nav_tstamp.append(acq.acquisition_time_stamp)
                nav_indices.append(count)
            else:
                nav_angles.append(180*cmath.phase(complex(acq.traj[25,0],acq.traj[25,1]))/math.pi)
                acq_tstamp.append(acq.acquisition_time_stamp)
                data_indices.append(count)
                kencode_step.append(acq.idx.kspace_encode_step_1)
                if(len(acquisition_0)<1):
                    acquisition_0.append(acq)
                connection.send(acq)
        count= count+1
    timeAcq=(nav_tstamp[-1]-time0)*2.5

    eprint("Acquisition: Total time {} ms , number: {} ".format(timeAcq,count))

    samples=len(nav_indices)
    [number_channels,nav_sample]=nav_data[0].shape
    nav_data=cp.concatenate(nav_data,axis=1)

    if len(nav_data.shape)!=3:
        nav_data = cp.reshape(nav_data,(number_channels,samples,nav_sample))
    
    nav_tstamp=cp.array(nav_tstamp)
    acq_tstamp=cp.array(acq_tstamp)
    nav_indices=cp.array(nav_indices)
    kencode_step=cp.array(kencode_step)
    nav_angles=cp.array(nav_angles)

    if number_of_sets >1 :
        fset=lambda x: cp.mean(cp.reshape(x,[int(x.shape[0]/number_of_sets),number_of_sets]),1)
        fset_nav=lambda x: cp.mean(cp.reshape(x,[number_channels,int(samples/number_of_sets),number_of_sets,nav_sample]),2)
        nav_data=fset_nav(nav_data)
        nav_angles=fset(nav_angles)
        acq_tstamp_set=fset(acq_tstamp)
        kencode_step=fset(kencode_step).astype(cp.int32)
        nav_indices=((fset(nav_indices)-0.5)/2).astype(cp.int32)
        nav_tstamp_set=fset(nav_tstamp)
    else:
        acq_tstamp_set=acq_tstamp
        nav_tstamp_set=nav_tstamp
    length_set=np.shape(nav_tstamp_set)[0]
    kstep_nav   = cp.concatenate((cp.asarray([kencode_step[0]]),kencode_step[nav_indices[1:]]),axis=0)
    angles_sorted = cp.zeros(cp.unique(nav_angles).shape)
    angles_sorted[kencode_step] = nav_angles
    nav_angles    = angles_sorted[kstep_nav]
    
    nav_data_copy   = nav_data.copy()
    nav_tstamp_copy = nav_tstamp_set.copy()
    nav_angles_copy = nav_angles.copy()

    acceptedTimes=[[] for n in range(params['numBins']*number_of_sets)]

    
    cardiac_waveform, samplingTime = estimateCardiacGatingSignal_set(nav_data_copy,nav_tstamp_copy,nav_angles_copy,kaiserBP=[0.6,0.65,2.0,2.1],angularCor=params['angularCorr']) 
    cp.cuda.runtime.deviceSynchronize()

    ecg_freq=estimated_initial_ecgfreq(cardiac_waveform,samplingTime,smooth=True,freqwindow=0.05)

    cardiac_waveform_smooth, samplingTime = estimateCardiacGatingSignal_set(nav_data_copy,nav_tstamp_copy,nav_angles_copy,kaiserBP=[ecg_freq-0.3,ecg_freq-0.25,ecg_freq+0.25,ecg_freq+0.3],angularCor=params['angularCorr']) 
    
    
    if params['version']=='v2':
        eprint('Version 2')
        bins_index,ecg_freq_final=cardiacbinning(cardiac_waveform_smooth,samplingTime,params['numBins'],ecg_freq,evenbins=params['evenbins'],phantomflag=params['phantom'],arrythmia_detection=params['arrythmia_detection'])
    else:
        eprint('Version 1')
        bins_index,ecg_freq_final=cardiacbinning(cardiac_waveform,samplingTime,params['numBins'],ecg_freq,evenbins=params['evenbins'],phantomflag=params['phantom'],arrythmia_detection=params['arrythmia_detection'])

    Index=np.arange(nav_tstamp_copy.shape[0])
    
    if params['phantom'] or params['respiRate']==100:
        aRespiTimesIndex=Index
    else:
        respiratory_waveform, samplingTime = estimateCardiacGatingSignal_set(nav_data_copy,nav_tstamp_copy,nav_angles_copy,kaiserBP=[0.08,0.1,0.45,0.50]) 
        aRespiTimes = binning(respiratory_waveform,nav_tstamp_copy.tolist(),params['respiRate'],False, True, False,1)  
        aRespiTimesIndex=Index[np.in1d(cp.asnumpy(nav_tstamp_copy),aRespiTimes[0])]
    
    for nbin in range(len(bins_index)):
        bin=np.array(bins_index[nbin])
        for set in range(number_of_sets):
            set_nav_tstamp=bin[np.in1d(bin,aRespiTimesIndex)]
            acceptedTimes[set*params['numBins']+nbin]=nav_tstamp[set_nav_tstamp*2+set]

    
    idx_to_send = []
    maxSize = 0 
    for ii in range(0,len(acceptedTimes)):
        temp = get_idx_to_send2(acq_tstamp,acceptedTimes[ii], samplingTime/number_of_sets)
    
        idx_to_send.append(np.concatenate(([np.array(temp.shape[0])],cp.asnumpy(temp))))
        if(idx_to_send[ii].shape[0] > maxSize):
            maxSize = idx_to_send[ii].shape[0]
    
            
    imageSize = pow(2,math.ceil(math.log2(math.sqrt(maxSize))))
    
    for ii in range(0,len(acceptedTimes)):
        data = np.zeros((imageSize*imageSize))
        data[range(0,len(idx_to_send[ii]))] = idx_to_send[ii].squeeze()
        data = np.reshape(data,(imageSize,imageSize))
        image = create_ismrmrd_image(data, acquisition_0[0], field_of_view, ii)
        connection.send(image)
    
    #Add cardiac time (1/ecg_freq/num_bins*1000) micros
    tframes=(1/ecg_freq_final/params['numBins'])*1000*1000

    data = np.zeros((imageSize*imageSize))
    data[0:2]=[1,tframes]
    data = np.reshape(data,(imageSize,imageSize))
    image = create_ismrmrd_image(data, acquisition_0[0], field_of_view,len(acceptedTimes))
    connection.send(image)

if __name__ == '__main__':
    gadgetron.external.listen(2000,CardiacBinningGadget)
    
    