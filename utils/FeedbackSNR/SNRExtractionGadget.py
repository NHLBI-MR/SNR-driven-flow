import numpy as np 
import os.path as op
import json
import ismrmrd as mrd
from  onnxruntime import InferenceSession
from time import time 
import gadgetron
import nibabel as nib
from SegmentationFlowGadget import create_ismrmrd_image


def _parse_params(xml):
    return {p.get('name'): p.get('value') for p in xml.iter('property')}


def SNRExtractionGadget(connection):
    imgs=[]
    masks=[]
    n=0
    hdr=connection.header
    field_of_view = hdr.encoding[0].reconSpace.fieldOfView_mm
    for item in connection:
        if item.data.dtype==np.complex64:
            imgs.append(item)
        if item.data.dtype==np.uint16:
            masks.append(item)
        print('--------------ITEM----------------')
        print(item.data.dtype)
        print(len(imgs))
        print(len(masks))
        n+=1
        if len(masks)==1 and len(imgs)==1:
            img=imgs.pop()
            mask=masks.pop()
            print('Mask and Image')
            print(mask.data.shape)
            print(img.data.shape)
            
            abs_img=np.squeeze(np.abs(img.data))
            mask_np=np.squeeze(mask.data)
            snr=np.nanmean(abs_img[mask_np==1])
            print(snr)
            snr_image=create_ismrmrd_image(snr*np.ones(img.data.shape),img.headers,field_of_view,n+10)   

            #connection.send(snr_image)
            connection.send(img)

    print('yes')


