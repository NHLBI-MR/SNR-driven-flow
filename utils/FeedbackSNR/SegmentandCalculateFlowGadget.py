import numpy as np 
import os.path as op
import json
import ismrmrd as mrd
from  onnxruntime import InferenceSession
from gadgetron_cmr_segmentation_util import prepare_case_onnx_numpy,predict_case_onnx,export_prediction_from_softmax_onnx
from time import time 
import gadgetron
import nibabel as nib
from scipy.ndimage import binary_erosion
import glob
import sys

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def load_json(file: str):
    with open(file, 'r') as f:
        a = json.load(f)
    return a

def _parse_params(xml):
    return {p.get('name'): p.get('value') for p in xml.iter('property')}

def SegmentAndCalculateFlowGadget(connection):

    hdr=connection.header
    field_of_view = hdr.encoding[0].reconSpace.fieldOfView_mm
    params = _parse_params(connection.config)
    matrixSize = hdr.encoding[0].reconSpace.matrixSize
    pxs_spacing=(10,field_of_view.x/matrixSize.x,field_of_view.y/matrixSize.y)

    params_init = _parse_params(connection.config)
    params={'path_onnx':'',
        'path_info_preprocess': 'info_preprocess.json',
        'erosion': 0,
        'savenii': False,
        'MaskThreshold':False,
        'Threshold': 10,
        }

    boolean_keys=['savenii','MaskThreshold']
    str_keys=['path_onnx','path_info_preprocess']
    int_keys=['erosion','Threshold']
    
    for bkey in boolean_keys:
        if bkey in params_init:
            params[bkey]=params_init[bkey]=='True'
    for skey in str_keys:
         if skey in params_init:
            params[skey]=params_init[skey]
    for ikey in int_keys:
        if ikey in params_init:
            params[ikey]=int(params_init[ikey])

    info_preprocess=load_json(op.abspath(params["path_info_preprocess"]))
    networks_path=glob.glob(op.abspath(params["path_onnx"]))
    eprint(networks_path)
    networks=[InferenceSession(path_network_onnx) for path_network_onnx in networks_path]

    n=3
    image_phase=[]
    image_mag=[]
    vds='VDSunknown'
    slines='Sunknown'
    for img in connection:
        if img.image_type ==mrd.IMTYPE_PHASE:
            if(len(image_phase)<1):
                image_phase.append(img)

        if img.image_type ==mrd.IMTYPE_MAGNITUDE:
            if(len(image_mag)<1):
                image_mag.append(img)      
        
        if len(image_mag)==1 and len(image_phase):
            img_mag=image_mag.pop()
            img_phase=image_phase.pop()
            t1=time()
            image_nnUnetFormat=np.abs(np.nan_to_num(img_mag.data)).transpose(0,1,3,2) #(phases,z,y,x)
            list_data_test,data_properties=prepare_case_onnx_numpy(image_nnUnetFormat,pxs_spacing,info_preprocess) # ! Modify image_nnUnetFormat data
            prediction=np.zeros(np.shape(image_nnUnetFormat),dtype=np.uint16)
            for frame in range(len(list_data_test)):
                data_test=list_data_test[frame][np.newaxis,...]   
                data_predicted=predict_case_onnx(data_test,info_preprocess,networks)
                prediction_frame=export_prediction_from_softmax_onnx(data_predicted,info_preprocess,data_properties)
                prediction_frame=prediction_frame.astype(np.uint16)
                if params["erosion"] >0 :
                        prediction_frame=binary_erosion(prediction_frame, structure=np.ones((1,params["erosion"],params["erosion"]))).astype(prediction_frame.dtype)
                prediction[frame,...]=prediction_frame.transpose(0,2,1)
            t2=time()
            eprint('nnUNet processes in : {} s'.format(t2-t1))
            prediction.transpose(0,1,3,2) 
            if params["MaskThreshold"]:
                prediction=(np.abs(np.nan_to_num(np.squeeze(img_mag.data)))>=params["Threshold"]).astype(prediction_frame.dtype)

                    
            
            if params["savenii"]:
                nii_volume = nib.Nifti1Image(image_nnUnetFormat.transpose(3,2,1,0),  np.diag((field_of_view.x/matrixSize.x,field_of_view.y/matrixSize.y,1,1)))
                output_path='/opt/data/gt_data/image_vds{}_{}_{}.nii.gz'.format(vds,slines,str(n).zfill(3))
                nib.save(nii_volume, output_path)
                nii_mask = nib.Nifti1Image(prediction.transpose(3,2,1,0), np.diag((field_of_view.x/matrixSize.x,field_of_view.y/matrixSize.y,1,1)))
                output_path='/opt/data/gt_data/mask_vds{}_{}_{}.nii.gz'.format(vds,slines,str(n).zfill(3))
                nib.save(nii_mask, output_path)

            mask=create_ismrmrd_image_fast(prediction,field_of_view,n)
            connection.send(img_phase)
            connection.send(img_mag)
            connection.send(mask)

def create_ismrmrd_image_fast(data, field_of_view, index):
        return mrd.image.Image.from_array(
            data,
            #acquisition=reference, #not working headers(inside a numpy array)
            image_series_index=index,
            image_type=mrd.IMTYPE_MAGNITUDE,
            field_of_view=(field_of_view.x, field_of_view.y, field_of_view.z),
            transpose=False
        )

if __name__ == '__main__':
    gadgetron.external.listen(2000,SegmentAndCalculateFlowGadget)
