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

def SegmentationFlowGadget(connection):
    # What are the dimension order of the input data ?

    hdr=connection.header
    field_of_view = hdr.encoding[0].reconSpace.fieldOfView_mm
    #To verify 
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
    n=0
    image_0=[]
    vds='VDSunknown'
    slines='Sunknown'
    for img in connection:
        if(len(image_0)<1):
            image_0.append(img)
        
        t1=time() 
        image_nnUnetFormat=np.abs(np.nan_to_num(img.data)).transpose(1,0)[np.newaxis,np.newaxis,...]
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
            
        prediction.transpose(0,1,3,2) 
        if params["MaskThreshold"]:
            prediction=(np.abs(np.nan_to_num(img.data))>=params["Threshold"]).astype(prediction_frame.dtype)
        
        t2=time()
        eprint('nnUNet processec SNR mean map in : {} s'.format(t2-t1))
        
        if params["savenii"]:
            nii_volume = nib.Nifti1Image(image_nnUnetFormat[0,...].transpose(2,1,0),  np.diag((field_of_view.x/matrixSize.x,field_of_view.y/matrixSize.y,1,1)))
            output_path='/opt/data/gt_data/image_vds{}_{}_{}.nii.gz'.format(vds,slines,str(n).zfill(3))
            nib.save(nii_volume, output_path)
            nii_volume = nib.Nifti1Image(img.data,  np.diag((field_of_view.x/matrixSize.x,field_of_view.y/matrixSize.y,1,1)))
            output_path='/opt/data/gt_data/rawimage_vds{}_{}_{}.nii.gz'.format(vds,slines,str(n).zfill(3))
            nib.save(nii_volume, output_path)
            nii_volume = nib.Nifti1Image(np.abs(img.data),  np.diag((field_of_view.x/matrixSize.x,field_of_view.y/matrixSize.y,1,1)))
            output_path='/opt/data/gt_data/rawimage_mag_vds{}_{}_{}.nii.gz'.format(vds,slines,str(n).zfill(3))
            nib.save(nii_volume, output_path)
            nii_volume = nib.Nifti1Image(np.angle(img.data),  np.diag((field_of_view.x/matrixSize.x,field_of_view.y/matrixSize.y,1,1)))
            output_path='/opt/data/gt_data/rawimage_phase_vds{}_{}_{}.nii.gz'.format(vds,slines,str(n).zfill(3))
            nib.save(nii_volume, output_path)
            nii_mask = nib.Nifti1Image(prediction, np.diag((field_of_view.x/matrixSize.x,field_of_view.y/matrixSize.y,1,1)))
            output_path='/opt/data/gt_data/mask_vds{}_{}_{}.nii.gz'.format(vds,slines,str(n).zfill(3))
            nib.save(nii_mask, output_path)

        n+=1
        mask=create_ismrmrd_image(prediction,field_of_view,n)
        mask._head.user_float[0]=(t2-t1)*1e3 #in ms
        connection.send(img)
        connection.send(mask)

def create_ismrmrd_image(data, field_of_view, index):
        return mrd.image.Image.from_array(
            data,
            image_index=index,
            image_type=mrd.IMTYPE_MAGNITUDE,
            field_of_view=(field_of_view.x, field_of_view.y, field_of_view.z),
            transpose=True
        )

if __name__ == '__main__':
    gadgetron.external.listen(2000,SegmentationFlowGadget)
