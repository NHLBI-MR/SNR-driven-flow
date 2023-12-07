import numpy as np 
import gadgetron
from compute_lung_seg_sandbox import lung_segmentation,create_ismrmrd_image
import ismrmrd as mrd
import copy
from time import time 
def _parse_params(xml):
    return {p.get('name'): p.get('value') for p in xml.iter('property')}

def SegmentationLung(connection):
    hdr=connection.header
    field_of_view = hdr.encoding[0].reconSpace.fieldOfView_mm
    matrixSize = hdr.encoding[0].reconSpace.matrixSize
    resolution = np.array(field_of_view.x/matrixSize.x)
    params_init = _parse_params(connection.config)
    print(matrixSize)
    params={
    'MaskThreshold':False,
    'Threshold': 5,
    }
    boolean_keys=['MaskThreshold']
    int_keys=['Threshold']
    
    for bkey in boolean_keys:
        if bkey in params_init:
            params[bkey]=params_init[bkey]=='True'
    for ikey in int_keys:
        if ikey in params_init:
            params[ikey]=int(params_init[ikey])


    for img in connection:   
        t1=time()    
        im=np.abs(np.nan_to_num(copy.deepcopy(img.data))).squeeze()
        print(np.shape(im))
        print(im.ndim)
        if im.ndim !=3:
            connection.send(img)
        else:
            if matrixSize.x != np.shape(im)[0]:
                lungmask = lung_segmentation(np.transpose(im,[1, 2, 0]), resolution, display=False).astype(np.uint16)
            else:
                 lungmask = lung_segmentation(im, resolution, display=False).astype(np.uint16)
            if params["MaskThreshold"]:
                    lungmask=(im>=params["Threshold"]).astype(lungmask.dtype)

            #lungmask = np.rot90(lungmask,-1, (0,1))
            lungmask = np.transpose(lungmask[np.newaxis, :, :, :],[0, 3, 2, 1])
            lungmask_img = create_ismrmrd_image_PD(lungmask, field_of_view, 2)

            t2=time()
            lungmask_img._head.user_float[0]=(t2-t1)*1e3 #in ms
            connection.send(img)
            connection.send(lungmask_img)

def create_ismrmrd_image_PD(data, field_of_view, index):
    return mrd.image.Image.from_array(
        data, 
        image_index=index, 
        image_type=mrd.IMTYPE_MAGNITUDE, 
        field_of_view=(field_of_view.x, field_of_view.y, field_of_view.z), 
        transpose=False, 
        repetition=False, 
        image_series_index=index)

if __name__ == '__main__':
    gadgetron.external.listen(2000,SegmentationLung)