#include <gadgetron/Node.h>
#include <gadgetron/mri_core_utility.h>
#include <gadgetron/mri_core_acquisition_bucket.h>
#include <ismrmrd/xml.h>
#include <gadgetron/gadgetron_mricore_export.h>
#include <gadgetron/mri_core_def.h>
#include <gadgetron/mri_core_data.h>
#include <gadgetron/hoNDArray_math.h>
#include <gadgetron/hoNDArray_utils.h>
#include <gadgetron/hoNDArray.h>
#include <ismrmrd/ismrmrd.h>
#include <gadgetron/Types.h>
#include <gadgetron/GadgetMRIHeaders.h>
#include <gadgetron/GadgetronTimer.h>
#include <gadgetron/FeedbackData.h>
#include "../../utils/gpu/cuda_utils.h"
#include <util_functions.h>
#include <gadgetron/ImageArraySendMixin.h>
using namespace Gadgetron;
using namespace Gadgetron::Core;


class ImageSNRExtractionAndFeedback : public ChannelGadget<Core::variant<Core::Image<unsigned short>, IsmrmrdImageArray>>
{

public:
    ImageSNRExtractionAndFeedback(const Core::Context &context, const Core::GadgetProperties &props) : ChannelGadget<Core::variant<Core::Image<unsigned short>, IsmrmrdImageArray>>(context, props)
    {
    }
    void process(InputChannel<Core::variant<Core::Image<unsigned short>, IsmrmrdImageArray>> &in, OutputChannel &out) override
    {

        bool maskflag = false;
        bool SNRflag = false;
        bool condition_stop = false;
        std::vector<Image<unsigned short>> mask_array;
        std::vector<size_t> images_index;
        std::vector<IsmrmrdImageArray> image_array;
        for (auto message : in)
        {

            if (holds_alternative<Core::Image<unsigned short>>(message))
            {
                mask_array.push_back(Core::get<Image<unsigned short>>(message));
                maskflag = true;
            }


            if (holds_alternative<IsmrmrdImageArray>(message))
            {
                image_array.push_back(Core::get<IsmrmrdImageArray>(message));
                SNRflag = true;
            }
            if (SNRflag && maskflag)
            {
                auto &[img,headerT,metaT,waveform,acqT]=image_array.back();
                auto &[header,mask,meta] = mask_array.back();

                auto numpixels = std::accumulate(mask.begin(),mask.end(),0);
                
                SNRflag = false;
                maskflag = false;
                
                if (numpixels == 0){
                    numpixels=1;
                } 
                auto dptr = img.get_data_ptr();
                auto mptr = mask.get_data_ptr();
                float SNR = 0;

                for (auto ii = 0; ii < img.get_number_of_elements(); ii++)
                {
                    SNR += ((abs(*(mptr + ii)) == 1) ? (abs(*(dptr + ii)) * float(*(mptr + ii)/float(numpixels))) : 0);
                }

                GDEBUG_STREAM( "SNR " << SNR);
                out.push(Gadgetron::FeedbackData{true, 0, 0,SNR});
                
                auto newHeader=headerT(0,0,0);
                newHeader.set = 0;
                newHeader.user_float[1]=SNR;
                newHeader.user_float[3]=header.user_float[0];
                newHeader.data_type = ISMRMRD::ISMRMRD_FLOAT;
                newHeader.image_index = (uint16_t)(4);
                newHeader.image_series_index = (uint16_t)(4);
                newHeader.image_type=ISMRMRD::ISMRMRD_IMTYPE_MAGNITUDE;

                auto tmp=abs(img);

                auto newHeaderM=newHeader;
                newHeaderM.data_type = ISMRMRD::ISMRMRD_USHORT;
                newHeaderM.image_index = (uint16_t)(5);
                newHeaderM.image_series_index = (uint16_t)(5);
                newHeaderM.image_type=ISMRMRD::ISMRMRD_IMTYPE_MAGNITUDE;
                mask*=100;

                out.push(Core::Image<float>(newHeader, std::move(tmp), std::optional<ISMRMRD::MetaContainer>()));
                out.push(Core::Image<unsigned short>(newHeaderM, std::move(mask), std::optional<ISMRMRD::MetaContainer>()));
                
                mask_array.pop_back();
                image_array.pop_back();
                

            }
            
        }
        
    }

protected:

};

GADGETRON_GADGET_EXPORT(ImageSNRExtractionAndFeedback)