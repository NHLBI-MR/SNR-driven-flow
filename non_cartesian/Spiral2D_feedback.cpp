/*
 * SpiralMocoRecon.cpp
 *
 *  Created on: September 17th, 2021
 *      Author: Ahsan Javed
 */

#include <gadgetron/Node.h>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/range/combine.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/lambda/bind.hpp>

#include <boost/hana/functional/iterate.hpp>
#include <numeric>
#include <random>
#include <sstream>
#include <fstream>
#include <iostream>

#include <gadgetron/mri_core_grappa.h>
#include <gadgetron/vector_td_utilities.h>
#include <gadgetron/NonCartesianTools.h>
#include <gadgetron/NFFTOperator.h>
#include <gadgetron/cgSolver.h>
#include <gadgetron/hoNDArray_math.h>
#include <gadgetron/hoNDArray_utils.h>
#include <gadgetron/hoNDArray_elemwise.h>
#include <gadgetron/hoNFFT.h>
#include <gadgetron/hoNDFFT.h>
#include <gadgetron/GadgetronTimer.h>
#include <gadgetron/mri_core_coil_map_estimation.h>
#include <gadgetron/generic_recon_gadgets/GenericReconBase.h>
#include <gadgetron/ImageArraySendMixin.h>
#include <gadgetron/mri_core_kspace_filter.h>
#include <gadgetron/ImageIOBase.h>
#include <gadgetron/hoNDArray_fileio.h>
#include <gadgetron/cuNDArray_math.h>
#include <gadgetron/cuNDArray.h>

#include <iterator>
#include <gadgetron/hoNDArray_utils.h>
#include <gadgetron/hoCgSolver.h>
#include <gadgetron/hoNDImage_util.h>
#include <gadgetron/FeedbackData.h>

#include "../spiral/SpiralBuffer.h"
#include "../utils/gpu/cuda_utils.h"
#include <util_functions.h>
#include "noncartesian_reconstruction.h"
#include "noncartesian_reconstruction_pseudo_replica.h"

#include "reconParams.h"

#include <omp.h>
#include <algorithm>
#include <cmath>

using namespace Gadgetron;
using namespace Gadgetron::Core;
using namespace Gadgetron::Indexing;
using namespace nhlbi_toolbox::reconstruction;

class Spiral2D_feedback : public ChannelGadget<Core::Acquisition>,
                          public ImageArraySendMixin<Spiral2D_feedback>
{
public:
    std::vector<size_t> image_dims_;
    uint64d3 image_dims_os_;
    float oversampling_factor_;
    float kernel_width_;
    bool verbose;

    boost::shared_ptr<cuNDArray<float_complext>> csm_;

    Spiral2D_feedback(const Core::Context &context, const Core::GadgetProperties &props) : ChannelGadget<Core::Acquisition>(context, props)
    {
        kernel_width_ = 3;
        oversampling_factor_ = oversampling_factor;
        verbose = false;
    }

    void process(InputChannel<Core::Acquisition> &in,
                 OutputChannel &out) override
    {

        int selectedDevice = nhlbi_toolbox::utils::selectCudaDevice();
        unsigned int warp_size = cudaDeviceManager::Instance()->warp_size();
        size_t RO, E1, E2, CHA, N, S, SLC;
        nhlbi_toolbox::utils::enable_peeraccess();

        ISMRMRD::AcquisitionHeader acqhdr;
        boost::shared_ptr<cuNDArray<float_complext>> csm;

        Gadgetron::reconParams recon_params;

        auto idx = 0;

        size_t acq_count = 0;

        auto maxAcq = ((header.encoding.at(0).encodingLimits.kspace_encoding_step_1.get().maximum + 1) *
                       (header.encoding[0].encodingLimits.kspace_encoding_step_2.get().maximum + 1) *
                       (header.encoding[0].encodingLimits.average.get().maximum + 1) *
                       (header.encoding[0].encodingLimits.repetition.get().maximum + 1) *
                       (header.encoding[0].encodingLimits.phase.get().maximum + 1) *
                       (header.encoding[0].encodingLimits.set.get().maximum + 1)); // use -1 for data acquired b/w 12/23 - 01/21
        GDEBUG_STREAM("maxAcq:  " << maxAcq);
        GDEBUG_STREAM("(header.encoding[0].encodingLimits.set.get().maximum + 1): " << (header.encoding[0].encodingLimits.set.get().maximum + 1));

        std::vector<Core::Acquisition> allAcq(maxAcq);
        std::vector<ISMRMRD::AcquisitionHeader> headers(maxAcq);
        uint32_t startTime=0;
        uint32_t startTime_real=0;
        // Collect all the data -- BEGIN()

        for (auto message : in)
        {

            auto &[head, data, traj] = message;
            allAcq[idx] = std::move((message));
            if(idx==0 || idx == crop_begin){
                startTime = head.acquisition_time_stamp; // Time used for feedback, reset at each time feedback is given
                startTime_real = head.acquisition_time_stamp; // Time used to calculate total acquisition time 

            }
            idx++;
            auto time = float(head.acquisition_time_stamp-startTime)*2.5; // ms
            auto time_acq= float(head.acquisition_time_stamp-startTime_real)*2.5; // ms

            if ((((int(time/reconStride)>0 && head.isFlagSet(ISMRMRD::ISMRMRD_ACQ_LAST_IN_SLICE)) || head.isFlagSet(ISMRMRD::ISMRMRD_ACQ_LAST_IN_MEASUREMENT)) && idx>crop_begin))
            {
                
                GDEBUG_STREAM("Time since last recon (ms) : " << time);
                auto &[headT, dataT, trajT] = allAcq[0];
                acqhdr = headT;
                
                startTime = head.acquisition_time_stamp;
                auto acq_toRecon = allAcq;
                acq_toRecon.resize(idx);
                if (crop_end > 0)
                    acq_toRecon.erase(acq_toRecon.begin() + crop_end, acq_toRecon.end());
                acq_toRecon.erase(acq_toRecon.begin(), acq_toRecon.begin() + crop_begin);

                GadgetronTimer timer("2D Recon:");

                cudaSetDevice(selectedDevice);
                auto &[headAcq, dataAcq, trajAcq] = acq_toRecon[0];
                RO = dataAcq.get_size(0);
                CHA = dataAcq.get_size(1);
                E2 = this->header.encoding.front().encodedSpace.matrixSize.z;
                N = dataAcq.get_size(3);
                S = 1;
                SLC = 1;

                recon_params.numberChannels = CHA;
                recon_params.RO = RO;
                recon_params.ematrixSize = this->header.encoding.front().encodedSpace.matrixSize;
                recon_params.rmatrixSize = this->header.encoding.front().reconSpace.matrixSize;
                recon_params.fov = this->header.encoding.front().encodedSpace.fieldOfView_mm;
                recon_params.oversampling_factor_ = oversampling_factor_;
                recon_params.kernel_width_ = kernel_width_;
                recon_params.selectedDevice = selectedDevice;
                recon_params.norm = 2;
                recon_params.useIterativeDCWEstimated = true;
                recon_params.oversampling_factor_dcf_ = 2.1; 
                recon_params.kernel_width_dcf_ = 5.5; 
                recon_params.iterations_dcf = 20;
                this->initialize_encoding_space_limits(this->header);

                noncartesian_reconstruction<2> reconstruction(recon_params);
                noncartesian_reconstruction_pseudo_replica<2> reconstruction_pr(recon_params);


                auto [cuData, traj_csm, dcf_in] = reconstruction.organize_data(&acq_toRecon);
                auto dcf = reconstruction.estimate_dcf(&traj_csm,&dcf_in);
                auto cuIimages = reconstruction_pr.reconstruct(&cuData, &traj_csm, &dcf, int(numReplicas));

                auto SNR_image = reconstruction_pr.calculate_SNR(&cuIimages);
               
                auto ho_images_all = hoNDArray<std::complex<float>>(std::move(*boost::reinterpret_pointer_cast<hoNDArray<std::complex<float>>>(SNR_image.to_host())));
                
                // Sending SNR images
                using namespace Gadgetron::Indexing;
                IsmrmrdImageArray imarray_sense;
                imarray_sense.data_ = ho_images_all;
                auto timePseudo=(float)timer.stop()/1000.0;
                GDEBUG_STREAM("Time (ms) : " << timePseudo);
                nhlbi_toolbox::utils::attachHeadertoImageArray(imarray_sense, acqhdr, this->header);
                prepare_image_array(imarray_sense, (size_t)0, ((int)series_counter), GADGETRON_IMAGE_SNR_MAP);
                imarray_sense.headers_(0, 0, 0).user_int[0]=head.idx.repetition;
                imarray_sense.headers_(0, 0, 0).user_float[0]=time;
                imarray_sense.headers_(0, 0, 0).user_float[2]=timePseudo;
                imarray_sense.headers_(0, 0, 0).data_type = ISMRMRD::ISMRMRD_CXFLOAT;
                imarray_sense.headers_(0, 0, 0).image_type = ISMRMRD::ISMRMRD_IMTYPE_COMPLEX;
                out.push(imarray_sense);
                series_counter++;
            }
        }
    }

protected:
    NODE_PROPERTY(oversampling_factor, float, "oversampling_factor", 2.1);
    NODE_PROPERTY(crop_begin, size_t, "crop_begin", 0);
    NODE_PROPERTY(crop_end, size_t, "crop_end", 0);
    NODE_PROPERTY(reconStride, size_t, "reconStride in ms", 20000);
    NODE_PROPERTY(numReplicas, size_t, "numReplicas", 100);
    int series_counter = 0;
};

GADGETRON_GADGET_EXPORT(Spiral2D_feedback)
