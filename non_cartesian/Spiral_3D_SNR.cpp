/** \file       Spiral_3D_SNR.cpp
    \brief      To fill up
    \author     Ahsan Javed, Pierre Daude
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

class Spiral_3D_SNR : public ChannelGadget<Core::variant<Core::Acquisition, std::vector<std::vector<size_t>>, std::vector<size_t>>>,
                        public ImageArraySendMixin<Spiral_3D_SNR>
{
public:
    std::vector<size_t> image_dims_;
    uint64d3 image_dims_os_;
    float oversampling_factor_;
    float kernel_width_;
    bool verbose;


    Spiral_3D_SNR(const Core::Context &context, const Core::GadgetProperties &props) : ChannelGadget<Core::variant<Core::Acquisition, std::vector<std::vector<size_t>>, std::vector<size_t>>>(context, props)
    {
        kernel_width_ = 3;
        oversampling_factor_ = oversampling_factor;
        verbose = false;
    }

    void process(InputChannel<Core::variant<Core::Acquisition, std::vector<std::vector<size_t>>, std::vector<size_t>>> &in,
                 OutputChannel &out) override
    {

        int selectedDevice = nhlbi_toolbox::utils::selectCudaDevice();
        size_t RO, E1, E2, CHA, N;
        nhlbi_toolbox::utils::enable_peeraccess();

        IsmrmrdImageArray imarray;
        ISMRMRD::AcquisitionHeader acqhdr;
        Gadgetron::reconParams recon_params;

        auto idx = 0;

        auto maxAcq = ((header.encoding.at(0).encodingLimits.kspace_encoding_step_1.get().maximum + 1) *
                       (header.encoding[0].encodingLimits.kspace_encoding_step_2.get().maximum + 1) *
                       (header.encoding[0].encodingLimits.average.get().maximum + 1) *
                       (header.encoding[0].encodingLimits.repetition.get().maximum + 1)); 
        GDEBUG_STREAM("maxAcq:  " << maxAcq);
        std::vector<std::vector<size_t>> idx_phases;
        std::vector<Core::Acquisition> allAcq(maxAcq);
        std::vector<ISMRMRD::AcquisitionHeader> headers(maxAcq);
        std::vector<size_t> vect_idx_max;
        auto idx_max = 0; 
        auto linesMeasured=0;
        bool SNR_to_estimate = true;

        // Collect all the data -- BEGIN()

        for (auto message : in)
        {
            if (holds_alternative<Core::Acquisition>(message) && idx < maxAcq)
            {
                auto &[head, data, traj] = Core::get<Core::Acquisition>(message);
                allAcq[idx] = std::move(Core::get<Core::Acquisition>(message));
                idx++;
            }
            if (holds_alternative<std::vector<std::vector<size_t>>>(message))
            {
                idx_phases = Core::get<std::vector<std::vector<size_t>>>(message);
            }

            if (holds_alternative<std::vector<size_t>>(message))
            {
                vect_idx_max = Core::get<std::vector<size_t>>(message);
                idx_max = vect_idx_max.back();
                linesMeasured = vect_idx_max.front();
                GDEBUG_STREAM("Idx " << idx_max);
                GDEBUG_STREAM("Lines Measured " << linesMeasured);
            }
            if ((!vect_idx_max.empty()) && (idx_max==idx) && (SNR_to_estimate))
            {
                SNR_to_estimate = false;
                GadgetronTimer timer("Pseudo Replica Recon :");
                cudaSetDevice(selectedDevice);
                auto &[headAcq, dataAcq, trajAcq] = allAcq[0];
                acqhdr = headAcq;
                RO = dataAcq.get_size(0);
                CHA = dataAcq.get_size(1);
                E2 = this->header.encoding.front().encodedSpace.matrixSize.z;
                N = dataAcq.get_size(3);

                recon_params.numberChannels = CHA;
                recon_params.RO = RO;
                recon_params.ematrixSize = this->header.encoding.front().encodedSpace.matrixSize;
                recon_params.rmatrixSize = this->header.encoding.front().reconSpace.matrixSize;
                recon_params.fov = this->header.encoding.front().encodedSpace.fieldOfView_mm;
                recon_params.oversampling_factor_ = oversampling_factor_;
                recon_params.kernel_width_ = kernel_width_;
                recon_params.selectedDevice = selectedDevice;
                recon_params.norm = 2;

                this->initialize_encoding_space_limits(this->header);

                noncartesian_reconstruction reconstruction(recon_params);
                allAcq.resize(idx);
                
                if (idx_phases.empty() || idx_phases[0].size() == 0){
                    GDEBUG_STREAM("Binning was not done \n");
                    
                } else {
                    GDEBUG_STREAM("Idx_phases: " << idx_phases.size());
                    // Calculate elements in each bin and sum of all elements
                    auto sumall = 0;
                    std::vector<size_t> nelem_idx;
                    for (auto iph = 0; iph < idx_phases.size(); iph++)
                    {
                        sumall += idx_phases[iph].size();
                        nelem_idx.push_back(idx_phases[iph].size());
                        GDEBUG_STREAM("nelem_idx: " << idx_phases[iph].size());

                    }
                }
                auto [cuData_csm, traj_csm, dcw_csm] = reconstruction.organize_data(&allAcq);
                auto dcf = reconstruction.estimate_dcf(&traj_csm,&dcw_csm);
                noncartesian_reconstruction_pseudo_replica<3> reconstruction_pr(recon_params);
                auto cuIimages = reconstruction_pr.reconstruct(&cuData_csm, &traj_csm, &dcf, int(numReplicas));
                
                auto SNR_image = reconstruction_pr.calculate_SNR(&cuIimages);
                auto SNR_image_crop = reconstruction.crop_to_recondims<float_complext>(SNR_image);
                auto ho_images_all = hoNDArray<std::complex<float>>(std::move(*boost::reinterpret_pointer_cast<hoNDArray<std::complex<float>>>(SNR_image_crop.to_host())));
                    
                // Sending SNR images
                using namespace Gadgetron::Indexing;
                imarray.data_ = ho_images_all;
                nhlbi_toolbox::utils::attachHeadertoImageArray(imarray, acqhdr, this->header);
                prepare_image_array(imarray, (size_t)0, ((int)series_counter), GADGETRON_IMAGE_SNR_MAP);
                auto timePseudo=(float)timer.stop()/1000.0;
                GDEBUG_STREAM("Time (ms) : " << timePseudo);
                imarray.headers_(0, 0, 0).user_int[0] = linesMeasured;
                imarray.headers_(0, 0, 0).user_float[0] = timePseudo;
                imarray.headers_(0, 0, 0).data_type = ISMRMRD::ISMRMRD_CXFLOAT;
                imarray.headers_(0, 0, 0).image_type = ISMRMRD::ISMRMRD_IMTYPE_COMPLEX;
                out.push(imarray);
                series_counter++;
            }

        }
        
    }

protected:
    NODE_PROPERTY(oversampling_factor, float, "oversampling_factor", 2.1);
    NODE_PROPERTY(numReplicas, size_t, "numReplicas", 100);
    int series_counter = 0;
};

GADGETRON_GADGET_EXPORT(Spiral_3D_SNR)