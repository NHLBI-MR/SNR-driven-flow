
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
#include <gadgetron/mri_core_def.h>
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

#include "../spiral/SpiralBuffer.h"
#include "../utils/gpu/cuda_utils.h"
#include <util_functions.h>
#include <noncartesian_reconstruction_2Dtimes.h>

#include <cuNDArray_reductions.h>
#include "reconParams.h"

#include <omp.h>
#include <algorithm>
#include <cmath>

#include <string>


using namespace Gadgetron;
using namespace Gadgetron::Core;
using namespace Gadgetron::Indexing;
using namespace nhlbi_toolbox::reconstruction;

class Spiral2DFlow_CGSense : public ChannelGadget<Core::variant<Core::Acquisition, std::vector<std::vector<size_t>>>>,
                        public ImageArraySendMixin<Spiral2DFlow_CGSense>
{
public:
    std::vector<size_t> image_dims_;
    uint64d3 image_dims_os_;
    float oversampling_factor_;
    float kernel_width_;
    bool verbose;

    Spiral2DFlow_CGSense(const Core::Context &context, const Core::GadgetProperties &props) : ChannelGadget<Core::variant<Core::Acquisition, std::vector<std::vector<size_t>>>>(context, props)
    {
        kernel_width_ = 3;
        oversampling_factor_ = oversampling_factor;
        verbose = false;
    }

    void process(InputChannel<Core::variant<Core::Acquisition, std::vector<std::vector<size_t>>>> &in,
                 OutputChannel &out) override
    {

        int selectedDevice = nhlbi_toolbox::utils::selectCudaDevice();

        size_t RO, E1, E2, CHA, N, S, SLC;
        nhlbi_toolbox::utils::enable_peeraccess();

        ISMRMRD::AcquisitionHeader acqhdrset1;
        Gadgetron::reconParams recon_params;
        boost::shared_ptr<cuNDArray<float_complext>> csm ;

        auto idx = 0;
        std::vector<std::vector<size_t>> idx_phases;
        size_t acq_count = 0;

        bool weightsEstimated = false;
        auto maxZencode = header.encoding.front().encodingLimits.kspace_encoding_step_2.get().maximum + 1;
        auto maxAcq = ((header.encoding.at(0).encodingLimits.kspace_encoding_step_1.get().maximum + 1) *
                       (header.encoding[0].encodingLimits.kspace_encoding_step_2.get().maximum + 1) *
                       (header.encoding[0].encodingLimits.average.get().maximum + 1) *
                       (header.encoding[0].encodingLimits.repetition.get().maximum + 1) *
                       (header.encoding[0].encodingLimits.phase.get().maximum + 1) *
                       (header.encoding[0].encodingLimits.set.get().maximum + 1));
        
        GDEBUG_STREAM("maxAcq: " << maxAcq);
        std::vector<Core::Acquisition> allAcq(maxAcq);
        std::vector<ISMRMRD::AcquisitionHeader> headers(maxAcq);
        uint32_t startTime=0;

        auto sets =(unsigned long)(header.encoding[0].encodingLimits.set.get().maximum + 1);

        // Collect all the data -- BEGIN()
        for (auto message : in)
        {
            if (holds_alternative<Core::Acquisition>(message) && (idx < maxAcq))
            {
                auto &[head, data, traj] = Core::get<Core::Acquisition>(message);
                allAcq[idx] = std::move(Core::get<Core::Acquisition>(message));
                idx++;

            }
            if (holds_alternative<std::vector<std::vector<size_t>>>(message))
            {
                idx_phases = Core::get<std::vector<std::vector<size_t>>>(message);
            }
        }
        {   
            // Check if binning data was sent -- cannot proceed without it really ! Use different Gadget
            if (idx_phases.empty() || idx_phases[0].size() == 0)
                GERROR("binning was not done\n");
            else
                GDEBUG_STREAM("Idx_phases:" << idx_phases.size());
            
            auto &[headT, dataT, trajT] = allAcq[0];
            acqhdrset1 = headT;
            startTime = acqhdrset1.acquisition_time_stamp;
            auto acq_toRecon = allAcq;
            acq_toRecon.resize(idx);

            GadgetronTimer timer("2D Recon + times:");

            cudaSetDevice(selectedDevice);
            auto &[headAcq, dataAcq, trajAcq] = acq_toRecon[0];
            RO = dataAcq.get_size(0);
            CHA = dataAcq.get_size(1);
            E2 = this->header.encoding.front().encodedSpace.matrixSize.z;
            N = dataAcq.get_size(3);
            S = 1;
            SLC = 1;

            // Calculate elements in each bin and sum of all elements
            auto sumall = 0;
            std::vector<size_t> nelem_idx;
            int32_t tframes=idx_phases[idx_phases.size()-1][0];
            idx_phases.pop_back();
            auto cardiac_frames=idx_phases.size()/sets;
            auto shots_per_time = hoNDArray<size_t> ({idx_phases.size()/sets,sets});
            for (auto set = 0; set < sets; set++)
            {
                for (auto iph = 0; iph < cardiac_frames; iph++)
                    {
                        sumall += idx_phases[iph+set*cardiac_frames].size();
                        nelem_idx.push_back(idx_phases[iph+set*cardiac_frames].size());
                        shots_per_time(iph,set) = (nelem_idx[iph+set*cardiac_frames]);
                    }
            }

            // NUFFT Recon parameters 
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
            recon_params.oversampling_factor_dcf_ = 5.5; //2.1;
            recon_params.kernel_width_dcf_ = 5.5; //5.5;
            recon_params.iterations_dcf = 20;

            this->initialize_encoding_space_limits(this->header);
            noncartesian_reconstruction<2> reconstruction(recon_params);

            auto [cuData, traj_in, dcf_in,num_elements] = reconstruction.organize_data(&acq_toRecon,idx_phases);

            auto dcf_csm = reconstruction.estimate_dcf(&traj_in,&dcf_in);
            auto dims_reco = reconstruction.get_recon_dims();
            cuNDArray<float_complext> channel_images(dims_reco,selectedDevice);
            reconstruction.reconstruct(&cuData, &channel_images, &traj_in, &dcf_csm);
            
            csm= reconstruction.generateRoemerCSM(&channel_images);
        
            //auto csm = *reconstruction.generateMcKenzieCSM(&channel_images);

            if(writeTmpData)
            {
                 auto csm2 = *reconstruction.generateMcKenzieCSM(&channel_images);
                nhlbi_toolbox::utils::write_gpu_nd_array(channel_images,"/opt/data/gt_data/channelImages.complex");
                nhlbi_toolbox::utils::write_gpu_nd_array(dcf_csm,"/opt/data/gt_data/dcf_csm.real");

            }

            auto trajVec = reconstruction.arraytovector(&traj_in, num_elements);
            auto dcwEstVec = reconstruction.arraytovector(&dcf_in, num_elements);
            std::vector<cuNDArray<float>> dcwVec;
            std::vector<size_t> num_elements_cha;
            for (auto iph = 0; iph < idx_phases.size(); iph++)
            {

                num_elements_cha.push_back(num_elements[iph]*recon_params.numberChannels);
                auto trajII = trajVec[iph];
                auto dcwII = dcwEstVec[iph];
                auto dcfII = reconstruction.estimate_dcf(&trajII, &dcwII);
                sqrt_inplace(&(dcfII));
                dcwVec.push_back(dcfII);
            }
            //auto cuDataP=permute(cuData,{0, 2, 1});
            //auto cuDataVec = reconstruction.arraytovector(&cuDataP, num_elements_cha);

            //
            //auto shots_per_time = hoNDArray<size_t>({nelem_idx.size(),1},nelem_idx.data());
            recon_params.shots_per_time = shots_per_time;
            recon_params.iterations = iterationsSense;
            recon_params.tolerance = tolSense;
            recon_params.selectedDevice = selectedDevice;
            recon_params.lambda_spatial = lambda;
            recon_params.lambda_time = lambdat;
            recon_params.norm = 2;
            nhlbi_toolbox::reconstruction::noncartesian_reconstruction_2Dtimes reconstruction2Dt(recon_params);
            //nhlbi_toolbox::reconstruction::noncartesian_reconstruction_fc<2> reconstruction4D(recon_params);
            //auto cuIimages = reconstruction4D.reconstruct_CGSense_fc(&cuData, &trajVec, &dcwVec, csm);
            auto cuIimages = reconstruction2Dt.reconstruct_CGSense(&cuData, &trajVec, &dcwVec, csm);
            /*
            long ro= RO;
            long channels =CHA;
            std::vector<cuNDArray<floatd3>> traj3dVec;
            for (auto ii = 0; ii < num_elements.size(); ii++)
            {   
                cuNDArray<floatd3> traj3Dii(trajVec[ii].get_dimensions());
                float z=0;
                for(auto xy = 0; xy < trajVec[ii].get_number_of_elements(); xy++){
                    floatd3 trajXYZ;
                    trajXYZ[0]=trajVec[ii][xy][0];
                    trajXYZ[1]=trajVec[ii][xy][1];
                    trajXYZ[2]=0;
                    traj3Dii[xy]=trajXYZ;
                }
                traj3dVec.push_back(std::move(traj3Dii));

            }

            // traj 3D
            nhlbi_toolbox::reconstruction::noncartesian_reconstruction_4D reconstruction4D(recon_params);
            auto cuIimages = reconstruction4D.reconstruct(&cuData, &traj3dVec, &dcwEstVec, csm);
            */

            //
            auto dims = reconstruction.get_recon_dims();
            dims.pop_back();
            dims.push_back(1);
            dims.push_back(num_elements.size()/sets);
            dims.push_back(sets);
            
            //cuNDArray<float_complext> images_all(dims,selectedDevice);
            GDEBUG_STREAM("Size 0:" << cuIimages.get_size(0));
            GDEBUG_STREAM("Size: 1" << cuIimages.get_size(1));
            GDEBUG_STREAM("Size: 2" << cuIimages.get_size(2));
            GDEBUG_STREAM("Size: 3" << cuIimages.get_size(3));
            cuIimages.reshape(dims);   
            using namespace Gadgetron::Indexing;
            IsmrmrdImageArray imarray_sense;
            auto ho_images_all = hoNDArray<std::complex<float>>(std::move(*boost::reinterpret_pointer_cast<hoNDArray<std::complex<float>>>(cuIimages.to_host())));

            for (auto ii = 0; ii < ho_images_all.get_size(4); ii++)
            {
                auto tmp = hoNDArray<std::complex<float>>(ho_images_all(slice, slice, slice, slice,ii));
                ISMRMRD::ImageHeader newHeader = ISMRMRD::ImageHeader();
                newHeader.clearAllFlags();
                newHeader.matrix_size[0] = (uint16_t)ho_images_all.get_size(0);
                newHeader.matrix_size[1] = (uint16_t)ho_images_all.get_size(1);
                newHeader.matrix_size[2] = (uint16_t)ho_images_all.get_size(2);
                newHeader.field_of_view[0] = recon_params.fov.x;
                newHeader.field_of_view[1] = recon_params.fov.y;
                newHeader.field_of_view[2] = recon_params.fov.z;
                newHeader.channels = (uint16_t)ho_images_all.get_size(3);
                newHeader.set = (uint16_t)ii;
                newHeader.user_int[0]=tframes;
                memcpy(newHeader.position, acqhdrset1.position, sizeof(float) * 3);
                memcpy(newHeader.read_dir, acqhdrset1.read_dir, sizeof(float) * 3);
                memcpy(newHeader.phase_dir, acqhdrset1.phase_dir, sizeof(float) * 3);
                memcpy(newHeader.slice_dir, acqhdrset1.slice_dir, sizeof(float) * 3);
                memcpy(newHeader.patient_table_position, acqhdrset1.patient_table_position, sizeof(float) * 3);
                newHeader.data_type = ISMRMRD::ISMRMRD_CXFLOAT;
                newHeader.image_index = (uint16_t)(series_counter);
                newHeader.image_series_index = (uint16_t)series_counter;
                auto newMetaContainer = std::optional<ISMRMRD::MetaContainer>(); 
                out.push(Core::Image<std::complex<float>>(newHeader, std::move(tmp), newMetaContainer));

            }
            series_counter++;

            bool condition_stop = false;
        }
    }

protected:
    NODE_PROPERTY(oversampling_factor, float, "oversampling_factor", 2.1);
    NODE_PROPERTY(writeTmpData, bool, "Write data for debugging ", false);
    NODE_PROPERTY(iterationsSense, size_t, "Number of Iterations Sense", 5);
    NODE_PROPERTY(tolSense, float, "Tolerance", 1e-6);
    NODE_PROPERTY(lambdat, float, "lambdat", 0);
    NODE_PROPERTY(lambda, float, "lambda", 0);
    int series_counter = 0;
};

GADGETRON_GADGET_EXPORT(Spiral2DFlow_CGSense)

