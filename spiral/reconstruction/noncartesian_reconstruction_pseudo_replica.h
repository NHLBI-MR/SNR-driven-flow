#pragma once

#include "noncartesian_reconstruction.h"
#include "reconParams.h"
#include <gadgetron/cuNDArray_elemwise.h>
using namespace Gadgetron;
using namespace Gadgetron::Core;
using namespace Gadgetron::Indexing;

namespace nhlbi_toolbox
{
    namespace reconstruction
    {
        template <size_t D>
        class noncartesian_reconstruction_pseudo_replica : public noncartesian_reconstruction<D>
        {
        public:
            noncartesian_reconstruction_pseudo_replica(reconParams recon_params) : noncartesian_reconstruction<D>(recon_params){};

            cuNDArray<float_complext> reconstruct(cuNDArray<float_complext> *data, cuNDArray<vector_td<float, D>> *trajectory_in, cuNDArray<float> *dcf_in, int num_iterations);
            cuNDArray<float_complext> calculate_SNR(cuNDArray<float_complext> *images);
            void preprocess(cuNDArray<vector_td<float,D>> *trajectory);
            size_t dataSize;
        protected:
            std::vector<std::vector<nhlbi_toolbox::reconstruction::noncartesian_reconstruction<D>>> nr_vector; //gpus<it>
            std::vector<int> eligibleGPUs;

        private:
        };
    }
}