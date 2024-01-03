#include "noncartesian_reconstruction.h"
#include <gadgetron/cuNonCartesianSenseOperator.h>
#include <gadgetron/cuSbcCgSolver.h>

using namespace Gadgetron;
using namespace Gadgetron::Core;
using namespace Gadgetron::Indexing;

namespace nhlbi_toolbox
{
    namespace reconstruction
    {
        class noncartesian_reconstruction_2Dtimes : public noncartesian_reconstruction<2>
        {
        public:
            noncartesian_reconstruction_2Dtimes(reconParams recon_params) : noncartesian_reconstruction<2>(recon_params){};
            
            using noncartesian_reconstruction::reconstruct;
            using noncartesian_reconstruction::organize_data;            

            cuNDArray<float_complext> reconstruct_CGSense(cuNDArray<float_complext> *data,
            std::vector<cuNDArray<vector_td<float, 2>>> *traj,std::vector<cuNDArray<float>> *dcw,
            boost::shared_ptr<cuNDArray<float_complext>> csm);
            
    
        };
    }
}