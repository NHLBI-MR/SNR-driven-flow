#include "noncartesian_reconstruction_2Dtimes.h"
#include "util_functions.h"
#include "cuNonCartesianTSenseOperator.h"
#include <gadgetron/cuSbcCgSolver.h>
#include <gadgetron/cuNDArray_elemwise.h>
#include "cuNDArray_elemwise.h"
using namespace Gadgetron;
namespace nhlbi_toolbox
{
    namespace reconstruction
    {
        
        cuNDArray<float_complext> noncartesian_reconstruction_2Dtimes::reconstruct_CGSense(
            cuNDArray<float_complext> *data,
            std::vector<cuNDArray<vector_td<float, 2>>> *traj,
            std::vector<cuNDArray<float>> *dcw,
            boost::shared_ptr<cuNDArray<float_complext>> csm)
        {
            auto data_dims = *data->get_dimensions();

            auto stride = std::accumulate(data_dims.begin(), data_dims.end() - 1, size_t(1), std::multiplies<size_t>());
            
            // prep data and dcw
            for (auto ii = 0; ii < dcw->size(); ii++)
            {
                auto inter_acc = std::accumulate(recon_params.shots_per_time.begin(), recon_params.shots_per_time.begin() + (ii), size_t(0)) * data_dims[0];

                for (auto iCHA = 0; iCHA < recon_params.numberChannels; iCHA++)
                {
                    auto dataview = cuNDArray<complext<float>>((*dcw)[ii].get_dimensions(), data->data() + inter_acc + stride * iCHA);
                    dataview *= (*dcw)[ii];
                }
            }

            auto E_ = boost::shared_ptr<cuNonCartesianTSenseOperator<float, 2>>(new cuNonCartesianTSenseOperator<float, 2>(ConvolutionType::ATOMIC));
            auto D_ = boost::shared_ptr<cuCgPreconditioner<float_complext>>(new cuCgPreconditioner<float_complext>());
            //recon_params.shots_per_time.get_size(0): cardiac frames ; recon_params.shots_per_time.get_size(1):set
            recon_dims = {image_dims_[0], image_dims_[1],recon_params.shots_per_time.get_size(0),recon_params.shots_per_time.get_size(1)};
            
            cuGpBbSolver<float_complext> solver_;

            solver_.set_max_iterations(recon_params.iterations);
            cuNDArray<float_complext> reg_image(recon_dims);
            E_->set_shots_per_time(recon_params.shots_per_time);
            E_->setup(from_std_vector<size_t, 2>(image_dims_), from_std_vector<size_t, 2>(image_dims_os_), recon_params.kernel_width_);
            E_->set_codomain_dimensions(&data_dims);
            E_->set_domain_dimensions(&recon_dims);
            E_->set_csm(csm);
            E_->set_dcw(*dcw);
            E_->preprocess(*traj);

            solver_.set_encoding_operator(E_);
            solver_.set_tc_tolerance(recon_params.tolerance);
            solver_.set_output_mode(decltype(solver_)::OUTPUT_VERBOSE);
            boost::shared_ptr<cuNDArray<float>> _precon_weights;
            _precon_weights = sum(abs_square(csm.get()).get(), 2);
            reciprocal_sqrt_inplace(_precon_weights.get());

            boost::shared_ptr<cuNDArray<float_complext>> precon_weights = boost::make_shared<cuNDArray<float_complext>>(*real_to_complex<float_complext>(_precon_weights.get()));
            D_->set_weights(precon_weights);
            solver_.set_preconditioner(D_);

            boost::shared_ptr<cuPartialDerivativeOperator<float_complext, 4>>
                Rt(new cuPartialDerivativeOperator<float_complext, 4>(2));

            Rt->set_weight(recon_params.lambda_time);
            Rt->set_domain_dimensions(&recon_dims);
            Rt->set_codomain_dimensions(&recon_dims);
            solver_.add_regularization_operator(Rt, recon_params.norm);

            boost::shared_ptr<cuPartialDerivativeOperator<float_complext, 4>>
                Rx(new cuPartialDerivativeOperator<float_complext, 4>(0));
            Rx->set_weight(recon_params.lambda_spatial);
            Rx->set_domain_dimensions(&recon_dims);
            Rx->set_codomain_dimensions(&recon_dims);
            solver_.add_regularization_operator(Rx, recon_params.norm);
            boost::shared_ptr<cuPartialDerivativeOperator<float_complext, 4>>
                Ry(new cuPartialDerivativeOperator<float_complext, 4>(1));
            Ry->set_weight(recon_params.lambda_spatial);
            Ry->set_domain_dimensions(&recon_dims);
            Ry->set_codomain_dimensions(&recon_dims);
            solver_.add_regularization_operator(Ry, recon_params.norm);

            int selectedDevice = nhlbi_toolbox::utils::selectCudaDevice();

            GDEBUG_STREAM("Data_device:" << data->get_device());
            GDEBUG_STREAM("gpus_input_possible[0]:" << selectedDevice);

            if (selectedDevice != data->get_device())
            {
                std::vector<int> gpus_input({data->get_device(), selectedDevice});
                solver_.set_gpus(gpus_input);
            }
            reg_image = *solver_.solve(data);

        return reg_image;
        }

    }
}