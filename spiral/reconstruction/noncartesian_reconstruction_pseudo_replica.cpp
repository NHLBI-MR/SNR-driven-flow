#include "noncartesian_reconstruction_pseudo_replica.h"
#include "util_functions.h"
#include <curand.h>
#include <cuNDArray_reductions.h>
using namespace Gadgetron;

namespace nhlbi_toolbox
{
    namespace reconstruction
    {
        template <size_t D>
        cuNDArray<float_complext> noncartesian_reconstruction_pseudo_replica<D>::reconstruct(cuNDArray<float_complext> *data, cuNDArray<vector_td<float, D>> *trajectory_in, cuNDArray<float> *dcf_in, int num_iterations)
        {
            using namespace nhlbi_toolbox::utils;

            int cur_device;
            cur_device = data->get_device();
            cudaSetDevice(cur_device);

            auto data_dims = *data->get_dimensions();

            auto dataSizeT = std::accumulate(data_dims.begin(), data_dims.end(), size_t(1), std::multiplies<size_t>()) * 4 * 2;

            dataSize = dataSizeT + dataSizeT / (2 * this->recon_params.numberChannels) + 3 * dataSizeT / (2 * this->recon_params.numberChannels); // add dcw and traj

            auto image_dims = this->image_dims_;
            auto imageSize = std::accumulate(image_dims.begin(), image_dims.end(), size_t(1), std::multiplies<size_t>()) * 4 * 2;
            dataSize += imageSize;

            auto eligibleGPUs = nhlbi_toolbox::utils::FindCudaDevices(dataSize);

            image_dims.push_back(this->recon_params.numberChannels);


            // preprocess(trajectory_in);

            auto iters_per_gpu = int(std::ceil(float(num_iterations) / float(eligibleGPUs.size())));

            auto im_dims = this->image_dims_;
            im_dims.push_back((iters_per_gpu * eligibleGPUs.size()) - (eligibleGPUs.size()-1));
            cuNDArray<float_complext> images_all(im_dims, cur_device);

            GDEBUG("eligibleGPUs: %d \n", eligibleGPUs.size());
            {
                GadgetronTimer timer("Pseudo Replica Reconstruction");

#pragma omp parallel for num_threads(eligibleGPUs.size()) schedule(dynamic)
                for (auto ii = 0; ii < eligibleGPUs.size(); ii++)
                {
                    cudaSetDevice(eligibleGPUs[ii]);
                 //   GDEBUG("GPU: %d", eligibleGPUs[ii]);
                    im_dims.pop_back();
                    im_dims.push_back(this->recon_params.numberChannels);

                    cuNDArray<float_complext> csm(im_dims, eligibleGPUs[ii]);
                    //  GadgetronTimer timer("reconstruct_todevice");
                    auto tdata = nhlbi_toolbox::utils::set_device(data, eligibleGPUs[ii]);

                    // auto timage = set_device(&images, deviceNo);

                    cuNDArray<float_complext> timage(image_dims, eligibleGPUs[ii]);
                    auto ttraj = nhlbi_toolbox::utils::set_device(trajectory_in, eligibleGPUs[ii]);
                    auto tdcf = nhlbi_toolbox::utils::set_device(dcf_in, eligibleGPUs[ii]);
                    
                    nhlbi_toolbox::reconstruction::noncartesian_reconstruction<D> nr(this->recon_params);

                    cuNDArray<float_complext> temp(tdata.get_dimensions(), eligibleGPUs[ii]);
                    auto cuData = boost::make_shared<cuNDArray<float_complext>>(temp);
                    
                    auto dims_tocreate = *tdata.get_dimensions();
                    dims_tocreate.push_back(2);
                    
                    cuNDArray<float> gpuDatar(dims_tocreate);
                    dims_tocreate.pop_back();
                    auto stride_data = std::accumulate(dims_tocreate.begin(), dims_tocreate.end(), 1, std::multiplies<size_t>());

                    

                    for (auto jj = 0; jj < iters_per_gpu; jj++)
                    {

                        cudaMemcpy(cuData.get()->get_data_ptr(),
                                   tdata.get_data_ptr(),
                                   tdata.get_number_of_elements() * sizeof(float_complext), cudaMemcpyDefault);

                        if ((jj != 0))
                        {   // better to generate large noise matrix this is likely good enough
                            curandGenerator_t gen;

                            time_t seconds;
                            time(&seconds);
                            srand((unsigned int)seconds);
                            curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937);
                            curandSetPseudoRandomGeneratorSeed(gen, seconds * (eligibleGPUs[ii] + 1) * (num_iterations + jj));
                            curandGenerateNormal(gen, gpuDatar.get_data_ptr(), gpuDatar.get_number_of_elements(), 0.0, 1);
                                
                            auto inputr = cuNDArray<float>(dims_tocreate, gpuDatar.data());
                            auto inputi = cuNDArray<float>(dims_tocreate, gpuDatar.data()+stride_data);

                            *cuData += *real_to_complex<float_complext>(&inputr);
                            *cuData += *imag_to_complex<float_complext>(&inputi);
                            curandDestroyGenerator(gen);
                        }
                        nr.reconstruct(cuData.get(), &timage, &ttraj, &tdcf);

                        if (jj == 0)
                            csm = *nr.generateRoemerCSM(&timage);

                        timage *= *conj(&csm);
                        auto combined = *sum(&timage, timage.get_number_of_dimensions() - 1);

                        // all this is to prevent copy of extra samples without noise
                        size_t stride;
                        if (ii == 0)
                            stride = (jj + ii * iters_per_gpu);
                        else if (ii == 1)
                            stride = ((jj - 1) + ii * iters_per_gpu);
                        else if (ii > 1)
                            stride = ((jj - 1) + (ii - 1) * (iters_per_gpu - 1) + iters_per_gpu);
                        if (jj != 0 || (ii == 0 && jj == 0))
                        {   // this is not a prob takes < 1ms on the order of 100us
                            cudaMemcpy(images_all.get_data_ptr() + stride * combined.get_number_of_elements(), combined.get_data_ptr(),
                                       (combined).get_number_of_elements() * sizeof(float_complext), cudaMemcpyDefault);
                        }
                    }
                }
            }

            return (images_all);
        }

        template <size_t D>
        cuNDArray<float_complext> noncartesian_reconstruction_pseudo_replica<D>::calculate_SNR(cuNDArray<float_complext> *images)
        {

            auto dims = *images->get_dimensions();
            auto num_elements = std::accumulate(dims.begin(), dims.end() - 1, size_t(1), std::multiplies<size_t>());

            auto num_images = dims[dims.size() - 1];

            dims.pop_back();
            cuNDArray<float_complext> no_noise_image(dims);
            dims.push_back(num_images - 1);
            cuNDArray<float_complext> noise_images(dims);

            cudaMemcpy(no_noise_image.get_data_ptr(), images->get_data_ptr(),
                       num_elements * sizeof(float_complext), cudaMemcpyDefault);

            cudaMemcpy(noise_images.get_data_ptr(), images->get_data_ptr() + num_elements,
                       num_elements * (num_images - 1) * sizeof(float_complext), cudaMemcpyDefault);

            dims.pop_back();
            hoNDArray<float> mean_nums(dims);

            auto ho_noise_images = hoNDArray<float_complext>(std::move(*boost::reinterpret_pointer_cast<hoNDArray<std::complex<float>>>(noise_images.to_host())));
            auto std_dev_ims = nhlbi_toolbox::utils::std_complex(ho_noise_images, D);

            auto SNR = no_noise_image;
            SNR /= cuNDArray<float_complext>(std_dev_ims);
            return SNR;
        }

        template class noncartesian_reconstruction_pseudo_replica<2>;
        template class noncartesian_reconstruction_pseudo_replica<3>;
    }

}