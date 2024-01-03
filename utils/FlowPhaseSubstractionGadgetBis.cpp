


#ifdef USE_OMP
#include <omp.h>
#endif
#include "Gadget.h"
#include "GadgetMRIHeaders.h"
#include "hoNDArray.h"
#include "gadgetron_mricore_export.h"
#include "ismrmrd/xml.h"
#include <queue>
#include <ismrmrd/ismrmrd.h>
#include <complex>

using namespace Gadgetron;
using namespace Gadgetron::Core;

  
    class FlowPhaseSubtractionGadgetBis : public ChannelGadget<Core::Image<std::complex<float>>>
    {

    public:

        FlowPhaseSubtractionGadgetBis(const Core::Context &context, const Core::GadgetProperties &props) : ChannelGadget<Core::Image<std::complex<float>>>(context, props)
    {
    }
        void process(InputChannel<Core::Image<std::complex<float>>>& in,
                                            OutputChannel& out) override{

        const auto e_limits = this->header.encoding[0].encodingLimits;
        const auto sets = e_limits.set ? e_limits.set->maximum + 1 : 1;
        GDEBUG_STREAM( "Flow");
        if (sets > 2)
            throw std::runtime_error("Phase subtraction only implemented for two sets");

        if (sets < 2) {
            std::move(in.begin(), in.end(), out.begin());
            return;
        }
        
        std::map<int, std::queue<Core::Image<std::complex<float>>>> queues;

        for (auto [header, data, meta] : in) {
            queues[header.set].emplace(header, std::move(data), std::move(meta));

            if (queues[0].empty() || queues[1].empty())
                continue;

            auto [header1, data1, meta1] = std::move(queues[0].front());
            auto [header2, data2, meta2] = std::move(queues[1].front());
            queues[0].pop();
            queues[1].pop();

            if (header1.image_index != header2.image_index)
                throw std::runtime_error("Mismatch in input indices detected");
            if (data1.size() != data2.size())
                throw std::runtime_error("Images must have same number of elements");

    #ifdef USE_OMP
    #pragma omp parallel for
    #endif
            for (long i = 0; i < (long)data2.size(); i++) {
                std::complex<float> tmp =
                    std::polar((std::abs(data1[i]) + std::abs(data2[i])) / 2.0f, std::arg(data1[i]) - std::arg(data2[i]));
                data2[i] = tmp;
            }

            header2.set = 0;
            header2.image_type=ISMRMRD::ISMRMRD_IMTYPE_PHASE;
            out.push(Core::Image<std::complex<float>>(header2, std::move(data2), meta2));
            header1.image_series_index +=1;
            header1.image_type=ISMRMRD::ISMRMRD_IMTYPE_MAGNITUDE;
            GDEBUG_STREAM( "DataType "<<header1.data_type);
            out.push(Core::Image<std::complex<float>>(header1, std::move(data1), meta2));
        }
    }

    protected:;
    }; 
GADGETRON_GADGET_EXPORT(FlowPhaseSubtractionGadgetBis)