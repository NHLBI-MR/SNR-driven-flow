/**
    \brief  Passes through an acquisition to the next gadget in the pipeline if the acquisition is below a certain time
*/
#include <gadgetron/Node.h>
#include <gadgetron/Gadget.h>
#include <gadgetron/hoNDArray.h>
#include <gadgetron/Types.h>
#include <ismrmrd/ismrmrd.h>
#include <ismrmrd/xml.h>
using namespace Gadgetron;
using namespace Gadgetron::Core;

class AcquisitionPassthroughTimingGadget : public ChannelGadget<Core::Acquisition> 
{

public:
    AcquisitionPassthroughTimingGadget(const Core::Context &context, const Core::GadgetProperties &props) : ChannelGadget<Core::Acquisition>(context, props)
    {
    }
    void process(Core::InputChannel<Core::Acquisition>& in, Core::OutputChannel& out) override
    {
        bool time_limit_exceeded = false;
        bool repetition_last = false;
        auto idx = 0;
        uint32_t startTime=0;
        std::vector<size_t> info_to_send;
        bool plot_index =true;
        auto repMeasured =0;
        auto time =0.0f;
        for (auto message : in)
        {
            if ((time_limit_exceeded && repetition_last)){
                if ((plot_index)){
                    auto time_per_rep=time/float(repMeasured);
                    GDEBUG_STREAM( "Last Index " << idx);
                    GDEBUG_STREAM( "Measured repetition " << repMeasured);
                    GDEBUG_STREAM( "Times per rep " << time_per_rep);
                    info_to_send.push_back(repMeasured);
                    info_to_send.push_back(idx);
                    out.push(info_to_send);
                    plot_index = false;
                }
                continue;
            }         
                        
            auto &head = std::get<ISMRMRD::AcquisitionHeader>(message);

            if ((head.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_NAVIGATION_DATA) || head.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_RTFEEDBACK_DATA))){
                out.push(message);
                continue;
            }

            if(idx==0){
                startTime = head.acquisition_time_stamp;
            }
            
            time = float(head.acquisition_time_stamp-startTime)*2.5; // ms
            time_limit_exceeded=time > timit_limit;
            repetition_last = head.isFlagSet(ISMRMRD::ISMRMRD_ACQ_LAST_IN_SLICE);
            if (repetition_last)
            {
                repMeasured++;
            }
            
            out.push(message);
            idx++;
            
        }
                
    }

protected:
NODE_PROPERTY(timit_limit, float, "Time in ms", 20000.0f);
};
GADGETRON_GADGET_EXPORT(AcquisitionPassthroughTimingGadget)
