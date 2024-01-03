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

class AcquisitionPassthroughSpiralShotsGadget : public ChannelGadget<Core::Acquisition> 
{

public:
    AcquisitionPassthroughSpiralShotsGadget(const Core::Context &context, const Core::GadgetProperties &props) : ChannelGadget<Core::Acquisition>(context, props)
    {
    }
    void process(Core::InputChannel<Core::Acquisition>& in, Core::OutputChannel& out) override
    {
        bool kstep_last = false;
        auto idx = 0;
        auto kstep_max = this->header.encoding[0].encodingLimits.kspace_encoding_step_2.get().maximum; // Need to be fix
        auto kstep_max_ping_pong = 1; // Need to be fix
        uint32_t startTime=0;
        std::vector<size_t> info_to_send;
        bool plot_index =true;
        auto linesMeasured =0;
        for (auto message : in)
        {
            if ((linesMeasured == max_spiral_shots)){
                if ((plot_index)){
                GDEBUG_STREAM( "Last Index " << idx);
                GDEBUG_STREAM( "Measured Lines " << linesMeasured);
                GDEBUG_STREAM( "Times Last " << time);
                info_to_send.push_back(linesMeasured);
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
            auto time = float(head.acquisition_time_stamp-startTime)*2.5; // ms
            kstep_last = (head.idx.kspace_encode_step_2 == kstep_max*kstep_max_ping_pong) || head.isFlagSet(ISMRMRD::ISMRMRD_ACQ_LAST_IN_ENCODE_STEP2);
            if (kstep_last)
            {
                linesMeasured++; 

                if (Ping_pong_kz){
                    kstep_max_ping_pong= (kstep_max_ping_pong+1) %2;
                }
            }
            
            out.push(message);
            idx++;
            
        }
                
    }

protected:
NODE_PROPERTY(Ping_pong_kz, bool, "Ping Pong kz sampling ", true);
NODE_PROPERTY(max_spiral_shots, uint32_t, "Maximum spiral shots ", 0);
};
GADGETRON_GADGET_EXPORT(AcquisitionPassthroughSpiralShotsGadget)
