#include "FeedbackWriter.h"
#include "io/primitives.h"
#include "../MessageID.h"
#include "../FeedbackData.h"

namespace Gadgetron::Core::Writers {
    
        void FeedbackWriter::serialize(
                std::ostream &stream,
                const Gadgetron::FeedbackData &fdata

        ) {

        IO::write(stream, GADGET_MESSAGE_ISMRMRD_FEEDBACK);
        IO::write(stream,uint32_t(sizeof("MyFeedback")));
        IO::write(stream,"MyFeedback");
        IO::write(stream,uint32_t(sizeof(fdata)));
        IO::write(stream,fdata);
            
        }
    
        GADGETRON_WRITER_EXPORT(FeedbackWriter)


}

