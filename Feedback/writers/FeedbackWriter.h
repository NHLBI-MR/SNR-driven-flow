#pragma once

#include <ismrmrd/ismrmrd.h>

#include <gadgetron/hoNDArray.h>

#include <gadgetron/Types.h>
#include <gadgetron/Writer.h>
#include "../FeedbackData.h"
namespace Gadgetron::Core::Writers {

    class FeedbackWriter
            : public Core::TypedWriter<Gadgetron::FeedbackData> {
    protected:
        void serialize(
                std::ostream &stream,
                const Gadgetron::FeedbackData &fdata
        );
    };
}