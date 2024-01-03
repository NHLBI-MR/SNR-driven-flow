#pragma once

#include "gpBbSolver.h"
#include <gadgetron/cuNDArray_operators.h>
#include <gadgetron/cuNDArray_elemwise.h>
#include <gadgetron/cuNDArray_blas.h>
#include <gadgetron/real_utilities.h>
#include <gadgetron/vector_td_utilities.h>


#include <gadgetron/cuSolverUtils.h>
using namespace nhlbi_toolbox;
namespace nhlbi_toolbox{

  template <class T> class cuGpBbSolver : public nhlbi_toolbox::gpBbSolver<cuNDArray<T> >
  {
  public:

    cuGpBbSolver() : gpBbSolver<cuNDArray<T> >() {}
    virtual ~cuGpBbSolver() {}
  };
}
