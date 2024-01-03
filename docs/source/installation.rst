Installation instructions
=========================

First of all, you will need to install Gadgetron. Information are available over here : `Gadgetron repository <https://gadgetron.readthedocs.io/en/latest/obtaining.html>`_
 `Gadgetron documentation <https://github.com/gadgetron/gadgetron>`_

.. code-block:: console

    mamba activate gadgetron
    mamba env update -f environment.yml

The code can then be built 

.. code-block:: console

    mkdir build && cd build && cmake ../ -GNinja -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} -DC_PREFIX_PATH=${CONDA_PREFIX} -DUSE_CUDA=ON -DUSE_MKL=ON


Once built, the package can be used with gadgetron using the config xml files provided with this repository (`config files repository<https://github.com/NHLBI-MR/SNR-driven-flow/tree/main/config>`_).

Test
====

Unfortunately the test dataset is not yet available. It should be in the following days.
