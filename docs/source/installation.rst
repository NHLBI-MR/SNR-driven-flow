Installation instructions
=========================

Hardware and software requirements
----------------------------------
* Gadgetron-python version 1.4.1
* Gadgetron version 4.5.1
* Tested on  GPUs : NVIDIA A100-SXM (80Gb), NVIDIA Quadro RTX 8000 (50Gb) 

Installation
------------

We recommend to different ways of obtaining and running this project : using a conda environment and build it or using Docker container

Installing in conda environment
-------------------------------

First of all, you will need to install Gadgetron. Information are available over here : 

`Gadgetron repository <https://gadgetron.readthedocs.io/en/latest/obtaining.html>`_ and `Gadgetron documentation <https://github.com/gadgetron/gadgetron>`_

.. code-block:: console

    mamba activate gadgetron
    mamba env update -f environment.yml

The code can then be built 

.. code-block:: console

    mkdir build && cd build && cmake ../ -GNinja -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} -DC_PREFIX_PATH=${CONDA_PREFIX} -DUSE_CUDA=ON -DUSE_MKL=ON


Once built, the package can be used with gadgetron using the config xml files provided with this repository (`config files repository <https://github.com/NHLBI-MR/SNR-driven-flow/tree/main/config>`_).

After activating the environment (with conda activate gadgetron), you should be able to check that everything is working with ``gadgetron --info`` and ``gadgetron_ismrmrd_client_feedback -h``

Docker container 
----------------

Alternatively, you can test the code by pulling the provided docker image using the following command:

.. code-block:: console

    docker pull gadgetronnhlbi/ubuntu_2004_cuda117_public_snrdrivenflow:built_rt


This image can be deployed with: 

.. code-block:: console

    docker run --gpus all  --name=deploy_rt -ti -p 9063:9002 --volume=[LOCAL_DATA_FOLDER]:/opt/data --restart unless-stopped --detach gadgetronnhlbi/ubuntu_2004_cuda117_public_snrdrivenflow:built_rt`

where **LOCAL_DATA_FOLDER** is the path to a folder containing raw data that can be used for testing the reconstruction. 

Test the code 
-------------

Once the docker container is running, you can start a bash terminal inside the container using: 

.. code-block:: console

    docker exec -ti deploy_rt bash 

and you can simply ou can simply navigate to `/opt/data/` and test the code :

.. code-block:: console

    cd /opt/data
    gadgetron_ismrmrd_client_feedback -p 9002 -f DATA_FILE -c spiral_2d_AO.xml -o OUTPUT_FILENAME.h5` 


In another terminal session you can monitor the logs from the container 

.. code-block:: console

    docker logs -f deploy_rt`


Please note that if you are using the gadgetron_ismrmrd_client_feedback from outside the container then you may need to specify the server address with **-a SERVER_ADDRESS** and the port **-p 9063**

.. code-block:: console

    cd LOCAL_DATA_FOLDER
    gadgetron_ismrmrd_client_feedback -a SERVER_ADDRESS -p 9063 -f DATA_FILE -c spiral_2d_AO.xml -o OUTPUT_FILENAME.h5` 


.. warning::

    gadgetron_ismrmrd_client will not work with our `config files repository <https://github.com/NHLBI-MR/SNR-driven-flow/tree/main/config>`_ 
    because the gadget **ImageSNRExtractionAndFeedback** is sending 
    a Feedback message which MessageID is unknown for Gadgetron alone.

Dataset
-------

The test data can be downloaded from zenodo: 10.5281/zenodo.10525047


Test dataset output
-------------------

.. code-block:: console

	gadgetron_ismrmrd_client_feedback -p 9002 -f AO_spiral_flow_pd_s32_vds100_FA25_smax40_noBF_206026_30000023100310273443400000003_30000023100310273443400000003_58_20231020-151547.h5 -c spiral_2d_AO.xml -o OUTPUT_FILENAME.h5
	Gadgetron ISMRMRD client
	  -- host            :      localhost
	  -- port            :      9002
	  -- hdf5 file  in   :      AO_spiral_flow_pd_s32_vds100_FA25_smax40_noBF_206026_30000023100310273443400000003_30000023100310273443400000003_58_20231020-151547.h5
	  -- hdf5 group in   :      /dataset
	  -- conf            :      spiral_2d_AO.xml
	  -- loop            :      1
	  -- hdf5 file out   :      OUTPUT_FILENAME.h5
	  -- hdf5 group out  :      2024-02-29 21:55:51
	This measurement has dependent measurements
	  SenMap : 206026_30000023100310273443400000003_30000023100310273443400000003_49
	  Noise : 206026_30000023100310273443400000003_30000023100310273443400000003_49
	Querying the Gadgetron instance for the dependent measurement: 206026_30000023100310273443400000003_30000023100310273443400000003_49
	Message received with ID: 1019
	WARNING: Dependent noise measurement not found on Gadgetron server. Was the noise data processed?
	Message received with ID: 1028
	size_str: 11
	Message: 0x7fcac3be5d10
	size_data: 21
	Feedback mybool: 1
	Feedback myint: 0
	Feedback myint2: 0
	Feedback myfloat: 58.9305
	Message received with ID: 1022
	Message received with ID: 1022
	Message received with ID: 1028
	size_str: 11
	Message: 0x7fcac3be5d10
	size_data: 21
	Feedback mybool: 1
	Feedback myint: 0
	Feedback myint2: 0
	Feedback myfloat: 83.1711
	Message received with ID: 1022
	Message received with ID: 1022
	Message received with ID: 1028
	size_str: 11
	Message: 0x7fcac3be5d10
	size_data: 21
	Feedback mybool: 1
	Feedback myint: 0
	Feedback myint2: 0
	Feedback myfloat: 101.61
	Message received with ID: 1022
	Message received with ID: 1022
	Message received with ID: 1028
	size_str: 11
	Message: 0x7fcac3be5d10
	size_data: 21
	Feedback mybool: 1
	Feedback myint: 0
	Feedback myint2: 0
	Feedback myfloat: 118.086
	Message received with ID: 1022
	Message received with ID: 1022
	Message received with ID: 1028
	size_str: 11
	Message: 0x7fcac3be5d10
	size_data: 21
	Feedback mybool: 1
	Feedback myint: 0
	Feedback myint2: 0
	Feedback myfloat: 132.117
	Message received with ID: 1022
	Message received with ID: 1022
	Message received with ID: 1028
	size_str: 11
	Message: 0x7fcac3be5d10
	size_data: 21
	Feedback mybool: 1
	Feedback myint: 0
	Feedback myint2: 0
	Feedback myfloat: 145.143
	Message received with ID: 1022
	Message received with ID: 1022
	Message received with ID: 1028
	size_str: 11
	Message: 0x7fcac3be5d10
	size_data: 21
	Feedback mybool: 1
	Feedback myint: 0
	Feedback myint2: 0
	Feedback myfloat: 155.582
	Message received with ID: 1022
	Message received with ID: 1022
	Message received with ID: 1028
	size_str: 11
	Message: 0x7fcac3be5d10
	size_data: 21
	Feedback mybool: 1
	Feedback myint: 0
	Feedback myint2: 0
	Feedback myfloat: 166
	Message received with ID: 1022
	Message received with ID: 1022
	Message received with ID: 1028
	size_str: 11
	Message: 0x7fcac3be5d10
	size_data: 21
	Feedback mybool: 1
	Feedback myint: 0
	Feedback myint2: 0
	Feedback myfloat: 177.593
	Message received with ID: 1022
	Message received with ID: 1022
	Message received with ID: 1028
	size_str: 11
	Message: 0x7fcac3be5d10
	size_data: 21
	Feedback mybool: 1
	Feedback myint: 0
	Feedback myint2: 0
	Feedback myfloat: 185.791
	Message received with ID: 1022
	Message received with ID: 1022
	Message received with ID: 1022
	Message received with ID: 1022
	Message received with ID: 1022
	Message received with ID: 1028
	size_str: 11
	Message: 0x7fcac3be5d10
	size_data: 21
	Feedback mybool: 1
	Feedback myint: 0
	Feedback myint2: 0
	Feedback myfloat: 194.667
	Message received with ID: 1022
	Message received with ID: 1022
	Message received with ID: 1028
	size_str: 11
	Message: 0x7fcac3be5d10
	size_data: 21
	Feedback mybool: 1
	Feedback myint: 0
	Feedback myint2: 0
	Feedback myfloat: 204.155
	Message received with ID: 1022
	Message received with ID: 1022
	Message received with ID: 1028
	size_str: 11
	Message: 0x7fcac3be5d10
	size_data: 21
	Feedback mybool: 1
	Feedback myint: 0
	Feedback myint2: 0
	Feedback myfloat: 211.391
	Message received with ID: 1022
	Message received with ID: 1022
	Message received with ID: 1028
	size_str: 11
	Message: 0x7fcac3be5d10
	size_data: 21
	Feedback mybool: 1
	Feedback myint: 0
	Feedback myint2: 0
	Feedback myfloat: 218.067
	Message received with ID: 1022
	Message received with ID: 1022
	Message received with ID: 1028
	size_str: 11
	Message: 0x7fcac3be5d10
	size_data: 21
	Feedback mybool: 1
	Feedback myint: 0
	Feedback myint2: 0
	Feedback myfloat: 221.655
	Message received with ID: 1022
	Message received with ID: 1022
	
