
#include "WaveformToTrajectory.h"

#include <gadgetron/log.h>
#include <gadgetron/mri_core_data.h>
#include <gadgetron/hoNDFFT.h>
#include <math.h>
#include <stdio.h>
#include <ismrmrd/xml.h>
#include <boost/algorithm/string.hpp>
#include <iostream>
#include <string>
#include <fstream>
#include <boost/filesystem/fstream.hpp>

#include <gadgetron/mri_core_utility.h>
#include <gadgetron/GadgetronTimer.h>

constexpr double PI = boost::math::constants::pi<double>();
using namespace Gadgetron;
using namespace Gadgetron::Core;
using namespace nhlbi_toolbox::utils;
WaveformToTrajectory::WaveformToTrajectory(const Core::Context &context, const Core::GadgetProperties &props)
    : ChannelGadget(context, props), header{context.header}, trajParams{context.header} {}

namespace
{
  bool is_noise(Core::Acquisition &acq)
  {
    return std::get<ISMRMRD::AcquisitionHeader>(acq).isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_NOISE_MEASUREMENT);
  }
} // namespace

// Convert to Templates to deal with view arrays etc.
hoNDArray<float> WaveformToTrajectory::combineTrajDCW(
    hoNDArray<floatd2> *traj_input, hoNDArray<float> *dcw_input, int iSL)
{

  std::vector<size_t> dims = *traj_input->get_dimensions();

  dims.insert(dims.begin(), (this->header.encoding.front().encodedSpace.matrixSize.z > 1) ? 4 : 3); // make the dimensions of traj+dcw be 4 is 3D and 3 is 2D using z encoding to check this
  auto traj = hoNDArray<float>(dims);

  auto traj_ptr = traj.get_data_ptr();
  auto ptr = traj_input->get_data_ptr();
  auto dcwptr = dcw_input->get_data_ptr();
  // std::ofstream ofs("/tmp/traj_grad_flat.log");
  for (size_t i = 0; i < traj_input->get_number_of_elements(); i++)
  {
    if ((this->header.encoding.front().encodedSpace.matrixSize.z > 1)) // is  3D
    {
      auto zencoding = float(-0.5 + iSL * 1 / ((float)this->header.encoding.front().encodedSpace.matrixSize.z));

      traj_ptr[i * 4] = ptr[i][0];
      traj_ptr[1 + i * 4] = ptr[i][1];
      traj_ptr[2 + i * 4] = zencoding;
      traj_ptr[3 + i * 4] = dcwptr[i];
    }
    else
    {
      traj_ptr[i * 3] = ptr[i][0];
      traj_ptr[1 + i * 3] = ptr[i][1];
      traj_ptr[2 + i * 3] = dcwptr[i];
    }
  }

  return traj;
}
hoNDArray<float> WaveformToTrajectory::combineTrajDCW(
    hoNDArray<float> *traj_input, hoNDArray<float> *dcw_input, int iSL)
{

  std::vector<size_t> dims = {(this->header.encoding.front().encodedSpace.matrixSize.z > 1) ? 4 : 3, traj_input->get_size(1)}; // make the dimensions of traj+dcw be 4 is 3D and 3 is 2D using z encoding to check this

  // dims.insert(dims.begin(), 3);
  auto traj = hoNDArray<float>(dims);

  auto traj_ptr = traj.get_data_ptr();
  auto ptr = traj_input->get_data_ptr();
  auto dcwptr = dcw_input->get_data_ptr();
  // std::ofstream ofs("/tmp/traj_grad_flat.log");
  for (size_t i = 0; i < traj_input->get_number_of_elements(); i++)
  {
    if ((this->header.encoding.front().encodedSpace.matrixSize.z > 1)) // is  3D
    {
      auto zencoding = float(-0.5 + iSL * 1 / ((float)this->header.encoding.front().encodedSpace.matrixSize.z));

      traj_ptr[i * 4] = ptr[i * 3];
      traj_ptr[1 + i * 4] = ptr[i * 3 + 1];
      traj_ptr[2 + i * 4] = zencoding;
      traj_ptr[3 + i * 4] = dcwptr[i];
    }
    else
    {
      traj_ptr[i * 3] = ptr[i * 2];
      traj_ptr[1 + i * 3] = ptr[i * 2 + 1];
      traj_ptr[2 + i * 3] = dcwptr[i];
    }
  }

  return traj;
}
void WaveformToTrajectory ::process(
    Core::InputChannel<Core::variant<Core::Acquisition, Core::Waveform>> &in, Core::OutputChannel &out)
{
  using namespace Gadgetron::Indexing;

  int waveForm_samples;
  int upsampleFactor;
  std::vector<Core::Acquisition> acquisitionsVec;
  auto fov = this->header.encoding.front().encodedSpace.fieldOfView_mm;
  auto matrixsize = this->header.encoding.front().encodedSpace.matrixSize;

  kspace_scaling = 1e-3 * fov.x / matrixsize.x;

  hoNDArray<float> traj_dcw;
  std::pair<hoNDArray<floatd2>, hoNDArray<float>> tw_gen;
  hoNDArray<floatd2> trajgen;
  hoNDArray<float> dcwgen;
  double Tsamp_us = 0.0;
  int trajecLen = 0;
  bool traj_not_generated = true;
  auto str_model = (header.acquisitionSystemInformation).get().systemModel->c_str();
  GDEBUG("systemModel:                   %s\n", str_model);

  if (perform_GIRF && this->girf_kernel.get_number_of_elements() == 0)
    if (strstr(str_model, "MAGNETOM eMeRge-XL"))
      this->girf_kernel = nhlbi_toolbox::corrections::readGIRFKernel(GIRF_folder + "GIRF_fmax_"); // AJ fix for now
    else
      this->girf_kernel = nhlbi_toolbox::corrections::readGIRFKernel(GIRF_folder + "GIRF"); // Read GIRF Kernel from file

  {
    GadgetronTimer timer("WaveformToTrajectory");
    // #pragma omp parallel
    // #pragma omp for

    for (auto message : in)
    {

      if (holds_alternative<Core::Waveform>(message) && !generateTraj)
      {
        auto &temp_waveform = Core::get<Core::Waveform>(message);
        auto &wave_head = std::get<ISMRMRD::WaveformHeader>(Core::get<Core::Waveform>(message));

        if (wave_head.waveform_id >= 10 && wave_head.waveform_id < 15)
        {
          waveForm_samples = wave_head.number_of_samples - 16;
          gradient_wave_store.insert(std::pair<size_t, Core::Waveform>(wave_head.scan_counter, std::move(Core::get<Core::Waveform>(message))));
        }
        else
        {
          // out.push(Core::get<Core::Waveform>(message));
        }

        continue;
      }

      if (Core::holds_alternative<Core::Acquisition>(message))
      {
        if (is_noise(Core::get<Core::Acquisition>(message)))
          continue;

        auto &[head, data, traj] = Core::get<Core::Acquisition>(message);
        if ((head.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_NAVIGATION_DATA) || head.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_HPFEEDBACK_DATA) || head.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_RTFEEDBACK_DATA)))
        {
          out.push(Core::Acquisition(std::move(head), std::move(data), std::move(traj)));
          continue;
        }

        head.trajectory_dimensions = (header.encoding.front().encodedSpace.matrixSize.z > 1) ? 4 : 3; // Nees for prepare_trajectory_from_waveforms;
        hoNDArray<float> trajectory_and_weights;

        if (generateTraj)
        {

          if (traj_not_generated)
          {
            if (perform_GIRF) // && trajParams.get_girf_kernel().get_number_of_elements() == 0)
            {
              if (strstr(str_model, "MAGNETOM eMeRge-XL"))
              {
                GDEBUG_STREAM("reading Emerge ")
                trajParams.read_girf_kernel(GIRF_folder + "GIRF_fmax_"); // AJ fix for now
              }
              else
                trajParams.read_girf_kernel(GIRF_folder + "GIRF");
              trajParams.set_girf_sampling_time(GIRF_samplingtime);
            }
            GadgetronTimer timer("Trajectory Gen:");
            tw_gen = trajParams.calculate_trajectories_and_weight(head);
            traj_not_generated = false;
            trajgen = std::get<0>(tw_gen);
            dcwgen = std::get<1>(tw_gen);
            nhlbi_toolbox::utils::normalize_trajectory(&trajgen);
            Tsamp_us=trajParams.get_Tsampling_us();
          }
          head.sample_time_us=Tsamp_us;
          if (head.discard_pre > 0)
            head.discard_pre = 0;
          auto traj = hoNDArray<floatd2>(trajgen(slice, head.idx.kspace_encode_step_1));
          auto densitycw = hoNDArray<float>(dcwgen(slice, head.idx.kspace_encode_step_1));
          traj_dcw = combineTrajDCW(&traj, &densitycw,
                                    head.idx.kspace_encode_step_2);
          trajecLen = trajgen.get_size(0);
          trajectory_and_weights = traj_dcw;
          for (int ii = 0; ii < trajectory_and_weights.get_size(1); ii++)
          {
            trajectory_and_weights(0, ii) = traj_dcw(0, ii);
            trajectory_and_weights(1, ii) = traj_dcw(1, ii);
            trajectory_and_weights(2, ii) = traj_dcw(2, ii);
              size_t num = 0;
              if (abs(trajectory_and_weights(0, ii)) > 0.5f || abs(trajectory_and_weights(1, ii)) > 0.5f)
              {
                if (ii == 0)
                  GERROR("To Prevent recon failure setting to ±0.5 \n");

                if (trajectory_and_weights(0, ii) > 0.5f)
                  {
                    trajectory_and_weights(0, ii) = 0.5f;
                  }
                  else if (trajectory_and_weights(0, ii) < -0.5f)
                  {
                    trajectory_and_weights(0, ii) = -0.5f;
                  }
                  if (trajectory_and_weights(1, ii) > 0.5f)
                  {
                    trajectory_and_weights(1, ii) = 0.5f;
                  }
                  else if (trajectory_and_weights(1, ii) < -0.5f)
                  {
                    trajectory_and_weights(1, ii) = -0.5f;
                  }
                num++;
              }
          }
        }
        // Prepare Trajectory for each acq and push the bucked through
        else
        {
          if (attachWaveform)
          {
            if (trajectory_map.find(head.idx.kspace_encode_step_1) == trajectory_map.end())
            {
              if (gradient_wave_store.find(head.scan_counter) != gradient_wave_store.end())
              {
                prepare_trajectory_from_waveforms(gradient_wave_store.find(head.scan_counter)->second, head);
              }
              else
              {
                acquisitionsVec.push_back(Core::get<Core::Acquisition>(message));
                continue;
              }
            }
            trajectory_and_weights = trajectory_map.find(head.idx.kspace_encode_step_1)->second;
            upsampleFactor = head.number_of_samples / waveForm_samples;

            int extraSamples = head.number_of_samples - waveForm_samples * upsampleFactor;
          }
          else
          {

            if (header.encoding.front().encodedSpace.matrixSize.z > 1)
            {
              if (head.discard_pre == 0) // UTE sequence doesn't set this parameter and FIRE incorrectly appends to the end so need this hack for now
              {
                head.discard_post = 10;
              }
              trajectory_and_weights = hoNDArray<float>({4, (*traj).get_size(1) - head.discard_post});
            }
            else
              trajectory_and_weights = hoNDArray<float>({3, (*traj).get_size(1) - head.discard_post});

            auto temp = permute(*traj, {0, 1});
            trajectory_and_weights.fill(0.0);
            for (int ii = 0; ii < trajectory_and_weights.get_size(1); ii++)
            {
              trajectory_and_weights(0, ii) = temp(0, ii);
              trajectory_and_weights(1, ii) = temp(1, ii);
              trajectory_and_weights(2, ii) = temp(2, ii);
              if (!perform_GIRF) // only do this if not doing apply girf else apply girf takes care of this
              {
                size_t num = 0;
                if (abs(trajectory_and_weights(0, ii)) > 0.5f || abs(trajectory_and_weights(1, ii)) > 0.5f)
                {
                  if (ii == 0)
                    GERROR("To Prevent recon failure setting to ±0.5 \n");

                  if (trajectory_and_weights(0, ii) > 0.5f)
                  {
                    trajectory_and_weights(0, ii) = 0.5f;
                  }
                  else if (trajectory_and_weights(0, ii) < -0.5f)
                  {
                    trajectory_and_weights(0, ii) = -0.5f;
                  }
                  if (trajectory_and_weights(1, ii) > 0.5f)
                  {
                    trajectory_and_weights(1, ii) = 0.5f;
                  }
                  else if (trajectory_and_weights(1, ii) < -0.5f)
                  {
                    trajectory_and_weights(1, ii) = -0.5f;
                  }
                  num++;
                }
              }
              if (header.encoding.front().encodedSpace.matrixSize.z > 1)
                trajectory_and_weights(3, ii) = temp(3, ii);
            }

            if (perform_GIRF) // do_girf
            {
              applyGIRF(trajectory_and_weights, head, header, kspace_scaling, this->girf_kernel);
            }

            //   trajectory_and_weights = permute(trajectory_and_weights, {1, 0});
          }
        }


        if (head.discard_pre == 0 && setPre)
          head.discard_pre = pre_cutoff_manual;

        out.push(Core::Acquisition(head, data, trajectory_and_weights));
      }
    }
  }
  // This is really needed because gadgetron_ismrmrd client is sending waveforms after data something has changed !
  if (attachWaveform)
  {
    GadgetronTimer timer("WaveformToTrajectory Secondary");
#pragma omp parallel
#pragma omp for
    for (auto message : acquisitionsVec)
    {
      auto &[head, data, traj] = message;
      head.trajectory_dimensions = 4; // Nees for prepare_trajectory_from_waveforms;
      hoNDArray<float> trajectory_and_weights;
      int extraSamples = 0;
      if (attachWaveform)
      {
        if (trajectory_map.find(head.idx.kspace_encode_step_1) == trajectory_map.end())
        {
          prepare_trajectory_from_waveforms(gradient_wave_store.find(head.scan_counter)->second, head);
        }

        trajectory_and_weights = trajectory_map.find(head.idx.kspace_encode_step_1)->second;

        head.trajectory_dimensions = 4; // Code to gen 3D traj forDavid set this to 4
        upsampleFactor = head.number_of_samples / waveForm_samples;

        extraSamples = head.number_of_samples - waveForm_samples * upsampleFactor;
      }
      else
      {
        trajectory_and_weights = *traj;

        extraSamples = head.number_of_samples - traj->get_size(1);
      }
      std::vector<size_t> tmp_dims;
      tmp_dims.push_back(head.number_of_samples);
      tmp_dims.push_back(head.active_channels);
      data.reshape(tmp_dims);
      head.number_of_samples = head.number_of_samples - extraSamples;

      auto acq = Core::Acquisition(std::move(head), std::move(data), std::move(trajectory_and_weights));
      out.push(acq);
    }
  }
  GadgetronTimer timer1("WaveformToTrajectory");
}
void WaveformToTrajectory::prepare_trajectory_from_waveforms(Core::Waveform &grad_waveform, const ISMRMRD::AcquisitionHeader &head)
{
  using namespace Gadgetron::Indexing;

  arma::fmat33 rotation_matrix;
  rotation_matrix(0, 0) = head.read_dir[0];
  rotation_matrix(1, 0) = head.read_dir[1];
  rotation_matrix(2, 0) = head.read_dir[2];
  rotation_matrix(0, 1) = head.phase_dir[0];
  rotation_matrix(1, 1) = head.phase_dir[1];
  rotation_matrix(2, 1) = head.phase_dir[2];
  rotation_matrix(0, 2) = head.slice_dir[0];
  rotation_matrix(1, 2) = head.slice_dir[1];
  rotation_matrix(2, 2) = head.slice_dir[2];

  auto TE_ = header.sequenceParameters.get().TE.get().at(0);
  auto &[wave_head, wave_data] = grad_waveform;

  hoNDArray<float> wave_data_float(wave_data.size() / 3, 3);
  auto wave_data_floatx = reinterpret_cast<const float *>(wave_data.get_data_ptr());

  int numberofGradSamples = wave_data_floatx[0];
  // auto wave_data_float = hoNDArray<const float>(wave_data_floatx);
  int sizeofCustomHeader = (wave_data.size() - 3 * numberofGradSamples) / 3;

  std::copy(wave_data_floatx, wave_data_floatx + wave_data.size(), wave_data_float.begin());

  int upsampleFactor = head.number_of_samples / numberofGradSamples;

  hoNDArray<floatd2> gradients(numberofGradSamples);

  size_t size_gradOVS = numberofGradSamples * upsampleFactor;

  auto trajectory_and_weights = hoNDArray<float>(4, size_gradOVS);
  trajectory_and_weights.fill(0.0f);
  for (int ii = 0; ii < numberofGradSamples; ii++)
  {
    gradients(ii)[0] = wave_data_float(sizeofCustomHeader + ii, 0); // / std::numeric_limits<uint32_t>::max()) * 80 - 40;
    gradients(ii)[1] = wave_data_float(sizeofCustomHeader + ii, 1); // / std::numeric_limits<uint32_t>::max()) * 80 - 40;
  }

  auto gradients_interpolated = zeroHoldInterpolation(gradients, upsampleFactor);

  if (perform_GIRF)
    gradients_interpolated = nhlbi_toolbox::corrections::girf_correct(gradients_interpolated, this->girf_kernel, rotation_matrix, 2e-6, 10e-6, 0.85e-6);

  auto zencoding = float(-0.5 + head.idx.kspace_encode_step_2 * 1 / ((float)this->header.encoding.front().encodedSpace.matrixSize.z));
  trajectory_and_weights(0, 0) = (gradients_interpolated(0)[0]) * GAMMA * 10 * 2 / 1000000 * kspace_scaling;
  trajectory_and_weights(1, 0) = (gradients_interpolated(0)[1]) * GAMMA * 10 * 2 / 1000000 * kspace_scaling;
  trajectory_and_weights(2, 0) = zencoding;

  for (int ii = 1; ii < size_gradOVS; ii++)
  {

    trajectory_and_weights(0, ii) = ((gradients_interpolated(ii)[0]) * GAMMA * 10 * 2 / 1000000 * kspace_scaling + trajectory_and_weights(0, ii - 1)); // mT/m * Hz/G * 10G * 2e-6
    trajectory_and_weights(1, ii) = ((gradients_interpolated(ii)[1]) * GAMMA * 10 * 2 / 1000000 * kspace_scaling + trajectory_and_weights(1, ii - 1)); // mT/m * Hz/G * 10G * 2e-6
    trajectory_and_weights(2, ii) = zencoding;                                                                                                         // Code to gen 3D traj forDavid
                                                                                                                                                       //  trajectory_and_weights(2, ii)  = float(-0.5 + head.idx.kspace_encode_step_2 * 1 / ((float)this->header.encoding.front().encodedSpace.matrixSize.z)); // Need to make it compatible with full waveform streaming and 2D
    if (abs(trajectory_and_weights(0, ii)) > 0.5 || abs(trajectory_and_weights(1, ii)) > 0.5)
    {
      GERROR("Trajectory outof bounds ±0.5 \n");
      GERROR("To Prevent recon failure setting to ±0.5 \n");
      if (trajectory_and_weights(0, ii) > 0.5f)
        trajectory_and_weights(0, ii) = 0.5f;
      else if (trajectory_and_weights(0, ii) < -0.5f)
        trajectory_and_weights(0, ii) = -0.5f;

      if (trajectory_and_weights(1, ii) > 0.5f)
        trajectory_and_weights(1, ii) = 0.5f;
      else if (trajectory_and_weights(1, ii) < -0.5f)
        trajectory_and_weights(1, ii) = -0.5f;
    }
  }
  float maxTx;
  float minTx;
  auto temp = permute(trajectory_and_weights, {1, 0});
  maxValue(hoNDArray<float>(temp(slice, 0)), maxTx);
  minValue(hoNDArray<float>(temp(slice, 0)), minTx);

  if (maxTx > 0.5 || minTx < -0.5)
    GERROR("Trajectory out of bounds");

  hoNDArray<float> trajectories_temp(3, trajectory_and_weights.get_size(1));
  auto temp2 = permute(trajectory_and_weights, {1, 0});
  trajectories_temp(0, slice) = hoNDArray<float>(temp2(slice, 0));
  trajectories_temp(1, slice) = hoNDArray<float>(temp2(slice, 1));
  trajectories_temp(2, slice) = hoNDArray<float>(temp2(slice, 2));

  trajectory_and_weights(3, slice) = calculate_weights_Hoge(gradients_interpolated, trajectories_temp);

  trajectory_map.insert(std::pair<size_t, hoNDArray<float>>(head.idx.kspace_encode_step_1, trajectory_and_weights));
}

hoNDArray<floatd2> WaveformToTrajectory::sincInterpolation(const hoNDArray<floatd2> input, int zpadFactor)
{
  hoNDArray<floatd2> output(input.size() * zpadFactor);
  hoNDArray<std::complex<float>> cinput = hoNDArray<std::complex<float>>(input.size());
  hoNDArray<std::complex<float>> coutput = hoNDArray<std::complex<float>>(input.size() * zpadFactor);

  for (int jj = 0; jj < 2; jj++)
  {

    std::fill(coutput.begin(), coutput.end(), 0);

    for (int zz = 0; zz < cinput.size(); zz++)
    {
      cinput(zz) = (input(zz)[jj]);
    }

    hoNDFFT<float>::instance()->fft1c(cinput);

    for (int ii = 0; ii < coutput.size(); ii++)
    {
      if (ii > coutput.size() / 2 - cinput.size() / 2 - 1 && ii < coutput.size() / 2 + (cinput.size() / 2))
      {
        coutput(ii) = cinput(ii - (output.size() / 2 - cinput.size() / 2));
      }
    }

    hoNDFFT<float>::instance()->ifft1c(coutput);
    coutput *= sqrt(zpadFactor);
    for (int zz = 0; zz < coutput.size(); zz++)
    {
      output(zz)[jj] = real(coutput(zz));
    }
  }
  // output *= sqrt(zpadFactor);
  return output;
}
hoNDArray<floatd2> WaveformToTrajectory::zeroHoldInterpolation(const hoNDArray<floatd2> input, int zpadFactor)
{
  hoNDArray<floatd2> output(input.size() * zpadFactor);

  for (int ii = 0; ii < input.size() * zpadFactor; ii++)
  {
    output(ii) = input(int(ii / zpadFactor));
  }
  return output;
}

hoNDArray<float> WaveformToTrajectory::calculate_weights_Hoge(const hoNDArray<floatd2> &gradients, const hoNDArray<float> &trajectories)
{

  using namespace Gadgetron::Indexing;
  hoNDArray<float> weights(trajectories.get_size(1), 1);
  for (int ii = 0; ii < trajectories.get_size(1); ii++)
  {

    auto abs_g = sqrt(gradients(ii)[0] * gradients(ii)[0] + gradients(ii)[1] * gradients(ii)[1]);
    auto abs_t = sqrt(trajectories(0, ii) * trajectories(0, ii) + trajectories(1, ii) * trajectories(1, ii));
    auto ang_g = atan2(gradients(ii)[1], gradients(ii)[0]);
    auto ang_t = atan2(trajectories(1, ii), trajectories(0, ii));
    weights(ii) = abs(cos(ang_g - ang_t)) * abs_g * abs_t;
  }

  return weights;
}

void WaveformToTrajectory::printGradtoFile(std::string fname_grad, hoNDArray<floatd2> grad_traj)
{
  std::ofstream of(fname_grad);
  for (auto ele : grad_traj)
    of << ele[0] << "\t" << ele[1] << "\n";
  of.close();
}

void WaveformToTrajectory::printTrajtoFile(std::string fname_grad, hoNDArray<float> grad_traj)
{
  std::ofstream of(fname_grad);
  for (int i = 0; i < grad_traj.get_size(1); i++)
    of << grad_traj(0, i) << "\t" << grad_traj(1, i) << "\n";
  of.close();
}
void WaveformToTrajectory::applyGIRF(hoNDArray<float> &trajectory_and_weights, ISMRMRD::AcquisitionHeader head, ISMRMRD::IsmrmrdHeader header, float kspace_scaling, hoNDArray<std::complex<float>> girf_kernel)
{
  arma::fmat33 rotation_matrix;
  rotation_matrix(0, 0) = head.read_dir[0];
  rotation_matrix(1, 0) = head.read_dir[1];
  rotation_matrix(2, 0) = head.read_dir[2];
  rotation_matrix(0, 1) = head.phase_dir[0];
  rotation_matrix(1, 1) = head.phase_dir[1];
  rotation_matrix(2, 1) = head.phase_dir[2];
  rotation_matrix(0, 2) = head.slice_dir[0];
  rotation_matrix(1, 2) = head.slice_dir[1];
  rotation_matrix(2, 2) = head.slice_dir[2];
  if (header.encoding.front().encodedSpace.matrixSize.z > 1)
  {

    auto traj_dcw = nhlbi_toolbox::utils::separate_traj_and_dcw_3D(&trajectory_and_weights, head.idx.kspace_encode_step_2, header.encoding.front().encodedSpace.matrixSize.z);
    auto traj_sep = std::move(*std::get<0>(traj_dcw).get());
    auto dcw_sep = std::move(*std::get<1>(traj_dcw).get());

    auto gradients = nhlbi_toolbox::utils::traj2grad_3D2D(traj_sep, kspace_scaling);
    gradients = nhlbi_toolbox::corrections::girf_correct(gradients, girf_kernel, rotation_matrix, head.sample_time_us * 1e-6, 10e-6, 0.85e-6);

    auto zencoding = float(-0.5 + head.idx.kspace_encode_step_2 * 1 / ((float)header.encoding.front().encodedSpace.matrixSize.z));
    trajectory_and_weights(0, 0) = (gradients(0)[0]) * GAMMA * 10 * 2 / 1000000 * kspace_scaling;
    trajectory_and_weights(1, 0) = (gradients(0)[1]) * GAMMA * 10 * 2 / 1000000 * kspace_scaling;
    trajectory_and_weights(2, 0) = zencoding;
    trajectory_and_weights(3, 0) = dcw_sep(0);

    for (int ii = 1; ii < trajectory_and_weights.get_size(1); ii++)
    {

      trajectory_and_weights(0, ii) = ((gradients(ii)[0]) * GAMMA * 10 * 2 / 1000000 * kspace_scaling + trajectory_and_weights(0, ii - 1)); // mT/m * Hz/G * 10G * 2e-6
      trajectory_and_weights(1, ii) = ((gradients(ii)[1]) * GAMMA * 10 * 2 / 1000000 * kspace_scaling + trajectory_and_weights(1, ii - 1)); // mT/m * Hz/G * 10G * 2e-6
      trajectory_and_weights(2, ii) = zencoding;                                                                                            // Code to gen 3D traj forDavid
      trajectory_and_weights(3, ii) = dcw_sep(ii);

      size_t num = 0;
      if (abs(trajectory_and_weights(0, ii)) > 0.5 || abs(trajectory_and_weights(1, ii)) > 0.5f)
      {
        if (ii == 1)
          GERROR("To Prevent recon failure setting to ±0.5 \n");

        if (trajectory_and_weights(0, ii) > 0.5f)
        {
          trajectory_and_weights(0, ii) = 0.5f;
        }
        else if (trajectory_and_weights(0, ii) < -0.5f)
        {
          trajectory_and_weights(0, ii) = -0.5f;
        }
        if (trajectory_and_weights(1, ii) > 0.5f)
        {
          trajectory_and_weights(1, ii) = 0.5f;
        }
        else if (trajectory_and_weights(1, ii) < -0.5)
        {
          trajectory_and_weights(1, ii) = -0.5f;
        }
        num++;
      }
    }
  }
  else
  {
    auto traj_dcw = nhlbi_toolbox::utils::separate_traj_and_dcw_all<2>(&trajectory_and_weights, 0, header.encoding.front().encodedSpace.matrixSize.z);
    auto traj_sep = std::move(*std::get<0>(traj_dcw).get());
    auto dcw_sep = std::move(*std::get<1>(traj_dcw).get());

    auto gradients = nhlbi_toolbox::utils::traj2grad(traj_sep, kspace_scaling);
    gradients = nhlbi_toolbox::corrections::girf_correct(gradients, girf_kernel, rotation_matrix, head.sample_time_us * 1e-6, 10e-6, 0.85e-6);

    trajectory_and_weights(0, 0) = (gradients(0)[0]) * GAMMA * 10 * 2 / 1000000 * kspace_scaling;
    trajectory_and_weights(1, 0) = (gradients(0)[1]) * GAMMA * 10 * 2 / 1000000 * kspace_scaling;
    trajectory_and_weights(2, 0) = dcw_sep(0);

    for (int ii = 1; ii < trajectory_and_weights.get_size(1); ii++)
    {

      trajectory_and_weights(0, ii) = ((gradients(ii)[0]) * GAMMA * 10 * 2 / 1000000 * kspace_scaling + trajectory_and_weights(0, ii - 1)); // mT/m * Hz/G * 10G * 2e-6
      trajectory_and_weights(1, ii) = ((gradients(ii)[1]) * GAMMA * 10 * 2 / 1000000 * kspace_scaling + trajectory_and_weights(1, ii - 1)); // mT/m * Hz/G * 10G * 2e-6
      trajectory_and_weights(2, ii) = dcw_sep(ii);
      
      if (abs(trajectory_and_weights(0, ii)) > 0.5f || abs(trajectory_and_weights(1, ii)) > 0.5f)
      {
        GERROR("Trajectory outof bounds ±0.5 \n");
        GERROR("To Prevent recon failure setting to ±0.5 \n");
        if (trajectory_and_weights(0, ii) > 0.5f)
          trajectory_and_weights(0, ii) = 0.5f;
        else if (trajectory_and_weights(0, ii) < -0.5f)
          trajectory_and_weights(0, ii) = -0.5f;

        if (trajectory_and_weights(1, ii) > 0.5f)
          trajectory_and_weights(1, ii) = 0.5f;
        else if (trajectory_and_weights(1, ii) < -0.5f)
          trajectory_and_weights(1, ii) = -0.5f;
      }
    }
  }
}
GADGETRON_GADGET_EXPORT(WaveformToTrajectory);