# Intel Unnati Industrial Training 2025

## Project Title: DL Streamer Pipeline Setup and Execution on Intel GPU via WSL2

### Overview

This project explores the deployment and execution of a deep learning pipeline using Intel DL Streamer and OpenVINO Toolkit in a WSL2 environment with an Intel integrated GPU (iGPU). It is developed under the Intel Unnati Industrial Training 2025 program.

### About the Author

This project was developed by a highly motivated data science enthusiast with a strong interest in AI, ML, and system-level integration. The contributor has a keen interest in deploying real-time inference solutions, mastering open-source tooling, and troubleshooting complex hardware-software configurations. This project showcases their hands-on learning, resilience, and practical understanding of deploying DL pipelines in constrained environments like WSL2.

### System Configuration

* **Laptop Brand**: Lenovo LOQ
* **Processor**: Intel(R) Core(TM) i5
* **GPU**: Intel(R) UHD Graphics (iGPU)
* **OS**: Windows 11 Home Single Language
* **WSL2 Kernel**: 6.6.87.2-microsoft-standard-WSL2
* **WSL Distro**: Ubuntu 20.04
* **Docker**: Docker Desktop with WSL2 integration

### Objectives

* Set up a deep learning video pipeline using Intel DL Streamer.
* Run the pipeline with GPU acceleration (Intel iGPU).
* Troubleshoot and validate OpenVINO and DL Streamer GPU inference in WSL2.

### Tasks Performed

1. **Environment Setup**

   * Installed Docker Desktop with WSL2 support.
   * Installed required Intel GPU drivers and OpenCL/Level-Zero runtimes.
   * Enabled virtualization features: `VirtualMachinePlatform`, `Microsoft-Windows-Subsystem-Linux`.
   * Installed OpenVINO Toolkit and DL Streamer Docker image.

2. **Validation Commands**

   * Verified OpenCL device via `clinfo`.
   * Checked OpenGL renderer using `glxinfo`.
   * Validated GPU driver version and device detection.

3. **Pipeline Execution**

   * Attempted to run a 4-stream DL Streamer pipeline for person detection and classification using:

     ```bash
     GST_DEBUG=fpsdisplaysink:5 gst-launch-1.0 \
     filesrc location=stream1.mp4 ! decodebin ! \
     gvadetect model=intel/person-detection-retail-0013/... device=GPU ! \
     gvaclassify model=intel/vehicle-attributes-recognition-barrier-0039/... device=GPU ! \
     gvawatermark ! videoconvert ! fpsdisplaysink name=fps1 ...
     ```
9. Execution Steps
# Navigate to the project directory
        cd ~/dlstreamer_project

# Activate the OpenVINO environment
       source openvino-env/bin/activate

# Start the DL Streamer Docker container
     docker run -it --rm \
     --device /dev/dri \
    --privileged \
    --user root \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "$PWD":/home/dlstreamer \
    intel/dlstreamer:latest


# Pipeline Execution Command for 3 Streams 
                 GST_DEBUG=fpsdisplaysink:5 gst-launch-1.0 \
                 filesrc location=stream1.mp4 ! decodebin ! \
               	 gvadetect model=intel/person-detection-retail-0013/FP32/person-detection-retail-0013.xml device=CPU ! \
                 gvaclassify model=intel/vehicle-attributes-recognition-barrier-0039/FP32/vehicle-attributes-recognition-barrier-0039.xml device=CPU ! \
            	 gvawatermark ! videoconvert ! fpsdisplaysink name=fps1 text-overlay=false video-sink=fakesink sync=false \
           	 filesrc location=stream2.mp4 ! decodebin ! \
		gvadetect model=intel/person-detection-retail-0013/FP32/person-detection-retail-0013.xml device=CPU ! \
		gvaclassify model=intel/vehicle-attributes-recognition-barrier-0039/FP32/vehicle-attributes-recognition-barrier-0039.xml device=CPU ! \
		gvawatermark ! videoconvert ! fpsdisplaysink name=fps2 text-overlay=false video-sink=fakesink sync=false \
		filesrc location=stream3.mp4 ! decodebin ! \
		gvadetect model=intel/person-detection-retail-0013/FP32/person-detection-retail-0013.xml device=CPU ! \
		gvaclassify model=intel/vehicle-attributes-recognition-barrier-0039/FP32/vehicle-attributes-recognition-barrier-0039.xml device=CPU ! \
		gvawatermark ! videoconvert ! fpsdisplaysink name=fps3 text-overlay=false video-sink=fakesink sync=false \
		filesrc location=stream4.mp4 ! decodebin ! \
		gvadetect model=intel/person-detection-retail-0013/FP32/person-detection-retail-0013.xml device=CPU ! \
		gvaclassify model=intel/vehicle-attributes-recognition-barrier-0039/FP32/vehicle-attributes-recognition-barrier-0039.xml device=CPU ! \
		gvawatermark ! videoconvert ! fpsdisplaysink name=fps4 text-overlay=false video-sink=fakesink sync=false


4. **Troubleshooting Performed**

   * Configured BIOS settings for hybrid/dGPU modes.
   * Attempted multiple installations of GPU drivers (including Arc & Iris Xe Graphics).
   * Enabled iGPU support via BIOS and Windows Device Manager.
   * Used multiple versions of OpenVINO and Docker containers.
   * Attempted `/dev/dri` mapping to enable GPU access in WSL2.

### Issue Faced

Despite extensive driver configuration, OpenCL verification, and BIOS tuning:

* \`\` was not present in WSL2, blocking DL Streamer’s GPU access.
* DL Streamer pipeline failed to acquire the GPU device via OpenVINO backend.
* Pipeline execution defaulted to CPU fallback mode.

> Note: Due to the `/dev/dri` device path not being exposed inside the WSL2 container, GPU-based inference with DL Streamer could not be validated. This prevented the actual execution of GPU-accelerated workloads within the DL pipeline.

### Notes

* CUDA installation and `nvcc` tests were also explored as alternatives.
* The CUDA toolchain worked successfully in WSL2, but DL Streamer GPU inference via NVIDIA is **not officially supported**.

### Conclusion

Due to the inability to expose `/dev/dri` in WSL2, the DL Streamer pipeline could not be tested on Intel iGPU. The environment was validated, and all necessary dependencies were installed, but device access remained restricted due to current WSL2 limitations or driver configuration constraints.

### Future Work

* Consider testing on native Ubuntu via dual-boot for full GPU access.
* Explore Intel Developer Cloud for GPU workloads.
* Monitor updates to WSL2 for improved iGPU support.
