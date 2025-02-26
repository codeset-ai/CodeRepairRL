# TTC - Test Time Compute for Program Repair

## GPU Precision Auto-Detection

The training configuration now automatically detects the GPU architecture and sets the appropriate precision settings:

- For Ampere (SM 8.0) and newer GPUs (e.g., A100, A6000, RTX 3000/4000 series), BF16 precision will be used
- For Pascal (SM 6.0) to Turing (SM 7.5) GPUs (e.g., GTX 1000 series, RTX 2000 series), FP16 precision will be used

