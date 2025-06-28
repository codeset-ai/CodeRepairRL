#!/bin/bash
#SBATCH --job-name=crrl-test
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err
#SBATCH --gpus 1
#SBATCH --time=0:30:00
#SBATCH -C "thin"

echo "CUDA availability:"
apptainer exec --nv crrl.sif python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
apptainer exec --nv crrl.sif python3 -c "import torch; print(f'CUDA device count: {torch.cuda.device_count()}')"
echo "Flash-attn version:"
apptainer exec --nv crrl.sif python3 -c "import flash_attn_cuda; print(f'Flash-attn version: {flash_attn_cuda.__version__}')"

# Assuming device 0 is used if CUDA_VISIBLE_DEVICES is not set or only 1 GPU is available
apptainer exec --nv crrl.sif python3 -c "import torch; print(f'CUDA device name: {torch.cuda.get_device_name(0)}')"

apptainer exec --nv crrl.sif python3 -c "from transformers import AutoModelForCausalLM; import torch; model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-8B', torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2'); print(model)"

echo "\nRunning test job: small repair task on stack dataset..."
apptainer exec --nv crrl.sif \
    python3 -m src.train_grpo \
    run=repair \
    model=small_qwen \
    grpo.use_vllm=false \
    grpo.max_steps=2 \
    grpo.logging_steps=1 \
    grpo.per_device_train_batch_size=1 \
    grpo.gradient_accumulation_steps=1

echo "\nTest job finished."