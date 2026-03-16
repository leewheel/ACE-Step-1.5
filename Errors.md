[Update] Checking for updates...
[Update] Already up to date (7f845ee).

Starting ACE-Step Gradio Web UI...
Server will be available at: http://127.0.0.1:7860

[Environment] Embedded Python not found, checking for uv...
[Environment] Using uv package manager...

Starting ACE-Step Gradio UI...

Loaded configuration from C:\WoWserver\AI\Ace-Step1.5\.env.example (fallback)
Skipping import of cpp extensions due to incompatible torch version 2.7.1+cu128 for torchao version 0.15.0             Please see https://github.com/pytorch/ao/issues/2919 for more info
W0316 13:30:14.369000 51468 Lib\site-packages\torch\distributed\elastic\multiprocessing\redirects.py:29] NOTE: Redirects are currently not supported in Windows or MacOs.
2026-03-16 13:30:15.670 | WARNING  | acestep.training.trainer:<module>:40 - bitsandbytes not installed. Using standard AdamW.
2026-03-16 13:30:19.583 | INFO     | acestep.gpu_config:get_gpu_memory_gb:466 - CUDA GPU detected: NVIDIA GeForce RTX 4060 Laptop GPU (8.0 GB)

============================================================
GPU Configuration Detected:
============================================================
  GPU Memory: 8.00 GB
  Configuration Tier: tier3
  Max Duration (with LM): 480s (8 min)
  Max Duration (without LM): 600s (10 min)
  Max Batch Size (with LM): 2
  Max Batch Size (without LM): 2
  Default LM Init: True
  Available LM Models: ['acestep-5Hz-lm-0.6B']
============================================================

Auto-enabling CPU offload (GPU 8.0GB < 20.0GB threshold)
Output directory: C:/WoWserver/AI/Ace-Step1.5/gradio_outputs
Initializing service from command line...
Initializing DiT model: acestep-v15-turbo on auto...
2026-03-16 13:30:19.657 | INFO     | acestep.core.generation.handler.init_service_loader:_load_main_model_from_checkpoint:161 - [initialize_service] Attempting to load model with attention implementation: sdpa
2026-03-16 13:30:21.262 | INFO     | acestep.core.generation.handler.init_service_loader:_apply_dit_quantization:110 - [initialize_service] DiT quantized with: int8_weight_only
DiT model initialized successfully
Auto-setting init_llm to True based on GPU configuration
Initializing 5Hz LM: acestep-5Hz-lm-0.6B on auto...
2026-03-16 13:30:23.660 | INFO     | acestep.llm_inference:initialize:548 - loading 5Hz LM tokenizer... it may take 80~90s
2026-03-16 13:30:34.333 | INFO     | acestep.llm_inference:initialize:552 - 5Hz LM tokenizer loaded successfully in 10.67 seconds
2026-03-16 13:30:34.333 | INFO     | acestep.llm_inference:initialize:557 - Initializing constrained decoding processor...
2026-03-16 13:30:34.334 | INFO     | acestep.llm_inference:initialize:563 - Setting constrained decoding max_duration to 480s based on GPU config (tier: tier3)
2026-03-16 13:30:34.994 | DEBUG    | acestep.constrained_logits_processor:_precompute_audio_code_tokens:577 - Found 1535 audio code tokens with values outside valid range [0, 63999]
2026-03-16 13:30:36.759 | INFO     | acestep.llm_inference:initialize:571 - Constrained processor initialized in 2.43 seconds
2026-03-16 13:30:36.760 | INFO     | acestep.llm_inference:initialize:598 - flash_attn not installed: disabling CUDA graph capture for nano-vllm (SDPA fallback uses .item() calls in paged-cache decode that are incompatible with CUDA graph capture)
2026-03-16 13:30:36.760 | INFO     | acestep.gpu_config:get_gpu_memory_gb:466 - CUDA GPU detected: NVIDIA GeForce RTX 4060 Laptop GPU (8.0 GB)
2026-03-16 13:30:37.317 | INFO     | acestep.gpu_config:get_lm_gpu_memory_ratio:899 - [get_lm_gpu_memory_ratio] model=0.6B, free=6.93GB, current_usage=1.07GB, lm_target=2.10GB, usable_for_lm=2.10GB, ratio=0.396
2026-03-16 13:30:37.317 | INFO     | acestep.llm_inference:get_gpu_memory_utilization:172 - Adaptive LM memory allocation: model=C:\WoWserver\AI\Ace-Step1.5\checkpoints\acestep-5Hz-lm-0.6B, target=1.2GB, ratio=0.396, total_gpu=8.0GB
2026-03-16 13:30:37.317 | INFO     | acestep.llm_inference:_initialize_5hz_lm_vllm:725 - Initializing 5Hz LM with model: C:\WoWserver\AI\Ace-Step1.5\checkpoints\acestep-5Hz-lm-0.6B, enforce_eager: True, tensor_parallel_size: 1, max_model_len: 2048, gpu_memory_utilization: 0.396
`torch_dtype` is deprecated! Use `dtype` instead!
2026-03-16 13:30:39.383 | INFO     | acestep.llm_inference:initialize:667 - 5Hz LM status message: ❌ Error initializing 5Hz LM: Cannot find a working triton installation. Either the package is not installed or it is too old. More information on installing Triton can be found at: https://github.com/triton-lang/triton

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"


Traceback:
Traceback (most recent call last):
  File "C:\WoWserver\AI\Ace-Step1.5\acestep\llm_inference.py", line 727, in _initialize_5hz_lm_vllm
    self.llm = LLM(
               ^^^^
  File "C:\WoWserver\AI\Ace-Step1.5\.venv\Lib\site-packages\nanovllm\engine\llm_engine.py", line 40, in __init__
    self.model_runner = ModelRunner(config, 0, self.events)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\WoWserver\AI\Ace-Step1.5\.venv\Lib\site-packages\nanovllm\engine\model_runner.py", line 129, in __init__
    self.warmup_model()
  File "C:\WoWserver\AI\Ace-Step1.5\.venv\Lib\site-packages\nanovllm\engine\model_runner.py", line 229, in warmup_model
    self.run(seqs, True)
  File "C:\WoWserver\AI\Ace-Step1.5\.venv\Lib\site-packages\nanovllm\engine\model_runner.py", line 594, in run
    logits = self.run_model(input_ids, positions, is_prefill)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\WoWserver\AI\Ace-Step1.5\.venv\Lib\site-packages\torch\utils\_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "C:\WoWserver\AI\Ace-Step1.5\.venv\Lib\site-packages\nanovllm\engine\model_runner.py", line 423, in run_model
    out = self.model.compute_logits(self.model(input_ids, positions))
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\WoWserver\AI\Ace-Step1.5\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\WoWserver\AI\Ace-Step1.5\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\WoWserver\AI\Ace-Step1.5\.venv\Lib\site-packages\nanovllm\models\qwen3.py", line 224, in forward
    return self.model(input_ids, positions)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\WoWserver\AI\Ace-Step1.5\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\WoWserver\AI\Ace-Step1.5\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\WoWserver\AI\Ace-Step1.5\.venv\Lib\site-packages\nanovllm\models\qwen3.py", line 181, in forward
    hidden_states, residual = layer(positions, hidden_states, residual)
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\WoWserver\AI\Ace-Step1.5\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\WoWserver\AI\Ace-Step1.5\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\WoWserver\AI\Ace-Step1.5\.venv\Lib\site-packages\nanovllm\models\qwen3.py", line 153, in forward
    hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\WoWserver\AI\Ace-Step1.5\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\WoWserver\AI\Ace-Step1.5\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\WoWserver\AI\Ace-Step1.5\.venv\Lib\site-packages\nanovllm\layers\layernorm.py", line 48, in forward
    return self.rms_forward(x)
           ^^^^^^^^^^^^^^^^^^^
  File "C:\WoWserver\AI\Ace-Step1.5\.venv\Lib\site-packages\torch\_dynamo\eval_frame.py", line 663, in _fn
    raise e.remove_dynamo_frames() from None  # see TORCHDYNAMO_VERBOSE=1
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\WoWserver\AI\Ace-Step1.5\.venv\Lib\site-packages\torch\_inductor\scheduler.py", line 3957, in create_backend
    raise TritonMissing(inspect.currentframe())
torch._inductor.exc.TritonMissing: Cannot find a working triton installation. Either the package is not installed or it is too old. More information on installing Triton can be found at: https://github.com/triton-lang/triton

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"


2026-03-16 13:30:39.385 | WARNING  | acestep.llm_inference:initialize:676 - Falling back to PyTorch backend
2026-03-16 13:30:40.263 | INFO     | acestep.llm_inference:_load_pytorch_model:361 - 5Hz LM initialized successfully using PyTorch backend on cuda
5Hz LM initialized successfully
Service initialization completed successfully!
Creating Gradio interface with language: en...
Enabling queue for multi-user support...
Launching server on 127.0.0.1:7860...
* Running on local URL:  http://127.0.0.1:7860
Error launching Gradio: Couldn't start the app because 'http://127.0.0.1:7860/gradio_api/startup-events' failed (code 502). Check your network or proxy settings to ensure localhost is accessible.
Traceback (most recent call last):
  File "C:\WoWserver\AI\Ace-Step1.5\acestep\acestep_v15_pipeline.py", line 669, in main
    demo.launch(
  File "C:\WoWserver\AI\Ace-Step1.5\.venv\Lib\site-packages\gradio\blocks.py", line 2761, in launch
    raise Exception(
Exception: Couldn't start the app because 'http://127.0.0.1:7860/gradio_api/startup-events' failed (code 502). Check your network or proxy settings to ensure localhost is accessible.

[Retry] Online dependency resolution failed, retrying in offline mode...

Loaded configuration from C:\WoWserver\AI\Ace-Step1.5\.env.example (fallback)
Skipping import of cpp extensions due to incompatible torch version 2.7.1+cu128 for torchao version 0.15.0             Please see https://github.com/pytorch/ao/issues/2919 for more info
W0316 13:30:55.422000 116592 Lib\site-packages\torch\distributed\elastic\multiprocessing\redirects.py:29] NOTE: Redirects are currently not supported in Windows or MacOs.
2026-03-16 13:30:56.728 | WARNING  | acestep.training.trainer:<module>:40 - bitsandbytes not installed. Using standard AdamW.
2026-03-16 13:31:00.658 | INFO     | acestep.gpu_config:get_gpu_memory_gb:466 - CUDA GPU detected: NVIDIA GeForce RTX 4060 Laptop GPU (8.0 GB)

============================================================
GPU Configuration Detected:
============================================================
  GPU Memory: 8.00 GB
  Configuration Tier: tier3
  Max Duration (with LM): 480s (8 min)
  Max Duration (without LM): 600s (10 min)
  Max Batch Size (with LM): 2
  Max Batch Size (without LM): 2
  Default LM Init: True
  Available LM Models: ['acestep-5Hz-lm-0.6B']
============================================================

Auto-enabling CPU offload (GPU 8.0GB < 20.0GB threshold)
Output directory: C:/WoWserver/AI/Ace-Step1.5/gradio_outputs
Initializing service from command line...
Initializing DiT model: acestep-v15-turbo on auto...
2026-03-16 13:31:00.737 | INFO     | acestep.core.generation.handler.init_service_loader:_load_main_model_from_checkpoint:161 - [initialize_service] Attempting to load model with attention implementation: sdpa
2026-03-16 13:31:02.345 | INFO     | acestep.core.generation.handler.init_service_loader:_apply_dit_quantization:110 - [initialize_service] DiT quantized with: int8_weight_only
DiT model initialized successfully
Auto-setting init_llm to True based on GPU configuration
Initializing 5Hz LM: acestep-5Hz-lm-0.6B on auto...
2026-03-16 13:31:04.934 | INFO     | acestep.llm_inference:initialize:548 - loading 5Hz LM tokenizer... it may take 80~90s
2026-03-16 13:31:15.456 | INFO     | acestep.llm_inference:initialize:552 - 5Hz LM tokenizer loaded successfully in 10.52 seconds
2026-03-16 13:31:15.456 | INFO     | acestep.llm_inference:initialize:557 - Initializing constrained decoding processor...
2026-03-16 13:31:15.457 | INFO     | acestep.llm_inference:initialize:563 - Setting constrained decoding max_duration to 480s based on GPU config (tier: tier3)
2026-03-16 13:31:16.125 | DEBUG    | acestep.constrained_logits_processor:_precompute_audio_code_tokens:577 - Found 1535 audio code tokens with values outside valid range [0, 63999]
2026-03-16 13:31:17.900 | INFO     | acestep.llm_inference:initialize:571 - Constrained processor initialized in 2.44 seconds
2026-03-16 13:31:17.900 | INFO     | acestep.llm_inference:initialize:598 - flash_attn not installed: disabling CUDA graph capture for nano-vllm (SDPA fallback uses .item() calls in paged-cache decode that are incompatible with CUDA graph capture)
2026-03-16 13:31:17.901 | INFO     | acestep.gpu_config:get_gpu_memory_gb:466 - CUDA GPU detected: NVIDIA GeForce RTX 4060 Laptop GPU (8.0 GB)
2026-03-16 13:31:18.486 | INFO     | acestep.gpu_config:get_lm_gpu_memory_ratio:899 - [get_lm_gpu_memory_ratio] model=0.6B, free=6.93GB, current_usage=1.07GB, lm_target=2.10GB, usable_for_lm=2.10GB, ratio=0.396
2026-03-16 13:31:18.486 | INFO     | acestep.llm_inference:get_gpu_memory_utilization:172 - Adaptive LM memory allocation: model=C:\WoWserver\AI\Ace-Step1.5\checkpoints\acestep-5Hz-lm-0.6B, target=1.2GB, ratio=0.396, total_gpu=8.0GB
2026-03-16 13:31:18.498 | INFO     | acestep.llm_inference:_initialize_5hz_lm_vllm:725 - Initializing 5Hz LM with model: C:\WoWserver\AI\Ace-Step1.5\checkpoints\acestep-5Hz-lm-0.6B, enforce_eager: True, tensor_parallel_size: 1, max_model_len: 2048, gpu_memory_utilization: 0.396
`torch_dtype` is deprecated! Use `dtype` instead!
2026-03-16 13:31:20.576 | INFO     | acestep.llm_inference:initialize:667 - 5Hz LM status message: ❌ Error initializing 5Hz LM: Cannot find a working triton installation. Either the package is not installed or it is too old. More information on installing Triton can be found at: https://github.com/triton-lang/triton

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"


Traceback:
Traceback (most recent call last):
  File "C:\WoWserver\AI\Ace-Step1.5\acestep\llm_inference.py", line 727, in _initialize_5hz_lm_vllm
    self.llm = LLM(
               ^^^^
  File "C:\WoWserver\AI\Ace-Step1.5\.venv\Lib\site-packages\nanovllm\engine\llm_engine.py", line 40, in __init__
    self.model_runner = ModelRunner(config, 0, self.events)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\WoWserver\AI\Ace-Step1.5\.venv\Lib\site-packages\nanovllm\engine\model_runner.py", line 129, in __init__
    self.warmup_model()
  File "C:\WoWserver\AI\Ace-Step1.5\.venv\Lib\site-packages\nanovllm\engine\model_runner.py", line 229, in warmup_model
    self.run(seqs, True)
  File "C:\WoWserver\AI\Ace-Step1.5\.venv\Lib\site-packages\nanovllm\engine\model_runner.py", line 594, in run
    logits = self.run_model(input_ids, positions, is_prefill)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\WoWserver\AI\Ace-Step1.5\.venv\Lib\site-packages\torch\utils\_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "C:\WoWserver\AI\Ace-Step1.5\.venv\Lib\site-packages\nanovllm\engine\model_runner.py", line 423, in run_model
    out = self.model.compute_logits(self.model(input_ids, positions))
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\WoWserver\AI\Ace-Step1.5\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\WoWserver\AI\Ace-Step1.5\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\WoWserver\AI\Ace-Step1.5\.venv\Lib\site-packages\nanovllm\models\qwen3.py", line 224, in forward
    return self.model(input_ids, positions)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\WoWserver\AI\Ace-Step1.5\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\WoWserver\AI\Ace-Step1.5\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\WoWserver\AI\Ace-Step1.5\.venv\Lib\site-packages\nanovllm\models\qwen3.py", line 181, in forward
    hidden_states, residual = layer(positions, hidden_states, residual)
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\WoWserver\AI\Ace-Step1.5\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\WoWserver\AI\Ace-Step1.5\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\WoWserver\AI\Ace-Step1.5\.venv\Lib\site-packages\nanovllm\models\qwen3.py", line 153, in forward
    hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\WoWserver\AI\Ace-Step1.5\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\WoWserver\AI\Ace-Step1.5\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\WoWserver\AI\Ace-Step1.5\.venv\Lib\site-packages\nanovllm\layers\layernorm.py", line 48, in forward
    return self.rms_forward(x)
           ^^^^^^^^^^^^^^^^^^^
  File "C:\WoWserver\AI\Ace-Step1.5\.venv\Lib\site-packages\torch\_dynamo\eval_frame.py", line 663, in _fn
    raise e.remove_dynamo_frames() from None  # see TORCHDYNAMO_VERBOSE=1
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\WoWserver\AI\Ace-Step1.5\.venv\Lib\site-packages\torch\_inductor\scheduler.py", line 3957, in create_backend
    raise TritonMissing(inspect.currentframe())
torch._inductor.exc.TritonMissing: Cannot find a working triton installation. Either the package is not installed or it is too old. More information on installing Triton can be found at: https://github.com/triton-lang/triton

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"


2026-03-16 13:31:20.576 | WARNING  | acestep.llm_inference:initialize:676 - Falling back to PyTorch backend
2026-03-16 13:31:21.457 | INFO     | acestep.llm_inference:_load_pytorch_model:361 - 5Hz LM initialized successfully using PyTorch backend on cuda
5Hz LM initialized successfully
Service initialization completed successfully!
Creating Gradio interface with language: en...
Enabling queue for multi-user support...
Launching server on 127.0.0.1:7860...
* Running on local URL:  http://127.0.0.1:7860
Error launching Gradio: Couldn't start the app because 'http://127.0.0.1:7860/gradio_api/startup-events' failed (code 502). Check your network or proxy settings to ensure localhost is accessible.
Traceback (most recent call last):
  File "C:\WoWserver\AI\Ace-Step1.5\acestep\acestep_v15_pipeline.py", line 669, in main
    demo.launch(
  File "C:\WoWserver\AI\Ace-Step1.5\.venv\Lib\site-packages\gradio\blocks.py", line 2761, in launch
    raise Exception(
Exception: Couldn't start the app because 'http://127.0.0.1:7860/gradio_api/startup-events' failed (code 502). Check your network or proxy settings to ensure localhost is accessible.

========================================
[Error] Failed to start ACE-Step
========================================

Both online and offline modes failed.
Please check:
  1. Your internet connection (for first-time setup)
  2. If dependencies were previously installed (offline mode requires a prior successful install)
  3. Try running: uv sync --offline
