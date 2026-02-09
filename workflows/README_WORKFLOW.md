# Base Workflow Setup

The base workflow JSON (`base_workflow.json`) should be placed in this directory.

## How to export from ComfyUI:

1. Open your workflow in ComfyUI
2. Click the **Save (API Format)** button or use **Developer > Save API Format**
3. Save the file as `workflows/base_workflow.json`

Alternatively, use the standard UI export and the workflow builder will
auto-convert from UI format to API format.

## Node ID Reference

The workflow builder maps these node IDs for parameter injection:

| Node ID | Function | What Gets Injected |
|---------|----------|--------------------|
| 46 | POSITIVE prompt | Character-specific positive prompt |
| 48 | NEGATIVE prompt | Lane-aware negative prompt |
| 49 | Input Parameters | steps, cfg, sampler, scheduler |
| 58 | Width | Resolution width |
| 61 | Height | Resolution height |
| 65 | Batch Size | 1 for master, configurable for vault |
| 73 | Checkpoint | fabricatedXL_v70.safetensors |
| 87 | Seed | Per-image unique seed |
| 96 | KSampler Primary | Main generation pass |
| 59 | KSampler Secondary | Refinement pass |
| 109 | Image Saver Main | Output path + metadata |
| 110 | Image Saver Upscaled | Upscaled output |
| 158 | Load FaceID Image | Character FaceID reference |
| 149 | Load Advanced Image | IP-Adapter Advanced reference |
| 100 | Load Style Image | Style transfer reference |
| 28-32 | Detailer Chain | Hand→Body→NSFW→Face→Eyes |
