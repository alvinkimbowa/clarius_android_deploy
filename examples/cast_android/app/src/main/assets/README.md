# Assets Directory

Place your PyTorch Mobile segmentation model file here.

## Required File

- **nnunet_xtiny_2_final.ptl** - Your converted PyTorch Mobile model

## Model Requirements

- Input: Single channel grayscale image (1x512x512)
- Input range: Normalized pixel values [0, 1]
- Output: Binary segmentation mask (1x512x512)
- Format: PyTorch Mobile (.ptl)

## Conversion Command

```bash
# Run the test script to convert your model
python test_model_conversion.py

# Or manually convert using PyTorch
python -c "
import torch
model = YourModel()
model.load_state_dict(torch.load('your_model.pth'))
model.eval()
dummy_input = torch.randn(1, 1, 512, 512)
traced_model = torch.jit.trace(model, dummy_input)
traced_model.save('nnunet_xtiny_2_final.ptl')
"
```

## File Naming

Make sure the filename matches the `MODEL_ASSET_NAME` constant in `CastService.java`:

```java
private static final String MODEL_ASSET_NAME = "nnunet_xtiny_2_final.ptl";
```

