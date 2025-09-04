#!/usr/bin/env python3
"""
Test script to verify and convert your PyTorch model for Android
"""

import torch
import os

def test_model_file(model_path):
    """Test if the model file can be loaded"""
    print(f"Testing model file: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    try:
        # Try to load the model
        model = torch.jit.load(model_path, map_location='cpu')
        print(f"✅ Model loaded successfully")
        print(f"   Model type: {type(model)}")
        
        # Test with dummy input
        dummy_input = torch.randn(1, 1, 512, 512)
        print(f"   Testing with input shape: {dummy_input.shape}")
        
        with torch.no_grad():
            output = model(dummy_input)
            print(f"✅ Model inference successful")
            print(f"   Output shape: {output.shape}")
            print(f"   Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def convert_model_if_needed():
    """Convert a regular PyTorch model to PyTorch Mobile format"""
    print("\n" + "="*50)
    print("MODEL CONVERSION GUIDE")
    print("="*50)
    
    print("""
If your model is not in PyTorch Mobile format, you need to convert it:

1. Load your original model:
   model = YourModel()
   model.load_state_dict(torch.load('your_model.pth'))
   model.eval()

2. Create dummy input:
   dummy_input = torch.randn(1, 1, 512, 512)

3. Trace the model:
   traced_model = torch.jit.trace(model, dummy_input)

4. Save as PyTorch Mobile:
   traced_model.save('nnunet_xtiny_2_final.ptl')

5. Test the converted model:
   python test_model.py
""")

if __name__ == "__main__":
    model_path = "app/src/main/assets/nnunet_xtiny_2_final.ptl"
    
    print("PyTorch Model Test for Android")
    print("="*40)
    
    if test_model_file(model_path):
        print("\n✅ Your model is ready for Android!")
    else:
        print("\n❌ Model needs to be converted or fixed")
        convert_model_if_needed()
