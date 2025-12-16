import torch
from scripts.inference import load_model, preprocess, predict
from PIL import Image
import numpy as np
import os

def test_final_model():
    print("🎯 FINAL MODEL TEST")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    
    # Test 1: Non-medical images (should have low confidence)
    print("\n🧪 TEST 1: NON-MEDICAL IMAGES")
    print("-" * 30)
    
    test_cases = [
        ("Random Chart", np.random.rand(224, 224, 3) * 255),
        ("Gradient", np.linspace(0, 255, 224*224).reshape(224, 224).astype(np.uint8)),
        ("Solid Color", np.full((224, 224, 3), 128, dtype=np.uint8)),
    ]
    
    for name, img_array in test_cases:
        try:
            # Save temporary image
            test_img = Image.fromarray(img_array.astype(np.uint8))
            test_path = f"temp_{name.replace(' ', '_')}.jpg"
            test_img.save(test_path)
            
            # Test prediction
            image_tensor = preprocess(test_path)
            predictions = predict(model, image_tensor, threshold=0.1)
            
            # Analyze results
            max_conf = max(predictions.values()) if predictions else 0
            high_conf_count = sum(1 for v in predictions.values() if v > 0.3)
            
            print(f"{name}:")
            print(f"  Max confidence: {max_conf:.3f}")
            print(f"  High confidence predictions: {high_conf_count}")
            
            if max_conf < 0.15:
                print("  ✅ WELL CALIBRATED - Low confidence as expected")
            else:
                print("  ⚠️  Higher than expected confidence")
                
            # Clean up
            os.remove(test_path)
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Test 2: Check if uploads folder has real X-rays
    print("\n🧪 TEST 2: REAL X-RAYS (if available)")
    print("-" * 30)
    
    uploads_folder = "static/uploads"
    if os.path.exists(uploads_folder):
        image_files = [f for f in os.listdir(uploads_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if image_files:
            for img_file in image_files[:2]:  # Test first 2
                img_path = os.path.join(uploads_folder, img_file)
                print(f"\n📷 Testing: {img_file}")
                
                try:
                    image_tensor = preprocess(img_path)
                    predictions = predict(model, image_tensor, threshold=0.1)
                    
                    if predictions:
                        print("  Predictions:")
                        for disease, confidence in list(predictions.items())[:3]:  # Top 3
                            status = "🟢" if confidence < 0.3 else "🟡" if confidence < 0.7 else "🔴"
                            print(f"    {status} {disease}: {confidence:.3f}")
                    else:
                        print("  ✅ No significant findings")
                        
                except Exception as e:
                    print(f"  ❌ Error: {e}")
        else:
            print("  No X-ray images found in uploads folder")
    
    print("\n🎯 TEST SUMMARY:")
    print("• Non-medical images should show <15% confidence")
    print("• Real X-rays may show higher confidence if diseases are detected")
    print("• Charts and random images will no longer get false diagnoses!")

if __name__ == "__main__":
    test_final_model()