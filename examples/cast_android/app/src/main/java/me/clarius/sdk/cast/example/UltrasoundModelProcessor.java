package me.clarius.sdk.cast.example;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.Log;

import org.pytorch.executorch.EValue;
import org.pytorch.executorch.Module;
import org.pytorch.executorch.Tensor;

// Import Java classes for timing analysis
import me.clarius.sdk.cast.example.TimingAnalyzer;
import me.clarius.sdk.cast.example.TimingAnalyzer.TimingAnalysis;
import java.io.File;
import java.util.Arrays;
import java.util.Set;
import java.util.stream.Collectors;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

public class UltrasoundModelProcessor {
    // private static final String MODEL_ASSET_NAME = "nnunet_final.ptl";
    // private static final String MODEL_ASSET_NAME = "nnunet_xtiny_2_final.pte";
    private static final String MODEL_ASSET_NAME = "nnunet_xtiny_4_final.pte";
    private static final String DEBUG_IMAGE_DIR = "/storage/emulated/0/Download/clarius_debug/images/";
    private static final boolean SAVE_DEBUG_IMAGES = false;
    private static final boolean ENABLE_TIMING_ANALYSIS = false;
    private static int debugImageCounter = 0;
    private static final int SAVE_DEBUG_INTERVAL = 10;
    // Model Configuration - matches your Python example
    private static final int MODEL_INPUT_WIDTH = 256;
    private static final int MODEL_INPUT_HEIGHT = 256;
    // Overlay Configuration
    private static final int MASK_COLOR = Color.argb(128, 255, 255, 0); // Semi-transparent yellow
    private static final String TAG = "UltrasoundModelProcessor";
    private final Context context;
    private String modelPath;
    private Module model;
    private boolean modelLoaded = false;
    private final Object modelLock = new Object();
    private final TimingAnalyzer timingAnalyzer;

    public UltrasoundModelProcessor(Context context) {
        this.context = context;
        this.timingAnalyzer = ENABLE_TIMING_ANALYSIS ? new TimingAnalyzer(context, MODEL_ASSET_NAME) : null;
    }

    // Class to store crop coordinates for later padding
    private static class CropCoordinates {
        final int left, top, right, bottom;
        final int originalWidth, originalHeight;
        
        CropCoordinates(int left, int top, int right, int bottom, int originalWidth, int originalHeight) {
            this.left = left;
            this.top = top;
            this.right = right;
            this.bottom = bottom;
            this.originalWidth = originalWidth;
            this.originalHeight = originalHeight;
        }
    }
    
    /**
     * Efficiently crop image to remove zero padding and return crop coordinates
     * @param bitmap Input bitmap (800x800 with zero padding)
     * @return Array: [croppedBitmap, cropCoordinates]
     */
    private Object[] cropImageContent(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        
        // Convert to pixel array for efficient processing
        int[] pixels = new int[width * height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);
        
        // Find top boundary (first non-zero row)
        int top = 0;
        for (int y = 0; y < height; y++) {
            boolean hasNonZero = false;
            for (int x = 0; x < width; x++) {
                if (pixels[y * width + x] != 0) {
                    hasNonZero = true;
                    break;
                }
            }
            if (hasNonZero) {
                top = y;
                break;
            }
        }
        
        // Find bottom boundary (last non-zero row)
        int bottom = height - 1;
        for (int y = height - 1; y >= 0; y--) {
            boolean hasNonZero = false;
            for (int x = 0; x < width; x++) {
                if (pixels[y * width + x] != 0) {
                    hasNonZero = true;
                    break;
                }
            }
            if (hasNonZero) {
                bottom = y;
                break;
            }
        }
        
        // Find left boundary (first non-zero column)
        int left = 0;
        for (int x = 0; x < width; x++) {
            boolean hasNonZero = false;
            for (int y = top; y <= bottom; y++) {
                if (pixels[y * width + x] != 0) {
                    hasNonZero = true;
                    break;
                }
            }
            if (hasNonZero) {
                left = x;
                break;
            }
        }
        
        // Find right boundary (last non-zero column)
        int right = width - 1;
        for (int x = width - 1; x >= 0; x--) {
            boolean hasNonZero = false;
            for (int y = top; y <= bottom; y++) {
                if (pixels[y * width + x] != 0) {
                    hasNonZero = true;
                    break;
                }
            }
            if (hasNonZero) {
                right = x;
                break;
            }
        }
        
        // Create cropped bitmap
        int cropWidth = right - left + 1;
        int cropHeight = bottom - top + 1;
        Bitmap croppedBitmap = Bitmap.createBitmap(bitmap, left, top, cropWidth, cropHeight);
        
        // Store crop coordinates
        CropCoordinates cropCoords = new CropCoordinates(left, top, right, bottom, width, height);
        
        return new Object[]{croppedBitmap, cropCoords};
    }
    
    /**
     * Pad segmentation mask back to original image size
     * @param maskBitmap Cropped segmentation mask
     * @param cropCoords Original crop coordinates
     * @return Padded mask matching original image size
     */
    private Bitmap padMaskToOriginalSize(Bitmap maskBitmap, CropCoordinates cropCoords) {
        // Create bitmap with original dimensions
        Bitmap paddedMask = Bitmap.createBitmap(cropCoords.originalWidth, cropCoords.originalHeight, Bitmap.Config.ARGB_8888);
        
        // Fill with transparent (zero) pixels
        paddedMask.eraseColor(Color.TRANSPARENT);
        
        // Draw the cropped mask at the original position
        Canvas canvas = new Canvas(paddedMask);
        canvas.drawBitmap(maskBitmap, cropCoords.left, cropCoords.top, null);
        
        return paddedMask;
    }
    
    /**
     * Extract the largest connected component from segmentation mask
     * @param maskBitmap Input segmentation mask
     * @return Mask with only the largest connected component
     */
    private Bitmap extractLargestComponent(Bitmap maskBitmap) {
        int width = maskBitmap.getWidth();
        int height = maskBitmap.getHeight();
        
        // Convert to pixel array
        int[] pixels = new int[width * height];
        maskBitmap.getPixels(pixels, 0, width, 0, 0, width, height);
        
        // Create visited array and component labels
        boolean[] visited = new boolean[width * height];
        int[] componentLabels = new int[width * height];
        int currentLabel = 1;
        int maxComponentSize = 0;
        int maxComponentLabel = 0;
        
        // Find all connected components using flood fill
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int index = y * width + x;
                
                // If pixel is part of segmentation (not transparent) and not visited
                if (pixels[index] != Color.TRANSPARENT && !visited[index]) {
                    int componentSize = floodFill(pixels, visited, componentLabels, 
                                                x, y, width, height, currentLabel);
                    
                    // Track largest component
                    if (componentSize > maxComponentSize) {
                        maxComponentSize = componentSize;
                        maxComponentLabel = currentLabel;
                    }
                    currentLabel++;
                }
            }
        }
        
        // Create new mask with only the largest component
        int[] resultPixels = new int[width * height];
        for (int i = 0; i < width * height; i++) {
            if (componentLabels[i] == maxComponentLabel) {
                resultPixels[i] = pixels[i]; // Keep original color
            } else {
                resultPixels[i] = Color.TRANSPARENT; // Remove other components
            }
        }
        
        // Create result bitmap
        Bitmap resultBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        resultBitmap.setPixels(resultPixels, 0, width, 0, 0, width, height);
        
        return resultBitmap;
    }
    
    /**
     * Flood fill algorithm to find connected component
     * @return Size of the connected component
     */
    private int floodFill(int[] pixels, boolean[] visited, int[] componentLabels,
                         int startX, int startY, int width, int height, int label) {
        int componentSize = 0;
        int targetColor = pixels[startY * width + startX];
        
        // Use stack for iterative flood fill (more memory efficient than recursion)
        java.util.Stack<int[]> stack = new java.util.Stack<>();
        stack.push(new int[]{startX, startY});
        
        while (!stack.isEmpty()) {
            int[] point = stack.pop();
            int x = point[0];
            int y = point[1];
            int index = y * width + x;
            
            // Check bounds and conditions
            if (x < 0 || x >= width || y < 0 || y >= height || 
                visited[index] || pixels[index] != targetColor) {
                continue;
            }
            
            // Mark as visited and label
            visited[index] = true;
            componentLabels[index] = label;
            componentSize++;
            
            // Add neighbors to stack (4-connected)
            stack.push(new int[]{x + 1, y});
            stack.push(new int[]{x - 1, y});
            stack.push(new int[]{x, y + 1});
            stack.push(new int[]{x, y - 1});
        }
        
        return componentSize;
    }
    
    public Bitmap processImage(Bitmap originalBitmap) {
        if (!modelLoaded) {
            try {
                loadModel();
            } catch (Exception e) {
                Log.e(TAG, "Error loading model", e);
                return originalBitmap;
            }
        }

        if (model == null) {
            Log.w(TAG, "Model is null during inference, returning original image");
            return originalBitmap.copy(originalBitmap.getConfig(), true);
        }

        debugImageCounter++;
        // Save a sample original bitmap to the storage. Overwrite if it already exists
        if (debugImageCounter % SAVE_DEBUG_INTERVAL == 0) {
            saveDebugImage(originalBitmap, "original");
        }

         // 1. Crop image to remove zero padding and remember coordinates
        Log.d(TAG, "Starting image processing - input size: " + originalBitmap.getWidth() + "x" + originalBitmap.getHeight());
        long startTime = System.currentTimeMillis();
        
        // Crop the image to remove zero padding
        Object[] cropResult = cropImageContent(originalBitmap);
        Bitmap croppedBitmap = (Bitmap) cropResult[0];
        CropCoordinates cropCoords = (CropCoordinates) cropResult[1];
        
        // 2. Resize cropped image for model input
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(croppedBitmap, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT, true);
        Tensor inputTensor = bitmapToDynamicZScoreTensorOptimized(resizedBitmap, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT);
        if (debugImageCounter % SAVE_DEBUG_INTERVAL == 0) {
            saveDebugImage(resizedBitmap, "input");
        }
        Log.d(TAG, "Input tensor shape: " + java.util.Arrays.toString(inputTensor.shape()));
        
        // 2. Model inference using ExecuTorch API
        Log.d(TAG, "Running model inference (ExecuTorch)...");
        long inferenceStartTime = System.currentTimeMillis();
        EValue[] outputs;
        Tensor outputTensor;
        synchronized (modelLock) {
            outputs = model.forward(EValue.from(inputTensor));
            outputTensor = outputs[0].toTensor();
        }
        
        // 3. Post-process the output
        long postProcessingStartTime = System.currentTimeMillis();
        
        long[] outputShape = outputTensor.shape();
        Log.d(TAG, "Model output tensor shape: " + java.util.Arrays.toString(outputShape));

        int[] labels = getPredictionLabels(outputTensor);
        if (labels == null) {
            Log.e(TAG, "Failed to extract prediction labels, returning original image");
            return originalBitmap.copy(originalBitmap.getConfig(), true);
        }        
        // Check if all predictions are zeros (model corruption detection)
        // TODO: Fix bug
        // The app currently has a bug where proceeding segmentations become all zeros
        // after an all zero prediction.
        boolean allZeros = true;
        for (int label : labels) {
            if (label != 0) {
                allZeros = false;
                break;
            }
        }
        if (allZeros) {
            Log.w(TAG, "Model output is all zeros, reloading model");
            try {
                loadModel();
            } catch (Exception e) {
                Log.e(TAG, "Error reloading model", e);
            }
        }
        
        // Convert labels -> mask Bitmap (label 0 -> transparent)
        Bitmap maskBitmap = maskFromArgmaxOutput(labels, outputShape);
        
        // Extract largest connected component to remove noise
        Bitmap cleanedMaskBitmap = extractLargestComponent(maskBitmap);
        
        // Scale mask to cropped image size first
        Bitmap scaledMaskBitmap = Bitmap.createScaledBitmap(cleanedMaskBitmap, croppedBitmap.getWidth(), croppedBitmap.getHeight(), true);
        
        // Pad the mask back to original image size using crop coordinates
        Bitmap paddedMaskBitmap = padMaskToOriginalSize(scaledMaskBitmap, cropCoords);
        
        // Overlay the padded mask on the original image
        Bitmap finalBitmap = overlaySegmentation(originalBitmap, paddedMaskBitmap);
        if (debugImageCounter % SAVE_DEBUG_INTERVAL == 0) {
            saveDebugImage(finalBitmap, "output");
        }
        long endTime = System.currentTimeMillis();
        Log.d(TAG, "Model forward completed");

        // Calculate timing values
        long prepTime = inferenceStartTime - startTime;
        long inferenceTime = postProcessingStartTime - inferenceStartTime;
        long postProcessingTime = endTime - postProcessingStartTime;
        long totalTime = endTime - startTime;
        // Log output tensor information
        Log.d(TAG, "Prep time: " + prepTime);
        Log.d(TAG, "Inference time: " + inferenceTime);
        Log.d(TAG, "Post processing time: " + postProcessingTime);
        Log.d(TAG, "Total time: " + totalTime);

        // Record timing data (only in debug builds)
        if (ENABLE_TIMING_ANALYSIS && timingAnalyzer != null) {
            timingAnalyzer.recordTiming(prepTime, inferenceTime, postProcessingTime, totalTime);
            
            // Save CSV files periodically (every 100 frames)
            if (debugImageCounter % 100 == 0) {
                timingAnalyzer.saveToCSV();
            }
        }

        return finalBitmap;
    }

    private void loadModel() throws Exception {
        Log.i(TAG, "Lazily loading model: " + MODEL_ASSET_NAME);
        
        // First check if the asset exists
        try {
            context.getAssets().open(MODEL_ASSET_NAME).close();
            Log.i(TAG, "Model file exists in assets: " + MODEL_ASSET_NAME);
        } catch (IOException e) {
            Log.e(TAG, "Model file not found in assets: " + MODEL_ASSET_NAME, e);
            throw e;
        }
        
        // Copy to file path
        Log.i(TAG, "Loading model from assets...");
        modelPath = assetFilePath(context, MODEL_ASSET_NAME);
        Log.i(TAG, "Model path: " + modelPath);
        
        // Check if the copied file exists
        java.io.File modelFile = new java.io.File(modelPath);
        if (!modelFile.exists()) {
            Log.e(TAG, "Model file does not exist at path: " + modelPath);
            throw new IOException("Model file does not exist");
        }
        Log.i(TAG, "Model file exists at path, size: " + modelFile.length() + " bytes");
        
        // Load the model (thread-safe)
        Log.i(TAG, "Attempting to load model with Module.load()...");
        model = Module.load(modelPath);
        modelLoaded = true;
        Log.i(TAG, "Module.load() completed successfully");
        Log.i(TAG, "ExecuTorch model loaded successfully: " + MODEL_ASSET_NAME);

        // Log model information for debugging (no dummy inference)
        logModelInfo(MODEL_ASSET_NAME, modelFile);
    }

    // Optimized: Build tensor [1,1,H,W] from Bitmap with grayscale conversion and z-score normalization
    private Tensor bitmapToDynamicZScoreTensorOptimized(Bitmap src, int modelW, int modelH) {
        int[] pixels = new int[modelW * modelH];
        src.getPixels(pixels, 0, modelW, 0, 0, modelW, modelH);
        float[] buf = new float[modelW * modelH];
        
        // Single pass: convert to grayscale and calculate mean
        float sum = 0f;
        for (int i = 0; i < pixels.length; i++) {
            int c = pixels[i];
            int r = (c >> 16) & 0xFF;
            int g = (c >> 8) & 0xFF;
            int b = c & 0xFF;
            float lum = 0.299f * r + 0.587f * g + 0.114f * b; // Keep in 0-255 range
            buf[i] = lum;
            sum += lum;
        }
        float mean = sum / pixels.length;
        
        // Second pass: calculate std and apply z-score normalization
        float sumSquaredDiff = 0f;
        for (int i = 0; i < pixels.length; i++) {
            float diff = buf[i] - mean;
            sumSquaredDiff += diff * diff;
        }
        float std = (float) Math.sqrt(sumSquaredDiff / pixels.length);
        
        // Apply z-score normalization in place
        for (int i = 0; i < pixels.length; i++) {
            buf[i] = (buf[i] - mean) / std;
        }
        
        return Tensor.fromBlob(buf, new long[]{1, 1, modelH, modelW});
    }

    // Convert labels -> mask Bitmap (label 0 -> transparent)
    private Bitmap maskFromArgmaxOutput(int[] labels, long[] shape) {
        int outH, outW;
        if (shape.length == 4) {
            outH = (int) shape[2]; // [1,1,H,W] or [1,C,H,W] after argmax should be [1, H, W]
            outW = (int) shape[3];
        } else if (shape.length == 3) {
            outH = (int) shape[1]; // [1,H,W]
            outW = (int) shape[2];
        } else {
            throw new RuntimeException("Unexpected output shape: " + java.util.Arrays.toString(shape));
        }
        
        int[] pixels = new int[outH * outW];
        // Map any non-zero label -> MASK_COLOR (keeps it simple). Change mapping if you want per-class colors.
        for (int i = 0; i < labels.length; i++) {
            int label = labels[i];
            pixels[i] = (label == 0) ? Color.TRANSPARENT : MASK_COLOR;
        }
        Bitmap mask = Bitmap.createBitmap(outW, outH, Bitmap.Config.ARGB_8888);
        mask.setPixels(pixels, 0, outW, 0, 0, outW, outH);
        return mask;
    }

    private Bitmap overlaySegmentation(Bitmap originalBitmap, Bitmap maskBitmap) {
        // Create a copy of the original bitmap to draw on
        Bitmap resultBitmap = originalBitmap.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(resultBitmap);
        
        // Draw the segmentation mask overlay
        Paint paint = new Paint();
        paint.setAlpha(128); // Semi-transparent
        
        canvas.drawBitmap(maskBitmap, 0, 0, paint);
        
        return resultBitmap;
    }

    // Helper method to extract prediction labels for analysis
    private int[] getPredictionLabels(org.pytorch.executorch.Tensor outputTensor) {
        try {
            // Try to get as float array first (in case it's actually float)
            float[] floatData = outputTensor.getDataAsFloatArray();
            int[] labels = new int[floatData.length];
            for (int i = 0; i < floatData.length; i++) {
                labels[i] = (int) floatData[i];
            }
            return labels;
        } catch (IllegalStateException e) {
            // If that fails, get as long array (int64) and convert to int
            try {
                long[] longData = outputTensor.getDataAsLongArray();
                int[] labels = new int[longData.length];
                for (int i = 0; i < longData.length; i++) {
                    labels[i] = (int) longData[i];
                }
                return labels;
            } catch (Exception e2) {
                Log.e(TAG, "Failed to extract prediction labels", e2);
                return null;
            }
        }
    }

    private static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }
        try (InputStream is = context.getAssets().open(assetName);
             OutputStream os = new FileOutputStream(file)) {
            byte[] buffer = new byte[64 * 1024]; // Increased buffer size to 64KB for better performance
            int read;
            while ((read = is.read(buffer)) != -1) {
                os.write(buffer, 0, read);
            }
            os.flush();
        }
        return file.getAbsolutePath();
    }

    private void logModelInfo(String modelName, File modelFile) {
        Log.d(TAG, "=== MODEL INFO ===");
        Log.d(TAG, "Model name: " + modelName);
        Log.d(TAG, "Model file size: " + modelFile.length() + " bytes");
        Log.d(TAG, "Model file path: " + modelFile.getAbsolutePath());
        Log.d(TAG, "==================");
    }

    private void saveDebugImage(Bitmap bitmap, String stage) {
        if (!SAVE_DEBUG_IMAGES || bitmap == null) {
            return;
        }

        try {
            // Create debug directory if it doesn't exist
            File debugDir = new File(DEBUG_IMAGE_DIR);
            if (!debugDir.exists()) {
                debugDir.mkdirs();
            }

            // Generate filename with timestamp and counter
            SimpleDateFormat sdf = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault());
            String timestamp = sdf.format(new Date());
            String filename = String.format("%s_%03d_%s.png", timestamp, debugImageCounter, stage);

            // Save the bitmap
            File imageFile = new File(debugDir, filename);
            FileOutputStream fos = new FileOutputStream(imageFile);
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, fos);
            fos.close();

            Log.d(TAG, "Debug image saved: " + imageFile.getAbsolutePath() +
                    " (size: " + bitmap.getWidth() + "x" + bitmap.getHeight() + ")");

        } catch (Exception e) {
            Log.e(TAG, "Failed to save debug image: " + stage, e);
        }
    }
    
    /**
     * Cleanup method to save final timing results
     */
    public void cleanup() {
        if (ENABLE_TIMING_ANALYSIS && timingAnalyzer != null) {
            timingAnalyzer.saveToCSV();
            Log.i(TAG, "Final timing results saved to CSV");
        }
    }

}
