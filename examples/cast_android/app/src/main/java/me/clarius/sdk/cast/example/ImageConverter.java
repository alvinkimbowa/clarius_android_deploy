package me.clarius.sdk.cast.example;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import java.nio.ByteBuffer;
import java.util.concurrent.Executor;

import me.clarius.sdk.ImageFormat;
import me.clarius.sdk.ProcessedImageInfo;

/**
 * Convert images in a separate thread to avoid blocking the producer (the SDK)
 */

public class ImageConverter {
    private static final String TAG = "ImageConverter";
    
    private final Executor executor;
    private final Callback callback;
    private final UltrasoundModelProcessor modelProcessor;

    ImageConverter(Executor executor, Callback callback, UltrasoundModelProcessor modelProcessor) {
        this.executor = executor;
        this.callback = callback;
        this.modelProcessor = modelProcessor;
    }



    public void convertImage(ByteBuffer buffer, ProcessedImageInfo info) {
        executor.execute(() -> {
            try {
                Log.d(TAG, "Starting image conversion - buffer size: " + buffer.capacity() + ", image size: " + info.imageSize);
                Bitmap bitmap = convert(buffer, info);
                Log.d(TAG, "Bitmap created - width: " + bitmap.getWidth() + ", height: " + bitmap.getHeight());
                
                // Process through segmentation model if available (always call processImage to trigger lazy load)
                Log.d(TAG, "Checking model processor - modelProcessor: " + (modelProcessor != null) + ", isModelLoaded: " + (modelProcessor != null ? modelProcessor.isModelLoaded() : "N/A"));
                
                if (modelProcessor != null) {
                    Log.d(TAG, "Processing image through segmentation model");
                    try {
                        Bitmap segmentedBitmap = modelProcessor.processImage(bitmap);
                        if (segmentedBitmap != null && segmentedBitmap != bitmap) {
                            Log.d(TAG, "Segmentation processing completed successfully");
                            // Only recycle the original bitmap if we got a different one back
                            bitmap.recycle();
                            bitmap = segmentedBitmap;
                        } else if (segmentedBitmap == bitmap) {
                            Log.d(TAG, "Segmentation processing returned original bitmap (no changes or skipped)");
                            // Don't recycle since it's the same bitmap
                        } else {
                            Log.w(TAG, "Segmentation processing returned null bitmap, using original");
                            // Keep the original bitmap if segmentation returns null
                        }
                    } catch (Exception e) {
                        Log.e(TAG, "Error processing image with segmentation model", e);
                        // Continue with original image if segmentation fails
                    }
                } else {
                    Log.w(TAG, "Model processor is null - skipping segmentation");
                }

                callback.onResult(bitmap, info.tm);
            } catch (Exception e) {
                callback.onError(e);
            }
        });
    }

    private Bitmap convert(ByteBuffer buffer, ProcessedImageInfo info) {
        boolean isCompressed = info.format != ImageFormat.Uncompressed;
        Bitmap bitmap;
        if (isCompressed) {
            if (buffer.hasArray()) {
                byte[] bytes = buffer.array();
                int offset = buffer.arrayOffset();
                int length = info.imageSize;
                assert offset + length < bytes.length;
                bitmap = BitmapFactory.decodeByteArray(bytes, offset, length);
            } else {
                byte[] bytes = new byte[buffer.capacity()];
                buffer.get(bytes);
                bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.length);
            }
        } else {
            bitmap = Bitmap.createBitmap(info.width, info.height, Bitmap.Config.ARGB_8888);
            bitmap.copyPixelsFromBuffer(buffer);
        }
        if (bitmap == null)
            throw new AssertionError("bad image data");
        return bitmap;
    }

    interface Callback {
        void onResult(Bitmap bitmap, long timestamp);

        void onError(Exception e);
    }
}
