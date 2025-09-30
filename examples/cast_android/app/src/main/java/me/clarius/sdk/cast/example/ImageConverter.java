package me.clarius.sdk.cast.example;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import java.nio.ByteBuffer;
import java.util.concurrent.Executor;
import java.util.concurrent.atomic.AtomicBoolean;

import me.clarius.sdk.ImageFormat;
import me.clarius.sdk.ProcessedImageInfo;
import me.clarius.sdk.cast.example.UltrasoundModelProcessor;

/**
 * Convert images in a separate thread to avoid blocking the producer (the SDK)
 */

public class ImageConverter {
    private static final String TAG = "ImageConverter";
    
    private final Executor executor;
    private final Callback callback;
    private final UltrasoundModelProcessor modelProcessor;
    private final AtomicBoolean inferBusy = new AtomicBoolean(false);
    
    ImageConverter(Context context, Executor executor, Callback callback) {
        this.executor = executor;
        this.callback = callback;
        this.modelProcessor = new UltrasoundModelProcessor(context);
    }



    public void convertImage(ByteBuffer buffer, ProcessedImageInfo info) {
        // Non-blocking policy: if an inference is running, drop this frame
        if (!inferBusy.compareAndSet(false, true)) {
            Log.d(TAG, "Dropping frame: inference in progress");
            return;
        }

        executor.execute(() -> {
            try {
                Log.d(TAG, "Starting image conversion - buffer size: " + buffer.capacity() + ", image size: " + info.imageSize);
                Bitmap bitmap = convert(buffer, info);
                Bitmap processedBitmap = modelProcessor.processImage(bitmap);
                // Update service caches for mask and original via binder is not directly available here,
                // so expose getters through the model processor and let the service pull if needed.
                callback.onResult(processedBitmap, info.tm);
            } catch (Exception e) {
                callback.onError(e);
            } finally {
                inferBusy.set(false);
            }
        });
    }

    public Bitmap getLastMask() {
        return modelProcessor.getLastMask();
    }

    public Bitmap getLastOriginal() {
        return modelProcessor.getLastOriginal();
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
