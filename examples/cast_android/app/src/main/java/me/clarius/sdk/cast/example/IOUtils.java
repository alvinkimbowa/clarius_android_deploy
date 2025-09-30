package me.clarius.sdk.cast.example;

import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Context;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Build;
import android.provider.MediaStore;

import androidx.annotation.RequiresApi;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.time.LocalDateTime;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;

public class IOUtils {
    @RequiresApi(api = Build.VERSION_CODES.Q)
    private static Uri save(ByteBuffer buffer, String prefix, Context context) throws IOException {
        final String fileName = String.format("%s_%s.tar", prefix,
                LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss")));
        final ContentValues contentValues = new ContentValues();
        contentValues.put(MediaStore.MediaColumns.DISPLAY_NAME, fileName);
        contentValues.put(MediaStore.MediaColumns.MIME_TYPE, "application/x-tar");
        contentValues.put(MediaStore.MediaColumns.RELATIVE_PATH, "Documents/Clarius");
        final ContentResolver contentResolver = context.getContentResolver();
        final Uri uri = MediaStore.Files.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY);
        final Uri itemUri = contentResolver.insert(uri, contentValues);
        if (itemUri == null) {
            throw new IOException("Failed to create the raw data file in the Documents folder");
        }
        try (OutputStream dest = contentResolver.openOutputStream(itemUri)) {
            dest.write(buffer.array());
        }
        return itemUri;
    }

    /**
     * Save the given byte buffer in the Documents folder.
     * <p>
     * NOTE: this method uses the MediaStore.Files.getContentUri() API which is only available on Android 10 and later.
     * Calling this method on older Android will raise an exception.
     *
     * @param buffer     the byte buffer to save.
     * @param context    the context to retrieve the Documents folder.
     * @return the saved file location.
     * @throws IOException
     */
    public static Uri saveInDocuments(ByteBuffer buffer, String prefix, Context context) throws IOException {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            return save(buffer, prefix, context);
        } else {
            throw new IOException("Saving only supported on Android 10 and later (API Q)");
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.Q)
    private static Uri saveBitmap(Bitmap bitmap, String prefix, Context context) throws IOException {
        final String dateFolder = LocalDate.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd"));
        final String fileName = String.format("%s_%s.png",
                LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss")),
                prefix);
        final ContentValues contentValues = new ContentValues();
        contentValues.put(MediaStore.MediaColumns.DISPLAY_NAME, fileName);
        contentValues.put(MediaStore.MediaColumns.MIME_TYPE, "image/png");
        contentValues.put(MediaStore.MediaColumns.RELATIVE_PATH, "Pictures/Clarius/" + dateFolder);
        final ContentResolver contentResolver = context.getContentResolver();
        final Uri imagesUri = MediaStore.Images.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY);
        final Uri itemUri = contentResolver.insert(imagesUri, contentValues);
        if (itemUri == null) {
            throw new IOException("Failed to create the image file in the Pictures folder");
        }
        try (OutputStream dest = contentResolver.openOutputStream(itemUri)) {
            if (!bitmap.compress(Bitmap.CompressFormat.PNG, 100, dest)) {
                throw new IOException("Failed to compress bitmap to PNG");
            }
        }
        return itemUri;
    }

    /**
     * Save a Bitmap as PNG in the Pictures/Clarius folder (Android 10+).
     */
    public static Uri saveBitmapInPictures(Bitmap bitmap, String prefix, Context context) throws IOException {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            return saveBitmap(bitmap, prefix, context);
        } else {
            throw new IOException("Saving only supported on Android 10 and later (API Q)");
        }
    }

    // Overload that accepts a fixed timestamp string (format: yyyy-MM-dd_HH-mm-ss) to ensure paired files share the same time.
    public static Uri saveBitmapInPictures(Bitmap bitmap, String prefix, String timestamp, Context context) throws IOException {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
            throw new IOException("Saving only supported on Android 10 and later (API Q)");
        }
        // Derive folder from the date part of the timestamp
        final String dateFolder;
        if (timestamp != null && timestamp.length() >= 10) {
            dateFolder = timestamp.substring(0, 10); // yyyy-MM-dd
        } else {
            dateFolder = LocalDate.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd"));
        }

        final String fileName = String.format("%s_%s.png", timestamp, prefix);
        final ContentValues contentValues = new ContentValues();
        contentValues.put(MediaStore.MediaColumns.DISPLAY_NAME, fileName);
        contentValues.put(MediaStore.MediaColumns.MIME_TYPE, "image/png");
        contentValues.put(MediaStore.MediaColumns.RELATIVE_PATH, "Pictures/Clarius/" + dateFolder);
        final ContentResolver contentResolver = context.getContentResolver();
        final Uri imagesUri = MediaStore.Images.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY);
        final Uri itemUri = contentResolver.insert(imagesUri, contentValues);
        if (itemUri == null) {
            throw new IOException("Failed to create the image file in the Pictures folder");
        }
        try (OutputStream dest = contentResolver.openOutputStream(itemUri)) {
            if (!bitmap.compress(Bitmap.CompressFormat.PNG, 100, dest)) {
                throw new IOException("Failed to compress bitmap to PNG");
            }
        }
        return itemUri;
    }
}
