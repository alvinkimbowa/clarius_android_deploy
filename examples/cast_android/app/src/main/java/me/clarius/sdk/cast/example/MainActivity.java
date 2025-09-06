package me.clarius.sdk.cast.example;

import android.graphics.Bitmap;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.ImageView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.navigation.NavController;
import androidx.navigation.Navigation;
import androidx.navigation.ui.AppBarConfiguration;
import androidx.navigation.ui.NavigationUI;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import me.clarius.sdk.cast.example.databinding.ActivityMainBinding;

public class MainActivity extends AppCompatActivity {

    private AppBarConfiguration appBarConfiguration;
    private ActivityMainBinding binding;

    private static final String TAG_MAIN_ACTIVITY = "MainActivity";
    private ImageView ultrasoundImageView;
    private UltrasoundModelProcessor modelProcessor;
    private ExecutorService executorService;
    private Handler mainThreadHandler;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        setSupportActionBar(binding.toolbar);

        NavController navController = Navigation.findNavController(this, R.id.nav_host_fragment_content_main);
        appBarConfiguration = new AppBarConfiguration.Builder(navController.getGraph()).build();
        NavigationUI.setupActionBarWithNavController(this, navController, appBarConfiguration);

        executorService = Executors.newSingleThreadExecutor();
        mainThreadHandler = new Handler(Looper.getMainLooper());
    }

    public void onNewUltrasoundImage(Bitmap ultrasoundBitmap) {
        if (ultrasoundBitmap != null && modelProcessor != null) {
            // Copy the bitmap if the original buffer might be reused/recycled by the SDK
            Bitmap immutableBitmap = ultrasoundBitmap.copy(ultrasoundBitmap.getConfig(), false);
            processAndDisplayImage(immutableBitmap);
        } else {
            if (ultrasoundBitmap == null) Log.w(TAG_MAIN_ACTIVITY, "Received null bitmap in onNewUltrasoundImage");
            if (modelProcessor == null) Log.w(TAG_MAIN_ACTIVITY, "modelProcessor is null in onNewUltrasoundImage");
        }
    }

    private void processAndDisplayImage(final Bitmap bitmapToProcess) {
        if (executorService == null || executorService.isShutdown()) {
            Log.w(TAG_MAIN_ACTIVITY, "ExecutorService not available for image processing.");
            return;
        }
        executorService.execute(() -> {
            try {
                final Bitmap resultBitmap = modelProcessor.processImage(bitmapToProcess);
                mainThreadHandler.post(() -> {
                    if (ultrasoundImageView != null && resultBitmap != null) {
                        ultrasoundImageView.setImageBitmap(resultBitmap);
                    } else {
                        if (ultrasoundImageView == null) Log.w(TAG_MAIN_ACTIVITY, "ultrasoundImageView is null, cannot display image.");
                        if (resultBitmap == null) Log.w(TAG_MAIN_ACTIVITY, "resultBitmap is null after processing.");
                    }
                });
            } catch (Exception e) {
                Log.e(TAG_MAIN_ACTIVITY, "Error processing image or updating UI", e);
            }
            // Note: Do not recycle bitmapToProcess here if it was the immutableCopy
            // and resultBitmap is different, as resultBitmap might be using parts of it
            // if processImage modified it in-place (which it shouldn't for the original).
            // The current processImage returns a new bitmap.
        });
    }


    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }

    @Override
    public boolean onSupportNavigateUp() {
        NavController navController = Navigation.findNavController(this, R.id.nav_host_fragment_content_main);
        return NavigationUI.navigateUp(navController, appBarConfiguration)
                || super.onSupportNavigateUp();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        // --- Added for Model Processing ---
        if (modelProcessor != null) {
            modelProcessor.close();
        }
        if (executorService != null) {
            executorService.shutdown();
        }
        // --- End Added ---
    }
}
