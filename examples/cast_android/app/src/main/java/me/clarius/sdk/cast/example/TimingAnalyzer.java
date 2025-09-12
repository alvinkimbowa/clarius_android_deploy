package me.clarius.sdk.cast.example;

import android.content.Context;
import android.util.Log;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * TimingAnalyzer - A utility class to collect and compute average inference times
 * for research purposes. This class collects timing data from the model processor
 * and computes statistics including averages, min, max, and standard deviation.
 */
public class TimingAnalyzer {
    
    private static final String TAG = "TimingAnalyzer";
    private static final String RESULTS_DIR = "/storage/emulated/0/Download/clarius_timing/";
    
    private final Context context;
    private final String modelName;
    private final String resultsFile;
    private final String csvFile;
    
    // Thread-safe collections to store timing data
    private final ConcurrentLinkedQueue<Long> prepTimes = new ConcurrentLinkedQueue<>();
    private final ConcurrentLinkedQueue<Long> inferenceTimes = new ConcurrentLinkedQueue<>();
    private final ConcurrentLinkedQueue<Long> postProcessingTimes = new ConcurrentLinkedQueue<>();
    private final ConcurrentLinkedQueue<Long> totalTimes = new ConcurrentLinkedQueue<>();
    
    // Statistics tracking
    private int sampleCount = 0;
    private long startTime = System.currentTimeMillis();
    
    public TimingAnalyzer(Context context, String modelName) {
        this.context = context;
        this.modelName = modelName != null ? modelName : "unknown_model";
        this.resultsFile = "timing_analysis_" + sanitizeModelName(this.modelName) + ".txt";
        this.csvFile = "timing_data_" + sanitizeModelName(this.modelName) + ".csv";
    }
    
    /**
     * Sanitize model name for use in filenames
     */
    private String sanitizeModelName(String name) {
        return name.replaceAll("[^a-zA-Z0-9_-]", "_")
                  .replaceAll("_{2,}", "_") // Replace multiple underscores with single
                  .replaceAll("^_|_$", "") // Remove leading/trailing underscores
                  .toLowerCase();
    }
    
    /**
     * Record timing data for a single inference
     */
    public void recordTiming(long prepTime, long inferenceTime, long postProcessingTime, long totalTime) {
        prepTimes.offer(prepTime);
        inferenceTimes.offer(inferenceTime);
        postProcessingTimes.offer(postProcessingTime);
        totalTimes.offer(totalTime);
        sampleCount++;
        
        Log.d(TAG, "Recorded timing - Prep: " + prepTime + "ms, Inference: " + inferenceTime + 
              "ms, Post: " + postProcessingTime + "ms, Total: " + totalTime + "ms");
        
        // Save data periodically (every 10 samples) to avoid losing data
        if (sampleCount % 10 == 0) {
            saveToCSV();
        }
    }
    
    /**
     * Compute statistics for a list of timing values
     */
    private TimingStats computeStats(List<Long> times) {
        if (times.isEmpty()) {
            return new TimingStats(0, 0, 0, 0, 0.0);
        }
        
        // Sort for min/max
        List<Long> sorted = new ArrayList<>(times);
        sorted.sort(Long::compareTo);
        
        long min = sorted.get(0);
        long max = sorted.get(sorted.size() - 1);
        
        // Calculate average
        long sum = 0;
        for (long time : times) {
            sum += time;
        }
        long average = sum / times.size();
        
        // Calculate standard deviation
        double variance = 0.0;
        for (long time : times) {
            double diff = time - average;
            variance += diff * diff;
        }
        variance /= times.size();
        double stdDev = Math.sqrt(variance);
        
        return new TimingStats(times.size(), min, max, average, stdDev);
    }
    
    /**
     * Get current statistics for all timing categories
     */
    public TimingAnalysis getCurrentStats() {
        TimingStats prepStats = computeStats(new ArrayList<>(prepTimes));
        TimingStats inferenceStats = computeStats(new ArrayList<>(inferenceTimes));
        TimingStats postProcessingStats = computeStats(new ArrayList<>(postProcessingTimes));
        TimingStats totalStats = computeStats(new ArrayList<>(totalTimes));
        
        return new TimingAnalysis(
            prepStats, inferenceStats, postProcessingStats, totalStats,
            sampleCount, System.currentTimeMillis() - startTime
        );
    }
    
    /**
     * Save timing data to CSV files (both per-frame and summary)
     */
    public void saveToCSV() {
        try {
            File resultsDir = new File(RESULTS_DIR);
            if (!resultsDir.exists()) {
                resultsDir.mkdirs();
            }
            
            // Save per-frame detailed results
            savePerFrameCSV(resultsDir);
            
            // Save summary statistics
            saveSummaryCSV(resultsDir);
            
            // Clear the data after saving to avoid duplicates
            clearData();
            
            Log.i(TAG, "Timing data saved to CSV files in: " + resultsDir.getAbsolutePath());
            
        } catch (Exception e) {
            Log.e(TAG, "Failed to save timing data to CSV", e);
        }
    }
    
    /**
     * Save detailed per-frame timing data to CSV
     */
    private void savePerFrameCSV(File resultsDir) {
        try {
            String baseFilename = csvFile.replace(".csv", "");
            File perFrameFile = new File(resultsDir, baseFilename + "_perframe.csv");
            FileWriter writer = new FileWriter(perFrameFile, true); // Append mode
            
            // Write header if file is empty
            if (perFrameFile.length() == 0) {
                writer.write("timestamp,frame,prep_time_ms,inference_time_ms,post_processing_time_ms,total_time_ms\n");
            }
            
            // Write data
            String timestamp = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.getDefault()).format(new Date());
            
            // Convert queues to lists for iteration
            List<Long> prepList = new ArrayList<>(prepTimes);
            List<Long> inferenceList = new ArrayList<>(inferenceTimes);
            List<Long> postList = new ArrayList<>(postProcessingTimes);
            List<Long> totalList = new ArrayList<>(totalTimes);
            
            int minSize = Math.min(Math.min(prepList.size(), inferenceList.size()), 
                                 Math.min(postList.size(), totalList.size()));
            
            for (int i = 0; i < minSize; i++) {
                writer.write(timestamp + "," + (i + 1) + "," + prepList.get(i) + "," + inferenceList.get(i) + 
                           "," + postList.get(i) + "," + totalList.get(i) + "\n");
            }
            
            writer.close();
            Log.i(TAG, "Per-frame timing data saved to: " + perFrameFile.getAbsolutePath());
            
        } catch (IOException e) {
            Log.e(TAG, "Failed to save per-frame timing data to CSV", e);
        }
    }
    
    /**
     * Save summary statistics to CSV
     */
    private void saveSummaryCSV(File resultsDir) {
        try {
            String baseFilename = csvFile.replace(".csv", "");
            File summaryFile = new File(resultsDir, baseFilename + "_summary.csv");
            FileWriter writer = new FileWriter(summaryFile, true); // Append mode
            
            // Write header if file is empty
            if (summaryFile.length() == 0) {
                writer.write("timestamp,session_duration_ms,sample_count," +
                           "prep_avg_ms,prep_min_ms,prep_max_ms,prep_stddev_ms," +
                           "inference_avg_ms,inference_min_ms,inference_max_ms,inference_stddev_ms," +
                           "post_avg_ms,post_min_ms,post_max_ms,post_stddev_ms," +
                           "total_avg_ms,total_min_ms,total_max_ms,total_stddev_ms," +
                           "prep_percent,inference_percent,post_percent,average_fps\n");
            }
            
            // Write data
            String timestamp = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.getDefault()).format(new Date());
            
            // Compute statistics
            TimingAnalysis stats = getCurrentStats();
            
            // Calculate percentages
            double avgTotal = stats.totalTime.average;
            double prepPercentage = (stats.prepTime.average / avgTotal) * 100;
            double inferencePercentage = (stats.inferenceTime.average / avgTotal) * 100;
            double postPercentage = (stats.postProcessingTime.average / avgTotal) * 100;
            double avgFPS = 1000.0 / avgTotal;
            
            // Write summary data
            writer.write(String.format("%s,%d,%d," +
                "%d,%d,%d,%.2f," +
                "%d,%d,%d,%.2f," +
                "%d,%d,%d,%.2f," +
                "%d,%d,%d,%.2f," +
                "%.1f,%.1f,%.1f,%.2f\n",
                timestamp,
                stats.sessionDuration,
                stats.sampleCount,
                stats.prepTime.average, stats.prepTime.min, stats.prepTime.max, stats.prepTime.standardDeviation,
                stats.inferenceTime.average, stats.inferenceTime.min, stats.inferenceTime.max, stats.inferenceTime.standardDeviation,
                stats.postProcessingTime.average, stats.postProcessingTime.min, stats.postProcessingTime.max, stats.postProcessingTime.standardDeviation,
                stats.totalTime.average, stats.totalTime.min, stats.totalTime.max, stats.totalTime.standardDeviation,
                prepPercentage, inferencePercentage, postPercentage, avgFPS
            ));
            
            writer.close();
            Log.i(TAG, "Summary timing data saved to: " + summaryFile.getAbsolutePath());
            
        } catch (IOException e) {
            Log.e(TAG, "Failed to save summary timing data to CSV", e);
        }
    }
    
    /**
     * Save comprehensive timing analysis to text file
     */
    public void saveToFile() {
        try {
            File resultsDir = new File(RESULTS_DIR);
            if (!resultsDir.exists()) {
                resultsDir.mkdirs();
            }
            
            File analysisFile = new File(resultsDir, resultsFile);
            FileWriter writer = new FileWriter(analysisFile);
            
            TimingAnalysis stats = getCurrentStats();
            String timestamp = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(new Date());
            
            writer.write("=== CLARIUS MODEL TIMING ANALYSIS ===\n");
            writer.write("Model: " + modelName + "\n");
            writer.write("Generated: " + timestamp + "\n");
            writer.write("Session Duration: " + stats.sessionDuration + "ms (" + (stats.sessionDuration / 1000.0) + "s)\n");
            writer.write("Total Samples: " + stats.sampleCount + "\n\n");
            
            // Prep time statistics
            writer.write("PRE-PROCESSING TIME:\n");
            writer.write("  Count: " + stats.prepTime.count + "\n");
            writer.write("  Average: " + stats.prepTime.average + "ms\n");
            writer.write("  Min: " + stats.prepTime.min + "ms\n");
            writer.write("  Max: " + stats.prepTime.max + "ms\n");
            writer.write("  Std Dev: " + String.format("%.2f", stats.prepTime.standardDeviation) + "ms\n\n");
            
            // Inference time statistics
            writer.write("INFERENCE TIME:\n");
            writer.write("  Count: " + stats.inferenceTime.count + "\n");
            writer.write("  Average: " + stats.inferenceTime.average + "ms\n");
            writer.write("  Min: " + stats.inferenceTime.min + "ms\n");
            writer.write("  Max: " + stats.inferenceTime.max + "ms\n");
            writer.write("  Std Dev: " + String.format("%.2f", stats.inferenceTime.standardDeviation) + "ms\n\n");
            
            // Post-processing time statistics
            writer.write("POST-PROCESSING TIME:\n");
            writer.write("  Count: " + stats.postProcessingTime.count + "\n");
            writer.write("  Average: " + stats.postProcessingTime.average + "ms\n");
            writer.write("  Min: " + stats.postProcessingTime.min + "ms\n");
            writer.write("  Max: " + stats.postProcessingTime.max + "ms\n");
            writer.write("  Std Dev: " + String.format("%.2f", stats.postProcessingTime.standardDeviation) + "ms\n\n");
            
            // Total time statistics
            writer.write("TOTAL PROCESSING TIME:\n");
            writer.write("  Count: " + stats.totalTime.count + "\n");
            writer.write("  Average: " + stats.totalTime.average + "ms\n");
            writer.write("  Min: " + stats.totalTime.min + "ms\n");
            writer.write("  Max: " + stats.totalTime.max + "ms\n");
            writer.write("  Std Dev: " + String.format("%.2f", stats.totalTime.standardDeviation) + "ms\n\n");
            
            // Performance insights
            writer.write("PERFORMANCE INSIGHTS:\n");
            double avgInference = stats.inferenceTime.average;
            double avgTotal = stats.totalTime.average;
            double inferencePercentage = (avgInference / avgTotal) * 100;
            double prepPercentage = (stats.prepTime.average / avgTotal) * 100;
            double postPercentage = (stats.postProcessingTime.average / avgTotal) * 100;
            
            writer.write("  Inference time is " + String.format("%.1f", inferencePercentage) + "% of total time\n");
            writer.write("  Pre-processing time is " + String.format("%.1f", prepPercentage) + "% of total time\n");
            writer.write("  Post-processing time is " + String.format("%.1f", postPercentage) + "% of total time\n");
            writer.write("  Average FPS: " + String.format("%.2f", 1000.0 / avgTotal) + "\n");
            
            writer.close();
            Log.i(TAG, "Timing analysis saved to: " + analysisFile.getAbsolutePath());
            
        } catch (IOException e) {
            Log.e(TAG, "Failed to save timing analysis", e);
        }
    }
    
    /**
     * Clear all collected data
     */
    public void clearData() {
        prepTimes.clear();
        inferenceTimes.clear();
        postProcessingTimes.clear();
        totalTimes.clear();
        sampleCount = 0;
        startTime = System.currentTimeMillis();
        Log.i(TAG, "Timing data cleared");
    }
    
    /**
     * Force save all data and generate final report
     */
    public void generateFinalReport() {
        saveToCSV();
        Log.i(TAG, "Final timing report generated");
    }
    
    /**
     * Data class to hold timing statistics for a single category
     */
    public static class TimingStats {
        public final int count;
        public final long min;
        public final long max;
        public final long average;
        public final double standardDeviation;
        
        public TimingStats(int count, long min, long max, long average, double standardDeviation) {
            this.count = count;
            this.min = min;
            this.max = max;
            this.average = average;
            this.standardDeviation = standardDeviation;
        }
    }
    
    /**
     * Data class to hold complete timing analysis
     */
    public static class TimingAnalysis {
        public final TimingStats prepTime;
        public final TimingStats inferenceTime;
        public final TimingStats postProcessingTime;
        public final TimingStats totalTime;
        public final int sampleCount;
        public final long sessionDuration;
        
        public TimingAnalysis(TimingStats prepTime, TimingStats inferenceTime, 
                            TimingStats postProcessingTime, TimingStats totalTime,
                            int sampleCount, long sessionDuration) {
            this.prepTime = prepTime;
            this.inferenceTime = inferenceTime;
            this.postProcessingTime = postProcessingTime;
            this.totalTime = totalTime;
            this.sampleCount = sampleCount;
            this.sessionDuration = sessionDuration;
        }
    }
}
