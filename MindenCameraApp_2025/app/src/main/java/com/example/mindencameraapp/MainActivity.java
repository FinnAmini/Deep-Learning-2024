/**
 * ---------------------------------------------------------------------------
 * Vorlesung: Deep Learning for Computer Vision (SoSe 2023)
 * Thema:     Test App for CameraX & TensorFlow Lite
 *
 * @author Jan Rexilius
 * @date   02/2023
 * ---------------------------------------------------------------------------
 */

package com.example.mindencameraapp;

import androidx.annotation.NonNull;

import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.PixelFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.media.ImageReader;
import android.net.Uri;
import android.os.Environment;
import android.util.Log;
import android.util.Size;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.AspectRatio;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.core.VideoCapture;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.recyclerview.widget.RecyclerView;

import android.annotation.SuppressLint;
import android.content.ContentValues;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.Surface;
import android.view.View;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.TextView;
import android.widget.Toast;

import com.google.common.util.concurrent.ListenableFuture;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.RequestBody;
import okhttp3.ResponseBody;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;
import retrofit2.http.Multipart;
import retrofit2.http.POST;
import retrofit2.http.Part;
import retrofit2.http.Query;


// ----------------------------------------------------------------------
// main class
public class MainActivity extends AppCompatActivity implements ImageAnalysis.Analyzer {

    private static final String TAG = "LOGGING:";
    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    PreviewView previewView;
    private ImageAnalysis imageAnalyzer;
    // image buffer
    private Bitmap bitmapBuffer;

    ImageButton buttonTakePicture;
    ImageButton buttonGallery;
    TextView classificationResults;
    private ImageCapture imageCapture;
    private VideoCapture videoCapture;
    private ExecutorService cameraExecutor;

    private int REQUEST_CODE_PERMISSIONS = 10;
    private final String[] REQUIRED_PERMISSIONS = new String[]{
            "android.permission.CAMERA",
            "android.permission.WRITE_EXTERNAL_STORAGE"
    };

    private final Object task = new Object();
    // add your filename here (model file)
    List<String> clasifierLabels = null;




    // ----------------------------------------------------------------------
    // set gui elements and start workflow
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        classificationResults = findViewById(R.id.classificationResults);
        buttonTakePicture = findViewById(R.id.buttonCapture);
        buttonTakePicture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                captureImage(false);
            }
        });
        buttonGallery = findViewById(R.id.buttonGallery);
        buttonGallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                captureImage(true);
            }
        });

        previewView = findViewById(R.id.previewView);

        // check permissions and start camera if all permissions available
        checkPermissions();
    }

    private boolean allPermissionsGranted() {
        for (String permission : REQUIRED_PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    // ----------------------------------------------------------------------
    // check app permissions
    private void checkPermissions() {
        if (allPermissionsGranted()) {
            cameraProviderFuture = ProcessCameraProvider.getInstance(this);
            cameraProviderFuture.addListener(() -> {
                try {
                    ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                    startCameraX(cameraProvider);
                } catch (ExecutionException e) {
                    e.printStackTrace();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

            }, getExecutor());
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS);
        }
    }

    private Executor getExecutor() {
        return ContextCompat.getMainExecutor(this);
    }


    // ----------------------------------------------------------------------
    // start camera
    @SuppressLint("RestrictedApi")
    private void startCameraX(ProcessCameraProvider cameraProvider) {

        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_FRONT)
                .build();

        Preview preview = new Preview.Builder().build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());

        imageCapture = new ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .setFlashMode(ImageCapture.FLASH_MODE_AUTO)
                .build();

        imageAnalyzer = new ImageAnalysis.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setTargetRotation(previewView.getDisplay().getRotation())
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build();
        imageAnalyzer.setAnalyzer(getExecutor(), this);

        // unbind before binding
        cameraProvider.unbindAll();
        try {
            cameraProvider.bindToLifecycle(this, cameraSelector, imageAnalyzer, preview, imageCapture);
        } catch (Exception exc) {
            Log.e(TAG, "Use case binding failed", exc);
        }
    }


    // ----------------------------------------------------------------------
    // capture single image
    private void captureImage(Boolean recognize) {
        imageCapture.takePicture(getExecutor(), new ImageCapture.OnImageCapturedCallback() {
            @Override
            public void onCaptureSuccess(@NonNull ImageProxy image) {
                super.onCaptureSuccess(image);
                Log.d("TAG", "Capture Image");
                sendImageToBackend(image, recognize);
            }
        });
    }


    // ----------------------------------------------------------------------
    // recognize image
    private void recognizeImage() {
        Log.d("Debug", "Pressing button 2...");

        long timeStamp = System.currentTimeMillis();
        ContentValues contentValues = new ContentValues();
        contentValues.put(MediaStore.MediaColumns.DISPLAY_NAME, timeStamp);
        contentValues.put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg");

        imageCapture.takePicture(
                new ImageCapture.OutputFileOptions.Builder(
                        getContentResolver(),
                        MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                        contentValues
                ).build(),
                getExecutor(),
                new ImageCapture.OnImageSavedCallback() {
                    @Override
                    public void onImageSaved(@NonNull ImageCapture.OutputFileResults outputFileResults) {
                        Toast.makeText(MainActivity.this,"Saving...",Toast.LENGTH_SHORT).show();
                    }

                    @Override
                    public void onError(@NonNull ImageCaptureException exception) {
                        Toast.makeText(MainActivity.this,"Error: "+exception.getMessage(),Toast.LENGTH_SHORT).show();


                    }
                });
    }

    // ----------------------------------------------------------------------
    // send iamge to backend
    private void sendImageToBackend(@NonNull ImageProxy imageProxy, Boolean recognize) {
        Log.d("Debug", "Pressing button 1...");

        ByteBuffer buffer = imageProxy.getPlanes()[0].getBuffer();
        byte[] arr = new byte[buffer.remaining()];
        buffer.get(arr);
        Bitmap bitmapImage = BitmapFactory.decodeByteArray(arr, 0, buffer.capacity());
        imageProxy.close();

        int width  = bitmapImage.getWidth();
        int height = bitmapImage.getHeight();

        // image size set to 224x224 (use bilinear interpolation)
        int size = height > width ? width : height;
        ImageProcessor imageProcessor = new ImageProcessor.Builder()
                .add(new Rot90Op(1))
                .add(new ResizeWithCropOrPadOp(size, size))
                .add(new ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
                .build();

        TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
        tensorImage.load(bitmapImage);
        tensorImage = imageProcessor.process(tensorImage);

        // Convert processed TensorImage back to Bitmap
        Bitmap processedBitmap = tensorImage.getBitmap();

        // Save the processed image locally
        if (!recognize) {
            saveImageToLocalStorage(processedBitmap);
        }

        // Send the image to the backend
        sendImage(processedBitmap, recognize);
    }

    private void saveImageToLocalStorage(Bitmap bitmap) {
        FileOutputStream outStream = null;
        try {
            // Get the Pictures directory
            File storageDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES);
            if (!storageDir.exists()) {
                storageDir.mkdirs();
            }

            // Create a file for the processed image
            String fileName = "processed_image_" + System.currentTimeMillis() + ".png";
            File imageFile = new File(storageDir, fileName);

            // Write the bitmap to the file
            outStream = new FileOutputStream(imageFile);
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, outStream);
            outStream.flush();

            Intent mediaScanIntent = new Intent(Intent.ACTION_MEDIA_SCANNER_SCAN_FILE);
            Uri contentUri = Uri.fromFile(imageFile);
            mediaScanIntent.setData(contentUri);
            sendBroadcast(mediaScanIntent);

            Log.d("Debug", "Image saved to: " + imageFile.getAbsolutePath());
        } catch (IOException e) {
            Log.e("Debug", "Error saving image", e);
        } finally {
            if (outStream != null) {
                try {
                    outStream.close();
                } catch (IOException e) {
                    Log.e("Debug", "Error closing output stream", e);
                }
            }
        }
    }

    // Method to send the image to the backend
    private void sendImage(Bitmap bitmap, Boolean recognize) {
        // Convert Bitmap to File
        File file = createTempFileFromBitmap(bitmap);

        if (file != null) {
            // Create Retrofit instance
            Retrofit retrofit = new Retrofit.Builder()
                    .baseUrl("https://yourbackend.com/") // Replace with your backend URL
                    .addConverterFactory(GsonConverterFactory.create())
                    .build();

            // Get API service
            ApiService apiService = retrofit.create(ApiService.class);

            // Create RequestBody and MultipartBody.Part
            RequestBody requestBody = RequestBody.create(MediaType.parse("image/png"), file);
            MultipartBody.Part body = MultipartBody.Part.createFormData("file", file.getName(), requestBody);

            // Make the API call
            Call<ResponseBody> call = apiService.uploadImage(body, recognize);
            Log.d("Debug", "Image enqueue");
            call.enqueue(new Callback<ResponseBody>() {
                @Override
                public void onResponse(@NonNull Call<ResponseBody> call, @NonNull Response<ResponseBody> response) {
                    if (response.isSuccessful() && response.body() != null) {
                        Log.d("Debug", "Image uploaded successfully");
                    } else {
                        Log.e("Debug", "Failed to upload image: " + (response.errorBody() != null ? response.errorBody().toString() : "Unknown error"));
                    }
                }

                @Override
                public void onFailure(@NonNull Call<ResponseBody> call, @NonNull Throwable t) {
                    Log.e("Debug", "Error uploading image", t);
                }
            });
        } else {
            Log.d("Debug", "File is null");
        }
    }

    public interface ApiService {
        @Multipart
        @POST("api/recognize")
        Call<ResponseBody> uploadImage(
                @Part MultipartBody.Part file,
                @Query("recognize") boolean recognize
        );
    }


    // Helper method to convert Bitmap to a temporary file
    private File createTempFileFromBitmap(Bitmap bitmap) {
        try {
            File tempFile = File.createTempFile("processed_image", ".png");
            FileOutputStream outStream = new FileOutputStream(tempFile);
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, outStream);
            outStream.flush();
            outStream.close();
            return tempFile;
        } catch (IOException e) {
            Log.e("Debug", "Error creating temp file", e);
            return null;
        }
    }

    @Override
    public void analyze(@NonNull ImageProxy imageProxy) {
        if ( imageProxy.getFormat()== PixelFormat.RGBA_8888){
            Bitmap bitmapImage = Bitmap.createBitmap(imageProxy.getWidth(),imageProxy.getHeight(),Bitmap.Config.ARGB_8888);
            bitmapImage.copyPixelsFromBuffer(imageProxy.getPlanes()[0].getBuffer());

            int rotation = imageProxy.getImageInfo().getRotationDegrees();

            imageProxy.close();
            int width  = bitmapImage.getWidth();
            int height = bitmapImage.getHeight();

            int size = height > width ? width : height;
            // image size set to 256x256 (use bilinear interpolation)
            ImageProcessor imageProcessor = new ImageProcessor.Builder()
                    .add(new Rot90Op(1))
                    .add(new ResizeWithCropOrPadOp(size, size))
                    .add(new ResizeOp(256, 256, ResizeOp.ResizeMethod.BILINEAR))
                    .build();

            TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
            tensorImage.load(bitmapImage);
            tensorImage = imageProcessor.process(tensorImage);
            TensorBuffer probabilityBuffer =
                    TensorBuffer.createFixedSize(new int[]{1, 15}, DataType.FLOAT32);

            TensorProcessor probabilityProcessor =
                    new TensorProcessor.Builder()/*.add(new NormalizeOp(0, 255))*/.build();

            String resultString = " ";
            if (null != clasifierLabels) {
                // Map of labels and their corresponding probability
                TensorLabel labels = new TensorLabel(clasifierLabels,
                        probabilityProcessor.process(probabilityBuffer));

                // Create a map to access the result based on label
                Map<String, Float> floatMap = labels.getMapWithFloatValue();
                resultString = getBestResult(floatMap);
                //Log.d("classifyImage", "RESULT: " + resultString);
                classificationResults.setText(resultString);
                //Toast.makeText(MainActivity.this, resultString, Toast.LENGTH_SHORT).show();
            }
        }
        // close image to get next one
        imageProxy.close();
    }


    // ----------------------------------------------------------------------
    // get 3 best keys & values from TF results
    public static String getResultString(Map<String, Float> mapResults){
        // max value
        Map.Entry<String, Float> entryMax1 = null;
        // 2nd max value
        Map.Entry<String, Float> entryMax2 = null;
        // 3rd max value
        Map.Entry<String, Float> entryMax3 = null;
        for(Map.Entry<String, Float> entry: mapResults.entrySet()){
            if (entryMax1 == null || entry.getValue().compareTo(entryMax1.getValue()) > 0){
                entryMax1 = entry;
            } else if (entryMax2 == null || entry.getValue().compareTo(entryMax2.getValue()) > 0){
                entryMax2 = entry;
            } else if (entryMax3 == null || entry.getValue().compareTo(entryMax3.getValue()) > 0){
                entryMax3 = entry;
            }
        }
        // result string includes the first three best values
        String result = entryMax1.getKey().trim() + " " + entryMax1.getValue().toString() + "\n" +
                        entryMax2.getKey().trim() + " " + entryMax2.getValue().toString() + "\n" +
                        entryMax3.getKey().trim() + " " + entryMax3.getValue().toString() + "\n";
        return result;
    }


    // ----------------------------------------------------------------------
    // get best key & value from TF results
    public static String getBestResult(@NonNull Map<String, Float> mapResults){
        // max value
        Map.Entry<String, Float> entryMax = null;
        for(Map.Entry<String, Float> entry: mapResults.entrySet()){
            if (entryMax == null || entry.getValue().compareTo(entryMax.getValue()) > 0) {
                entryMax = entry;
            }
        }
        int val = (int)(entryMax.getValue()*100.0f);
        entryMax.setValue((float)val);
        // result string includes the first three best values
        String result = "  " + entryMax.getKey().trim() + "   (" + Integer.toString(val) + "%)";
        return result;
    }

} // class