package com.example.mindencameraapp;

import androidx.annotation.NonNull;

import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.widget.ImageButton;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.RequestBody;
import okhttp3.ResponseBody;
import okhttp3.logging.HttpLoggingInterceptor;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;
import retrofit2.http.Multipart;
import retrofit2.http.POST;
import retrofit2.http.Part;
import retrofit2.http.Query;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "LOGGING:";
    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    private PreviewView previewView;
    private ImageCapture imageCapture;
    private int currentCamera = CameraSelector.LENS_FACING_FRONT;
    private ImageButton buttonSwitchCamera;


    ImageButton buttonTakePicture;
    ImageButton buttonGallery;
    TextView classificationResults;

    private final int REQUEST_CODE_PERMISSIONS = 10;
    private final String[] REQUIRED_PERMISSIONS = new String[]{
            "android.permission.CAMERA",
            "android.permission.WRITE_EXTERNAL_STORAGE"
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        classificationResults = findViewById(R.id.classificationResults);
        buttonTakePicture = findViewById(R.id.buttonCapture);
        buttonTakePicture.setOnClickListener(v -> captureImage(false));
        buttonGallery = findViewById(R.id.buttonGallery);
        buttonGallery.setOnClickListener(v -> captureImage(true));

        previewView = findViewById(R.id.previewView);

        buttonSwitchCamera = findViewById(R.id.buttonSwitchCamera);
        buttonSwitchCamera.setOnClickListener(v -> {
            currentCamera = (currentCamera == CameraSelector.LENS_FACING_FRONT)
                    ? CameraSelector.LENS_FACING_BACK
                    : CameraSelector.LENS_FACING_FRONT;

            startCameraX(); // Restart the camera with the updated lens
        });


        // Check permissions and start camera if all permissions are granted
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

    private void checkPermissions() {
        if (allPermissionsGranted()) {
            cameraProviderFuture = ProcessCameraProvider.getInstance(this);
            cameraProviderFuture.addListener(this::startCameraX, getExecutor());
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS);
        }
    }

    private Executor getExecutor() {
        return ContextCompat.getMainExecutor(this);
    }

    private void startCameraX() {
        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();

                CameraSelector cameraSelector = new CameraSelector.Builder()
                        .requireLensFacing(currentCamera)
                        .build();

                Preview preview = new Preview.Builder().build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());

                imageCapture = new ImageCapture.Builder()
                        .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                        .setFlashMode(ImageCapture.FLASH_MODE_AUTO)
                        .build();

                cameraProvider.unbindAll();
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageCapture);
            } catch (Exception exc) {
                Log.e(TAG, "Use case binding failed", exc);
            }
        }, getExecutor());
    }


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

    private void sendImageToBackend(@NonNull ImageProxy imageProxy, Boolean recognize) {
        Log.d("Debug", "Pressing button 1...");

        ByteBuffer buffer = imageProxy.getPlanes()[0].getBuffer();
        byte[] arr = new byte[buffer.remaining()];
        buffer.get(arr);
        Bitmap bitmapImage = BitmapFactory.decodeByteArray(arr, 0, buffer.capacity());
        imageProxy.close();

        // Send the image to the backend
        sendImage(bitmapImage, recognize);
    }

    private void sendImage(Bitmap bitmap, Boolean recognize) {
        File file = createTempFileFromBitmap(bitmap);

        if (file != null) {
            Retrofit retrofit = getRetrofitInstance();

            ApiService apiService = retrofit.create(ApiService.class);

            RequestBody requestBody = RequestBody.create(MediaType.parse("image/png"), file);
            MultipartBody.Part body = MultipartBody.Part.createFormData("image", file.getName(), requestBody);

            Call<ResponseBody> call = apiService.uploadImage(body, recognize);
            Log.d("Debug", "Image enqueue");
            call.enqueue(new Callback<ResponseBody>() {
                @Override
                public void onResponse(@NonNull Call<ResponseBody> call, @NonNull Response<ResponseBody> response) {
                    if (response.isSuccessful() && response.body() != null) {
                        try {
                            String responseBody = response.body().string();
                            Log.d("Debug", "Response: " + responseBody);

                            JSONObject jsonResponse = new JSONObject(responseBody);
                            String message = jsonResponse.optString("message", "No message received");

                            if (recognize) {
                                JSONArray closestArray = jsonResponse.optJSONArray("closest");
                                if (closestArray != null && closestArray.length() > 0) {
                                    JSONObject closestObject = closestArray.getJSONObject(0);
                                    boolean recognized = closestObject.optBoolean("recognized", false);

                                    if (recognized) {
                                        runOnUiThread(() -> Toast.makeText(MainActivity.this, "Recognition successful: " + message, Toast.LENGTH_LONG).show());
                                    } else {
                                        runOnUiThread(() -> Toast.makeText(MainActivity.this, "Recognition failed: No matches found.", Toast.LENGTH_LONG).show());
                                    }
                                } else {
                                    runOnUiThread(() -> Toast.makeText(MainActivity.this, "Recognition failed: No data received.", Toast.LENGTH_LONG).show());
                                }
                            } else {
                                runOnUiThread(() -> Toast.makeText(MainActivity.this, message, Toast.LENGTH_LONG).show());
                            }
                        } catch (IOException | JSONException e) {
                            Log.e("Debug", "Error parsing response body", e);
                            runOnUiThread(() -> Toast.makeText(MainActivity.this, "An error occurred while processing the response.", Toast.LENGTH_LONG).show());
                        }
                    } else {
                        Log.e("Debug", "Failed to upload image: " + (response.errorBody() != null ? response.errorBody().toString() : "Unknown error"));
                        runOnUiThread(() -> Toast.makeText(MainActivity.this, "Failed to upload the image.", Toast.LENGTH_LONG).show());
                    }
                }

                @Override
                public void onFailure(@NonNull Call<ResponseBody> call, @NonNull Throwable t) {
                    Log.e("Debug", "Error uploading image", t);
                    runOnUiThread(() -> Toast.makeText(MainActivity.this, "Failed to upload the image.", Toast.LENGTH_LONG).show());
                }
            });
        } else {
            Log.d("Debug", "File is null");
            runOnUiThread(() -> Toast.makeText(MainActivity.this, "File is null.", Toast.LENGTH_LONG).show());
        }
    }

    private Retrofit getRetrofitInstance() {
        HttpLoggingInterceptor loggingInterceptor = new HttpLoggingInterceptor();
        loggingInterceptor.setLevel(HttpLoggingInterceptor.Level.BODY);

        OkHttpClient client = new OkHttpClient.Builder()
                .addInterceptor(loggingInterceptor)
                .build();

        return new Retrofit.Builder()
                .baseUrl("http://192.168.137.1:5000/")
                .client(client)
                .addConverterFactory(GsonConverterFactory.create())
                .build();
    }

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

    public interface ApiService {
        @Multipart
        @POST("api/recognize")
        Call<ResponseBody> uploadImage(
                @Part MultipartBody.Part file,
                @Query("recognize") boolean recognize
        );
    }
}
