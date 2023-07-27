package com.example.whole_video_app;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.CaptureResult;
import android.hardware.camera2.TotalCaptureResult;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.MediaPlayer;
import android.media.MediaRecorder;
import org.apache.commons.io.FileUtils;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Base64;
import android.util.Log;
import android.util.Size;
import android.util.SparseIntArray;
import android.view.Surface;
import android.view.TextureView;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.Chronometer;
import android.widget.CompoundButton;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.ToggleButton;
import android.widget.VideoView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.LinkedList;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Timer;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "REC123";

    private static final int REQUEST_CAMERA_PERMISSION_RESULT = 0;
    private static final int REQUEST_WRITE_EXTERNAL_STORAGE_PERMISSION_RESULT = 1;
    private static final int STATE_PREVIEW = 0;
    private static final int STATE_WAIT_LOCK = 1;
    private int mCaptureState = STATE_PREVIEW;
    private TextureView mTextureView;
    private TextView textView, text_intro;
    private View rectangle;
    private Button replay, anotherone, start, back, save;
    private ProgressBar progressbar;

    private TextView title;
    private TextView keeprec;
    private ImageView err;

    StringBuilder response = new StringBuilder();

    //private String state_switch;

    private Switch switch_btt;
    private VideoView videoView;
    private Runnable r_show, r_save;
    private Runnable r_end;
    public String old_filename;

    public ToggleButton check_taggle_btn;

    public int tipo;

    public boolean flag = false;

    public final String server = "http://172.20.10.5:3535/";

    public Queue queue_response = new LinkedList();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        verifyStoragePermissions(this);

        createVideoFolder();
        //createImageFolder();

        mChronometer = (Chronometer) findViewById(R.id.chronometer);
        mTextureView = (TextureView) findViewById(R.id.textureView);
        videoView = (VideoView) findViewById(R.id.vv);
        textView = (TextView) findViewById(R.id.tt);
        text_intro = findViewById(R.id.intro);
        switch_btt = (Switch) findViewById(R.id.simpleSwitch);
        rectangle = findViewById(R.id.rectangle);
        replay = findViewById(R.id.restart);
        anotherone = findViewById(R.id.anotherone);
        progressbar = findViewById(R.id.progressbar);

        title = findViewById(R.id.title);
        keeprec = findViewById(R.id.keeprec);
        err = findViewById(R.id.er);
        start = findViewById(R.id.start);
        back = findViewById(R.id.back);
        save = findViewById(R.id.save);

        check_taggle_btn = findViewById(R.id.check_toggle_btn);

        keeprec.setVisibility(View.INVISIBLE);
        err.setVisibility(View.INVISIBLE);

        title.setText("Make the statues talk");
        text_intro.setText("Just follow the instructions:\n\n" +
                "1) choose a subject and a modality (fast or precise)\n\n" +
                "2) record a video (with \"start\"). It is important to clearly record the face\n\n" +
                "3) the deepfake will be reproduced and the statue will talk about its history ");
        text_intro.setVisibility(View.VISIBLE);
        title.setVisibility(View.VISIBLE);

        save.setVisibility(View.INVISIBLE);
        replay.setVisibility(View.INVISIBLE);
        anotherone.setVisibility(View.INVISIBLE);
        rectangle.setVisibility(View.INVISIBLE);
        progressbar.setVisibility(View.INVISIBLE);


        switch_btt.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {

            @Override
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                if (isChecked) {
                    switch_btt.setText("STATUE");
                    switch_btt.setTextColor(Color.WHITE);
                } else {
                    switch_btt.setText("PEOPLE");
                    switch_btt.setTextColor(Color.BLUE);
                }
            }

        });

        check_taggle_btn.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                if (isChecked) {
                    check_taggle_btn.setText("PRECISE");
                    check_taggle_btn.setTextColor(Color.parseColor("#ffffff"));
                } else {
                    check_taggle_btn.setText("FAST");
                    check_taggle_btn.setTextColor(Color.parseColor("#000000"));
                }
            }
        });


        videoView.setVisibility(View.INVISIBLE);

        //set textview
        //textView.setText("Let's start! choose the subject and record with the button below");
        textView.setVisibility(View.VISIBLE);

        //runnable only to end (statues)
        r_end = new Runnable() {
            @Override
            public void run() {
                //stop recording
                mChronometer.stop();
                mChronometer.setVisibility(View.INVISIBLE);
                rectangle.setVisibility(View.INVISIBLE);
                progressbar.setVisibility(View.VISIBLE);
                mIsRecording = false;
                mIsTimelapse = false;
                //mRecordImageButton.setImageResource(R.mipmap.btn_video_online);

                Log.i(TAG, "STOPPED");

                mMediaRecorder.stop();
                mMediaRecorder.reset();
                old_filename = mVideoFileName;
                //put it in another trade (?)
                Intent mediaStoreUpdateIntent = new Intent(Intent.ACTION_MEDIA_SCANNER_SCAN_FILE);
                mediaStoreUpdateIntent.setData(Uri.fromFile(new File(old_filename)));
                sendBroadcast(mediaStoreUpdateIntent);

                //gallery_video_process(old_filename);
                Send_video myThread = new Send_video();
                myThread.start();

            }

        };

        r_show = new Runnable() {
            @Override
            public void run() {
                Show_video show_thread = new Show_video();
                show_thread.start();
            }

        };

        r_save = new Runnable() {
            public void run() {
                Save_video save_video = new Save_video();
                save_video.start();
            }
        };


        //mRecordImageButton
        start.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                switch_btt.setVisibility(View.INVISIBLE);
                check_taggle_btn.setVisibility(View.INVISIBLE);
                start.setVisibility(View.INVISIBLE);
                back.setVisibility(View.INVISIBLE);
                mTextureView.setVisibility(View.VISIBLE);
                Log.i(TAG, "Sate switch on click: " + switch_btt.isChecked());

                start_cycle();

            }
        });

        start.setOnLongClickListener(new View.OnLongClickListener() {
            @Override
            public boolean onLongClick(View v) {
                Log.i(TAG, "Inside the onLongClick");
                /*
                mIsTimelapse =true;
                start.setImageResource(R.mipmap.btn_timelapse);
                checkWriteStoragePermission();

                 */
                return true;
            }
        });

        anotherone.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Log.i(TAG, "Start another deepfake");
                Intent intent = new Intent(MainActivity.this, MenuActivity.class);
                startActivity(intent);
                finish();

            }
        });

        replay.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                replay.setVisibility(View.INVISIBLE);
                save.setVisibility(View.INVISIBLE);
                anotherone.setVisibility(View.INVISIBLE);
                r_show.run();
            }
        });

        save.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                r_save.run();
            }
        });

        back.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(MainActivity.this, MenuActivity.class);
                startActivity(intent);
                finish();
            }
        });
    }


    private CameraDevice mCameraDevice;
    private CameraDevice.StateCallback mCameraDeviceStateCallback = new CameraDevice.StateCallback() {
        @Override
        public void onOpened(CameraDevice camera) {
            mCameraDevice = camera;
            mMediaRecorder = new MediaRecorder();
            if(mIsRecording) {
                try {
                    createVideoFileName();
                } catch (Exception e) {
                    Log.i(TAG, "Error on opening camera/file: " + e);
                }
                startRecord();
                mMediaRecorder.start();
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        rectangle.setVisibility(View.VISIBLE);
                        mChronometer.setBase(SystemClock.elapsedRealtime());
                        mChronometer.setVisibility(View.VISIBLE);
                        mChronometer.start();
                    }
                });
            }
        }

        @Override
        public void onDisconnected(CameraDevice camera) {
            camera.close();
            mCameraDevice = null;
        }

        @Override
        public void onError(CameraDevice camera, int error) {
            camera.close();
            mCameraDevice = null;
        }
    };
    private HandlerThread mBackgroundHandlerThread;
    private Handler mBackgroundHandler;
    private String mCameraId;
    private Size mPreviewSize;
    private Size mVideoSize;
    private MediaRecorder mMediaRecorder;
    private Chronometer mChronometer;
    private int mTotalRotation;
    private CameraCaptureSession mPreviewCaptureSession;
    private CameraCaptureSession.CaptureCallback mPreviewCaptureCallback = new
            CameraCaptureSession.CaptureCallback() {

                private void process(CaptureResult captureResult) {
                    switch (mCaptureState) {
                        case STATE_PREVIEW:
                            // Do nothing
                            break;
                        case STATE_WAIT_LOCK:
                            mCaptureState = STATE_PREVIEW;
                            Integer afState = captureResult.get(CaptureResult.CONTROL_AF_STATE);
                            if(afState == CaptureResult.CONTROL_AF_STATE_FOCUSED_LOCKED ||
                                    afState == CaptureResult.CONTROL_AF_STATE_NOT_FOCUSED_LOCKED) {
                                Toast.makeText(getApplicationContext(), "AF Locked!", Toast.LENGTH_SHORT).show();
                                //startStillCaptureRequest();
                            }
                            break;
                    }
                }

                @Override
                public void onCaptureCompleted(CameraCaptureSession session, CaptureRequest request, TotalCaptureResult result) {
                    super.onCaptureCompleted(session, request, result);

                    process(result);
                }
            };
    private CameraCaptureSession mRecordCaptureSession;
    private CameraCaptureSession.CaptureCallback mRecordCaptureCallback = new
            CameraCaptureSession.CaptureCallback() {

                private void process(CaptureResult captureResult) {
                    switch (mCaptureState) {
                        case STATE_PREVIEW:
                            // Do nothing
                            break;
                        case STATE_WAIT_LOCK:
                            mCaptureState = STATE_PREVIEW;
                            Integer afState = captureResult.get(CaptureResult.CONTROL_AF_STATE);
                            if(afState == CaptureResult.CONTROL_AF_STATE_FOCUSED_LOCKED ||
                                    afState == CaptureResult.CONTROL_AF_STATE_NOT_FOCUSED_LOCKED) {
                                Toast.makeText(getApplicationContext(), "AF Locked!", Toast.LENGTH_SHORT).show();
                                //startStillCaptureRequest();
                            }
                            break;
                    }
                }

                @Override
                public void onCaptureCompleted(CameraCaptureSession session, CaptureRequest request, TotalCaptureResult result) {
                    super.onCaptureCompleted(session, request, result);
                    process(result);
                }
            };
    private CaptureRequest.Builder mCaptureRequestBuilder;

    //private ImageButton mRecordImageButton;
    private ImageButton mStillImageButton;
    private boolean mIsRecording = false;
    private boolean mIsTimelapse = false;
    private boolean first = true;
    private Timer fileSwitchTimer;

    private Handler handler;

    private File mVideoFolder;
    private String mVideoFileName;
    private File mImageFolder;
    private String mImageFileName;

    private static SparseIntArray ORIENTATIONS = new SparseIntArray();
    static {
        ORIENTATIONS.append(Surface.ROTATION_0, 0);
        ORIENTATIONS.append(Surface.ROTATION_90, 90);
        ORIENTATIONS.append(Surface.ROTATION_180, 180);
        ORIENTATIONS.append(Surface.ROTATION_270, 270);
    }

    private static class CompareSizeByArea implements Comparator<Size> {

        @Override
        public int compare(Size lhs, Size rhs) {
            return Long.signum( (long)(lhs.getWidth() * lhs.getHeight()) - (long)(rhs.getWidth() * rhs.getHeight()));
        }
    }

    public void start_cycle() {

        //start
        mMediaRecorder = null;
        mIsRecording = true;
        //mRecordImageButton.setImageResource(R.mipmap.btn_video_busy);
        checkWriteStoragePermission(); //start record
        // mMediaRecorder.start();
        mChronometer.setBase(SystemClock.elapsedRealtime());
        mChronometer.setVisibility(View.VISIBLE);
        mChronometer.start();

        new Handler().postDelayed(r_end,13000); //4
        //new Handler().postDelayed(r_show,90000); //40
    }


    @Override
    protected void onResume() {
        super.onResume();

        startBackgroundThread();

        if(mTextureView.isAvailable()) {
            setupCamera(mTextureView.getWidth(), mTextureView.getHeight());
            connectCamera();
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if(requestCode == REQUEST_WRITE_EXTERNAL_STORAGE_PERMISSION_RESULT) {
            if(grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                if(mIsRecording || mIsTimelapse) {
                    mIsRecording = true;
                    //mRecordImageButton.setImageResource(R.mipmap.btn_video_busy);
                }
            }
        }
    }

    @Override
    protected void onPause() {
        closeCamera();
        stopBackgroundThread();
        super.onPause();
    }

    private void setupCamera(int width, int height) {
        CameraManager cameraManager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        try {
            for(String cameraId : cameraManager.getCameraIdList()){
                CameraCharacteristics cameraCharacteristics = cameraManager.getCameraCharacteristics(cameraId);
                if(cameraCharacteristics.get(CameraCharacteristics.LENS_FACING) ==
                        CameraCharacteristics.LENS_FACING_FRONT){
                    continue;
                }
                StreamConfigurationMap map = cameraCharacteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
                int deviceOrientation = getWindowManager().getDefaultDisplay().getRotation();
                mTotalRotation = sensorToDeviceRotation(cameraCharacteristics, deviceOrientation);
                boolean swapRotation = mTotalRotation == 90 || mTotalRotation == 270;
                int rotatedWidth = width;
                int rotatedHeight = height;
                if(swapRotation) {
                    rotatedWidth = height;
                    rotatedHeight = width;
                }

                mVideoSize = chooseOptimalSize(map.getOutputSizes(MediaRecorder.class), rotatedWidth, rotatedHeight);
                mCameraId = cameraId;
                return;
            }
        } catch (Exception e) {
            Log.i(TAG, "Error on setting up camera: " + e);
        }
    }

    private void connectCamera() {
        CameraManager cameraManager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        try {
            if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                if(ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA) ==
                        PackageManager.PERMISSION_GRANTED) {
                    cameraManager.openCamera(mCameraId, mCameraDeviceStateCallback, mBackgroundHandler);
                } else {
                    if(shouldShowRequestPermissionRationale(android.Manifest.permission.CAMERA)) {
                        Toast.makeText(this, "Video app required access to camera", Toast.LENGTH_SHORT).show();
                    }
                    requestPermissions(new String[] {android.Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO
                    }, REQUEST_CAMERA_PERMISSION_RESULT);
                }

            } else {
                cameraManager.openCamera(mCameraId, mCameraDeviceStateCallback, mBackgroundHandler);
            }
        } catch (Exception e) {
            Log.i(TAG, "Error on connecting camera: " + e);
        }
    }

    private void startRecord() {
        try {
            if(mIsRecording) {
                setupMediaRecorder();
            }

            SurfaceTexture surfaceTexture = mTextureView.getSurfaceTexture();
            //surfaceTexture.setDefaultBufferSize(mPreviewSize.getWidth(), mPreviewSize.getHeight());
            Surface previewSurface = new Surface(surfaceTexture);
            Surface recordSurface = mMediaRecorder.getSurface();
            mCaptureRequestBuilder = mCameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_RECORD);
            mCaptureRequestBuilder.addTarget(previewSurface);
            mCaptureRequestBuilder.addTarget(recordSurface);

            mCameraDevice.createCaptureSession(Arrays.asList(previewSurface, recordSurface),//, mImageReader.getSurface()),
                    new CameraCaptureSession.StateCallback() {
                        @Override
                        public void onConfigured(CameraCaptureSession session) {
                            mRecordCaptureSession = session;
                            try {
                                mRecordCaptureSession.setRepeatingRequest(
                                        mCaptureRequestBuilder.build(), null, null
                                );
                            } catch (Exception e) {
                                Log.i(TAG, "Error on configuration: " + e);
                            }
                        }

                        @Override
                        public void onConfigureFailed(CameraCaptureSession session) {
                            Log.i(TAG, "Error configuration failed");
                        }
                    }, null);

        } catch (Exception e) {
            Log.i(TAG, "Error on start recording: " + e);
        }
    }

    private void closeCamera() {
        if(mCameraDevice != null) {
            mCameraDevice.close();
            mCameraDevice = null;
        }
        if(mMediaRecorder != null) {
            mMediaRecorder.release();
            mMediaRecorder = null;
        }
    }

    private void startBackgroundThread() {
        mBackgroundHandlerThread = new HandlerThread("Camera2VideoImage");
        mBackgroundHandlerThread.start();
        mBackgroundHandler = new Handler(mBackgroundHandlerThread.getLooper());
    }

    private void stopBackgroundThread() {
        mBackgroundHandlerThread.quitSafely();
        try {
            mBackgroundHandlerThread.join();
            mBackgroundHandlerThread = null;
            mBackgroundHandler = null;
        } catch (Exception e) {
            Log.i(TAG, "Error on stop background thread: " + e);
        }
    }

    private static int sensorToDeviceRotation(CameraCharacteristics cameraCharacteristics, int deviceOrientation) {
        int sensorOrienatation = cameraCharacteristics.get(CameraCharacteristics.SENSOR_ORIENTATION);
        deviceOrientation = ORIENTATIONS.get(deviceOrientation);
        return (sensorOrienatation + deviceOrientation + 360) % 360;
    }

    private static Size chooseOptimalSize(Size[] choices, int width, int height) {
        List<Size> bigEnough = new ArrayList<Size>();
        for(Size option : choices) {
            if(option.getHeight() == option.getWidth() * height / width &&
                    option.getWidth() >= width && option.getHeight() >= height) {
                bigEnough.add(option);
            }
        }
        if(bigEnough.size() > 0) {
            return Collections.min(bigEnough, new CompareSizeByArea());
        } else {
            return choices[0];
        }
    }

    private void createVideoFolder() {
        File movieFile = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES); //change directory
        mVideoFolder = new File(movieFile, "camera2VideoImage");
        if(!mVideoFolder.exists()) {
            mVideoFolder.mkdirs();
        }
    }

    private File createVideoFileName() throws IOException { //change name (too long)
        String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String prepend = "VIDEO_" + timestamp + "_";
        File videoFile = File.createTempFile(prepend, ".mp4", mVideoFolder);
        mVideoFileName = videoFile.getAbsolutePath();
        return videoFile;
    }

    @SuppressLint("SetTextI18n")
    private void checkWriteStoragePermission() {
        //set view and text while buffering
        videoView.setVisibility(View.INVISIBLE);
        text_intro.setVisibility(View.INVISIBLE);
        title.setVisibility(View.INVISIBLE);
        textView.setVisibility(View.INVISIBLE);
        keeprec.setText("Keep the statue face inside the rectangle: soon will appear something!");
        keeprec.setVisibility(View.VISIBLE);

        if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if(ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED) {
                try {
                    createVideoFileName();
                } catch (Exception e) {
                    Log.i(TAG, "Error on check write storage permissions: " + e);
                }
                if(mIsTimelapse || mIsRecording) {
                    startRecord();
                    mMediaRecorder.start();
                    mChronometer.setBase(SystemClock.elapsedRealtime());
                    mChronometer.setVisibility(View.VISIBLE);
                    mChronometer.start();
                }
            } else {
                if(shouldShowRequestPermissionRationale(Manifest.permission.WRITE_EXTERNAL_STORAGE)) {
                    Toast.makeText(this, "app needs to be able to save videos", Toast.LENGTH_SHORT).show();
                }
                requestPermissions(new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, REQUEST_WRITE_EXTERNAL_STORAGE_PERMISSION_RESULT);
            }
        } else {
            try {
                createVideoFileName();
            } catch (Exception e) {
                Log.i(TAG, "Error on create video file name: " + e);
            }
            if(mIsRecording || mIsTimelapse) {
                startRecord();
                mMediaRecorder.start();
                mChronometer.setBase(SystemClock.elapsedRealtime());
                mChronometer.setVisibility(View.VISIBLE);
                mChronometer.start();
            }
        }
    }

    private void setupMediaRecorder() throws IOException {
        mMediaRecorder.setVideoSource(MediaRecorder.VideoSource.SURFACE);
        mMediaRecorder.setAudioSource(MediaRecorder.AudioSource.MIC);
        mMediaRecorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);
        mMediaRecorder.setOutputFile(mVideoFileName);
        mMediaRecorder.setVideoEncodingBitRate(1000000);
        mMediaRecorder.setVideoFrameRate(30);
        mMediaRecorder.setVideoSize(1280, 720);//(mVideoSize.getWidth(), mVideoSize.getHeight());
        mMediaRecorder.setVideoEncoder(MediaRecorder.VideoEncoder.H264);
        mMediaRecorder.setAudioEncoder(MediaRecorder.AudioEncoder.AAC);
        mMediaRecorder.setOrientationHint(mTotalRotation);
        //mMediaRecorder.setMaxDuration(3000); //set the length of the video (stop automatically)
        mMediaRecorder.prepare();
    }


    /** check and methods for http requests **/

    // Storage Permissions
    private static final int REQUEST_EXTERNAL_STORAGE = 1;
    private static String[] PERMISSIONS_STORAGE = {
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
    };
    public static void verifyStoragePermissions(Activity activity) {
        // Check if we have write permission
        int permission = ActivityCompat.checkSelfPermission(activity, Manifest.permission.WRITE_EXTERNAL_STORAGE);

        if (permission != PackageManager.PERMISSION_GRANTED) {
            // We don't have permission so prompt the user
            ActivityCompat.requestPermissions(
                    activity,
                    PERMISSIONS_STORAGE,
                    REQUEST_EXTERNAL_STORAGE
            );
        }
    }

    public class Send_video extends Thread {
        @Override
        public void run() {
            keeprec.setText("wait for the deep fake");
            keeprec.setVisibility(View.VISIBLE);
            mTextureView.setVisibility(View.INVISIBLE);
            rectangle.setVisibility(View.INVISIBLE);
            String path = old_filename;
            //check if send as statue or person
            String serverUrl = "";
            if (switch_btt.isChecked()) {
                serverUrl = server + "video_give_statue";
            } else {
                serverUrl = server + "video_give_people";
            }

            if (check_taggle_btn.isChecked()) {
                tipo = 1;
            } else {
                tipo = 0;
            }


            //Log.i(TAG, "file passed (original video): " + path);
            Log.i(TAG, "server call: " + serverUrl);
            Log.i(TAG, "start communication");

            response = new StringBuilder();

            try {
                Log.i(TAG, "start");
                File videoFile = new File(path);
                URL url = new URL(serverUrl);

                /** method 1 **/
                File video_received = new File(path);
                byte[] videoBytes = FileUtils.readFileToByteArray(video_received);
                // Encode the video bytes as a Base64 string
                String base64Video = Base64.encodeToString(videoBytes, Base64.DEFAULT);
                Log.i(TAG, "base64video length: " + base64Video.length());

                // Set up the HTTP connection
                HttpURLConnection connection = (HttpURLConnection) url.openConnection();
                connection.setRequestMethod("POST");
                //connection.setRequestProperty("Content-Type", "application/json");
                connection.setRequestProperty("Content-Type", "application/json; utf-8");
                connection.setRequestProperty("Accept", "application/json");

                Log.i(TAG, "connection set");


                // Construct the JSON request body with the Base64 video data
                JSONObject json_data = new JSONObject();
                json_data.put("index", 0);
                json_data.put("type", tipo);
                json_data.put("video", base64Video);


                String jsonString = json_data.toString();
                Log.i(TAG, "json string created: " + jsonString);
                byte[] postDataBytes = jsonString.getBytes(StandardCharsets.UTF_8);
                Log.i(TAG, "post data bytes: " + postDataBytes.length);
                Log.i(TAG, "try to write it");
                connection.getOutputStream().write(postDataBytes);
                Log.i(TAG, "json wrote on connection");


                Log.i(TAG, "start response");
                BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(connection.getInputStream(), StandardCharsets.UTF_8));
                String line;
                while ((line = bufferedReader.readLine()) != null) {
                    response.append(line);
                }
                bufferedReader.close();

                //read response
                int responseCode = connection.getResponseCode();

                String re = response.toString().intern();

                if (re == "Error") {
                    queue_response.add(false);
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            Log.i(TAG, "Error on showing video!");
                            textView.setText("The face of the statue is not detected, please retry");
                            textView.setVisibility(View.VISIBLE);
                            keeprec.setVisibility(View.INVISIBLE);
                            progressbar.setVisibility(View.INVISIBLE);
                            anotherone.setVisibility(View.VISIBLE);
                            mTextureView.setVisibility(View.INVISIBLE);
                            err.setVisibility(View.VISIBLE);
                            videoView.setVisibility(View.INVISIBLE);
                        }
                    });
                } else {
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            keeprec.setText("wait for the deep fake");
                            keeprec.setVisibility(View.VISIBLE);
                            videoView.setVisibility(View.INVISIBLE);
                            mTextureView.setVisibility(View.INVISIBLE);
                            progressbar.setVisibility(View.INVISIBLE);
                            switch_btt.setVisibility(View.INVISIBLE);
                            check_taggle_btn.setVisibility(View.INVISIBLE);
                            start.setVisibility(View.INVISIBLE);
                            back.setVisibility(View.INVISIBLE);

                            Log.i(TAG, "start communication");

                            String videoUrl = server + "video_display";

                            try {
                                keeprec.setVisibility(View.INVISIBLE);
                                Uri uri = Uri.parse(videoUrl);
                                videoView.setVideoURI(uri);
                                videoView.setVisibility(View.VISIBLE);
                                videoView.start();
                                videoView.setOnCompletionListener(new MediaPlayer.OnCompletionListener() {
                                    public void onCompletion(MediaPlayer mp) {
                                        replay.setVisibility(View.VISIBLE);
                                        save.setVisibility(View.VISIBLE);
                                        anotherone.setVisibility(View.VISIBLE);
                                    }
                                });
                            } catch (Exception e) {
                                Log.i(TAG, "Error on showing video: " + e);
                                textView.setText("The face of the statue is not detected, please retry");
                                textView.setVisibility(View.VISIBLE);
                                anotherone.setVisibility(View.VISIBLE);
                                mTextureView.setVisibility(View.INVISIBLE);
                                videoView.setVisibility(View.INVISIBLE);

                            }
                        }
                    });
                }
            } catch (Exception e) {
                Log.i(TAG, "Error: " + e);
            }

        }
    }


    public class Save_video extends Thread {
        public void run() {

            if (!flag) {

            // Reading input via BufferedReader class
            BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
            // Try block to check for exceptions
            try {

                // URL for microsoft cognitive server.
                URL url = new URL(server + "video_display");
                HttpURLConnection con = (HttpURLConnection) url.openConnection();
                con.setRequestMethod("GET");

                InputStream ip = con.getInputStream();
                BufferedReader br1 = new BufferedReader(new InputStreamReader(ip));

                StringBuilder response = new StringBuilder();
                String responseSingle = null;

                int responseCode = con.getResponseCode();
                if (responseCode == HttpURLConnection.HTTP_OK) {

                    String videoFileName = "video" + System.currentTimeMillis() + ".mp4"; // Replace with your desired file name and extension

                    ContentResolver contentResolver = getContentResolver();
                    ContentValues contentValues = new ContentValues();
                    contentValues.put(MediaStore.Video.Media.TITLE, videoFileName);
                    contentValues.put(MediaStore.Video.Media.DISPLAY_NAME, videoFileName);
                    contentValues.put(MediaStore.Video.Media.MIME_TYPE, "video/mp4");
                    contentValues.put(MediaStore.Video.Media.DATE_ADDED, System.currentTimeMillis());
                    contentValues.put(MediaStore.Video.Media.DATA, Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MOVIES) + "/" + videoFileName);

                    Uri videoFile = contentResolver.insert(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, contentValues);

                    Log.i(TAG, "Response: file path " + videoFile);


                    //OutputStream outputStream = Files.newOutputStream(videoFile.toPath());
                    OutputStream outputStream = contentResolver.openOutputStream(videoFile);
                    byte[] buffer = new byte[4096];
                    int bytesRead;

                    while ((bytesRead = con.getInputStream().read(buffer)) != -1) {
                        outputStream.write(buffer, 0, bytesRead);
                    }
                    outputStream.flush();
                    Log.i(TAG, "Response: wrote");
                }
                con.disconnect();

                Log.i(TAG, "Response: done");

                runOnUiThread(new Runnable() {
                    public void run() {
                        Toast.makeText(getApplicationContext(), "Video saved on gallery/movie!", Toast.LENGTH_SHORT).show();
                        flag = true;
                    }
                });
            } catch (Exception e) {
                Log.i(TAG, "Error on savin the video: " + e);
            }

        } else {
                runOnUiThread(new Runnable() {
                    public void run() {
                        Toast.makeText(getApplicationContext(), "Video already saved", Toast.LENGTH_SHORT).show();
                    }
                });
            }
        }
    }

    public class Show_video extends Thread {
        @Override
        public void run() {
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    keeprec.setText("wait for the deep fake");
                    keeprec.setVisibility(View.VISIBLE);
                    videoView.setVisibility(View.INVISIBLE);
                    mTextureView.setVisibility(View.INVISIBLE);
                    progressbar.setVisibility(View.INVISIBLE);
                    switch_btt.setVisibility(View.INVISIBLE);
                    check_taggle_btn.setVisibility(View.INVISIBLE);
                    start.setVisibility(View.INVISIBLE);
                    back.setVisibility(View.INVISIBLE);

                    Log.i(TAG, "start communication");

                    String videoUrl = server + "video_display";
                    try {
                        keeprec.setVisibility(View.INVISIBLE);
                        Uri uri = Uri.parse(videoUrl);
                        videoView.setVideoURI(uri);
                        videoView.setVisibility(View.VISIBLE);
                        videoView.start();
                        videoView.setOnCompletionListener(new MediaPlayer.OnCompletionListener() {
                            public void onCompletion(MediaPlayer mp) {
                                replay.setVisibility(View.VISIBLE);
                                save.setVisibility(View.VISIBLE);
                                anotherone.setVisibility(View.VISIBLE);
                            }
                        });
                    } catch (Exception e) {
                        Log.i(TAG, "Error on showing video: " + e);
                        textView.setText("The face of the statue is not detected, please retry");
                        textView.setVisibility(View.VISIBLE);
                        anotherone.setVisibility(View.VISIBLE);
                        mTextureView.setVisibility(View.INVISIBLE);
                        videoView.setVisibility(View.INVISIBLE);

                    }
                }
            });
        }
    }
}

