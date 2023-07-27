package com.example.whole_video_app;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.Resources;
import android.graphics.Color;
import android.graphics.SurfaceTexture;
import android.graphics.drawable.Drawable;
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

import androidx.annotation.DrawableRes;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.json.JSONObject;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Timer;

public class ExplanationActivity extends AppCompatActivity {

    private static final String TAG = "REC123";

    private static final int REQUEST_CAMERA_PERMISSION_RESULT = 0;
    private static final int REQUEST_WRITE_EXTERNAL_STORAGE_PERMISSION_RESULT = 1;
    private static final int STATE_PREVIEW = 0;
    private static final int STATE_WAIT_LOCK = 1;
    private int mCaptureState = STATE_PREVIEW;
    private TextureView mTextureView;
    private TextView textView;
    private View rectangle;
    private Button replay, start, back, deepfake, save;
    private Button anotherone;
    private ProgressBar progressbar;

    private TextView title, statue_name, statue_description, text_intro;
    private TextView keeprec;
    private ImageView err, statue_image;

    StringBuilder response = new StringBuilder();
    StringBuilder response_info = new StringBuilder();

    //private String state_switch;
    private VideoView videoView;
    private Runnable r_show;
    private Runnable r_end, r_save;
    public String old_filename;

    public boolean flag = false;

    public ToggleButton check_taggle_btn;

    public int tipo;

    public final String server = "http://172.20.10.5:3535/";

    public Queue queue_response = new LinkedList();


    public Map<String, String> definitions = new HashMap<>();
    public Map<String, String> names = new HashMap<>();

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        //cretae the definition dictionary
        definitions.put("atena", "Athena or Athene, often given the epithet Pallas, is an ancient Greek goddess associated with wisdom, warfare, and handicraft who was later syncretized with the Roman goddess Minerva. Athena was regarded as the patron and protectress of various cities across Greece, particularly the city of Athens, from which she most likely received her name. The Parthenon on the Acropolis of Athens is dedicated to her. Her major symbols include owls, olive trees, snakes, and the Gorgoneion. In art, she is generally depicted wearing a helmet and holding a spear.");
        definitions.put("arringatore", "The Orator, also known as L'Arringatore (Italian), Aule Meteli (Etruscan) or Aulus Metellus (Latin), is an Etruscan bronze sculpture from the late second or the early first century BC. Aulus Metellus was an Etruscan senator in the Roman republic, originally from Perugia or Cortona. The Aulus Metellus sculpture was found in 1566 with the exact location being debated, but all sources agree the sculpture was found either in or around Lake Trasimeno in the province of Perugia on the border between Umbria and Tuscany, 177 kilometers (110 miles) from Rome. The statue is exhibited in the National Archaeological Museum of Florence.");
        definitions.put("atena_armata", "The statue of Armed Athena (Ancient Greek: Παρθένος Ἀθηνᾶ, lit. 'Athena the Virgin') was a monumental chryselephantine sculpture of the goddess Athena. Attributed to Phidias and dated to the mid-fifth century BCE, it was an offering from the city of Athens to Athena, its tutelary deity. The naos of the Parthenon on the acropolis of Athens was designed exclusively to accommodate it.");
        definitions.put("demostene", "Demosthenes (/dɪˈmɒs.θəniːz/; Greek: Δημοσθένης, romanized: Dēmosthénēs; Attic Greek: [dɛːmosˈtʰenɛːs]; 384 – 12 October 322 BC) was a Greek statesman and orator in ancient Athens. His orations constitute a significant expression of contemporary Athenian intellectual prowess and provide insight into the politics and culture of ancient Greece during the 4th century BC. Demosthenes learned rhetoric by studying the speeches of previous great orators. He delivered his first judicial speeches at the age of 20, in which he successfully argued that he should gain from his guardians what was left of his inheritance. For a time, Demosthenes made his living as a professional speechwriter (logographer) and a lawyer, writing speeches for use in private legal suits.");
        definitions.put("dioniso", "In ancient Greek religion and myth, Dionysus (/daɪ.əˈnaɪsəs/; Ancient Greek: Διόνυσος Dionysos) is the god of wine-making, orchards and fruit, vegetation, fertility, festivity, insanity, ritual madness, religious ecstasy, and theatre. He was also known as Bacchus (/ˈbækəs/ or /ˈbɑːkəs/; Ancient Greek: Βάκχος Bacchos) by the Greeks (a name later adopted by the Romans) for a frenzy he is said to induce called baccheia. As Dionysus Eleutherios (\"the liberator\"), his wine, music, and ecstatic dance free his followers from self-conscious fear and care, and subvert the oppressive restraints of the powerful. His thyrsus, a fennel-stem sceptre, sometimes wound with ivy and dripping with honey, is both a beneficent wand and a weapon used to destroy those who oppose his cult and the freedoms he represents. Those who partake of his mysteries are believed to become possessed and empowered by the god himself.");
        definitions.put("era_barberini", "The Barberini Hera or Barberini Juno is a Roman sculpture of Hera or Juno, copied from a Greek original. The statue depicts the goddess standing, wearing a crown and peplos (which clings to show her form beneath and has dropped from her left shoulder, nearly revealing her breast) and now resting the weight of her restored right arm on a standing sceptre and carrying a patera in her left.");
        definitions.put("ercole", "Hercules (/ˈhɜːrkjʊˌliːz/, US: /-kjə-/) is the Roman equivalent of the Greek divine hero Heracles, son of Jupiter and the mortal Alcmena. In classical mythology, Hercules is famous for his strength and for his numerous far-ranging adventures. The Romans adapted the Greek hero's iconography and myths for their literature and art under the name Hercules. In later Western art and literature and in popular culture, Hercules is more commonly used than Heracles as the name of the hero. Hercules is a multifaceted figure with contradictory characteristics, which enabled later artists and writers to pick and choose how to represent him. This article provides an introduction to representations of Hercules in the later tradition.");
        definitions.put("kouros_da_tenea", "The grave statue of a youth from Tenea known as the Kouros of Tenea (formerly Apollo of Tenea) is now located in the Glyptothek in Munich, Germany. The archaic Kouros was created in North-East Peloponnese about 560 BC. The Parian marble statue was discovered in 1846, approximately twenty kilometers South of Ancient Corinth at the site of ancient Tenea. The Kouros was acquired by the Glyptothek in 1853.");
        definitions.put("minerva_tritonia", "She is Minerva, or the Greek Pallas Athena, goddess of Wisdom and War Strategy, protective and terrible goddess at the same time. She is called Minerva Tritonia because a triton accompanies her at her side, a very rare element in representations of Minerva, which refers to a little-known legend, according to which the goddess was raised near a river called Triton.");
        definitions.put("poseidone", "Poseidon (/pəˈsaɪdən, pɒ-, poʊ-/; Greek: Ποσειδῶν) is one of the Twelve Olympians in ancient Greek religion and mythology, presiding over the sea, storms, earthquakes and horses. He was the protector of seafarers and the guardian of many Hellenic cities and colonies. In pre-Olympian Bronze Age Greece, Poseidon was venerated as a chief deity at Pylos and Thebes, with the cult title \"earth shaker\"; in the myths of isolated Arcadia, he is related to Demeter and Persephone and was venerated as a horse, and as a god of the waters. Poseidon maintained both associations among most Greeks: He was regarded as the tamer or father of horses, who, with a strike of his trident, created springs (in the Greek language, the terms for both are related). His Roman equivalent is Neptune.");
        definitions.put("zeus", "Zeus (/zjuːs/; Ancient Greek: Ζεύς) is the sky and thunder god in ancient Greek religion, who rules as king of the gods on Mount Olympus. His name is cognate with the first element of his Roman equivalent Jupiter. His mythology and powers are similar, though not identical, to those of Indo-European deities such as Jupiter, Perkūnas, Perun, Indra, Dyaus, and Zojz. Zeus is the child of Cronus and Rhea, the youngest of his siblings to be born, though sometimes reckoned the eldest as the others required disgorging from Cronus's stomach. In most traditions, he is married to Hera, by whom he is usually said to have fathered Ares, Eileithyia, Hebe, and Hephaestus. At the oracle of Dodona, his consort was said to be Dione, by whom the Iliad states that he fathered Aphrodite.");
        definitions.put("altro", "The statue is not one of the “talking statues” project, so the story of this statue is still unknown, but, you can still generate its DeepFake!");


        //cretae the names dictionary
        names.put("atena", "Athena");
        names.put("arringatore", "The Orator");
        names.put("atena_armata", "Armed Athena");
        names.put("demostene", "Demosthenes");
        names.put("dioniso", "Dionysus");
        names.put("era_barberini", "Hera Barberini");
        names.put("ercole", "Hercules");
        names.put("kouros_da_tenea", "Kouros of Tenea");
        names.put("minerva_tritonia", "Minerva Tritonia");
        names.put("poseidone", "Poseidon");
        names.put("zeus", "Zeus");
        names.put("altro", "Unknown");


        super.onCreate(savedInstanceState);
        setContentView(R.layout.explanation);

        verifyStoragePermissions(this);

        createVideoFolder();
        //createImageFolder();

        mChronometer = (Chronometer) findViewById(R.id.chronometer);
        mTextureView = (TextureView) findViewById(R.id.textureView);
        videoView = (VideoView) findViewById(R.id.vv);
        textView = (TextView) findViewById(R.id.tt);
        text_intro = findViewById(R.id.intro);
        rectangle = findViewById(R.id.rectangle);
        replay = findViewById(R.id.restart);
        anotherone = findViewById(R.id.anotherone);
        progressbar = findViewById(R.id.progressbar);

        title = findViewById(R.id.title);
        keeprec = findViewById(R.id.keeprec);
        err = findViewById(R.id.er);

        check_taggle_btn = findViewById(R.id.check_toggle_btn);

        keeprec.setVisibility(View.INVISIBLE);
        err.setVisibility(View.INVISIBLE);

        title.setText("Know your statue");
        text_intro.setText("Just follow the instructions:\n\n" +
                "1) Choose a statue and a modality (fast or precise)\n\n" +
                "2) Record a video (with \"start\"), it’s important to clearly record the face\n\n" +
                "3) The information about the statue will appear\n\n" +
                "4) Afterwards you can generate the DeepFake if you want! (based on the prerecorded video)");
        text_intro.setVisibility(View.VISIBLE);
        title.setVisibility(View.VISIBLE);


        replay.setVisibility(View.INVISIBLE);
        anotherone.setVisibility(View.INVISIBLE);
        rectangle.setVisibility(View.INVISIBLE);
        progressbar.setVisibility(View.INVISIBLE);

        start = findViewById(R.id.start);
        save = findViewById(R.id.save);
        deepfake = findViewById(R.id.show_deepfake);
        statue_name = findViewById(R.id.statue_name);
        statue_description = findViewById(R.id.statue_description);
        statue_image = findViewById(R.id.statue_image);
        back = findViewById(R.id.back);
        statue_image.setVisibility(View.INVISIBLE);
        save.setVisibility(View.INVISIBLE); ////////////////////////////////7
        statue_name.setVisibility(View.INVISIBLE);
        deepfake.setVisibility(View.INVISIBLE);
        statue_description.setVisibility(View.INVISIBLE);



        videoView.setVisibility(View.INVISIBLE);

        //set textview
        //textView.setText("Let's start! choose the subject and record with the button below");
        textView.setVisibility(View.VISIBLE);


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

                // Starting the preview prior to stopping recording which should hopefully
                // resolve issues being seen in Samsung devices.
                //startPreview();
                mMediaRecorder.stop();
                mMediaRecorder.reset();
                old_filename = mVideoFileName;
                //put it in another trade (?)
                Intent mediaStoreUpdateIntent = new Intent(Intent.ACTION_MEDIA_SCANNER_SCAN_FILE);
                mediaStoreUpdateIntent.setData(Uri.fromFile(new File(old_filename)));
                sendBroadcast(mediaStoreUpdateIntent);

                //gallery_video_process(old_filename);
                Receive_info info = new Receive_info();
                info.start();
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
                start.setVisibility(View.INVISIBLE);
                check_taggle_btn.setVisibility(View.INVISIBLE);
                back.setVisibility(View.INVISIBLE);
                mTextureView.setVisibility(View.VISIBLE);

                start_cycle();

            }
        });

        save.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                r_save.run();
            }
        });

        start.setOnLongClickListener(new View.OnLongClickListener() {
            @Override
            public boolean onLongClick(View v) {
                Log.i(TAG, "Inside the onLongClick");
                return true;
            }
        });

        anotherone.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Log.i(TAG, "Start another deepfake");
                Intent intent = new Intent(ExplanationActivity.this, MenuActivity.class);
                startActivity(intent);
                finish();
            }
        });

        replay.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                queue_response.add(true);
                replay.setVisibility(View.INVISIBLE);
                anotherone.setVisibility(View.INVISIBLE);
                save.setVisibility(View.INVISIBLE);
                r_show.run();
            }
        });

        deepfake.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                statue_name.setVisibility(View.INVISIBLE);
                statue_image.setVisibility(View.INVISIBLE);
                statue_description.setVisibility(View.INVISIBLE);
                deepfake.setVisibility(View.INVISIBLE);
                back.setVisibility(View.INVISIBLE);

                keeprec.setText("wait for the deep fake");
                keeprec.setVisibility(View.VISIBLE);
                mTextureView.setVisibility(View.VISIBLE);
                progressbar.setVisibility(View.VISIBLE);
                r_show.run();

            }
        });

        back.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(ExplanationActivity.this, MenuActivity.class);
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
            } else {
                //startPreview();
            }
            // Toast.makeText(getApplicationContext(), "Camera connection made!", Toast.LENGTH_SHORT).show();
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

        new Handler().postDelayed(r_end,13000);
        //new Handler().postDelayed(r_show,40000);
    }


    @Override
    protected void onResume() {
        super.onResume();

        startBackgroundThread();

        if(mTextureView.isAvailable()) {
            setupCamera(mTextureView.getWidth(), mTextureView.getHeight());
            connectCamera();
        }
        /*
        else {
            mTextureView.setSurfaceTextureListener(mSurfaceTextureListener);
        }

        */
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

                //mPreviewSize = chooseOptimalSize(map.getOutputSizes(SurfaceTexture.class), rotatedWidth, rotatedHeight);
                mVideoSize = chooseOptimalSize(map.getOutputSizes(MediaRecorder.class), rotatedWidth, rotatedHeight);
                //mImageSize = chooseOptimalSize(map.getOutputSizes(ImageFormat.JPEG), rotatedWidth, rotatedHeight);


                Log.i(TAG, "rotate width: " + rotatedWidth);
                Log.i(TAG, "rotate height: " + rotatedHeight);
                //Log.i(TAG, "dimension preview: " + mPreviewSize);
                Log.i(TAG, "dimension video: " + mVideoSize);
                //Log.i(TAG, "dimension image: " + mImageSize);

                //mImageReader = ImageReader.newInstance(mImageSize.getWidth(), mImageSize.getHeight(), ImageFormat.JPEG, 1);
                //mImageReader.setOnImageAvailableListener(mOnImageAvailableListener, mBackgroundHandler);
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
            //String serverUrl = "http://172.20.10.2:3535/video_give_statue"; //"http://172.20.10.5:3535/"; //http://127.0.0.1:8080/
            String serverUrl = server + "video_give_statue";

            if (check_taggle_btn.isChecked()) {
                tipo = 1;
            } else {
                tipo = 0;
            }

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
                String res = response.toString();
                res = res.intern();
                int responseCode = connection.getResponseCode();
                //String responseMessage = connection.getResponseMessage();

                if (res == "Error") {
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
                    queue_response.add(true);
                }



                Log.i(TAG, "response code: " + responseCode);
                Log.i(TAG, "###############################response message: " + response);

            } catch (Exception e) {
                Log.i(TAG, "Error: " + e);
            }

        }
    }


    public class Receive_info extends Thread {
        String res = "";
        @Override
        public void run() {
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    keeprec.setText("wait for info");
                    keeprec.setVisibility(View.VISIBLE);
                    mTextureView.setVisibility(View.INVISIBLE);
                    rectangle.setVisibility(View.INVISIBLE);
                }
            });
            String path = old_filename;

            String serverUrl = server + "check_statue";


            Log.i(TAG, "start communication");

            response_info = new StringBuilder();

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
                BufferedReader bufferedReader1 = new BufferedReader(new InputStreamReader(connection.getInputStream(), StandardCharsets.UTF_8));
                String line;
                while ((line = bufferedReader1.readLine()) != null) {
                    response_info.append(line);
                }
                bufferedReader1.close();

                //read response
                int responseCode1 = connection.getResponseCode();
                res = response_info.toString();
                res = res.intern();
                Log.i(TAG, "Response obtained: " + res);

                //String responseMessage = connection.getResponseMessage();

                if (res == "Error") { //no faces detected
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
                    return;
                }

                runOnUiThread(new Runnable() {
                                  @Override
                                  public void run() {
                                      //to set imageView with the right statue (based on the response obtained)
                                      Resources resource = getResources();
                                      int resID = resource.getIdentifier(res, "drawable", getPackageName());
                                      Drawable drawable = ContextCompat.getDrawable(getApplicationContext(), resID);
                                      statue_image.setImageDrawable(drawable);


                                      Log.i(TAG, "Show info!");
                                      keeprec.setVisibility(View.INVISIBLE);
                                      progressbar.setVisibility(View.INVISIBLE);
                                      mTextureView.setVisibility(View.INVISIBLE);
                                      videoView.setVisibility(View.INVISIBLE);
                                      statue_description.setText(definitions.get(res)); //based on description
                                      //statue_image.setImageResource(R.drawable.alcor_logo); //set based on label returned
                                      statue_name.setText(names.get(res).toUpperCase()); //based on label
                                      statue_name.setVisibility(View.VISIBLE);
                                      statue_image.setVisibility(View.VISIBLE);
                                      statue_description.setVisibility(View.VISIBLE);
                                      deepfake.setVisibility(View.VISIBLE);
                                      back.setVisibility(View.VISIBLE);
                                  }
                });


                Log.i(TAG, "response code: " + responseCode1);
                Log.i(TAG, "###############################response message: " + res);

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
                    boolean value;
                    try {
                        value = (boolean)queue_response.remove();
                        if (value) {
                            keeprec.setText("wait for the deep fake");
                            keeprec.setVisibility(View.VISIBLE);
                            videoView.setVisibility(View.INVISIBLE);
                            mTextureView.setVisibility(View.INVISIBLE);
                            progressbar.setVisibility(View.INVISIBLE);
                            start.setVisibility(View.INVISIBLE);
                            check_taggle_btn.setVisibility(View.INVISIBLE);
                            back.setVisibility(View.INVISIBLE);
                            String path = old_filename;

                            Log.i(TAG, "start communication");

                            //String videoUrl = "http://172.20.10.2:3535/video_display";
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
                                textView.setText("Error on generating the deepfake. The face of the statue is not detected, please retry");
                                textView.setVisibility(View.VISIBLE);
                                keeprec.setVisibility(View.INVISIBLE);
                                progressbar.setVisibility(View.INVISIBLE);
                                anotherone.setVisibility(View.VISIBLE);
                                mTextureView.setVisibility(View.INVISIBLE);
                                err.setVisibility(View.VISIBLE);
                                videoView.setVisibility(View.INVISIBLE);
                            }
                        } else {
                            Log.i(TAG, "Error on showing video");
                            textView.setText("Error on generating the deepfake. The face of the statue is not detected, please retry");
                            textView.setVisibility(View.VISIBLE);
                            keeprec.setVisibility(View.INVISIBLE);
                            progressbar.setVisibility(View.INVISIBLE);
                            anotherone.setVisibility(View.VISIBLE);
                            mTextureView.setVisibility(View.INVISIBLE);
                            err.setVisibility(View.VISIBLE);
                            videoView.setVisibility(View.INVISIBLE);
                        }
                    } catch (Exception e) {
                        new Handler().postDelayed(r_show,3000);
                    }
                }
            });


        }
    }
}

