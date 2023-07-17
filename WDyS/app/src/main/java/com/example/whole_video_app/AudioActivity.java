package com.example.whole_video_app;

import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.media.MediaPlayer;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.os.Environment;
import android.os.SystemClock;
import android.util.Base64;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.Chronometer;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import static android.Manifest.permission.RECORD_AUDIO;
import static android.Manifest.permission.WRITE_EXTERNAL_STORAGE;

import org.apache.commons.io.FileUtils;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;

public class AudioActivity extends AppCompatActivity {
    private TextView title, text, recording;

    private String TAG = "REC123";

    public final String server = "http://172.20.10.5:3535/";

    private Button stop_record, play_rec, stop_rec, start_record, back, go_video, retake;
    private Chronometer mChronometer;

    public Runnable r_send;

    // creating a variable for media recorder object class.
    private MediaRecorder mRecorder;

    // creating a variable for mediaplayer class
    private MediaPlayer mPlayer;

    // string variable is created for storing a file name
    private static String mFileName = null;

    // constant for storing audio permission
    public static final int REQUEST_AUDIO_PERMISSION_CODE = 1;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_audio);

        //if hasn't permission
        if (!checkPermissions()) {
            ActivityCompat.requestPermissions(this, new String[]{RECORD_AUDIO}, REQUEST_AUDIO_PERMISSION_CODE);
        }

        back = findViewById(R.id.back);
        title = findViewById(R.id.title);
        text = findViewById(R.id.tt);
        mChronometer = (Chronometer) findViewById(R.id.chronometer);
        recording = findViewById(R.id.recording);
        go_video = findViewById(R.id.go_video);
        retake = findViewById(R.id.record_anotherone);
        start_record = findViewById(R.id.btnRecord);
        stop_record = findViewById(R.id.btnStop);
        play_rec = findViewById(R.id.btnPlay);
        stop_rec = findViewById(R.id.btnStopPlay);
        stop_record.setBackground(ContextCompat.getDrawable(this, R.drawable.gray_button));
        stop_record.setEnabled(false);

        text.setText("Start registering your audio and create your personal deepfake! (around 12 secs audio)");

        go_video.setVisibility(View.INVISIBLE);
        retake.setVisibility(View.INVISIBLE);
        play_rec.setVisibility(View.INVISIBLE);
        stop_rec.setVisibility(View.INVISIBLE);
        recording.setVisibility(View.INVISIBLE);



        r_send = new Runnable() {
            @Override
            public void run() {
                Send_audio send_thread = new Send_audio();
                send_thread.start();
            }

        };




        start_record.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                startRecording();
            }
        });

        stop_record.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                pauseRecording();
            }
        });

        play_rec.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                playAudio();
            }
        });

        stop_rec.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                pausePlaying();
            }
        });

        back.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(AudioActivity.this, MenuActivity.class);
                startActivity(intent);
                finish();
            }
        });

        go_video.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                r_send.run();
            }
        });

        retake.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                stop_record.setBackground(ContextCompat.getDrawable(getApplicationContext(), R.drawable.gray_button));
                start_record.setBackground(ContextCompat.getDrawable(getApplicationContext(), R.drawable.replay_button));
                start_record.setEnabled(true);
                stop_record.setEnabled(false);
                start_record.setVisibility(View.VISIBLE);
                stop_record.setVisibility(View.VISIBLE);
                text.setText("Start registering your audio and create your personal deepfake! (around 12 secs audio)");

                go_video.setVisibility(View.INVISIBLE);
                retake.setVisibility(View.INVISIBLE);
                play_rec.setVisibility(View.INVISIBLE);
                stop_rec.setVisibility(View.INVISIBLE);
                recording.setVisibility(View.INVISIBLE);
            }
        });
    }

    private void startRecording() {
        stop_record.setBackground(ContextCompat.getDrawable(this, R.drawable.replay_button));
        stop_record.setEnabled(true);
        start_record.setBackground(ContextCompat.getDrawable(this, R.drawable.gray_button));
        start_record.setEnabled(false);
        recording.setVisibility(View.VISIBLE);
        mChronometer.setBase(SystemClock.elapsedRealtime());
        mChronometer.setVisibility(View.VISIBLE);
        mChronometer.start();

        mFileName = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS).getAbsolutePath();//Environment.getExternalStorageDirectory().getAbsolutePath();
        mFileName += "/AudioRecording.3gp";

        mRecorder = new MediaRecorder();
        mRecorder.setAudioSource(MediaRecorder.AudioSource.MIC);
        mRecorder.setOutputFormat(MediaRecorder.OutputFormat.THREE_GPP);
        mRecorder.setAudioEncoder(MediaRecorder.AudioEncoder.AMR_NB);
        mRecorder.setOutputFile(mFileName);

        try {
            mRecorder.prepare();
            mRecorder.start();
            } catch (Exception e) {
                Log.i(TAG, "Error on recording audio: " + e);
            }
    }


    public void playAudio() {
        play_rec.setBackground(ContextCompat.getDrawable(this, R.drawable.gray_button));
        play_rec.setEnabled(false);
        stop_rec.setBackground(ContextCompat.getDrawable(this, R.drawable.replay_button));
        stop_rec.setEnabled(true);

        mPlayer = new MediaPlayer();
        try {
            mPlayer.setDataSource(mFileName);
            mPlayer.prepare();
            mPlayer.setOnCompletionListener(new MediaPlayer.OnCompletionListener() {
                @Override
                public void onCompletion(MediaPlayer mp) {
                    play_rec.setBackground(ContextCompat.getDrawable(getApplicationContext(), R.drawable.replay_button));
                    play_rec.setEnabled(true);
                    stop_rec.setBackground(ContextCompat.getDrawable(getApplicationContext(), R.drawable.gray_button));
                    stop_rec.setEnabled(false);
                }
            });
            mPlayer.start();
        } catch (Exception e) {
            Log.i(TAG, "Error on playing audio: " + e);
        }
    }

    public void pauseRecording() {
        play_rec.setBackground(ContextCompat.getDrawable(this, R.drawable.replay_button));
        play_rec.setEnabled(true);
        stop_rec.setBackground(ContextCompat.getDrawable(this, R.drawable.gray_button));
        stop_rec.setEnabled(false);
        play_rec.setVisibility(View.VISIBLE);
        stop_rec.setVisibility(View.VISIBLE);
        go_video.setVisibility(View.VISIBLE);
        retake.setVisibility(View.VISIBLE);

        mChronometer.stop();
        mChronometer.setVisibility(View.INVISIBLE);
        start_record.setVisibility(View.INVISIBLE);
        recording.setVisibility(View.INVISIBLE);
        stop_record.setVisibility(View.INVISIBLE);
        text.setText("You have recorded your audio! \nIf it’s ok press \"Record video\" and create a deepfake of the statue with your voice");

        mRecorder.stop();
        mRecorder.release();
        mRecorder = null;
    }

    public void pausePlaying() {
        play_rec.setBackground(ContextCompat.getDrawable(this, R.drawable.replay_button));
        play_rec.setEnabled(true);
        stop_rec.setBackground(ContextCompat.getDrawable(this, R.drawable.gray_button));
        stop_rec.setEnabled(false);

        mPlayer.release();
        mPlayer = null;
    }

    private boolean checkPermissions() {
        return ActivityCompat.checkSelfPermission(this, RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        switch (requestCode) {
            case REQUEST_AUDIO_PERMISSION_CODE:
                if (grantResults.length > 0) {
                    boolean permissionToRecord = grantResults[0] == PackageManager.PERMISSION_GRANTED;
                    //boolean permissionToStore = grantResults[1] == PackageManager.PERMISSION_GRANTED;
                    if (!permissionToRecord) {// && permissionToStore) {
                        //Toast.makeText(getApplicationContext(), "Permission Denied", Toast.LENGTH_LONG).show();
                        ActivityCompat.requestPermissions(AudioActivity.this, new String[]{RECORD_AUDIO, WRITE_EXTERNAL_STORAGE}, REQUEST_AUDIO_PERMISSION_CODE);
                    }
                }
                break;
        }
    }


    public class Send_audio extends Thread {
        @Override
        public void run() {
            runOnUiThread(new Runnable() {
                              @Override
                              public void run() {
                                  recording.setText("Sending audio...");
                                  recording.setTextColor(Color.WHITE);
                                  recording.setVisibility(View.VISIBLE);
                              }
                          });

            String serverUrl = server + "send_audio";

            Log.i(TAG, "start communication");

            StringBuilder response = new StringBuilder();
            try {
                File audioFile = new File(mFileName);
                URL url = new URL(serverUrl);
                byte[] videoBytes = FileUtils.readFileToByteArray(audioFile);
                // Encode the video bytes as a Base64 string
                String base64Audio = Base64.encodeToString(videoBytes, Base64.DEFAULT);
                Log.i(TAG, "base64video length: " + base64Audio.length());

                // Set up the HTTP connection
                HttpURLConnection connection = (HttpURLConnection) url.openConnection();
                connection.setRequestMethod("POST");
                //connection.setRequestProperty("Content-Type", "application/json");
                connection.setRequestProperty("Content-Type", "application/json; utf-8");
                connection.setRequestProperty("Accept", "application/json");

                Log.i(TAG, "connection set");

                // Construct the JSON request body with the Base64 video data
                JSONObject json_data = new JSONObject();
                json_data.put("audio", base64Audio);

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
                String re = response.toString().intern();
                //read response
                int responseCode = connection.getResponseCode();
                //String responseMessage = connection.getResponseMessage();

                if (re == "Error") {
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            Toast.makeText(getApplicationContext(), "The audio has some truobles, please record it again!", Toast.LENGTH_SHORT).show();
                            recording.setText("recording...");
                            recording.setTextColor(Color.RED);
                            recording.setVisibility(View.INVISIBLE);

                            stop_record.setBackground(ContextCompat.getDrawable(getApplicationContext(), R.drawable.gray_button));
                            start_record.setBackground(ContextCompat.getDrawable(getApplicationContext(), R.drawable.replay_button));
                            start_record.setEnabled(true);
                            stop_record.setEnabled(false);
                            start_record.setVisibility(View.VISIBLE);
                            stop_record.setVisibility(View.VISIBLE);
                            text.setText("Start registering your audio and create your personal deepfake! (around 12 secs audio)");

                            go_video.setVisibility(View.INVISIBLE);
                            retake.setVisibility(View.INVISIBLE);
                            play_rec.setVisibility(View.INVISIBLE);
                            stop_rec.setVisibility(View.INVISIBLE);
                            recording.setVisibility(View.INVISIBLE);
                        }
                    });
                } else {
                    Intent intent = new Intent(AudioActivity.this, VideoAudioActivity.class);
                    startActivity(intent);
                    finish();
                }

            } catch (Exception e) {
                Log.i(TAG, "Error on sending audio: " + e);
            }

        }
    }
}

