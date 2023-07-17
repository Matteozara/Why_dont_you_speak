package com.example.whole_video_app;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;

public class SplashActivity extends AppCompatActivity {
    public static int SPLASH_TIMER=2000;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_splash);

        new Handler().postDelayed(new Runnable() {
            @Override
            public void run() {
                startActivity(new Intent(SplashActivity.this, MenuActivity.class));
                //Intent intent = new Intent(SplashActivity.this, MainActivity.class);
                //startActivity(intent);
                finish();
            }
        }, SPLASH_TIMER); // Delay in milliseconds
    }
}
