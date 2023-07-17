package com.example.whole_video_app;

import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;

import androidx.appcompat.app.AppCompatActivity;

public class MenuActivity extends AppCompatActivity {

    private Button audio, deepfake, explanation;

    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.menu);

        // initialising all views through id defined above
        audio = findViewById(R.id.audio);
        deepfake = findViewById(R.id.deepfake);
        explanation = findViewById(R.id.explanation);

        // Set on Click Listener on Registration button
        audio.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(MenuActivity.this, AudioActivity.class);
                startActivity(intent);
                finish();
            }
        });

        deepfake.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(MenuActivity.this, MainActivity.class);
                startActivity(intent);
                finish();
            }
        });

        explanation.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(MenuActivity.this, ExplanationActivity.class);
                startActivity(intent);
                finish();
            }
        });
    }
}
