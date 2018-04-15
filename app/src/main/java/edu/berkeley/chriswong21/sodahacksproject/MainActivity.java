package edu.berkeley.chriswong21.sodahacksproject;

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.AsyncTask;
import android.os.Bundle;
import android.support.annotation.NonNull;
import android.text.SpannableString;
import android.text.method.ScrollingMovementMethod;
import android.text.style.RelativeSizeSpan;
import android.util.Log;
import android.view.KeyEvent;
import android.view.SurfaceView;
import android.view.View;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import com.google.api.client.extensions.android.http.AndroidHttp;
import com.google.api.client.json.gson.GsonFactory;
import com.google.api.services.translate.Translate;
import com.google.api.services.translate.model.TranslationsListResponse;
import com.google.api.services.translate.model.TranslationsResource;
import com.google.api.services.vision.v1.Vision;
import com.google.api.services.vision.v1.VisionRequestInitializer;
import com.google.api.services.vision.v1.model.AnnotateImageRequest;
import com.google.api.services.vision.v1.model.AnnotateImageResponse;
import com.google.api.services.vision.v1.model.BatchAnnotateImagesRequest;
import com.google.api.services.vision.v1.model.BatchAnnotateImagesResponse;
import com.google.api.services.vision.v1.model.EntityAnnotation;
import com.google.api.services.vision.v1.model.Feature;
import com.google.api.services.vision.v1.model.Image;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends Activity implements CameraBridgeViewBase.CvCameraViewListener2 {
    private final static int SCREEN_WIDTH = 1280;
    private final static int SCREEN_HEIGHT = 720;
    private final static String TAG = "MainActivity";
    private final static String CLOUD_API_KEY = "Put your API Key here if you want to use this";
    private final static StringEntry[] LANGUAGES = {
            new StringEntry("EN", "English"),
            new StringEntry("ES", "Spanish"),
            new StringEntry("DE", "German"),
            new StringEntry("JA", "Japanese"),
            new StringEntry("BG", "Bulgarian")
    };

    private CameraBridgeViewBase mOpenCvCameraView;
    private Mat mIntermediate;
    private boolean mTapped;
    private boolean mOCRPressed;
    private boolean mCloudIsProcessing;
    private boolean mInitVisibility = false;
    private boolean mShowRequest = false;
    private Toast mToast;
    private double mLabelHeight = 25;
    private int mLangIndex = 1;

    ProgressBar mLoadBar;
    TextView mTextOverlay;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                    mOpenCvCameraView.enableView();
                    break;
                default:
                    super.onManagerConnected(status);
                    break;
            }
        }
    };

    private static class StringEntry {
        String key;
        String val;

        public StringEntry(String key, String val) {
            this.key = key;
            this.val = val;
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        initializeCamera();
        initializeViews();
    }

    //Initialization methods
    private void initializeCamera() {
        setContentView(R.layout.activity_main);
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.main_activity_java_surface_view);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.setMaxFrameSize(SCREEN_WIDTH, SCREEN_HEIGHT);
        mOpenCvCameraView.disableFpsMeter();
        mCloudIsProcessing = false;
    }

    private void initializeViews() {
        mLoadBar = (ProgressBar) findViewById(R.id.progress_bar);
        mTextOverlay = (TextView) findViewById(R.id.text_box);
        mTextOverlay.setTextColor(Color.parseColor("#FFFFFF"));
        mTextOverlay.setMovementMethod(new ScrollingMovementMethod());
        mToast = Toast.makeText(getApplicationContext(), "", Toast.LENGTH_SHORT);
    }

    private void showToast(String msg) {
        if (mToast != null) {
            mToast.setText(msg);
            mToast.show();
        }
    }

    @Override
    public boolean onKeyDown(int keyCode, KeyEvent event) {
        if (keyCode == KeyEvent.KEYCODE_VOLUME_DOWN) {
            mLangIndex = mLangIndex - 1 + LANGUAGES.length;
            mLangIndex = mLangIndex % LANGUAGES.length;
            showToast("Language: " + LANGUAGES[mLangIndex].val);
            return true;
        } else if (keyCode == KeyEvent.KEYCODE_VOLUME_UP) {
            mLangIndex = mLangIndex + 1 + LANGUAGES.length;
            mLangIndex = mLangIndex % LANGUAGES.length;
            showToast("Language: " + LANGUAGES[mLangIndex].val);
            return true;
        }
        return false;
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
        } else {
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    private void setVisibility(final View view, final int visibility) {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                if (view != null) {
                    view.setVisibility(visibility);
                }
            }
        });
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mIntermediate = new Mat(height, width, CvType.CV_8UC4);
    }

    @Override
    public void onCameraViewStopped() {
        if (mIntermediate != null) {
            mIntermediate.release();
        }
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mIntermediate = inputFrame.rgba();
        if (!mInitVisibility) {
            setVisibility(mLoadBar, View.INVISIBLE);
            setVisibility(mTextOverlay, View.INVISIBLE);
            mInitVisibility = true;
        }
        if (mTapped && !mCloudIsProcessing) {
            setProcessing(true);
            mTapped = false;
            Bitmap bmp = matToBmp(mIntermediate);
            if (bmp != null) {
                callCloudVision(bmp, "LABEL_DETECTION");
            }
        } else if (mOCRPressed && !mCloudIsProcessing) {
            setProcessing(true);
            mOCRPressed = false;
            Bitmap bmp = matToBmp(mIntermediate);
            if (bmp != null) {
                callCloudVision(bmp, "TEXT_DETECTION");
            }
        }
        return mIntermediate;
    }

    public void onClick(View view) {
        if (mShowRequest) {
            setVisibility(mTextOverlay, View.INVISIBLE);
            showToast("Cleared previous request");
            mShowRequest = false;
        } else if (mCloudIsProcessing) {
            showToast("Request is still being processed.");
        } else {
            showToast("Making vision request...");
            mTapped = true;
        }
    }

    public void translatePressed(View view) {
        if (mShowRequest) {
            setVisibility(mTextOverlay, View.INVISIBLE);
            showToast("Cleared previous request");
            mShowRequest = false;
        } else if (mCloudIsProcessing) {
            showToast("Request is still being processed.");
        } else {
            showToast("Making OCR request...");
            mOCRPressed = true;
        }
    }

    @NonNull
    private Image getImageEncodeImage(Bitmap bitmap) {
        Image base64EncodedImage = new Image();
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.JPEG, 90, byteArrayOutputStream);
        byte[] imageBytes = byteArrayOutputStream.toByteArray();
        base64EncodedImage.encodeContent(imageBytes);
        return base64EncodedImage;
    }

    private Bitmap matToBmp(Mat m) {
        Bitmap bmp = null;
        try {
            bmp = Bitmap.createBitmap(m.cols(), m.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(m, bmp);
        } catch (Exception e) {
            Log.d(TAG + "::matToBmp", "Error while converting Mat to Bitmap", e);
        }
        return bmp;
    }

    /******************
     * VISION METHODS *
     ******************/

    private void callCloudVision(Bitmap bitmap, final String featureType) {
        final List<Feature> featureList = new ArrayList<>();
        final List<AnnotateImageRequest> annotateImageRequests = new ArrayList<>();
        Feature f = new Feature();
        f.setType(featureType);
        f.setMaxResults(15);
        featureList.add(f);

        AnnotateImageRequest annotateImageReq = new AnnotateImageRequest();
        annotateImageReq.setFeatures(featureList);
        annotateImageReq.setImage(getImageEncodeImage(bitmap));
        annotateImageRequests.add(annotateImageReq);

        new AsyncTask() {
            @Override
            protected String doInBackground(Object... params) {
                try {
                    Vision vision = getVision();
                    BatchAnnotateImagesRequest bair = new BatchAnnotateImagesRequest();
                    bair.setRequests(annotateImageRequests);
                    Vision.Images.Annotate annotateRequest = vision.images().annotate(bair);
                    annotateRequest.setDisableGZipContent(true);

                    List<String> labels = getLabels(annotateRequest.execute(), featureType);
                    setOverlayText(translate(labels));
                } catch (IOException e) {
                    Log.d(TAG + "::doInBackground()->Error!", "", e);
                }
                setProcessing(false);
                return "";
            }
        }.execute();
    }

    private void setProcessing(boolean processing) {
        mCloudIsProcessing = processing;
        if (processing) {
            setVisibility(mLoadBar, View.VISIBLE);
        } else {
            setVisibility(mLoadBar, View.INVISIBLE);
        }
    }

    private List<String> getLabels(BatchAnnotateImagesResponse response, String featureType) {
        ArrayList<String> labels = new ArrayList<>();
        if (response.getResponses().isEmpty()) return labels;
        AnnotateImageResponse imageResponses = response.getResponses().get(0);
        if (featureType == "TEXT_DETECTION" && imageResponses.getTextAnnotations() != null) {
            for (EntityAnnotation ea : imageResponses.getTextAnnotations()) {
                labels.add(ea.getDescription());
                break;
            }
        } else if (featureType == "LABEL_DETECTION" && imageResponses.getLabelAnnotations() != null) {
            for (EntityAnnotation ea : imageResponses.getLabelAnnotations()) {
                labels.add(ea.getDescription());
            }
        }
        return labels;
    }

    private Vision getVision() {
        Vision.Builder builder = new Vision.Builder(
                AndroidHttp.newCompatibleTransport(),
                GsonFactory.getDefaultInstance(),
                null);
        builder.setVisionRequestInitializer(new VisionRequestInitializer(CLOUD_API_KEY));
        return builder.build();
    }

    /***********************
     * TRANSLATION METHODS *
     ***********************/

    private List<StringEntry> translate(List<String> labels) {
        ArrayList<StringEntry> entries = new ArrayList<StringEntry>();
        try {
            TranslationsListResponse response = getTranslationResponse(labels);
            int index = 0;
            for (TranslationsResource resource : response.getTranslations()) {
                String translated = resource.getTranslatedText();
                entries.add(new StringEntry(labels.get(index), resource.getTranslatedText()));
                index++;
            }
        } catch (Exception e) {
            Log.d(TAG + "::translate", "Lost in translation", e);
        }
        return entries;
    }

    private TranslationsListResponse getTranslationResponse(List<String> labels) throws IOException {
        Translate t = getTranslate();
        Translate.Translations.List list = t.new Translations().list(labels,
                LANGUAGES[mLangIndex].key);
        list.setKey(CLOUD_API_KEY);
        return list.execute();
    }

    private Translate getTranslate() {
        Translate t = new Translate.Builder(
                AndroidHttp.newCompatibleTransport(),
                GsonFactory.getDefaultInstance(), null)
                .setApplicationName("SodaHacks Project")
                .build();
        return t;
    }

    /********************
     * OVERLAY CREATION *
     ********************/

    private void setOverlayText(final List<StringEntry> translations) {
        mShowRequest = true;
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                mTextOverlay.setText("");
                for (SpannableString ss : formatEntries(translations)) {
                    mTextOverlay.scrollTo(0, 0);
                    mTextOverlay.append(ss);
                }
            }
        });
        setVisibility(mTextOverlay, View.VISIBLE);
    }

    private List<SpannableString> formatEntries(List<StringEntry> translations) {
        ArrayList<SpannableString> spans = new ArrayList<>();
        for (StringEntry translation : translations) {
            spans.add(scaleString(String.format(" %s\n", translation.key), 1.1f));
            spans.add(scaleString(String.format("  (%s)\n", translation.val), 0.7f));
            spans.add(scaleString(String.format("\n", translation.key), 0.2f));
        }
        if (spans.isEmpty()) {
            spans.add(new SpannableString("Nothing of interest detected."));
        }
        return spans;
    }

    private SpannableString scaleString(String str, float f) {
        SpannableString sstr = new SpannableString(str);
        sstr.setSpan(new RelativeSizeSpan(f), 0, sstr.length(), 0);
        return sstr;
    }
}
