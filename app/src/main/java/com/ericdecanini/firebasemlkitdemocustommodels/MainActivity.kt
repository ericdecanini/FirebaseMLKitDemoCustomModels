package com.ericdecanini.firebasemlkitdemocustommodels

import android.graphics.drawable.Drawable
import android.support.v7.app.AppCompatActivity
import android.os.Bundle
import com.google.firebase.ml.custom.FirebaseModelManager
import com.google.firebase.ml.custom.FirebaseModelInputs
import android.graphics.Bitmap
import android.graphics.drawable.BitmapDrawable
import android.util.Log
import java.io.ByteArrayOutputStream
import com.google.firebase.ml.custom.FirebaseModelInterpreter
import com.google.firebase.ml.custom.FirebaseModelOptions
import com.google.firebase.ml.custom.FirebaseModelDataType
import com.google.firebase.ml.custom.FirebaseModelInputOutputOptions
import com.google.firebase.ml.custom.model.FirebaseLocalModelSource


class MainActivity : AppCompatActivity() {

    private val LOG_TAG = MainActivity::class.java.simpleName
    private lateinit var interpreter: FirebaseModelInterpreter

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        loadModel()
        classifyImage(R.drawable.panda)
    }

    private fun loadModel() {
        val localSource = FirebaseLocalModelSource.Builder("image-classification")
                .setAssetFilePath("image-classification.tflite")  // Or setFilePath if you downloaded from your host
                .build()
        FirebaseModelManager.getInstance().registerLocalModelSource(localSource)


        val options = FirebaseModelOptions.Builder()
                .setLocalModelName("image-classification")
                .build()
        interpreter = FirebaseModelInterpreter.getInstance(options)!!
    }

    private fun classifyImage(drawableRes: Int) {
        val d = getDrawable(drawableRes)
        var input = drawableToFloats(d, 224, 224)

        val inputs = FirebaseModelInputs.Builder()
                .add(input)  // add() as many input arrays as your model requires
                .build()

        val inputOutputOptions = FirebaseModelInputOutputOptions.Builder()
                .setInputFormat(0, FirebaseModelDataType.FLOAT32, intArrayOf(1, 224, 224, 3))
                .setOutputFormat(0, FirebaseModelDataType.FLOAT32, intArrayOf(1, 1001))
                .build()

        Log.d(LOG_TAG, "Running interpreter")

        interpreter.run(inputs, inputOutputOptions)
                .addOnSuccessListener {
                    val output = it.getOutput<Array<FloatArray>>(0)
                    Log.d(LOG_TAG, "--- Probabilities ---")
                    for (i in 0 until output[0].size) {
                        if (output[0][i] > 0) { Log.d(LOG_TAG, "$i: ${output[0][i]}") }
                    }
                }
                .addOnFailureListener {
                    // Task failed with an exception
                    Log.e(LOG_TAG, "Firebase ML Kit Failed: ${it.message}")
                    it.printStackTrace()
                }
    }

    private fun drawableToFloats(d: Drawable, width: Int, height: Int): Array<Array<Array<FloatArray>>> {
        val bitmap = (d as BitmapDrawable).bitmap
        val stream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream)
        val byteArray = stream.toByteArray()

        // Convert the byte array to a 4d array
        var colourCounter = 0
        var widthCounter = 0
        var heightCounter = 0

        var pixels = Array(1) { Array(224) { Array(224) { FloatArray(3) } } }

        for (i in 0 until byteArray.size) {
            val byte = byteArray[i]
            pixels[0][widthCounter][heightCounter][colourCounter] = byte.toFloat()

            colourCounter++
            if (colourCounter == 2) {
                colourCounter = 0
                widthCounter++
            }

            if (widthCounter == width - 1) {
                widthCounter = 0
                heightCounter++
            }

            if (heightCounter == height - 1) {
                break
            }

        }

        return pixels
    }

}
