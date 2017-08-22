package com.lenovo.ca.AcousticClassification.test;


import java.io.File;
import java.util.*;

import com.lenovo.ca.AcousticClassification.preprocessing.ExtractFeature;
import com.lenovo.ca.AcousticClassification.preprocessing.Preprocess;
import com.lenovo.ca.AcousticClassification.preprocessing.WavFile.*;



/**
 * Created by fubo5 on 2017/3/27.
 */
public class ExtractFeatureTest
{
    public static void main(String[] args) {
        try
        {
            WavFile wavFile = WavFile.openWavFile(new File("D:\\Projects\\AcousticClassification\\Data" +
                    "\\DIY\\Music\\clips\\temp\\phonemusic2_8.wav"));
            // Get the number of audio channels in the wav file
            int numChannels = wavFile.getNumChannels();
            long sampleRate = wavFile.getSampleRate();

            // Create a buffer of 1 second
            double[] buffer = new double[(int)sampleRate * numChannels];  //double[100 * numChannels];

            int framesRead;
            System.out.println("Printing the features:");
            do  // every iteration is one second
            {
                // Read one second sound into 'buffer'
                framesRead = wavFile.readFrames(buffer, (int)sampleRate);

                // Downsample to 8 kHz
                int NEW_SAMPLERATE = 8000;
                double[] bufferResample = Preprocess.resample(buffer, buffer.length, NEW_SAMPLERATE);
                //System.out.println(Arrays.toString(bufferResample));
                double[][] rawFrames = Preprocess.frameSignal(bufferResample, bufferResample.length, 0.064, 0.064);
                //double[][] keptFrames = Preprocess.frameAdmin(rawFrames);
                ExtractFeature soundFeatures = new ExtractFeature();
                double[] features = soundFeatures.extractFeatures(rawFrames);
//                ExtractFeature myFeatures = new ExtractFeature(keptFrames);
                System.out.println(Arrays.toString(features));
            } while (framesRead != 0);//(framesRead != 0);

//            // Loop through the keys and calculate the mean value for multiple (typical 3 to 10) seconds.
//            Set keys = ExtractFeature.features.keySet();
//            System.out.println(keys);
//            for(Object key: keys) {
//                System.out.print(key+": ");
//                System.out.printf("%.3f", Stat.mean(ExtractFeature.features.get(key)));
//                System.out.println();
//            }
//            System.out.println(ExtractFeature.features);
        } catch (Exception e)
        {
            System.err.println(e);
        }
    }
}

