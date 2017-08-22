package com.lenovo.ca.AcousticClassification.test;


import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;

import com.lenovo.ca.AcousticClassification.preprocessing.Preprocess;
import com.lenovo.ca.AcousticClassification.preprocessing.WavFile.WavFile;
import com.lenovo.ca.AcousticClassification.utils.ListFiles;

/**
 * Created by fubo5 on 2017/4/12.
 */
public class PreprocessTest {
    public static void testResample(String soundFolder, String rawDataFile, String resampleDataFile) {
        ArrayList<File> soundFiles = new ArrayList<File>();
        ListFiles.allDirs(soundFolder, soundFiles, ".wav");
        File rawDataPath = new File(rawDataFile);
        File resampleDataPath = new File(resampleDataFile);

        int NEW_SAMPLERATE = 8000;

        for (File file : soundFiles) {
            try {
                WavFile wavFile = WavFile.openWavFile(file);
                // Get the number of audio channels in the wav file
                int numChannels = wavFile.getNumChannels();
                long sampleRate = wavFile.getSampleRate();

                // Write original data to file
                StringBuilder rawDataString = new StringBuilder();

                // Create a buffer of 1 second
                double[] buffer = new double[(int) sampleRate * numChannels];  //double[100 * numChannels];
                int framesRead = wavFile.readFrames(buffer, (int)sampleRate);


                //rawDataString.append(buffer);
                System.out.println("before resample");
                System.out.println(Arrays.toString(buffer));
                System.out.println("after resample");
                System.out.println(Arrays.toString(Preprocess.resample(buffer, (int)sampleRate, 8000)));

            } catch (Exception e) {
                System.err.println(e);
            }
        }
    }

    public static void main(String[] args) {
        testResample("D:\\Projects\\AcousticClassification\\Data\\DIY\\Music\\clips\\temp", "a",
                "b");
    }
}
