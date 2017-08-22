package com.lenovo.ca.AcousticClassification.preprocessing;

import java.io.*;
import java.util.*;

import com.lenovo.ca.AcousticClassification.preprocessing.WavFile.WavFile;
import com.lenovo.ca.AcousticClassification.preprocessing.mfcc.MFCC;
import com.lenovo.ca.AcousticClassification.utils.ListFiles;
import com.lenovo.ca.AcousticClassification.utils.Stat;


/**
 * Created by fubo5 on 2017/3/2.
 * This code calculates the features from the frames of the sound signal.
 */

public class ExtractAgeFeature {

    /**
     * Count the number of zero-crossings in time-domain within a frame.
     *
     * @param frame 1D array of the sound signal.
     * @return number of zero-crossings
     */
    private static int zeroCrossRate(double[] frame) {
        int count = 0;
        for (int i=1; i<frame.length; i++) {
            count += Math.abs(Math.signum(frame[i]) - Math.signum(frame[i-1]));
        }
        return count/2;
    }

    /**
     * Count the number of zero-crossings in time-domain for each frames within the window
     * @param frames
     * @return
     */
    private static double[] zcrMultiple(double[][] frames) {
        double[] czc = new double[frames.length];
        for (int i=0; i<frames.length; i++) {
            czc[i] = zeroCrossRate(frames[i]);
        }
        return czc;
    }

    /**
     * Returns the peak of zcr in a window.
     */
    public static double zcrPeak(double[][] frames) {
        return Stat.max(zcrMultiple(frames));
    }

    /**
     * Calculate the standard deviation of zcr in a window.
     */
    public static double zcrStd(double[][] frames) {
        return Stat.std(zcrMultiple(frames));
    }

    /**
     * High zero-crossing rate ratio (HZCRR) is the ratio of the number of frames whose ZCR are
     * above 1.5 fold average zero-crossing rate in one-second window.
     * @param frames
     * @return
     */
    public static double hzcrr(double[][] frames) {
        // Calculate the average zcr of the window
        double[] zcrs = zcrMultiple(frames);
        double avgZcr = Stat.mean(zcrs);
        double hzcrrSum = 0;
        for (int i = 0; i < zcrs.length; i++) {
            if (zcrs[i] > (1.5 * avgZcr)) hzcrrSum++;
        }
        return hzcrrSum/zcrs.length;
    }

    /**
     * Calculate the number of frames within the window that
     * have an RMS value less than 50% of the mean RMS.
     *
     * @param frames 2D array of the sound signal.
     * @return the number of frames within the window that have an RMS value less than 50% of the mean RMS.
     */
    public static double lowEnergyFrameRate(double[][] frames) {
        // Calculate the rms for each frame
        double[] RMSs = new double[frames.length];
        for (int i=0; i<frames.length; i++) {
            RMSs[i] = Preprocess.rms(frames[i]);
        }

        // Calculate the mean RMS of the entire window
        double avgRMS = Stat.mean(RMSs);
        int count = 0;
        for (double RMS : RMSs) {
            if (RMS < 0.5 * avgRMS) count++;
        }

        // Return the percentage of low-energy frame number
        return ((double)count)/((double)frames.length);
    }

    /**
     * Calculate the L2-norm distance of two vectors. Two vectors need to be of the same length.
     */
    private static double l2Distance(double[] vector1, double[] vector2) {
        if (vector1.length != vector2.length) {
            System.out.println("Two vectors need to be of the same length.");
            return -1;
        } else {
            double distance = 0;
            for(int i=0; i<vector1.length; i++) {
                distance += Math.pow((vector1[i] - vector2[i]), 2);
            }
            return distance;
        }
    }

    /**
     * Calculate the L2-norm distance of two adjacent vectors.
     * This method returns a median result of multiple frames.
     * @param spectrums The probability density of the frames in one second
     */
    public static double spectralFlux(double[][] spectrums) {
        double[] distances = new double[spectrums.length-1];
        for(int i=1; i<spectrums.length; i++) {
            //System.out.println(Arrays.toString(spectrums[i-1]));
            distances[i-1] = l2Distance(spectrums[i], spectrums[i-1]);
        }
        //System.out.println(Arrays.toString(distances));
        return Stat.median(distances);
    }

    /**
     * The frequency bin below which 93% of the distribution is concentrated.
     *
     * @param spectrum 1D array of the spectrum of one sound frame.
     */
    private static double spectralRolloff(double[] spectrum) {
        double spectrumSum = 0;
        int i = 0;
        while (i<spectrum.length) {
            spectrumSum += spectrum[i];
            if (spectrumSum>0.93) break;
            i++;
        }
        return (double)i/spectrum.length;
    }

    /**
     * The balancing point of the spectral power distribution.
     */
    private static double spectralCentroid(double[] spectrum) {
        double sumWeightedPs = 0;
        double sumPs = 0;
        for (int i=0; i<spectrum.length; i++) {
            sumWeightedPs += (i+1)*Math.pow(spectrum[i], 2);
            sumPs += Math.pow(spectrum[i], 2);
        }
        return sumWeightedPs/sumPs;
    }

    /**
     * The width of the frequencies that the signal occupies.
     */
    private static double bandWidth(double[] spectrum) {
        double sc = spectralCentroid(spectrum);
        double sumWeightedPs = 0;
        double sumPs = 0;
        for (int i=0; i<spectrum.length; i++) {
            // note the calculation of sumWeightedPs is different from spectralCentroid() method
            sumWeightedPs += Math.pow(i+1-sc, 2)*Math.pow(spectrum[i], 2);
            sumPs += Math.pow(spectrum[i], 2);
        }
        return sumWeightedPs/sumPs;
    }

    /**
     * Calculate the phase deviations of the frequency bins in the spectrum weighted by their magnitude.
     * @param spectrum The probability density of the signal
     * @param phases The phase of the signal after fft transformation
     * @return
     */
    private static double normWeightedPhaseDev(double[] spectrum, double[] phases) {
        double sumNwpd = 0;
        double[] secDevPhases = Stat.diff(Stat.diff(phases));
        for (int i=0; i<spectrum.length; i++) {
            sumNwpd += spectrum[i] * secDevPhases[i];
        }
        return sumNwpd;
    }

    /**
     * Measures the history pattern in frequency domain. It is used to differentiate speeches.
     * @param spectrums The probability density of the frames in one second
     * @return a single number, relative spectral entropy
     */
    private static double relativeSpectralEntropy(double[][] spectrums) {
        //Initialize the history pattern m
        double[][] m = new double[spectrums.length][spectrums[0].length];
        m[0] = spectrums[0];
        double rse = 0;
        for (int t=1; t<spectrums.length; t++) {
            for (int i=0; i<spectrums[0].length; i++) {
                m[t][i] = m[t-1][i] * 0.9 + spectrums[t][i] * 0.1;
                rse += -spectrums[t][i] * (Math.log(spectrums[t][i]/m[t-1][i])/Math.log(2));
            }
        }
        return rse;
    }

    /**
     * Extract all features for the sound frames.
     * @param frames 2D array of the sound signal.
     */
    public static double[] extractFeatures(double[][] frames) {
        // Calculate the power densities of the input frames
        double[][] spectrums = new double[frames.length][frames[0].length];
        double[] spectrum;
        double[] phases;

        // Initialize the features that are calculated from a single frame
        double[] allSrf = new double[frames.length];
        double[] allSc = new double[frames.length];
        double[] allBw = new double[frames.length];
        double[] allNwpd = new double[frames.length];

        for (int i=0; i<frames.length; i++) {
//            System.out.println("original signal");
//            System.out.println(Arrays.toString(frames[i]));

            // Remove the offset
            double DC = Stat.mean(frames[i]);
            for (int j = 0; j < frames[i].length; j++) {
                frames[i][j] = frames[i][j] - DC;
            }

            // apply Hanning window before FFT transformation
            spectrum = Preprocess.pdf(Preprocess.applyHannWindow(frames[i]));
//            System.out.println("spectrum");
//            System.out.println(Arrays.toString(spectrum));
            phases = Preprocess.fftPhases(frames[i]);
            spectrums[i] = spectrum;
            allSrf[i] = spectralRolloff(spectrum);
            allSc[i] = spectralCentroid(spectrum);
            allBw[i] = bandWidth(spectrum);
            allNwpd[i] = normWeightedPhaseDev(spectrum, phases);
        }

        // Feature order: {"zcrPeak", "zcrStd", "hzcrr", "lefr", "sf", "srfMean", "scMean", "bwMean",
        //                  "nwpdMean(removed)", "rse", "srfVar", "scVar", "bwVar", "nwpdVar(removed)"}
        // Added the variance statistics of the features
        double[] features = {zcrPeak(frames), zcrStd(frames), hzcrr(frames), lowEnergyFrameRate(frames),
                spectralFlux(spectrums), Stat.mean(allSrf), Stat.mean(allSc), Stat.mean(allBw),
                Stat.mean(allNwpd), relativeSpectralEntropy(spectrums), Stat.var(allSrf), Stat.var(allSc),
                Stat.var(allBw), Stat.var(allNwpd)};
//        System.out.println("features:");
//        System.out.println(Arrays.toString(features));
        return features;
    }

    /**
     * Extract MFCC features for each window.
     * @param frames 2D array of the sound signal.
     * @return 1D array of the mean and standard deviation of the MFCCs of the frames
     */
    public static double[] mfccWindowFeatures(double[][] frames, double sampleRate) {
        MFCC mfcc = new MFCC(frames[0].length, sampleRate, 13, true);
        double[][] mel2d = new double[frames.length][13];
        // Calculate the MFCC features for each frame
        for (int i = 0; i < frames.length; i++) {
            //System.out.println(Arrays.toString(frames[i]));
            double[] frame = Preprocess.applyHammingWindow(frames[i]);
            mel2d[i] = mfcc.mfcc(frame);
            //System.out.println(mfsc2d[i].length);
        }
        // Remove the redundant Coefficient which is the volume
        double[][] mel2dSubset = new double[mel2d.length][];
        for (int i = 0; i < mel2d.length; i++) {
            mel2dSubset[i] = Arrays.copyOfRange(mel2d[i], 1, 13);
        }
        return mfcc.meanStd(mel2dSubset, false);
    }

    /**
     * Extract MFCC features for each frame.
     * @param frames 2D array of the sound signal.
     * @return 2D array of the MFCCs and Delta-MFCCs of the frames
     */
    public static double[][] mfccFrameFeatures(double[][] frames, double sampleRate) {
        MFCC mfcc = new MFCC(frames[0].length, sampleRate, 13, true);
        double[][] mel2d = new double[frames.length][13];
        // Calculate the MFCC features for each frame
        for (int i = 0; i < frames.length; i++) {
            //System.out.println(Arrays.toString(frames[i]));
            double[] frame = Preprocess.applyHammingWindow(frames[i]);
            mel2d[i] = mfcc.mfcc(frame);
            //System.out.println(mfsc2d[i].length);
        }
        // Remove the redundant Coefficient which is the volume
        double[][] mel2dSubset = new double[mel2d.length][];
        for (int i = 0; i < mel2d.length; i++) {
            mel2dSubset[i] = Arrays.copyOfRange(mel2d[i], 1, 13);
        }

        // Calculate the MFCC deltas
        double[][] dmfcc = mfcc.performDelta2D(mel2dSubset);

        // Merge mel2dSubset and dmfcc horizontally.
        // I.e., result.row = mel2dSusbset.row = dmfcc.row
        //       result.col = mel2dSubset.col + dmfcc.col
        double[][] result = appendH(mel2dSubset, dmfcc);
        return result;
    }

    /**
     * Append two 2D arrays horizontally.
     * @param a 2D array
     * @param b 2D array
     * @return 2D array of which the row is the same as a (or b), the columns is the sum of a and b.
     */
    public static double[][] appendH(double[][] a, double[][] b) {
        double[][] m = new double[a.length][a[0].length + b[0].length];
        for (int i = 0; i < a.length; i++) {
            // copy a
            for (int j = 0; j < a[0].length; j++) {
                m[i][j] = a[i][j];
            }
            // copy b
            for (int j = 0; j < b[0].length; j++) {
                m[i][j + a[0].length] = b[i][j];
            }
        }
        return m;
    }

    /**
     * Write the features to a file in CSV format separated by a comma. If the file already exists, append to the
     * end of the file.
     * @param features Features calculated from one-second long sound.
     * @param featureFile The CSV file where the features are written to.
     * @param featureName The name of the feature to add to the end of each line
     */
    public static void featureToFile(double[] features, String featureFile, String featureName) {
        StringBuilder rowFeature = new StringBuilder();
        for (int i=0; i<features.length; i++) {
            rowFeature.append(String.format("%.6f", features[i]));
//            System.out.println(features[i]);
//            System.out.println(String.format("%.6f", features[i]));
            //if (i!=features.length-1) rowFeature.append(",");
            rowFeature.append(",");
        }
        rowFeature.append(featureName);
        rowFeature.append("\n");

        // Write the StringBuilder to file
        File file = new File(featureFile);
        try {
            FileWriter pw = new FileWriter(file, file.exists());
            pw.write(rowFeature.toString());
            pw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Read all .wav files in the specific folder, calculate the features and write them to a file
     * @param trainFolder The folder contains the .wav files
     * @param featureFile The file path to write the features to
     */
    public static void generateCoarseFeatures(String trainFolder, String featureFile, String label, boolean overwrite) {
        // Load all wave files in the folder
        ArrayList<File> soundFiles = new ArrayList<File>();
        ListFiles.allDirs(trainFolder, soundFiles, ".wav");

        // Delete the file if existed
        File tempFile = new File(featureFile);
        if (overwrite) {
            if (tempFile.exists()) tempFile.delete();
        }

        int NEW_SAMPLERATE = 8000;
        int NUM_FRAMES_PER_WINDOW = 15;
        double FRAME_LENGTH = 0.064;
        double FRAME_STEP = 0.064;
        for (File file: soundFiles) {
            // Calculate features from each file
            System.out.print("Load file ");
            System.out.println(file);
            try
            {
                WavFile wavFile = WavFile.openWavFile(file);
                int numChannels = wavFile.getNumChannels();
                long sampleRate = wavFile.getSampleRate();
                int windowLength = (int)(sampleRate * ((NUM_FRAMES_PER_WINDOW - 1) * FRAME_STEP + FRAME_LENGTH));

                double[][] bufferStereo = new double[numChannels][windowLength];
                double[] bufferMono = new double[windowLength];
                double[] buffer = new double[windowLength];
                //System.out.println("Buffer: "+ buffer.length);

                int framesRead = 0;

//                //------------- Skip the offset (half a second) --------------------
//                // set an offset to start reading
//                double[][] bufferStereoOffset = new double[numChannels][windowLength/2];
//                double[] bufferMonoOffset = new double[windowLength/2];
//                double[] bufferShift = new double[windowLength/2];
//                if (numChannels == 1) {  // if mono, use the only channel
//                    // added in a half window offset
//                    framesRead = wavFile.readFrames(bufferMonoOffset, bufferShift.length);
//                } else {  // if stereo, use Channel 0
//                    framesRead = wavFile.readFrames(bufferStereoOffset, bufferShift.length);
//                }
//                // ---------- comment the above code if read from beginning --------

                do //(sampleRate < wavFile.getFramesRemaining())  // every iteration is one second
                {
                    // Read N frames into buffer
                    if (numChannels == 1) {  // if mono, use the only channel
                        framesRead = wavFile.readFrames(bufferMono, buffer.length);
                        buffer = bufferMono;
                    } else {  // if stereo, use Channel 0
                        framesRead = wavFile.readFrames(bufferStereo, buffer.length);
                        buffer = bufferStereo[0];
                    }

                    // Down sample to 8 kHz
                    double[] bufferResample = Preprocess.resample(buffer, (int)sampleRate, NEW_SAMPLERATE);
                    //System.out.println(Arrays.toString(bufferResample));

                    // Reshape one window to multiple frames
                    double[][] frames = Preprocess.frameSignal(bufferResample, NEW_SAMPLERATE,
                            FRAME_LENGTH, FRAME_LENGTH);
                    //System.out.println("frame: "+ frames[0].length + " num:" + frames.length);

                    // Discard the window if all frames are non-informative
                    if (Preprocess.frameAdmin(frames)) {
                        double[] features = extractFeatures(frames);
                        featureToFile(features, featureFile, label);
                        //System.out.println("informative");

                    } else {
                        System.out.println("non-informative");
                    }
                } while (framesRead == windowLength);  // Stop loading if not enough data
            } catch (Exception e)
            {
                System.err.println(e);
            }
        }
    }


    /**
     * Read all .wav files in the specific folder, calculate the MFCC for each frame, and then do the
     * statistic for each window and write the window feature to a file
     * @param trainFolder The folder contains the .wav files
     * @param mfccFile The file path to write the MFCCs to
     */
    public static void generateWindowFeature(String trainFolder, String mfccFile, String label, boolean overwrite, boolean shift) {
        // Load all wave files in the folder
        ArrayList<File> soundFiles = new ArrayList<File>();
        ListFiles.allDirs(trainFolder, soundFiles, ".wav");

        // Delete the file if existed
        File tempFile = new File(mfccFile);
        if(overwrite) {
            if (tempFile.exists()) tempFile.delete();
        }

        int NEW_SAMPLERATE = 8000;
        int NUM_FRAMES_PER_WINDOW = 60;
        double FRAME_LENGTH = 0.032;
        double FRAME_STEP = 0.016;
        int NUM_MFCC_COEFF = 13;

        for (File file: soundFiles) {
            // Calculate features from each file
            System.out.print("Load file ");
            System.out.println(file);
            System.out.println();

            try
            {
                WavFile wavFile = WavFile.openWavFile(file);
                int numChannels = wavFile.getNumChannels();
                long sampleRate = wavFile.getSampleRate();
                int windowLength = (int)(sampleRate * ((NUM_FRAMES_PER_WINDOW - 1) * FRAME_STEP + FRAME_LENGTH));

                double[][] bufferStereo = new double[numChannels][windowLength];
                double[] bufferMono = new double[windowLength];
                double[] buffer = new double[windowLength];

                // set an offset to start reading
                double[][] bufferStereoOffset = new double[numChannels][windowLength/2];
                double[] bufferMonoOffset = new double[windowLength/2];
                double[] bufferShift = new double[windowLength/2];
                //System.out.println("Buffer: "+ buffer.length);

//                // MFCC features
//                MFCC mfcc = new MFCC((int)(NEW_SAMPLERATE*FRAME_LENGTH), NEW_SAMPLERATE, NUM_MFCC_COEFF, true);
//                double[][] mel2d = new double[NUM_FRAMES_PER_WINDOW][NUM_MFCC_COEFF];

                int framesRead = 0;

                if (shift) {
                    //------------- Skip the offset (half a second) --------------------
                    if (numChannels == 1) {  // if mono, use the only channel
                        // added in a half window offset
                        framesRead = wavFile.readFrames(bufferMonoOffset, bufferShift.length);
                    } else {  // if stereo, use Channel 0
                        framesRead = wavFile.readFrames(bufferStereoOffset, bufferShift.length);
                    }
                    // ---------- comment the above code if read from beginning --------
                }

                // every iteration is one second
                do
                {
                    if (numChannels == 1) {  // if mono, use the only channel
                        // added in a half window offset
                        framesRead = wavFile.readFrames(bufferMono, buffer.length);
                        buffer = bufferMono;
                    } else {  // if stereo, use Channel 0
                        framesRead = wavFile.readFrames(bufferStereo, buffer.length);
                        buffer = bufferStereo[0];
                    }

                    // Down sample to 8 kHz
                    double[] bufferResample = Preprocess.resample(buffer, (int)sampleRate, NEW_SAMPLERATE);
                    //System.out.println(Arrays.toString(bufferResample));

                    // Reshape one window to multiple frames
                    double[][] frames = Preprocess.frameSignal(bufferResample, NEW_SAMPLERATE,
                            FRAME_LENGTH, FRAME_STEP);
                    //System.out.println("frame: "+ frames[0].length + " num:" + frames.length);

                    // Discard the window if all frames are non-informative
                    if (Preprocess.frameAdmin(frames, -50, 6, 0.3)) {

//                        // Calculate the MFCC features for each window
                        double[] melFeatures = mfccWindowFeatures(frames, NEW_SAMPLERATE);
                        featureToFile(melFeatures, mfccFile, label);
//                        for (double[] mel1d : mel2dSubset) {
//                            ExtractFeature.featureToFile(mel1d, mfccFile, label);
//                        }


                    } else {
                        System.out.println("non-informative");

//                        //record quiet period as well
//                        double[] empty = new double[24];
//                        ExtractFeature.featureToFile(empty, mfccFile, "quiet");
                    }
                } while (framesRead == windowLength);  // Stop loading if not enough data
            } catch (Exception e)
            {
                System.err.println(e);
            }
        }
    }

    public static void generateFrameFeatureFromFile(String trainFilePath, String mfccFile, String label, boolean overwrite) {
        // Delete the file if existed
        File tempFile = new File(mfccFile);
        if(overwrite) {
            if (tempFile.exists()) tempFile.delete();
        }

        int NEW_SAMPLERATE = 8000;
        int NUM_FRAMES_PER_WINDOW = 180;  // changed from 60 to 30
        double FRAME_LENGTH = 0.032;
        double FRAME_STEP = 0.016;
        int NUM_MFCC_COEFF = 13;
        String path = "";
        
        Map<String,Integer> ageCategoryMap =new HashMap<String,Integer>();

        File file = new File(trainFilePath);
            // Calculate features from each file
            System.out.print("Load file ");
            System.out.println(file);
            System.out.println();
            
            //add by zhuolei to label age
            label="";
            path= file.getPath();
//            if(path.contains("female")){
//            	label = "female";
//            }else{
//            	label = "male";
//            }
            
            if(path.contains("adult")){
            	label +="_adult";
            }else  if(path.contains("old")){
            	label +="_old";
            }else{
            	label +="_young";
            }

            try
            {
                WavFile wavFile = WavFile.openWavFile(file);
                int numChannels = wavFile.getNumChannels();
                long sampleRate = wavFile.getSampleRate();
                int windowLength = (int)(sampleRate * ((NUM_FRAMES_PER_WINDOW - 1) * FRAME_STEP + FRAME_LENGTH));

                double[][] bufferStereo = new double[numChannels][windowLength];
                double[] bufferMono = new double[windowLength];
                double[] buffer = new double[windowLength];

                // MFCC features
//                MFCC mfcc = new MFCC((int)(NEW_SAMPLERATE*FRAME_LENGTH), NEW_SAMPLERATE, NUM_MFCC_COEFF, true);
//                double[][] mel2d = new double[NUM_FRAMES_PER_WINDOW][NUM_MFCC_COEFF];

                int framesRead = 0;

                // every iteration is one second
                do
                {
                    if (numChannels == 1) {  // if mono, use the only channel
                        // added in a half window offset
                        framesRead = wavFile.readFrames(bufferMono, buffer.length);
                        buffer = bufferMono;
                    } else {  // if stereo, use Channel 0
                        framesRead = wavFile.readFrames(bufferStereo, buffer.length);
                        buffer = bufferStereo[0];
                    }

                    // Down sample to 8 kHz
                    double[] bufferResample = Preprocess.resample(buffer, (int)sampleRate, NEW_SAMPLERATE);
                    //System.out.println(Arrays.toString(bufferResample));

                    // Reshape one window to multiple frames
                    double[][] frames = Preprocess.frameSignal(bufferResample, NEW_SAMPLERATE,
                            FRAME_LENGTH, FRAME_STEP);
                    //System.out.println("frame: "+ frames[0].length + " num:" + frames.length);

                    // Discard the window if all frames are non-informative
                    if (Preprocess.frameAdmin(frames, -50, 6, 0)) {

//                        // Calculate the MFCC features (MFCC and DMFCC) for each frame
                        double[][] melFeatures = mfccFrameFeatures(frames, NEW_SAMPLERATE);
                        for (double[] melFeature : melFeatures) {
                            featureToFile(melFeature, mfccFile, label);
                            if(!ageCategoryMap.containsKey(label)){
                                ageCategoryMap.put(label, 1);
                            }else{
                            	ageCategoryMap.put(label, ageCategoryMap.get(label)+1);
                            }
                        }
//                        for (double[] mel1d : mel2dSubset) {
//                            ExtractFeature.featureToFile(mel1d, mfccFile, label);
//                        }
                    } else {
                        System.out.println("non-informative");

//                        //record quiet period as well
//                        double[] empty = new double[24];
//                        ExtractFeature.featureToFile(empty, mfccFile, "quiet");
                    }
                } while (framesRead == windowLength);  // Stop loading if not enough data
                

            } catch (Exception e)
            {
                System.err.println(e);
            }
        
        
        for(String key:ageCategoryMap.keySet()){
      	  System.out.println(key+":	"+ageCategoryMap.get(key));
      }
    }
    
    /**
     * Read all .wav files in the specific folder, calculate the MFCC and DMFCC for each frame and write them to a file
     * @param trainFolder The folder contains the .wav files
     * @param mfccFile The file path to write the MFCCs to
     */
    public static void generateFrameFeature(String trainFolder, String mfccFile, String label, boolean overwrite) {
        // Load all wave files in the folder
        ArrayList<File> soundFiles = new ArrayList<File>();
        ListFiles.allDirs(trainFolder, soundFiles, ".wav");

        // Delete the file if existed
        File tempFile = new File(mfccFile);
        if(overwrite) {
            if (tempFile.exists()) tempFile.delete();
        }

        int NEW_SAMPLERATE = 8000;
        int NUM_FRAMES_PER_WINDOW = 180;  // changed from 60 to 30
        double FRAME_LENGTH = 0.032;
        double FRAME_STEP = 0.016;
        int NUM_MFCC_COEFF = 13;
        String path = "";
        
        Map<String,Integer> ageCategoryMap =new HashMap<String,Integer>();

        for (File file: soundFiles) {
            // Calculate features from each file
            System.out.print("Load file ");
            System.out.println(file);
            System.out.println();
            
            //add by zhuolei to label age
            label="";
            path= file.getPath();
//            if(path.contains("female")){
//            	label = "female";
//            }else{
//            	label = "male";
//            }
            
            if(path.contains("adult")){
            	label +="_adult";
            }else  if(path.contains("old")){
            	label +="_old";
            }else{
            	label +="_young";
            }

            try
            {
                WavFile wavFile = WavFile.openWavFile(file);
                int numChannels = wavFile.getNumChannels();
                long sampleRate = wavFile.getSampleRate();
                int windowLength = (int)(sampleRate * ((NUM_FRAMES_PER_WINDOW - 1) * FRAME_STEP + FRAME_LENGTH));

                double[][] bufferStereo = new double[numChannels][windowLength];
                double[] bufferMono = new double[windowLength];
                double[] buffer = new double[windowLength];

                // MFCC features
//                MFCC mfcc = new MFCC((int)(NEW_SAMPLERATE*FRAME_LENGTH), NEW_SAMPLERATE, NUM_MFCC_COEFF, true);
//                double[][] mel2d = new double[NUM_FRAMES_PER_WINDOW][NUM_MFCC_COEFF];

                int framesRead = 0;

                // every iteration is one second
                do
                {
                    if (numChannels == 1) {  // if mono, use the only channel
                        // added in a half window offset
                        framesRead = wavFile.readFrames(bufferMono, buffer.length);
                        buffer = bufferMono;
                    } else {  // if stereo, use Channel 0
                        framesRead = wavFile.readFrames(bufferStereo, buffer.length);
                        buffer = bufferStereo[0];
                    }

                    // Down sample to 8 kHz
                    double[] bufferResample = Preprocess.resample(buffer, (int)sampleRate, NEW_SAMPLERATE);
                    //System.out.println(Arrays.toString(bufferResample));

                    // Reshape one window to multiple frames
                    double[][] frames = Preprocess.frameSignal(bufferResample, NEW_SAMPLERATE,
                            FRAME_LENGTH, FRAME_STEP);
                    //System.out.println("frame: "+ frames[0].length + " num:" + frames.length);

                    // Discard the window if all frames are non-informative
                    if (Preprocess.frameAdmin(frames, -50, 6, 0)) {

//                        // Calculate the MFCC features (MFCC and DMFCC) for each frame
                        double[][] melFeatures = mfccFrameFeatures(frames, NEW_SAMPLERATE);
                        for (double[] melFeature : melFeatures) {
                            featureToFile(melFeature, mfccFile, label);//写入csv文件
                            if(!ageCategoryMap.containsKey(label)){
                                ageCategoryMap.put(label, 1);
                            }else{
                            	ageCategoryMap.put(label, ageCategoryMap.get(label)+1);
                            }
                        }
//                        for (double[] mel1d : mel2dSubset) {
//                            ExtractFeature.featureToFile(mel1d, mfccFile, label);
//                        }
                    } else {
                        System.out.println("non-informative");

//                        //record quiet period as well
//                        double[] empty = new double[24];
//                        ExtractFeature.featureToFile(empty, mfccFile, "quiet");
                    }
                } while (framesRead == windowLength);  // Stop loading if not enough data
                

            } catch (Exception e)
            {
                System.err.println(e);
            }
        }
        
        for(String key:ageCategoryMap.keySet()){
      	  System.out.println(key+":	"+ageCategoryMap.get(key));
      }
    }

    public static void main(String[] args) {
//        String AMBIENT_PATH = "D:\\Projects\\AcousticClassification\\Data\\DIY\\Ambient\\train";
//        String MUSIC_PATH = "D:\\Projects\\AcousticClassification\\Data\\DIY\\Music\\train";
//        String VOICE_PATH = "D:\\Projects\\AcousticClassification\\Data\\DIY\\Voice\\train";
//        String TEST_PATH = "D:\\Projects\\AcousticClassification\\Data\\DIY\\Voice\\unused\\temp";
//        generateCoarseFeatures(TEST_PATH, "output/feature files/test_meeting3.arff", "voice", false);

        // Gender Classification {male, female}
       // String MALE_PATH = "D:\\Projects\\AcousticClassification\\Data\\DIY\\Voice\\train\\male";
        //String FEMALE_PATH = "D:\\Projects\\AcousticClassification\\Data\\DIY\\Voice\\train\\female";
        //String AGE_PATH = "E:\\workspace\\gitcode\\Context_Awareness\\Algorithm\\Acoustic_Classification\\data\\voice\\train\\male\\VoxForge\\hugh-20070606-bul\\wav\\bul0001.wav";
        // Scene Classification {windy,road,metro_station,car,bus,others}
    	String VoicePrint_Path="D:\\CA\\voiceData\\female1\\sp10001.wav";
        String TRAIN_PATH = "D:\\Projects\\AcousticClassification\\Data\\DIY\\Ambient\\train\\DCASE2016\\OTHERS";
//        generateWindowFeature(TRAIN_PATH, "output/sceneStd.arff", "others", false, true);
       // generateFrameFeature(FEMALE_PATH, "output/femaleFrames.csv", "female", true);
        generateFrameFeatureFromFile(VoicePrint_Path, "sp10001.csv", "", true);
        //generateFrameFeature(AGE_PATH, "output/ageFrames.csv", "", true);
    }
}
