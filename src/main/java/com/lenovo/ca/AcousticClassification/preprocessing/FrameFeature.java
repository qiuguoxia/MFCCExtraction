package com.lenovo.ca.AcousticClassification.preprocessing;


import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

import com.lenovo.ca.AcousticClassification.preprocessing.WavFile.WavFile;
import com.lenovo.ca.AcousticClassification.preprocessing.mfcc.MFCC;
import com.lenovo.ca.AcousticClassification.utils.Filtering;
import com.lenovo.ca.AcousticClassification.utils.ListFiles;
import com.lenovo.ca.AcousticClassification.utils.Stat;

/**
 * Created by fubo5 on 2017/6/28.
 * This class implements the feature extraction in the paper
 * "Content-based audio classification" by L. Lu et al 2003
 * The features are extracted from one frame (25 ms).
 */
public class FrameFeature {

    /**
     * Count the number of zero-crossings in time-domain within a frame.
     *
     * @param frame 1D array of the sound signal.
     * @return number of zero-crossings
     */
    private int zeroCrossRate(double[] frame) {
        int count = 0;
        for (int i=1; i<frame.length; i++) {
            count += Math.abs(Math.signum(frame[i]) - Math.signum(frame[i-1]));
        }
        return count/2;
    }

    /**
     * Calculate the number of frames within the window that
     * have an RMS value less than 50% of the mean RMS.
     *
     * @param frames 2D array of the sound signal.
     * @return the number of frames within the window that have an RMS value less than 50% of the mean RMS.
     */
    private double lowEnergyFrameRate(double[][] frames) {
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
     * Returns the total spectrum power of a frame. The logarithm is applied to the result.
     * @param spectrum 1D array of the power spectrum of the frame. No normalization should be applied.
     * @return
     */
    private double shortTimeEnergy(double[] spectrum) {
        return Math.log(Stat.sum(spectrum));
    }

    /**
     * Calculate the sub-band energy distribution of the frame.
     * The four sub-bands are [[0, w0/8], [w0/8, w0/4], [w0/4, w0/2], [w0/2, w0]] respectively.
     * @param spectrum 1D array of the power spectrum of the frame.
     * @param ste short time energy calculated from shortTimeEnergy method
     * @return 1D array of the energy for each sub band
     */
    private double[] subBandEnergyDistrb(double[] spectrum, double ste) {
        double[] D = new double[4];
        int w0 = spectrum.length; // w0 is half of the sampling frequency
        int[] subBands = {0, w0/8, w0/4, w0/2, w0};  // there are four sub-bands
        for (int i = 0; i < D.length; i++) {
            D[i] = Stat.sum(Arrays.copyOfRange(spectrum, subBands[i], subBands[i + 1]))/ste;
        }
        return D;
    }

    /**
     * Calculate the means and stds of D for each column.
     * @param allD a 2D array of energy distribution. Each row is one frame and each column is one sub band.
     * @return a 1D array. The first four elements are four means and the following 4 elements are stds.
     */
    private double[] flattenD(double[][] allD) {
        double[] result = new double[8];
        // TODO - calculate the mean and stds
        return result;
    }

    /**
     * Calculate the L2-norm distance of two vectors. Two vectors need to be of the same length.
     */
    private double l2Distance(double[] vector1, double[] vector2) {
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
    public double spectralFlux(double[][] spectrums) {
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
    private double spectralRolloff(double[] spectrum) {
        double spectrumSum = 0;
        int i = 0;
        while (i < spectrum.length) {
            spectrumSum += spectrum[i];
            if (spectrumSum > 0.93) break;
            i++;
        }
        return (double)i/spectrum.length;
    }

    /**
     * The balancing point of the spectral power distribution. Also known as the brightness.
     */
    private double spectralCentroid(double[] spectrum) {
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
    private double bandWidth(double[] spectrum) {
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

//    /**
//     * Calculate the phase deviations of the frequency bins in the spectrum weighted by their magnitude.
//     * @param spectrum The probability density of the signal
//     * @param phases The phase of the signal after fft transformation
//     * @return
//     */
//    private double normWeightedPhaseDev(double[] spectrum, double[] phases) {
//        double sumNwpd = 0;
//        double[] secDevPhases = Stat.diff(Stat.diff(phases));
//        for (int i=0; i<spectrum.length; i++) {
//            sumNwpd += spectrum[i] * secDevPhases[i];
//        }
//        return sumNwpd;
//    }

//    /**
//     * Measures the history pattern in frequency domain. It is used to differentiate speeches.
//     * @param spectrums The probability density of the frames in one second
//     * @return a single number, relative spectral entropy
//     */
//    private double relativeSpectralEntropy(double[][] spectrums) {
//        //Initialize the history pattern m
//        double[][] m = new double[spectrums.length][spectrums[0].length];
//        m[0] = spectrums[0];
//        double rse = 0;
//        for (int t=1; t<spectrums.length; t++) {
//            for (int i=0; i<spectrums[0].length; i++) {
//                m[t][i] = m[t-1][i] * 0.9 + spectrums[t][i] * 0.1;
//                rse += -spectrums[t][i] * (Math.log(spectrums[t][i]/m[t-1][i])/Math.log(2));
//            }
//        }
//        return rse;
//    }

    /**
     * Finds the index of the max value in the array
     * @param arr  1D array
     */
    private int argMax(double[] arr){
        double max = arr[0];
        int j = 0;
        for(int i = 0;i < arr.length;i++){
            if(arr[i] > max){
                max = arr[i];
                j = i;
            }
        }
        return j;
    }

    /**
     * The normalized correlation function of a frame
     * @param frames 2D array and each row is one frame
     * @return
     */
    private double[] normCorrelateFunc(double[][] frames) {
        double[] ncfs = new double[frames.length];
        for(int i = 0; i < frames.length; i++){
            double s = 0;
            double b = 0;
            double t = 0;
            int kPeak = argMax(frames[i]);
            int q = kPeak + i * frames[0].length;
            for(int j = 0; j < frames[0].length; j++){
                int p = 2 * i * frames[0].length + j - q;
                t += Math.pow(frames[i][j], 2);
                if(p >=0 ){
                    s += frames[p / frames[0].length][p%frames[0].length] * frames[i][j];
                    b += Math.pow(frames[p/frames[0].length][p % frames[0].length],2);
                }
            }
            ncfs[i] = s / (Math.sqrt(b) * Math.sqrt(t) + 1E-6); // Added by 1E-6 to escape the divide by 0 problem
        }
        return ncfs;
    }

    /**
     * Calculate the band periodicity of the input frames of the specific band
     * @param frames 2D array and each row is one frame
     * @return
     */
    private double bandPeriodicityPerBand(double[][] frames){
        double[] ncf = normCorrelateFunc(frames);
        return Stat.mean(ncf);
    }

    /**
     * Calculate the band periodicity of the input frames for the 4 sub bands
     * 500 ~ 1000 Hz, 1000 ~ 2000 Hz, 2000 ~ 3000 Hz, 3000 ~ 4000 Hz
     * @param window 1D array of the signal before framing
     * @param sampleRate the sampling rate of the signal
     * @return 1D array of 4 elements
     */
    private double[] bandPeriodicityAllBands(double[] window, int sampleRate){
        double[][] SUB_BANDS = {{500, 1000}, {1000, 2000}, {2000, 3000}, {3000, 4000}};
        double frame_width = 0.025;
        double frame_step = 0.025;
        double[] bp = new double[4];

        for (int i = 0; i < SUB_BANDS.length; i++) {
            // band pass the signal and Then frame the signal into multiple frames
            double[] filteredFrames = Filtering.bandpass(window,
                    sampleRate, SUB_BANDS[i][0], SUB_BANDS[i][1], 2, 0);
            double[][] frames = Preprocess.frameSignal(filteredFrames, sampleRate, frame_width, frame_step);
            bp[i] = bandPeriodicityPerBand(frames);
        }
        return bp;
    }

    /**
     * Returns the ratio of noise frames in a given window.
     * @param frames 2D array and each row is one frame
     * @return
     */
    private double noiseFrameRatio(double[][] frames) {
        double[] ncfs = normCorrelateFunc(frames);
        double numNoiseFrames = 0;
        for (double ncf : ncfs) {
            if (ncf < 0.3) numNoiseFrames++;
        }
        return numNoiseFrames/ncfs.length;
    }

    /**
     * Extract all features for the sound frames.
     * @param frames 2D array of the sound signal.
     */
    public double[] extractFeatures(double[][] frames) {
        // Calculate the power densities of the input frames
        double[][] spectrums = new double[frames.length][frames[0].length];
        double[] spectrum;

        // Initialize the features that are calculated from a single frame
        double[] allZcr = new double[frames.length];
        double[] allSte = new double[frames.length];
        double[][] allD = new double[frames.length][4];

        double[] allSrf = new double[frames.length];
        double[] allSc = new double[frames.length];
        double[] allBw = new double[frames.length];
        double[] allNwpd = new double[frames.length];

        for (int i=0; i<frames.length; i++) {
//            System.out.println("original signal");
//            System.out.println(Arrays.toString(frames[i]));

//            // Remove the offset
//            double DC = Stat.mean(frames[i]);
//            for (int j = 0; j < frames[i].length; j++) {
//                frames[i][j] = frames[i][j] - DC;
//            }

            // apply Hanning window before FFT transformation
            allZcr[i] = zeroCrossRate(frames[i]);
            spectrum = Preprocess.pdf(Preprocess.applyHannWindow(frames[i]));
//            System.out.println("spectrum");
//            System.out.println(Arrays.toString(spectrum));
            spectrums[i] = spectrum;

            allSte[i] = shortTimeEnergy(spectrum);
            allD[i] = subBandEnergyDistrb(spectrum, allSte[i]);

            allSrf[i] = spectralRolloff(spectrum);
            allSc[i] = spectralCentroid(spectrum);  // a.k.a. brightness
            allBw[i] = bandWidth(spectrum);
//            allNwpd[i] = normWeightedPhaseDev(spectrum, phases);
        }

        double[] resultsD = flattenD(allD);
        double sf = spectralFlux(spectrums);

        // TODO - Flattern the 2D allD to 4 means and 4 stds

        // Feature order: {"zcrMean", "zcrStd", "STEmean", "STEstd", "D1mean", "D1std", "D2mean", "D2std",
        // "D3mean", "D3std", "D4mean", "D4std", "brightnessMean", "brightnessStd", "bwMean", "bwStd",
        // "sf", "bp0", "bp1", "MFCC[8]Mean", "MFCC[8]Std"}
        // Added the variance statistics of the features
        double[] features = {Stat.mean(allZcr), Stat.std(allZcr), Stat.mean(allSte), Stat.std(allSte), resultsD[0],
                resultsD[4], resultsD[1], resultsD[5], resultsD[2], resultsD[6], resultsD[3], resultsD[7],
                Stat.mean(allSc), Stat.std(allSc), Stat.mean(allBw), Stat.std(allBw), sf
                // TODO - to be added: bp0, bp1, MFCC[0-8]means and stds
                };
//        System.out.println("features:");
//        System.out.println(Arrays.toString(features));
        return features;
    }

    /**
     * Overload the extractFeatures function without the window input
     * @param frames 2D array of the sound signal.
     * @param window 1D array of the signal before framing. This is used for calculating band periodicity.
     * @param sampleRate the sampling rate of the window signal. e.g. 44100, 8000
     *               TODO - This is a temple solution. Need to be changed back to one input.
     * @return
     */
    public double[] extractFeatures(double[][] frames, double[] window, int sampleRate) {
        // Calculate the original features
        double[] featuresOrig = extractFeatures(frames);

        int len = featuresOrig.length;
        double[] features = new double[len + 3];
        // Copy the temple features into the feature array
        System.arraycopy(featuresOrig, 0, features, 0, featuresOrig.length);

        // Calculate the band periodicity and keep the bp for
        // the sub bands of bp[0] (500~1000 Hz), bp[1] 1000~2000 Hz and bpSum}
        // TODO - need to remove the DC value first. It is not done in this function.
        window = Stat.abs(window);
        double[] bp = bandPeriodicityAllBands(window, sampleRate);
        features[len] = bp[0];
        features[len + 1] = bp[1];
        features[len + 2] = Stat.sum(bp);
        return features;
    }

    /**
     * Extract MFCC features for each window.
     * @param frames 2D array of the sound signal.
     * @return 1D array of the mean and standard deviation of the MFCCs of the frames
     */
    public double[] mfccWindowFeatures(double[][] frames, double sampleRate) {
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
    public double[][] mfccFrameFeatures(double[][] frames, double sampleRate) {
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
    public double[][] appendH(double[][] a, double[][] b) {
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
    public void featureToFile(double[] features, String featureFile, String featureName) {
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
                ExtractFeature soundFeatures = new ExtractFeature();

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
//                        double[] features = soundFeatures.extractFeatures(frames); //TODO -- to be tested
                        double[] features = soundFeatures.extractFeatures(frames, bufferResample, NEW_SAMPLERATE);
                        soundFeatures.featureToFile(features, featureFile, label);
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
    public void generateWindowFeature(String trainFolder, String mfccFile, String label, boolean overwrite, boolean shift) {
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

    /**
     * Read all .wav files in the specific folder, calculate the MFCC and DMFCC for each frame and write them to a file
     * @param trainFolder The folder contains the .wav files
     * @param mfccFile The file path to write the MFCCs to
     */
    public void generateFrameFeature(String trainFolder, String mfccFile, String label, boolean overwrite) {
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
    }

    public static void main(String[] args) {
        String AMBIENT_PATH = "D:\\Projects\\AcousticClassification\\Data\\DIY\\Ambient\\train";
        String MUSIC_PATH = "D:\\Projects\\AcousticClassification\\Data\\DIY\\Music\\train";
        String VOICE_PATH = "D:\\Projects\\AcousticClassification\\Data\\DIY\\Voice\\train";
        String TEST_PATH = "D:\\Projects\\AcousticClassification\\Data\\DIY\\Ambient\\train\\bus";
        generateCoarseFeatures(MUSIC_PATH, "output/feature_files/coarse5.3.arff", "music", false);

        // Gender Classification {male, female}
        String MALE_PATH = "D:\\Projects\\AcousticClassification\\Data\\DIY\\Voice\\train\\male";
        String FEMALE_PATH = "D:\\Projects\\AcousticClassification\\Data\\DIY\\Voice\\train\\female";

        // Scene Classification {windy,road,metro_station,car,bus,others}
        String TRAIN_PATH = "D:\\Projects\\AcousticClassification\\Data\\DIY\\Scene\\train\\DCASE2016\\LOCATION\\road";
//        generateWindowFeature(TRAIN_PATH, "output/sceneStd.arff", "others", false, true);
//        ExtractFeature soundFeatures = new ExtractFeature();
//        soundFeatures.generateFrameFeature(TRAIN_PATH, "output/feature_files/scene_0.1.csv", "road", false);
    }
}
