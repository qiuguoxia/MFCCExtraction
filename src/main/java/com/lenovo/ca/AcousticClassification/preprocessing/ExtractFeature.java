package com.lenovo.ca.AcousticClassification.preprocessing;


import java.io.*;
import java.util.*;

import com.lenovo.ca.AcousticClassification.preprocessing.WavFile.WavFile;
import com.lenovo.ca.AcousticClassification.preprocessing.mfcc.MFCC;
import com.lenovo.ca.AcousticClassification.utils.Filtering;
import com.lenovo.ca.AcousticClassification.utils.ListFiles;
import com.lenovo.ca.AcousticClassification.utils.Stat;

/**
 * Created by fubo5 on 2017/3/2.
 * This code calculates the features from the frames of the sound signal.
 */

public class ExtractFeature {

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
     * Count the number of zero-crossings in time-domain for each frames within the window
     * @param frames
     * @return
     */
    private double[] zcrMultiple(double[][] frames) {
        double[] czc = new double[frames.length];
        for (int i=0; i<frames.length; i++) {
            czc[i] = zeroCrossRate(frames[i]);
        }
        return czc;
    }

    /**
     * Returns the peak of zcr in a window.
     */
    private double zcrPeak(double[][] frames) {
        return Stat.max(zcrMultiple(frames));
    }

    /**
     * Calculate the standard deviation of zcr in a window.
     */
    private double zcrStd(double[][] frames) {
        return Stat.std(zcrMultiple(frames));
    }

    /**
     * High zero-crossing rate ratio (HZCRR) is the ratio of the number of frames whose ZCR are
     * above 1.5 fold average zero-crossing rate in one-second window.
     * @param frames
     * @return
     */
    private double hzcrr(double[][] frames) {
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
     * Calculate the L2-norm distance of two vectors after normalization.
     * Two vectors need to be of the same length.
     */
    private double l2DistanceNorm(double[] vector1, double[] vector2) {
        // Normalize the vector
        double max1 = Stat.max(vector1);
        double max2 = Stat.max(vector2);
//        double sum1 = Stat.sum(vector1);
//        double sum2 = Stat.sum(vector2);

        double distance = 0;
        for(int i=0; i<vector1.length; i++) {
            distance += Math.pow((vector1[i]/max1 - vector2[i]/max2), 2);
//            distance += Math.pow((vector1[i]/sum1 - vector2[i]/sum2), 2);
        }
        return distance;
    }

    /**
     * Calculate the L2-norm distance of two adjacent vectors.
     * This method returns a median result of multiple frames.
     * @param spectrums The probability density of the frames in one second
     */
    private double spectralFlux(double[][] spectrums) {
        double[] distances = new double[spectrums.length-1];
        for(int i=1; i<spectrums.length; i++) {
            //System.out.println(Arrays.toString(spectrums[i-1]));
            distances[i-1] = l2Distance(spectrums[i], spectrums[i-1]);
        }
        //System.out.println(Arrays.toString(distances));
        return Stat.mean(distances);
    }

    /**
     * Calculate the L2-norm distance of two adjacent vectors.
     * This method is modified according to L. Lu 2001 paper.
     * This method returns a median result of multiple frames.
     * @param spectrums The probability density of the frames in one second
     */
    private double spectralFluxNorm(double[][] spectrums) {
        double[] distances = new double[spectrums.length-1];
        for(int i=1; i<spectrums.length; i++) {
            //System.out.println(Arrays.toString(spectrums[i-1]));
            distances[i-1] = l2DistanceNorm(spectrums[i], spectrums[i-1]);
        }
        //System.out.println(Arrays.toString(distances));
        return Stat.mean(distances);
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
     * The balancing point of the spectral power distribution.
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

    /**
     * TODO - this feature seem unimportant and takes lots of computation. Thus we removed it from the feature vector.
     * Calculate the phase deviations of the frequency bins in the spectrum weighted by their magnitude.
     * @param spectrum The probability density of the signal
     * @param phases The phase of the signal after fft transformation
     * @return
     */
    private double normWeightedPhaseDev(double[] spectrum, double[] phases) {
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
    private double relativeSpectralEntropy(double[][] spectrums) {
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
     * Counts the sum of the peaks which are greater than the threshold in the spectrum.
     * The spectrum should be normalized by dividing by the max value.
     * @param spectrum
     * @param threshold the value above which the peak is counted. Otherwise the peak is ignored.
     * @return The sum of all peaks
     */
    private double spectrumPeakSum(double[] spectrum, double threshold) {
        double sum = 0;
        double max = Stat.max(spectrum);
        for (int i = 2; i < spectrum.length; i++) {
            if ((spectrum[i] < spectrum[i-1]) && (spectrum[i-1] > spectrum[i-2])
                    && spectrum[i-1]/max > threshold) sum += spectrum[i-1]/max;
        }
        return sum;
    }

    private double spectrumPeaks(double[][] spectrums) {
        double THRESHOLD = 0.1;
        double sum = 0;
        for (double[] spectrum : spectrums) {
            sum += spectrumPeakSum(spectrum, THRESHOLD);
        }
        return sum/spectrums.length;
    }

    /**
     * Calculate the autocorrelation of the signal.
     * The second half of the result is returned so that it is of the same length as the input signal.
     * @param arr input 1D array signal
     * @return 1D array of autocorrelated signal.
     */
    private double[] autocorrelation(double[] arr){
        int len = arr.length;
        double[] autocorrelation = new double[len];
        double means = Stat.mean(arr);
        double variance = Stat.var(arr);
        for(int k = 0; k < len; k++){
            double sum = 0.0;
            for(int i = 0; i < len - k; i++){
                sum += (arr[i] - means) * (arr[i + k] - means);
            }
            autocorrelation[k] = sum / ((len - k) * variance);
        }
        return autocorrelation;
    }

    /**
     * The auto-correlation coefficient is defined as the maximum value (exclusive the first one)
     * of the auto-correlation function.
     * @param frame The signal frame to calculate the auto-correlation. It shall be normalized first. (?)
     * @return
     */
    private double autoCoefficient(double[] frame) {
//        // Normalize the frame with the range (max - min)
//        double max = Stat.max(frame);
//        double min = Stat.min(frame);
//        double range = max - min;
//        for (int i = 0; i < frame.length; i++) {
//            frame[i] = (frame[i] - min) / range;
//        }

        double[] autocorrelation = autocorrelation(frame);
        return Stat.max(Arrays.copyOfRange(autocorrelation, 1, autocorrelation.length));
    }

    /**
     * Calculate the mean auto-correlation coefficients of the frames.
     * @param frames
     * @return
     */
    private double autoCoeffMean(double[][] frames) {
        double sum = 0;
        for (double[] frame : frames) {
            sum += autoCoefficient(frame);
        }
        return sum/frames.length;
    }

    /**
     *
     * @param arr  一维数组
     * @return  一维数组最大值的索引
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
     * 将二维数组转为一维数组
     * @param arr 二维数组
     * @return  一维数组
     */
    private double[]  dimensionTransformation(double[][] arr){
        int size = arr.length * arr[0].length;
        double[] singleArray = new double[size];
        int index = 0;
        for(int i = 0; i < arr.length; i++){
            for(int j = 0;j < arr[i].length;j++){
//		    	System.out.println(arr[i][j]);
                singleArray[index++] = arr[i][j];
            }
        }
        return singleArray;
    }


    /**
     * 一维数组的切片
     * @param array 一维数组, start起始切片索引, end结束切片索引
     * @return 切片后的结果
     */
    private double[] sliceArray(double[] array, int start, int end){
        double[] sliceResult = new double[end - start];
        int m = 0;
        for(; start < end; start++){
            sliceResult[m++] = array[start];
        }
        return sliceResult;
    }


    /**
     * 两个一维数组各个元素相乘后相加
     * @param arr1
     * @param arr2
     * @return
     */
    private double dotArray(double[] arr1, double[] arr2){
        double result = 0;
        for(int i = 0; i < arr1.length; i++){
            result += arr1[i] * arr2[i];
        }
        return result;
    }

    /**
     * 计算二维数组的normCorrelatedFunc
     * @param array
     * @return  一维数组 normCorrelatedFunc
     */
    private double[] normCorrelatedFunc(double[][] array){
        int frameSize = array[0].length;
        int frameNum = array.length;
        double[] bandPeriodicity = new double[frameNum - 2];
        double[]  globalList = dimensionTransformation(array);
        for(int i = 1; i < frameNum - 1; i++){
            int previousGlobalIndex = argMax(array[i - 1]) + (i - 1) * frameSize;
            int currentGlobalIndex = argMax(array[i]) + i * frameSize;
            double[] previousList = sliceArray(globalList, previousGlobalIndex, previousGlobalIndex + frameSize);
            double[] currentList = sliceArray(globalList, currentGlobalIndex, currentGlobalIndex + frameSize);
            double a = dotArray(previousList, currentList);
            double b = Math.sqrt(dotArray(previousList, previousList));
            double c = Math.sqrt(dotArray(currentList, currentList));
            bandPeriodicity[i - 1] = a / (b * c);
        }
        return bandPeriodicity;
    }

    /**
     * BandPeriodicity
     * @param array
     * @return
     */
    private double bandPeriodicity(double[][] array){
        double[] ncf = normCorrelatedFunc(array);
        double sum = 0;
        for(int i = 0; i < ncf.length; i++){
            sum += ncf[i];
        }
        return sum / ncf.length;
    }

    /**
     * Calculate the band periodicity of the input frames for the 4 sub bands
     * 500 ~ 1000 Hz, 1000 ~ 2000 Hz, 2000 ~ 3000 Hz, 3000 ~ 4000 Hz
     * @param window
     * @return 1D array of 4 elements
     */
    private double[] bandPeriodicityAllBands(double[] window, int sampleRate){
//        double[][] SUB_BANDS = {{500, 1000}, {1000, 2000}, {2000, 3000}, {3000, 4000}};
        // TODO - The result is not effctively improved for the sub bands of {2000, 3500} Hz.
        // TODO - We can remove the last sub band to save the computation power
        double[][] SUB_BANDS = {{0, 500}, {500, 1000}, {1000, 2000}, {2000, 3500}};
        double frame_width = 0.025;
        double frame_step = 0.025;
        double[] bp = new double[SUB_BANDS.length];
//        window = Preprocess.applyHannWindow(window); // Apply Hannning window before FFT to remove edge effect

        for (int i = 0; i < SUB_BANDS.length; i++) {
            // band pass the signal and Then frame the signal into multiple frames

            double[] filteredWindow = Filtering.bandpass(window,
                    sampleRate, SUB_BANDS[i][0], SUB_BANDS[i][1], 4, 0);
//            featureToFile(filteredWindow, "output/filteredSignal.csv", "bp");
            // Get the absolute signal
//            window = Stat.abs(filteredWindow);

            double[][] frames = Preprocess.frameSignal(filteredWindow, sampleRate, frame_width, frame_step);
            bp[i] = bandPeriodicity(frames);
        }
        return bp;
    }

    /**
     * Returns the ratio of noise frames in a given window.
     * @param frames 2D array and each row is one frame
     * @return
     */
    private double noiseFrameRatio(double[][] frames) {
        double[] ncfs = normCorrelatedFunc(frames);
        double numNoiseFrames = 0;
        for (double ncf : ncfs) {
            if (ncf < 0.2) numNoiseFrames++;
        }
        return numNoiseFrames/ncfs.length;
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
     * Extract all features for the sound frames.
     * @param frames 2D array of the sound signal.
     */
    public double[] extractFeatures(double[][] frames) {
        // Calculate the power densities of the input frames
        double[][] spectrums = new double[frames.length][frames[0].length/2];
        double[] spectrum;
//        double[] phases;

        // Initialize the features that are calculated from a single frame
        double[] allSrf = new double[frames.length];
        double[] allSc = new double[frames.length];
        double[] allBw = new double[frames.length];
        double[][] allD = new double[frames.length][4];
//        double[] allNwpd = new double[frames.length];

        for (int i=0; i<frames.length; i++) {
//            System.out.println("original signal");
//            System.out.println(Arrays.toString(frames[i]));

            // Remove the offset, TODO - Try using HighPass filter
            double DC = Stat.mean(frames[i]);
            for (int j = 0; j < frames[i].length; j++) {
                frames[i][j] = frames[i][j] - DC;
            }

            // apply Hanning window before FFT transformation
            spectrum = Preprocess.pdf(Preprocess.applyHannWindow(frames[i]));
//            System.out.println("spectrum");
//            System.out.println(Arrays.toString(spectrum));
//            phases = Preprocess.fftPhases(frames[i]);
            spectrums[i] = spectrum;
            allSrf[i] = spectralRolloff(spectrum);
            allSc[i] = spectralCentroid(spectrum);
            allBw[i] = bandWidth(spectrum);
//            allNwpd[i] = normWeightedPhaseDev(spectrum, phases);
        }

//        System.out.println(Arrays.deepToString(spectrums));
//        System.out.println(spectralFluxNorm(spectrums));
//        System.out.println(spectrumPeaks(spectrums));

        // Feature order: {"zcrPeak", "zcrStd", "hzcrr", "lefr", "sfLog", "srfMean", "scMean", "bwMean",
        //                  "nwpdMean(removed)", "rse", "srfVar", "scVar", "bwVar", "nwpdVar(removed)", "nfr",
        //                  "sp", "ac"}
        // Added the variance statistics of the features
        double[] features = {zcrPeak(frames), zcrStd(frames), hzcrr(frames), lowEnergyFrameRate(frames),
                spectralFluxNorm(spectrums), Stat.mean(allSrf), Stat.mean(allSc), Stat.mean(allBw),
                /*Stat.mean(allNwpd), */relativeSpectralEntropy(spectrums), Stat.var(allSrf), Stat.var(allSc),
                Stat.var(allBw)/*, Stat.var(allNwpd)*/, spectrumPeaks(spectrums), autoCoeffMean(frames)};
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

        // Add band periodicity features
        double[] bp = bandPeriodicityAllBands(window, sampleRate);
        features[len] = bp[0];
        features[len + 1] = bp[1];
        features[len + 2] = Stat.sum(bp);
//        System.out.println(Arrays.toString(bp));

        // Add short time energy and sub-band energy distribution features


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
        int NUM_FRAMES_PER_WINDOW = 40;
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
                        boolean write = true;
                        for (double element : features) {
                            if (Double.isNaN(element)) {
                                System.out.println("found one NaN");
                                write = false;
                            }
                        }
                        if (write) soundFeatures.featureToFile(features, featureFile, label);
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
        String HIGH_AMBIENT_PATH = "D:\\Projects\\AcousticClassification\\Data\\DIY\\Ambient\\train\\high\\office";
        String LOW_AMBIENT_PATH = "D:\\Projects\\AcousticClassification\\Data\\DIY\\Ambient\\train\\high\\crowd";
        String MUSIC_PATH = "D:\\Projects\\AcousticClassification\\Data\\DIY\\Music\\train";
        String VOICE_PATH = "D:\\Projects\\AcousticClassification\\Data\\DIY\\Voice\\train";
        String TEST_PATH = "D:\\Projects\\AcousticClassification\\Data\\DIY\\Ambient\\test\\office";

//        generateCoarseFeatures(HIGH_AMBIENT_PATH, "output/feature_files/coarse8.0_del.arff", "high_ambient", false);
//        generateCoarseFeatures(LOW_AMBIENT_PATH, "output/feature_files/coarse8.0_del.arff", "low_ambient", false);
//        generateCoarseFeatures(MUSIC_PATH, "output/feature_files/coarse8.0.arff", "music", false);
//        generateCoarseFeatures(VOICE_PATH, "output/feature_files/coarse8.0.arff", "voice", false);
        generateCoarseFeatures(TEST_PATH, "output/feature_files/test_office.arff", "high_ambient", false);

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
