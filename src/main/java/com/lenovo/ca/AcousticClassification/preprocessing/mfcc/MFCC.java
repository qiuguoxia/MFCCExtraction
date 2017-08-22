package com.lenovo.ca.AcousticClassification.preprocessing.mfcc;


import java.nio.Buffer;
import java.util.Arrays;
import java.util.Iterator;

import com.lenovo.ca.AcousticClassification.preprocessing.ExtractFeature;
import com.lenovo.ca.AcousticClassification.utils.Stat;

/**
 * Modified by fubo5 on 2017/5/3.
 * This code was folked from https://raw.githubusercontent.com/Sciss/SpeechRecognitionHMM/
 * master/src/main/java/org/ioe/tprsa/audio/feature/MFCC.java

 * Mel-Frequency Cepstrum Coefficients.
 *
 * @author Ganesh Tiwari
 * @author Hanns Holger Rutz
 *
 */
public class MFCC {

    private final static int       numMelFilters       = 24;    // how much滤波器个数
    private final static double    preEmphasisAlpha    = 0.97;
    private final static double    lowerFilterFreq     = 80.00; // FmelLow

    private final double    sampleRate;//8000
    private final double    upperFilterFreq;
    private final int       samplesPerFrame;

    private final boolean usePreEmphasis;

    final FFT fft;
    final DCT dct;

    int M;  // regression window size; i.e.,number of frames to take into account while taking delta

    /** Creates an MFCC processor with no pre-emphasis. */

    public MFCC(int samplesPerFrame, double sampleRate, int numCoefficients) {
        this(samplesPerFrame, sampleRate, numCoefficients, false);
    }

    public MFCC(int samplesPerFrame, double sampleRate, int numCoefficients, boolean preEmphasis) {
        this.samplesPerFrame    = samplesPerFrame;
        this.sampleRate         = sampleRate;
        this.usePreEmphasis     = preEmphasis;
        upperFilterFreq         = sampleRate / 2.0;
        fft = new FFT();
        dct = new DCT(numCoefficients, numMelFilters);
        this.M = 2;  // the number of skip frames when calculating MFCC deltas.
    }

    // Calculate the Mel-frequency Spectral Coefficients
    public double[] mfsc(double[] frame) {
        if (usePreEmphasis) {
            frame = preEmphasis(frame);
        }
        // Magnitude Spectrum
        final double[] bin = powerSpectrum(frame);
		/*
		 * cBin=frequencies of the channels in terms of FFT bin indices (cBin[i]
		 * for the i -th channel)
		 */

        // prepare filter for for melFilter
        final int cBin[] = fftBinIndices();// same for all
        // process Mel filter bank
        final double fBank[] = melFilter(bin, cBin);
        // magnitudeSpectrum and bin filter indices

        // System.out.println("after mel filter");
        // ArrayWriter.printDoubleArrayToConsole(fBank);

        // Non-linear transformation
        final double f[] = nonLinearTransformation(fBank);
        // System.out.println("after N L T");
        // ArrayWriter.printDoubleArrayToConsole(f);

        // Cepstral coefficients, by DCT
        // System.out.println("after DCT");
        // ArrayWriter.printDoubleArrayToConsole(cepc);

        return f;
    }

    /**
     * Apply dct to the result of mfsc and get the mfcc result
     * @param frame
     * @return
     */
    public double[] mfcc(double[] frame) {
        double f[] = mfsc(frame);
        double[] cepc=dct.perform(f);
        double[] cepstral=this.lifter(cepc);
        return cepstral;
    }



    private double[] lifter(double[] coeff){
        for(int i =0;i<coeff.length;i++){
            coeff[i] = coeff[i]* (1+0.5*numMelFilters*Math.sin(Math.PI*i/numMelFilters));
        }
        return coeff;
    }

    private double[] magnitudeSpectrum(double frame[]) {
        final double magSpectrum[] = new double[frame.length];
        // calculate FFT for current frame
        fft.process(frame);
        // System.err.println("FFT SUCCEED");
        // calculate magnitude spectrum
        for (int k = 0; k < frame.length; k++) {
            magSpectrum[k] = Math.sqrt(fft.real[k] * fft.real[k] + fft.imag[k] * fft.imag[k]);
            //magSpectrum[k] = (fft.real[k] * fft.real[k] + fft.imag[k] * fft.imag[k])/samplesPerFrame;
        }
        return magSpectrum;
    }

    private double[] powerSpectrum(double frame[]) {
        final double magSpectrum[] = new double[frame.length];
        // calculate FFT for current frame
        fft.process(frame);
        // System.err.println("FFT SUCCEED");
        // calculate magnitude spectrum
        for (int k = 0; k < frame.length; k++) {
            //magSpectrum[k] = Math.sqrt(fft.real[k] * fft.real[k] + fft.imag[k] * fft.imag[k]);
            magSpectrum[k] = (fft.real[k] * fft.real[k] + fft.imag[k] * fft.imag[k])/samplesPerFrame;
        }
        return magSpectrum;
    }

    /**
     * Emphasizes high freq signal
     */
    private double[] preEmphasis(double inputSignal[]) {
        final double outputSignal[] = new double[inputSignal.length];
        // apply pre-emphasis to each sample
        outputSignal[0] = inputSignal[0];
        for (int n = 1; n < inputSignal.length; n++) {
            outputSignal[n] = (double) (inputSignal[n] - preEmphasisAlpha * inputSignal[n - 1]);
        }
        return outputSignal;
    }

    private int[] fftBinIndices() {
        final int cBin[] = new int[numMelFilters + 2];
        cBin[0] = (int) Math.round(lowerFilterFreq / sampleRate * samplesPerFrame);// cBin0
        cBin[cBin.length - 1] = (samplesPerFrame / 2);// cBin24
        for (int i = 1; i <= numMelFilters; i++) {// from cBin1 to cBin23
            final double fc = centerFreq(i);// center freq for i th filter
            cBin[i] = (int) Math.round(fc / sampleRate * samplesPerFrame);
        }
        return cBin;
    }

    /**
     * Performs mel filter operation
     *
     * @param bin
     *            magnitude spectrum (| |)^2 of fft
     * @param cBin
     *            mel filter coefficients
     * @return mel filtered coefficients --> filter bank coefficients.
     */
    private double[] melFilter(double bin[], int cBin[]) {
        final double temp[] = new double[numMelFilters + 2];
        for (int k = 1; k <= numMelFilters; k++) {
            double num1 = 0.0, num2 = 0.0;
            for (int i = cBin[k - 1]; i <= cBin[k]; i++) {
                num1 += ((i - cBin[k - 1] + 1) / (cBin[k] - cBin[k - 1] + 1)) * bin[i];
            }

            for (int i = cBin[k] + 1; i <= cBin[k + 1]; i++) {
                num2 += (1 - ((i - cBin[k]) / (cBin[k + 1] - cBin[k] + 1))) * bin[i];
            }

            temp[k] = num1 + num2;
        }
        final double fBank[] = new double[numMelFilters];
        System.arraycopy(temp, 1, fBank, 0, numMelFilters);
        return fBank;
    }

    /**
     * performs nonlinear transformation
     *
     * @param fBank filter bank coefficients
     * @return f log of filter bac
     */
    private double[] nonLinearTransformation(double fBank[]) {
        double f[] = new double[fBank.length];
        final double FLOOR = -50;
        for (int i = 0; i < fBank.length; i++) {
            f[i] = Math.log(fBank[i]);
            // check if ln() returns a value less than the floor
            if (f[i] < FLOOR) {
                f[i] = FLOOR;
            }
        }
        return f;
    }

    private double centerFreq(int i) {
        final double melFLow    = freqToMel(lowerFilterFreq);
        final double melFHigh   = freqToMel(upperFilterFreq);
        final double temp       = melFLow + ((melFHigh - melFLow) / (numMelFilters + 1)) * i;
        return inverseMel(temp);
    }

    private double inverseMel(double x) {
        final double temp = Math.pow(10, x / 2595) - 1;
        return 700 * (temp);
    }

    protected double freqToMel(double freq) {
        return 2595 * log10(1 + freq / 700);
    }

    private double log10(double value) {
        return Math.log(value) / Math.log(10);
    }

    /**
     * Calculate the mean and variance of the 2D MFCC matrix.
     * @param coeff2d a 2D matrix, each row is one set of MFCC coefficients. Each column is one coefficient.
     * @param normalize determine if the feature vector is normalized by dividing the diff of the row.
     * @return a 1D array. The first half is the average vertically and the second half is the variance vertically
     */
    public double[] meanVariance(double[][] coeff2d, boolean normalize) {
        int nRow = coeff2d.length;
        int nCol = coeff2d[0].length;
        double[] mfccFeatures = new double[nCol * 2];

        // loop through every column to calculate the mean
        for (int j = 0; j < nCol; j++) {
            double sum = 0.0;
            for (int i = 0; i < coeff2d.length; i++) {
                sum += coeff2d[i][j];
            }
            mfccFeatures[j] = sum / (double) nRow;
        }

        // loop through every column to calculate the variance
        for (int j = 0; j < nCol; j++) {
            double powSum = 0.0;
            for (int i = 0; i < nRow; i++) {
                powSum += (coeff2d[i][j] - mfccFeatures[j]) * (coeff2d[i][j] - mfccFeatures[j]);
            }
            mfccFeatures[j + nCol] = powSum / (double) nRow;
        }

        // normalize by dividing the max of the row
        if (normalize) {
            double maxMean = Stat.max(Arrays.copyOfRange(mfccFeatures, 0, nCol));
            double minMean = Stat.min(Arrays.copyOfRange(mfccFeatures, 0, nCol));
            double maxVar = Stat.max(Arrays.copyOfRange(mfccFeatures, nCol, nCol*2));
            double minVar = Stat.min(Arrays.copyOfRange(mfccFeatures, nCol, nCol*2));
            for (int i = 0; i < nCol; i++) {
                mfccFeatures[i] = (mfccFeatures[i] - minMean) / (maxMean - minMean);  // normalize means
                mfccFeatures[i + nCol] = (mfccFeatures[i + nCol] - minVar) / (maxVar - minVar);  // normalize variances
            }
        }

        // normalize by subtracting the mean of the column
        return mfccFeatures;
    }

    /**
     * Calculate the mean and standard deviation of the 2D MFCC matrix.
     * @param coeff2d a 2D matrix, each row is one set of MFCC coefficients. Each column is one coefficient.
     * @param normalize determine if the feature vector is normalized by dividing the diff of the row.
     * @return a 1D array. The first half is the average vertically and the second half is the variance vertically
     */
    public double[] meanStd(double[][] coeff2d, boolean normalize) {
        int nRow = coeff2d.length;
        int nCol = coeff2d[0].length;
        double[] mfccFeatures = new double[nCol * 2];

        // loop through every column to calculate the mean
        for (int j = 0; j < nCol; j++) {
            double sum = 0.0;
            for (int i = 0; i < coeff2d.length; i++) {
                sum += coeff2d[i][j];
            }
            mfccFeatures[j] = sum / (double) nRow;
        }

        // loop through every column to calculate the standard deviation
        for (int j = 0; j < nCol; j++) {
            double powSum = 0.0;
            for (int i = 0; i < nRow; i++) {
                powSum += (coeff2d[i][j] - mfccFeatures[j]) * (coeff2d[i][j] - mfccFeatures[j]);
            }
            mfccFeatures[j + nCol] = Math.sqrt(powSum / (double) nRow);  // <-- Added a sqrt() here
        }

        // normalize by dividing the max of the row
        if (normalize) {
            double maxMean = Stat.max(Arrays.copyOfRange(mfccFeatures, 0, nCol));
            double minMean = Stat.min(Arrays.copyOfRange(mfccFeatures, 0, nCol));
            double maxVar = Stat.max(Arrays.copyOfRange(mfccFeatures, nCol, nCol*2));
            double minVar = Stat.min(Arrays.copyOfRange(mfccFeatures, nCol, nCol*2));
            for (int i = 0; i < nCol; i++) {
                mfccFeatures[i] = (mfccFeatures[i] - minMean) / (maxMean - minMean);  // normalize means
                mfccFeatures[i + nCol] = (mfccFeatures[i + nCol] - minVar) / (maxVar - minVar);  // normalize variances
            }
        }

        // normalize by subtracting the mean of the column
        return mfccFeatures;
    }

    /** fubo5: Perform the delta calculation for the input MFCCs.
     *
     * @param data 2D array of MFCC. The column is the MFCC coefficient and the row is one sample
     * @return 2D array of Delta-MFCC with the same shape of input data.
     */

    public double[][] performDelta2D(double[][] data) {
        int noOfMfcc = data[0].length;
        int frameCount = data.length;
        // 1. calculate sum of mSquare i.e., denominator
        double mSqSum = 0;
        for (int i = -M; i < M; i++) {
            mSqSum += Math.pow(i, 2);
        }
        // 2.calculate numerator
        double delta[][] = new double[frameCount][noOfMfcc];
        for (int i = 0; i < noOfMfcc; i++) {
            // handle the boundary
            // 0 padding results best result
            // from 0 to M
            for (int k = 0; k < M; k++) {
                // delta[k][i] = 0; //0 padding
                delta[k][i] = data[k][i]; // 0 padding
            }
            // from frameCount-M to frameCount
            for (int k = frameCount - M; k < frameCount; k++) {
                // delta[l][i] = 0;
                delta[k][i] = data[k][i];
            }
            for (int j = M; j < frameCount - M; j++) {
                // travel from -M to +M
                double sumDataMulM = 0;
                for (int m = -M; m <= +M; m++) {
                    // System.out.println("Current m -->\t"+m+
                    // "current j -->\t"+j + "data [m+j][i] -->\t"+data[m +
                    // j][i]);
                    sumDataMulM += m * data[m + j][i];
                }
                // 3. divide
                delta[j][i] = sumDataMulM / mSqSum;
            }
        }// end of loop


        return delta;
    }// end of fn

    public double[] performDelta1D(double[] data) {
        int frameCount = data.length;

        double mSqSum = 0;
        for (int i = -M; i < M; i++) {
            mSqSum += Math.pow(i, 2);
        }
        double[] delta = new double[frameCount];

        for (int k = 0; k < M; k++) {
            delta[k] = data[k]; // 0 padding
        }
        // from frameCount-M to frameCount
        for (int k = frameCount - M; k < frameCount; k++) {
            delta[k] = data[k];
        }
        for (int j = M; j < frameCount - M; j++) {
            // travel from -M to +M
            double sumDataMulM = 0;
            for (int m = -M; m <= +M; m++) {
                // System.out.println("Current m -->\t"+m+ "current j -->\t"+j +
                // "data [m+j][i] -->\t"+data[m + j][i]);
                sumDataMulM += m * data[m + j];
            }
            // 3. divide
            delta[j] = sumDataMulM / mSqSum;
        }
        // ArrayWriter.printDoubleArrayToConole(delta);
        return delta;
    }

    public static void main(String[] args) {
        MFCC mfcc = new MFCC(5,1,13);
        double[][] m = {{0,1,2,3,4}, {6,7,8,9,10}, {21,22,23,24,45}, {0,1,2,3,4}, {6,7,8,9,10}};
        System.out.println(Arrays.toString(mfcc.meanVariance(m, false)));
        System.out.println(Arrays.deepToString(mfcc.performDelta2D(m)));
        ExtractFeature soundFeatures = new ExtractFeature();
        System.out.println(Arrays.deepToString(soundFeatures.appendH(m, m)));
    }
}
