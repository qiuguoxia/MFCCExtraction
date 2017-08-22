package com.lenovo.ca.AcousticClassification.preprocessing;


import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;

import com.lenovo.ca.AcousticClassification.utils.Complex;
import com.lenovo.ca.AcousticClassification.utils.FFT;
import com.lenovo.ca.AcousticClassification.utils.Filtering;
import com.lenovo.ca.AcousticClassification.utils.LinearInterpolation;
import com.lenovo.ca.AcousticClassification.utils.Stat;

/**
 * Created by fubo5 on 2017/3/2.
 * Pre-process the data before extracting the features. The pre-processing includes crop the signal
 * into multiple frames, using the rms and entropy thresholds to keep the informational frames.
 *
 */
public class Preprocess {
    /**
     * Applies a Hanning Window to the data set.
     * Hanning Windows are used to increase the accuracy of the FFT.
     * One should always apply a window to a dataset before applying an FFT
     * @param signalIn The data you want to apply the window to
     * @param pos The starting index you want to apply a window from
     * @param size The size of the window
     * @return The windowed data set
     */
    public static double[] applyHannWindow(double[] signalIn, int pos, int size){
        for (int i = pos; i < pos + size; i++){
            int j = i - pos; // j = index into Hann window function
            signalIn[i] = signalIn[i] * 0.5 * (1.0 - Math.cos(2.0 * Math.PI * j / size));
        }
        return signalIn;
    }

    /**
     * Overload applyHanningWindow() method. If the parameter is not given. Apply Hanning Window to the whole window.
     */
    public static double[] applyHannWindow(double[] signalIn) {
        signalIn = applyHannWindow(signalIn, 0, signalIn.length);
        return signalIn;
    }

    /**
     * Applies a Hamming Window to the data set.
     * Hamming Windows are used to increase the accuracy of the FFT.
     * One should always apply a window to a dataset before applying an FFT
     * @param signalIn The data you want to apply the window to
     * @param pos The starting index you want to apply a window from
     * @param size The size of the window
     * @return The windowed data set
     */
    public static double[] applyHammingWindow(double[] signalIn, int pos, int size) {
       // double alpha = 0.54;
    	double alpha=0.46;
        double beta = 1 - alpha;
        for (int i = pos; i < pos + size; i++){
            int j = i - pos; // j = index into Hanmming window function
            signalIn[i] = signalIn[i] * (alpha - beta * Math.cos(2.0 * Math.PI * j / size));
        }
        return signalIn;
    }

    /**
     * Overload applyHammingWindow() method. If the parameter is not given. Apply Hamming Window to the whole window.
     */
    public static double[] applyHammingWindow(double[] signalIn) {
        signalIn = applyHammingWindow(signalIn, 0, signalIn.length);
        return signalIn;
    }

    /**
     * Normalize the window so that its average is 0 and the range is within -0.95 and 0.95.
     * @param window
     * @return The normalized window
     */
    public static double[] normalize(double[] window) {
        double mean = Stat.mean(window);
        double range = Stat.max(window) - Stat.min(window);
        for (int i = 0; i < window.length; i++) {
            window[i] = (window[i] - mean) / range * 0.95;
        }
        return window;
    }

    /**
     * Downsamples the signal to 8 kHz. The input signal should be one-second long.
     */
    public static double[] resample(double[] signal, int oldSampleRate, int newSampleRate) {
        // if newSampleRate = oldSampleRate, return the original signal
        if (oldSampleRate == newSampleRate) return signal;
        //System.out.println(Arrays.toString(signal));
        double[] lowPassSamples = Filtering.filterSignal(signal, oldSampleRate,
                newSampleRate/2, 2, 0, 0);
        return LinearInterpolation.interpolate(lowPassSamples, oldSampleRate, newSampleRate);
    }

    /**
     * Split the signal into sliding frames.
     *
     * @param  signal 1D array of the sound signal
     * @param  sampleRate sampling frequency. E.g. 44100
     * @param  length window length in seconds.
     * @param  step sliding step in second
     * @return a 2D array of the frames
     */
    public static double[][] frameSignal(double[] signal, int sampleRate, double length, double step) {
        int lengthInSample = (int) ((double) sampleRate * length);
        int stepInSample = (int) ((double) sampleRate * step);
        int numWindows = (signal.length - lengthInSample) / stepInSample + 1;
        System.out.println("numWindows"+numWindows+"lengthInSample"+lengthInSample);
        double[][] frames = new double[numWindows][lengthInSample];
        for (int i = 0; i < numWindows; i++) {
            for (int j = 0; j < lengthInSample; j++) {
                frames[i][j] = signal[i * stepInSample + j];
            }
        }
        return frames;
    }

    /**
     * Calculate the root mean square of the signal
     *
     * @param  frame 1D array of the sound signal
     * @return the root mean square of the input data array
     */
    public static double rms(double[] frame) {
        double sum = 0.0;
        for (double data : frame) {
            sum += Math.pow(data, 2);
        }
//        System.out.println("rms is:");
//        System.out.println(Math.pow(sum/frame.length, 0.5));
//        System.out.println("-----");
        return Math.pow(sum/frame.length, 0.5);
    }

    /**
     * Calculate the root mean square of the signal in dB.
     *
     * @param  frame 1D array of the sound signal
     * @return the root mean square of the input data array in dB.
     */
    private static double rmsLog(double[] frame) {

//        // Probe
//        System.out.print("RMS in LOG is: ");
//        System.out.println(20 * Math.log10(rms(frame)));

        return 20 * Math.log10(rms(frame));
    }

    /**
     * Convert the time-series signal to complex and calculate the fft transform.
     *
     * @param  frame 1D array of the sound signal
     * @return the fft transform in complex format.
     */
    private static Complex[] signalToFFT(double[] frame) {
        Complex[] frameComplex = new Complex[frame.length];
        for(int i = 0; i<frameComplex.length; i++){
            frameComplex[i] = new Complex(frame[i], 0);
        }
        return FFT.fft(frameComplex);
    }

    /**
     * Get the amplitudes of the signal after fft transformation.
     */
    public static double[] fftAmplitudes(double[] frame) {
        Complex[] frameFFT = signalToFFT(frame);
        double[] amplitudes = new double[frame.length/2]; // notice, we only keep the first half of the frequency domain
        for (int i=0; i<frame.length/2; i++) {
            amplitudes[i] = frameFFT[i].abs();
        }
        return amplitudes;
    }

    /**
     * Get the phases of the signal after fft tranformation.
     */
    public static double[] fftPhases(double[] frame) {
        Complex[] frameFFT = signalToFFT(frame);
        double[] phases = new double[frame.length/2];
        for (int i=0; i<frame.length/2; i++) {
            phases[i] = frameFFT[i].phase();
        }
        return phases;
    }

    /**
     * Calculate the probability distribution function of the signal.
     *
     * @param  frame 1D array of the sound signal
     * @return the root mean square of the input data array in dB.
     */
    public static double[] pdf(double[] frame) {
        double[] amplitudes = fftAmplitudes(frame);

        // Calculate the power of the fft signal
        double[] spectrum = new double[amplitudes.length];
        double powerSum = 0;
        for (int i=1; i<amplitudes.length; i++) {  // Discard the DC value and calculate from the 2nd FFT value
            spectrum[i] = Math.pow(amplitudes[i], 2);
            powerSum += spectrum[i];
        }

        // Normalize the power to probability density
        for (int i = 1; i < spectrum.length; i++) {
            spectrum[i] = spectrum[i] / powerSum;
        }

        // Replace the 1st element with a very small number in case of zero dividing error
        spectrum[0] = 1e-10;
        return spectrum;
    }

    /**
     * Calculate the power spectrum of the signal.
     *
     * @param  frame 1D array of the sound signal
     */
    public static double[] powerSpectrum(double[] frame) {
        double[] amplitudes = fftAmplitudes(frame);
        // Calculate the power of the fft signal
        double[] spectrum = new double[amplitudes.length];
        double powerSum = 0;
        for (int i=1; i<amplitudes.length; i++) {  // Discard the DC value and calculate from the 2nd FFT value
            spectrum[i] = Math.pow(amplitudes[i], 2);
            powerSum += spectrum[i];
        }
        // Replace the 1st element with a very small number in case of zero dividing error
        spectrum[0] = 1e-10;
        return spectrum;
    }
    /**
     * Calculate the entropy of the signal.
     *
     * @param  frame 1D array of the sound signal
     * @return the entropy of the input data array.
     */
    private static double entropy(double[] frame) {
        double[] probabilityDensity = pdf(applyHannWindow(frame));
        //System.out.println(Arrays.toString(probabilityDensity));
        double entropy = 0;
        for (double pd : probabilityDensity) {
            entropy += -pd * (Math.log(pd)/Math.log(2));
        }

//        // Probe
//        System.out.print("entropy is ");
//        System.out.println(entropy);

        return entropy;
    }

    /**
     * Calculate the entropy of the signal. If the percentage of informative frames is greater than percent,
     * then return True, other wise return False.
     *
     * @param frames 2D array of the sound signal. Each row is one frame.
     * @param thresholdRMSLog The rms threshold in logrithm. The frame should have an rms bigger than the threshold.
     * @param thresholdEntropy The entropy threshold. The frame should have an entropy smaller than the threshold.
     * @param percent The threshold above which the whole window is regarded as informative.
     *                if (numInform/numWin > percent) return True.
     * @return the kept frames.
     */
    public static boolean frameAdmin(double[][] frames, double thresholdRMSLog, double thresholdEntropy, double percent) {
        double totalFrames = frames.length;
        double infomativeFramesCounter = 0.0;
        for (double[] frame : frames) {
            // keep the whole frames if fulfill either conditions
//            System.out.println("RmsLog: " + rmsLog(frame) + ", Entropy: " + entropy(frame));
            if ((rmsLog(frame) > thresholdRMSLog) && (entropy(frame) < thresholdEntropy)) {
                infomativeFramesCounter++;
            } else { // if quiet
//                System.out.println("RmsLog: " + rmsLog(frame) + ", Entropy: " + entropy(frame));
            }
            if (infomativeFramesCounter/totalFrames > percent) {
                //System.out.println("informative");
                return true;
            }
        }
//        System.out.println("non-informative");
        return false;
    }

    /**
     * Overload frameAdmin() method with default parameters.
     */
    public static boolean frameAdmin(double[][] frames) {
        // todo - need to change the value according to the actual phones.
        return frameAdmin(frames, -60, 6, 0.0);
    }

    /**
     * Remove the quiet frames.
     *
     * @param frames 2D array of the sound signal. Each row is one frame.
     * @param thresholdRMSLog The rms threshold in logrithm. The frame should have an rms bigger than the threshold.
     * @return the kept frames.
     */
    public static double[][] removeQuietFrames(double[][] frames, double thresholdRMSLog) {
        int[] frameList = new int[frames.length];
        int count = 0;
        for (int i = 0; i < frames.length; i++) {
            if (rmsLog(frames[i]) > thresholdRMSLog) {
                frameList[i] = i;
                count++;
            }
        }

        int[] newFrameList = Arrays.copyOfRange(frameList, 0, count);

        // Create a new 2D array to store the informative frames
        double[][] newFrames = new double[count][frames[0].length];
        for (int i = 0; i < newFrameList.length; i++) {
            newFrames[i] = frames[newFrameList[i]];
        }
        System.out.println(frames.length);
        System.out.println(newFrames.length);
        return newFrames;
    }

    /**
     * Convert an array of bytes to an array of short data type.
     * @param bytes an array of bytes, every two bytes combine a short
     * @return
     */
    public static short[] byteToShort(byte[] bytes, boolean littleEndian) {
        short[] shorts = new short[bytes.length/2];
        if (littleEndian) {
            ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().get(shorts);
        } else {
            ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN).asShortBuffer().get(shorts);
        }
        return shorts;
    }

    /**
     * Use byteToShort function to convert bytes to double with rescale (-1, 1)
     * @param bytes
     * @return
     */
    public static double[] byteToDouble(byte[] bytes, boolean littleEndian) {
        short[] shorts = byteToShort(bytes, littleEndian);
        double[] doubles = new double[shorts.length];
        for (int i = 0; i < shorts.length; i++) {
            doubles[i] = (double)shorts[i]/32768;  // rescale by 1/(2^15)
        }
        return doubles;
    }

    public static void main(String[] args) {
        System.out.println("Test results:");
        MicSound testMic = MicSound.getInstance();
        int sampleRate = (int) MicSound.getAudioFormat().getSampleRate();
        System.out.println(sampleRate);

        double[] buffer;
        System.out.println("Test frame admin");
        for (int i=0; i<100; i++){
            // Read in one second
            buffer = testMic.nextBuffer();
            double[][] frames = frameSignal(buffer, sampleRate, 0.064, 0.064);
//            System.out.println(frames[0].length);
            //System.out.println(frameAdmin(frames) ? "Informative" : "Non-informative");
            double[][] newFrames = removeQuietFrames(frames, -35);
            System.out.println(Arrays.deepToString(newFrames));
        }
//
//        byte[] bytes = {106, -1, -106, -14, -66, -10, 21, -2};
//        System.out.println(Arrays.toString(byteToDouble(bytes, true)));
    }
}
