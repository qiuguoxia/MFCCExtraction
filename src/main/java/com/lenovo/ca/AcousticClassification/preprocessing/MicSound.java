package com.lenovo.ca.AcousticClassification.preprocessing;

/**
 * Created by fubo5 on 2017/3/20.
 * Read the sound from the PC microphone.
 */
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.DataLine;
import javax.sound.sampled.TargetDataLine;
import java.util.Arrays;


public class MicSound {
    private  TargetDataLine targetLine ;
    private byte[] targetData;
    private static final MicSound micSound = new MicSound();

    private MicSound() {
        // Initialize the microphone
        //AudioFormat format = new AudioFormat(8000, 16, 1, true, true);
        AudioFormat format = getAudioFormat();

        // windows size in bytes, for 16bit signal : 1second = sampleRate * 2
        // 1 window = 15 frames = 960 ms
        double windowSize = getAudioFormat().getSampleRate() * (getAudioFormat().getSampleSizeInBits()/8) * 0.064 * 15;
        DataLine.Info targetInfo = new DataLine.Info(TargetDataLine.class, format);
        try {
            targetLine = (TargetDataLine) AudioSystem.getLine(targetInfo);
            targetLine.open(format);
            targetLine.start();

            //System.out.println(targetLine.getBufferSize() / 5);
            targetData = new byte[(int) windowSize];
        } catch (Exception e) {
            System.err.println(e);
        }
    }

    public static MicSound getInstance(){
        // Create an instance to initialize the microphone
        return micSound;
    }

    public double[] nextBuffer(){
        // Read in a buffer data from the microphone
        int numBytesRead;
        try {
        numBytesRead = targetLine.read(targetData, 0, targetData.length);
        double[] data = bytesToDoubleArray(targetData);  //Convert the byte array to double array
        return data;
        } catch (Exception e) {
            System.err.println(e);
            System.err.println("Cannot find the microphone.");
            return new double[0];
        }
    }

    public static AudioFormat getAudioFormat() {
        float sampleRate = 44100.0F;
        //8000,11025,16000,22050,44100
        int sampleSizeInBits = 16;
        //8,16
        int channels = 1;
        //1,2
        boolean signed = true;
        //true,false
        boolean bigEndian = false;
        //true,false
        return new AudioFormat(sampleRate, sampleSizeInBits, channels, signed, bigEndian);
    }

    /**
     * Converts bytes from a TargetDataLine into a double[] allowing the information to be read.
     * NOTE: One byte is lost in the conversion so don't expect the arrays to be the same length!
     * @param bufferData The buffer read in from the target data line
     * @return The double[] that the buffer has been converted into.
     */
    private static double[] bytesToDoubleArray(byte[] bufferData){
        final int bytesRecorded = bufferData.length;
        final int bytesPerSample = getAudioFormat().getSampleSizeInBits()/8;
        final double amplification = 1.0; // choose a number as you like
        double[] micBufferData = new double[bytesRecorded / bytesPerSample];  // this was [bytesRecorded - bytesPerSample +1];
        for (int index = 0, floatIndex = 0; index < bytesRecorded; index += bytesPerSample, floatIndex++) {
            double sample = 0;
            for (int b = 0; b < bytesPerSample; b++) {
                int v = bufferData[index + b];
                if (b < bytesPerSample - 1 || bytesPerSample == 1) {
                    v &= 0xFF;
                }
                sample += v << (b * 8);
            }
            double sample32 = amplification * (sample / 32768.0);
            micBufferData[floatIndex] = sample32;
        }
        return micBufferData;
    }

    public static void main(String[] args) {
        MicSound testMic = MicSound.getInstance();
        double[] buffer;
        int i = 0;
        while(i<10){
            buffer = testMic.nextBuffer();
            System.out.println(Arrays.toString(buffer));
            i++;
        }
    }
}