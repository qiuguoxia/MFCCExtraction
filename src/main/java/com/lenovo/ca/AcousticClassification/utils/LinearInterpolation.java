package com.lenovo.ca.AcousticClassification.utils;


import java.util.Arrays;

/**
 * Construct new data points within the range of a discrete set of known data points by linear equation
 */
public class LinearInterpolation {

    public LinearInterpolation(){

    }

    /**
     * Do interpolation on the samples according to the original and destinated sample rates
     *
     * @param oldSampleRate sample rate of the original samples
     * @param newSampleRate sample rate of the interpolated samples
     * @param samples original samples
     * @return interpolated samples
     */
    public static double[] interpolate(double[] samples, int oldSampleRate, int newSampleRate) {

        if (oldSampleRate==newSampleRate){
            return samples;
        }

        int newLength=(int)Math.round(((float)samples.length/oldSampleRate*newSampleRate));
        float lengthMultiplier=(float)newLength/samples.length;
        double[] interpolatedSamples = new double[newLength];

        // interpolate the value by the linear equation y=mx+c
        for (int i = 0; i < newLength; i++){

            // get the nearest positions for the interpolated point
            float currentPosition = i / lengthMultiplier;
            int nearestLeftPosition = (int)currentPosition;
            int nearestRightPosition = nearestLeftPosition + 1;
            if (nearestRightPosition>=samples.length){
                nearestRightPosition=samples.length-1;
            }

            double slope=samples[nearestRightPosition]-samples[nearestLeftPosition]; // delta x is 1
            float positionFromLeft = currentPosition - nearestLeftPosition;

            interpolatedSamples[i] = (double)(slope*positionFromLeft+samples[nearestLeftPosition]); // y=mx+c
        }

        return interpolatedSamples;
    }
    public static void main(String[] args) {
        double[] rawSamples = {1,2,3,4,5,6,7,8,9,10,12,14,16,18,20};
        double[] newSamples = interpolate(rawSamples, 2, 1);
        System.out.println(Arrays.toString(newSamples));
    }
}