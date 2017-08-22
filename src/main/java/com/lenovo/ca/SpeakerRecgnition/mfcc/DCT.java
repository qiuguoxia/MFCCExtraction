package com.lenovo.ca.SpeakerRecgnition.mfcc;

public class DCT {
	
	  /**
     * number of mfcc coefficients
     */
    final int numCoefficients;
    /**
     * number of Mel Filters
     */
    final int M;

    /**
     * @param numCoefficients
     *            length of array, i.e., number of features
     * @param M
     *            number of Mel Filters
     */
    public DCT(int numCoefficients, int M) {
        this.numCoefficients = numCoefficients;
        this.M = M;
    }

    public double[] perform(double y[]) {
        final double cepc[] = new double[numCoefficients];
        // perform DCT
        for (int n = 1; n <= numCoefficients; n++) {
            for (int i = 1; i <= M; i++) {
                cepc[n - 1] += y[i - 1] * Math.cos(Math.PI * (n - 1) / M * (i - 0.5));
            }
        }
        return cepc;
    }

}
