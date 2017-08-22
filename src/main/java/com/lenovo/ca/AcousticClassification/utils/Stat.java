package com.lenovo.ca.AcousticClassification.utils;


import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by fubo5 on 2017/3/21.
 */
public class Stat {
    /**
     * Returns the sum of an array.
     */
    public static double sum(double[] m) {
        double sum = 0;
        for (double element : m) {
            sum += element;
        }
        return sum;
    }

    public static double sum(ArrayList<Double> m) {
        double sum = 0;
        for (double element : m) {
            sum += element;
        }
        return sum;
    }

    /**
     * Returns the mean value of an array.
     */
    public static double mean(double[] m) {
        return sum(m) / m.length;
    }

    public static double mean(ArrayList<Double> m) {
        return sum(m) / m.size();
    }


    /**
     * Returns the median value of an array.
     */
    public static double median(double[] m) {
        Arrays.sort(m);
        int middle = m.length/2;
        if (m.length%2 == 1) {
            return m[middle];
        } else {
            return (m[middle-1] + m[middle]) / 2.0;
        }
    }

    /**
     * Returns the variance of an array.
     */
    public static double var(double[] m) {
        double mean = mean(m);
        double powSum = 0;
        for (double x : m) {
            powSum += (x-mean) * (x-mean);
        }
        return powSum/ m.length;
    }

    /**
     * Returns the standard deviation of an array.
     */
    public static double std(double[] m) {
        return Math.sqrt(var(m));
    }

    /**
     * Returns the maximum value of an array.
     */
    public static double max(double[] m) {
        double maxValue = m[0];
        for (double item : m) {  // update maxValue if comes across a bigger value
            maxValue = (item > maxValue) ? item : maxValue;
        }
        return maxValue;
    }

    /**
     * Returns the minimum value of an array.
     */
    public static double min(double[] m) {
        double minValue = m[0];
        for (double item : m) {  // update minValue if comes across a smaller value
            minValue = (item < minValue) ? item : minValue;
        }
        return minValue;
    }

    /**
     * Returns the absolute value of an array
     * @param array
     * @return
     */
    public static double[] abs(double[] array) {
        for (int i = 0; i < array.length; i++) {
            array[i] = Math.abs(array[i]);
        }
        return array;
    }

    /**
     * Calculate the differential array of the original array.
     * newArray[0] = array[0];
     * newArray[1] = array[1] - array[0];
     * newArray[2] = array[2] - array[1]; and so on.
     *
     * @param m
     * @return a new array of the same length as the input array.
     */
    public static double[] diff(double[] m) {
        double[] mDiff = Arrays.copyOf(m, m.length);
        for (int i=1; i<m.length; i++) {
            mDiff[i] = m[i] - m[i-1];
        }
        return mDiff;
    }

    /**
     * Return the index of the max value in the array
     * @param m
     * @return
     */
    public static int maxIndex(double[] m) {
        int maxIndex = 0;
        for (int i = 1; i < m.length; i++) {
            double newnumber = m[i];
            if ((newnumber > m[maxIndex])) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    /**
     * Returns the most frequent item in an array.
     * @param m
     * @return One element that appears the most of the times.
     */
    public static double popularItem(double[] m) {
        double maxKey = -1;
        int maxValue = -1;

        Map<Double, Integer> pretenders = new HashMap<Double, Integer>();

        int masSize = m.length;

        for (int i = 0; i < masSize; i++) {
            if (!pretenders.containsKey(m[i]))
                pretenders.put(m[i], 1);
            else
                pretenders.put(m[i], pretenders.get(m[i])+1);
        }

        for (Map.Entry< Double, Integer> entry : pretenders.entrySet())
        {
            if (entry.getValue() > maxValue) {
                maxKey = entry.getKey();
                maxValue = entry.getValue();
            }
        }
        return maxKey;
    }

    /**
     * Returns the peak values of a 1D array
     * @param arr 1D array
     * @return
     */
    public static ArrayList<Double> peakValues(double[] arr) {
        ArrayList<Double> peaks = new ArrayList<>();
        for (int i = 2; i < arr.length; i++) {
            if ((arr[i] < arr[i-1]) && (arr[i-1] > arr[i-2])) peaks.add(arr[i-1]);
        }
        return peaks;
    }

    public static void main(String[] args) {
        double[] foo = {11,2,2,2,2,2,2,9,10,4,3,3,1,2,4,5,3,8,3,9,0,3,2};
        double[] bar = {1,2,5,4,7,9,1,0};
//        System.out.println(popularItem(foo));
//        System.out.println(max(foo));
//        System.out.println(min(foo));
        System.out.println(maxIndex(bar));
        System.out.println(peakValues(foo));
    }
}
