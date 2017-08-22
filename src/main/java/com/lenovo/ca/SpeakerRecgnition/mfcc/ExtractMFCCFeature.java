package com.lenovo.ca.SpeakerRecgnition.mfcc;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import com.lenovo.ca.AcousticClassification.preprocessing.Preprocess;
import com.lenovo.ca.AcousticClassification.preprocessing.WavFile.WavFile;

import com.lenovo.ca.AcousticClassification.utils.ListFiles;
import com.lenovo.ca.AcousticClassification.utils.Stat;

public class ExtractMFCCFeature {

	private final static int n_fftlength=256;
	
	//final FFT fft;
	
	
  

  

  
   

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
    * 12 MFCC's and additionally the 0-th coefficient
    * Extract MFCC features for each window.
    * @param frames 2D array of the sound signal.
    * @return 1D array of the mean and standard deviation of the MFCCs of the frames
    */
//   public static double[] mfccWindowFeatures(double[][] frames, double sampleRate) {
//	   MFCC mfcc = new MFCC(frames[0].length, sampleRate, 13, true);
//	   
//       double[][] mel2d = new double[frames.length][13];
//       // Calculate the MFCC features for each frame
//       for (int i = 0; i < frames.length; i++) {
//           //System.out.println(Arrays.toString(frames[i]));
//           double[] frame = Preprocess.applyHammingWindow(frames[i]);
//           mel2d[i] = mfcc.mfcc(frame);
//           //System.out.println(mfsc2d[i].length);
//       }
//       // Remove the redundant Coefficient which is the volume
//       double[][] mel2dSubset = new double[mel2d.length][];
//       for (int i = 0; i < mel2d.length; i++) {
//           mel2dSubset[i] = Arrays.copyOfRange(mel2d[i], 1, 13);
//       }
//       return mfcc.meanStd(mel2dSubset, false);
//   }

   /**
    * Extract MFCC features for each frame.
    * @param frames 2D array of the sound signal.
    * @return 2D array of the MFCCs and Delta-MFCCs of the frames
    */
   public static double[][] mfccFrameFeatures(double[][] frames, double sampleRate) {
	   
       MFCC mfcc = new MFCC(19,sampleRate,24,256,true,22,true);
         
       double[][] mel2d = new double[frames.length][19];
       double[] logPower = new double[frames.length];
       // Calculate the MFCC features for each frame
       for (int i = 0; i < frames.length; i++) {
    	   
    	   logPower[i]=logPower(frames[i]);
    	   
    	   //System.out.println("*****对数能量"+logPower[i]);
           //System.out.println(Arrays.toString(frames[i]));
           double[] frame = Preprocess.applyHammingWindow(frames[i]);//对每一帧加窗
           
         //  System.out.println("&&&&&&&&&&&&"+frame.length);
           double[] MelCoefficient = mfcc.getParameters(frame);
           mel2d[i]=MelCoefficient;
       }
       double[][] mel2dSubset = new double[mel2d.length][];
       mel2dSubset = mergeArrays(mel2d,logPower);
       
/*************************************************************************/      
       // Remove the redundant Coefficient which is the volume
//       for (int i = 0; i < mel2d.length; i++) {
//           mel2dSubset[i] = Arrays.copyOfRange(mel2d[i], 1, 19);
//           System.out.println("mel2dSubset维度"+mel2dSubset[i].length);
//       }
      // System.out.println("新的二维数组行指针个数"+mel2dSubset.length);
/************************************************************************/
       
       
       // Calculate the MFCC deltas
      
       double[][] dmfcc = mfcc.performDelta2D(mel2dSubset);
       double[][] dmfcc2 = mfcc.performDelta2D(dmfcc);
       System.out.println("二阶差分后行指针个数"+dmfcc2[0].length);
       
       // Merge mel2dSubset and dmfcc horizontally.
       // I.e., result.row = mel2dSusbset.row = dmfcc.row
       //       result.col = mel2dSubset.col + dmfcc.col
       
       double[][] firstresult = appendH(mel2dSubset, dmfcc);
       System.out.println("firstresultchangdu"+firstresult[0].length);
       double[][] result = appendH(firstresult,dmfcc2);
       System.out.println("resultchangdu"+result[0].length);
       return result;
      
   }
   
   public static double[][] mergeArrays(double[][] array,double []newArray){
	   
	   double[][] newArr = new double[array.length][20];
	   for(int i=0;i<array.length;i++){
		   int j=0;
		   for(;j<19;j++){
			   newArr[i][j]=array[i][j];
		   }
		   newArr[i][19]=newArray[i];
	   }
	   
	   return newArr;
   }
   
   public static double logPower(double[] frameSignal){
	   double sumPower=0.0;
	   for(int i=0;i<frameSignal.length;i++){
		  sumPower+=Math.pow(frameSignal[i],2); 
	   }
	   sumPower=Math.log10(sumPower)*10;
	return sumPower;
	   
   }
   
   public static double[] addArray(double[] array,double newParameter){
	   
	   double[] newArr = new double[array.length+1];
	   newArr = Arrays.copyOf(array, array.length);
	   //newArr = Array.copyOf(array,array.length);
	   newArr[array.length] = newParameter;//new variable
	   return newArr;
	  
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
   public static void featureToFile(double[] features, String featureFile) {
       StringBuilder rowFeature = new StringBuilder();
       for (int i=0; i<features.length; i++) {
           rowFeature.append(String.format("%.6f", features[i]));
           //System.out.println(features.length);
          // System.out.println(String.format("%.6f", features[i]));
           //if (i!=features.length-1) rowFeature.append(",");
           rowFeature.append(",");
       }
      // rowFeature.append(featureName);
       rowFeature.append("\n");

       // Write the StringBuilder to file
       File file = new File(featureFile);
       try {
       	//System.out.println("**************写入文件*****************");
           FileWriter pw = new FileWriter(file, file.exists());
           pw.write(rowFeature.toString());
          // System.out.println("#############写入文件##################");
           pw.close();
       } catch (IOException e) {
           e.printStackTrace();
       }
   }

  
   /**
    * Read all .wav files in the specific folder, calculate the MFCC for each frame, and then do the
    * statistic for each window and write the window feature to a file
    * @param trainFolder The folder contains the .wav files
    * @param mfccFile The file path to write the MFCCs to
    */
//   public static void generateWindowFeature(String trainFolder, String mfccFile, String label, boolean overwrite, boolean shift) {
//       // Load all wave files in the folder
//       ArrayList<File> soundFiles = new ArrayList<File>();
//       ListFiles.allDirs(trainFolder, soundFiles, ".wav");
//
//       // Delete the file if existed
//       File tempFile = new File(mfccFile);
//       if(overwrite) {
//           if (tempFile.exists()) tempFile.delete();
//       }
//
//       int NEW_SAMPLERATE = 8000;
//       int NUM_FRAMES_PER_WINDOW = 60;
//       double FRAME_LENGTH = 0.032;
//       double FRAME_STEP = 0.016;
//       int NUM_MFCC_COEFF = 13;
//
//       for (File file: soundFiles) {
//           // Calculate features from each file
//           System.out.print("Load file ");
//           System.out.println(file);
//           System.out.println();
//
//           try
//           {
//               WavFile wavFile = WavFile.openWavFile(file);
//               int numChannels = wavFile.getNumChannels();
//               long sampleRate = wavFile.getSampleRate();
//               int windowLength = (int)(sampleRate * ((NUM_FRAMES_PER_WINDOW - 1) * FRAME_STEP + FRAME_LENGTH));
//
//               double[][] bufferStereo = new double[numChannels][windowLength];
//               double[] bufferMono = new double[windowLength];
//               double[] buffer = new double[windowLength];
//
//               // set an offset to start reading
//               double[][] bufferStereoOffset = new double[numChannels][windowLength/2];
//               double[] bufferMonoOffset = new double[windowLength/2];
//               double[] bufferShift = new double[windowLength/2];
//               //System.out.println("Buffer: "+ buffer.length);
//
////               // MFCC features
////               MFCC mfcc = new MFCC((int)(NEW_SAMPLERATE*FRAME_LENGTH), NEW_SAMPLERATE, NUM_MFCC_COEFF, true);
////               double[][] mel2d = new double[NUM_FRAMES_PER_WINDOW][NUM_MFCC_COEFF];
//
//               int framesRead = 0;
//
//               if (shift) {
//                   //------------- Skip the offset (half a second) --------------------
//                   if (numChannels == 1) {  // if mono, use the only channel
//                       // added in a half window offset
//                       framesRead = wavFile.readFrames(bufferMonoOffset, bufferShift.length);
//                   } else {  // if stereo, use Channel 0
//                       framesRead = wavFile.readFrames(bufferStereoOffset, bufferShift.length);
//                   }
//                   // ---------- comment the above code if read from beginning --------
//               }
//
//               // every iteration is one second
//               do
//               {
//                   if (numChannels == 1) {  // if mono, use the only channel
//                       // added in a half window offset
//                       framesRead = wavFile.readFrames(bufferMono, buffer.length);
//                       buffer = bufferMono;
//                   } else {  // if stereo, use Channel 0
//                       framesRead = wavFile.readFrames(bufferStereo, buffer.length);
//                       buffer = bufferStereo[0];
//                   }
//
//                   // Down sample to 8 kHz
//                   double[] bufferResample = Preprocess.resample(buffer, (int)sampleRate, NEW_SAMPLERATE);
//                   //System.out.println(Arrays.toString(bufferResample));
//
//                   // Reshape one window to multiple frames
//                   double[][] frames = Preprocess.frameSignal(bufferResample, NEW_SAMPLERATE,
//                           FRAME_LENGTH, FRAME_STEP);
//                   //System.out.println("frame: "+ frames[0].length + " num:" + frames.length);
//
//                   // Discard the window if all frames are non-informative
//                   if (Preprocess.frameAdmin(frames, -50, 6, 0.3)) {
//
////                       // Calculate the MFCC features for each window
//                       double[] melFeatures = mfccWindowFeatures(frames, NEW_SAMPLERATE);
//                       featureToFile(melFeatures, mfccFile);
////                       for (double[] mel1d : mel2dSubset) {
////                           ExtractFeature.featureToFile(mel1d, mfccFile, label);
////                       }
//
//
//                   } else {
//                       System.out.println("non-informative");
//
////                       //record quiet period as well
////                       double[] empty = new double[24];
////                       ExtractFeature.featureToFile(empty, mfccFile, "quiet");
//                   }
//               } while (framesRead == windowLength);  // Stop loading if not enough data
//           } catch (Exception e)
//           {
//               System.err.println(e);
//           }
//       }
//   }

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
      
       int NUM_MFCC_COEFF=19;
       //String path = "";
       
       //Map<String,Integer> ageCategoryMap =new HashMap<String,Integer>();

       File file = new File(trainFilePath);
           // Calculate features from each file
           System.out.print("Load file ");
           System.out.println(file);
           System.out.println();
////           
////           //add by zhuolei to label age
           label="";
          // path= file.getPath();
//
//           
   //        if(path.contains("adult")){
////           	label +="_adult";
////           }else  if(path.contains("old")){
////           	label +="_old";
////           }else{
////           	label +="_young";
////           }

           try
           {
               WavFile wavFile = WavFile.openWavFile(file);
               int numChannels = wavFile.getNumChannels();
               long sampleRate = wavFile.getSampleRate();
               int windowLength = (int)(sampleRate * ((NUM_FRAMES_PER_WINDOW - 1) * FRAME_STEP + FRAME_LENGTH));
               
               //System.out.println("windowLength"+windowLength+"numChannels"+numChannels);
               
               double[][] bufferStereo = new double[numChannels][windowLength];
               double[] bufferMono = new double[windowLength];
               double[] buffer = new double[windowLength];

               // MFCC features
//               MFCC mfcc = new MFCC((int)(NEW_SAMPLERATE*FRAME_LENGTH), NEW_SAMPLERATE, NUM_MFCC_COEFF, true);
//               double[][] mel2d = new double[NUM_FRAMES_PER_WINDOW][NUM_MFCC_COEFF];

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
                   /***************************************************************/
                   //预加重
                   double[] preEmphasisAlpha=preEmphasis(bufferResample);
                   System.out.println("preEmphasisAlpha"+preEmphasisAlpha.length);
                   /***************************************************************/
                   // Reshape one window to multiple frames
                   //bufferResample 信号数据
                   /*************分帧***********************************************/
                   double[][] frames = Preprocess.frameSignal(preEmphasisAlpha, NEW_SAMPLERATE,
                           FRAME_LENGTH, FRAME_STEP);
                   
                   //System.out.println("frame: "+ frames[0].length + " num:" + frames.length);

                   // Discard the window if all frames are non-informative
                   if (Preprocess.frameAdmin(frames, -50, 6, 0)) {
                	   
                       System.out.println("&&&&&&&************");
//                       // Calculate the MFCC features (MFCC and DMFCC) for each frame
                       double[][] melFeatures = mfccFrameFeatures(frames, NEW_SAMPLERATE);
                       System.out.println("melFeature行指针个数"+melFeatures.length+"melFeatures维度"+melFeatures[0].length);
                       for (double[] melFeature : melFeatures) {
                           featureToFile(melFeature, mfccFile);
//                           
                       }
//                       for (double[] mel1d : mel2dSubset) {
//                           ExtractFeature.featureToFile(mel1d, mfccFile, label);
//                       }
                   } else {
                       System.out.println("non-informative");

//                       //record quiet period as well
//                       double[] empty = new double[24];
//                       ExtractFeature.featureToFile(empty, mfccFile, "quiet");
                   }
               } while (framesRead == windowLength);  // Stop loading if not enough data
               

           } catch (Exception e)
           {
               System.err.println(e);
           }
       
       
//       for(String key:ageCategoryMap.keySet()){
//     	  System.out.println(key+":	"+ageCategoryMap.get(key));
//     }
   }
   
   private static double[] preEmphasis(double inputSignal[]) {
	   double preEmphasisAlpha=0.97;
       final double outputSignal[] = new double[inputSignal.length];
       // apply pre-emphasis to each sample
       outputSignal[0] = inputSignal[0];
       for (int n = 1; n < inputSignal.length; n++) {
           outputSignal[n] = (double) (inputSignal[n] - preEmphasisAlpha * inputSignal[n - 1]);
       }
       return outputSignal;
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

       int NEW_SAMPLERATE = 8000;//声音频率
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
//           if(path.contains("female")){
//           	label = "female";
//           }else{
//           	label = "male";
//           }
           
//           if(path.contains("adult")){
//           	label +="_adult";
//           }else  if(path.contains("old")){
//           	label +="_old";
//           }else{
//           	label +="_young";
//           }

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
//               MFCC mfcc = new MFCC((int)(NEW_SAMPLERATE*FRAME_LENGTH), NEW_SAMPLERATE, NUM_MFCC_COEFF, true);
//               double[][] mel2d = new double[NUM_FRAMES_PER_WINDOW][NUM_MFCC_COEFF];

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

//                       // Calculate the MFCC features (MFCC and DMFCC) for each frame
                       double[][] melFeatures = mfccFrameFeatures(frames, NEW_SAMPLERATE);
                       for (double[] melFeature : melFeatures) {
                           featureToFile(melFeature, mfccFile);//写入csv文件
                           if(!ageCategoryMap.containsKey(label)){
                               ageCategoryMap.put(label, 1);
                           }else{
                           	ageCategoryMap.put(label, ageCategoryMap.get(label)+1);
                           }
                       }
//                       for (double[] mel1d : mel2dSubset) {
//                           ExtractFeature.featureToFile(mel1d, mfccFile, label);
//                       }
                   } else {
                       System.out.println("non-informative");

//                       //record quiet period as well
//                       double[] empty = new double[24];
//                       ExtractFeature.featureToFile(empty, mfccFile, "quiet");
                   }
               } while (framesRead == windowLength);  // Stop loading if not enough data
               

           } catch (Exception e)
           {
               System.err.println(e);
           }
       }
       
//       for(String key:ageCategoryMap.keySet()){
//     	  System.out.println(key+":	"+ageCategoryMap.get(key));
//     }
   }
	public static void main(String args[]){
		
		String VoicePrint_Path="D:\\CA\\voiceData\\motonew\\f\\21\\female21c005.wav";
		//System.out.println(VoicePrint_Path);
		generateFrameFeatureFromFile(VoicePrint_Path, "sp10002.csv", "", true);
		
		
	}

}
