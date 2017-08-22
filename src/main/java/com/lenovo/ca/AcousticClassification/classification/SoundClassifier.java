package com.lenovo.ca.AcousticClassification.classification;


/**
 * Created by fubo5 on 2017/6/6.
 */

import java.io.FileInputStream;
import java.io.InputStream;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;
import java.util.Queue;

import com.lenovo.ca.AcousticClassification.classification.gmm.GMMPredict;
import com.lenovo.ca.AcousticClassification.classification.weka.WekaClassifier;
import com.lenovo.ca.AcousticClassification.preprocessing.ExtractFeature;
import com.lenovo.ca.AcousticClassification.preprocessing.Preprocess;
import com.lenovo.ca.AcousticClassification.utils.Stat;

public class SoundClassifier {

    private int SAMPLE_RATE = 8000;
    // The emission matrix was learned from the decision tree
    public static double[][] COARSE_MATRIX = {
            {0.892, 0.072, 0.036},
            {0.064, 0.853, 0.083},
            {0.023, 0.060, 0.917}
    };
    public static double[][] GENDER_MATRIX = {
            {0.926, 0.074},
            {0.092, 0.908}
    };
    public static double[][] SCENE_MATRIX = {
            {0.785, 0.050, 0.021, 0.027, 0.021, 0.097},
            {0.00001, 0.812, 0.012, 0.00001, 0.001, 0.174},
            {0.00001, 0.010, 0.770, 0.00001, 0.009, 0.211},
            {0.00001, 0.00001, 0.00001, 0.975, 0.006, 0.018},
            {0.00001, 0.00001, 0.020, 0.015, 0.863, 0.102},
            {0.00001, 0.037, 0.010, 0.003, 0.004, 0.945}
    };
    public static double[][] VOICE_MATRIX = {
    	{0.884061906, 0.042628292, 0.020363834, 0.052945968},
        {0.024650922, 0.958972591, 0.006550595, 0.009825892},
        {0.021983274, 0.009318996, 0.946475508, 0.022222222},
        {0.023597399, 0.017336865, 0.015169757, 0.943895979}
    };

    private WekaClassifier voiceClf;  //声音粗分类 
    private WekaClassifier genderClf;  //性别分类
    private WekaClassifier sceneClf;   //场景分类
    private WekaClassifier ageClf;    // 年龄分类
    
    private GMMPredict gmm_adult = null;
    private GMMPredict gmm_old = null;
    private GMMPredict gmm_young = null;
    
    ClassifierApi classifierApi;  //分类结果
    
    Queue<Integer> voice_queue = new LinkedList<>();
    Queue<Integer> gender_queue = new LinkedList<>();
    Queue<Integer> scene_queue = new LinkedList<>();
    
    private String [] coarse_arr = {"环境1", "环境2", "音乐", "人声"};
    private String [] gender_arr = {"男","女"};
    private String [] scene_arr = {"windy","road","metro_station","car","bus","others"};

    /**
     * 
     * @param voiceClf
     * @param genderClf
     */
    private void Init(WekaClassifier voiceClf, WekaClassifier genderClf){
		if(null == this.voiceClf)
			this.voiceClf = voiceClf;
		if(null == this.genderClf)
			this.genderClf = genderClf;
		
		if(null == this.voice_queue)
			this.voice_queue = new LinkedList<Integer>();
		if(null == this.gender_queue)
			this.gender_queue = new LinkedList<Integer>();
		
		if(null == this.classifierApi){
			this.classifierApi = new ClassifierApi();
		}
	}
    
    public SoundClassifier(InputStream voiceIn, InputStream genderIn) throws Exception {
    	
    	//加载模型
    	WekaClassifier voiceClf = new WekaClassifier();
    	voiceClf.loadClassifier(voiceIn);
    	
    	WekaClassifier genderClf = new WekaClassifier();
//    	genderClf.loadClassifier(genderIn);
    	//初始化模型
    	this.Init(voiceClf, genderClf);
    }
    
    public SoundClassifier(String voiceModelPath, String genderModelPath) throws Exception {
    	
    	//加载模型
    	WekaClassifier voiceClf = new WekaClassifier();
    	InputStream voiceIn = new FileInputStream(voiceModelPath);
    	voiceClf.loadClassifier(voiceIn);
    	
    	WekaClassifier genderClf = new WekaClassifier();
    	InputStream genderIn = new FileInputStream(genderModelPath);
    	genderClf.loadClassifier(genderIn);
    	//初始化模型
    	this.Init(voiceClf, genderClf);
    }

    
    
    private void initGMMPredictModels(){
		gmm_young = new GMMPredict("gmm_age_young.model");
		
		gmm_adult = new GMMPredict("gmm_age_adult.model");
		
		gmm_old = new GMMPredict("gmm_age_old.model");
    }
    
    /**
     * 开发阶段的结果-_-
     * @return
     */
    public Map<String, String> getDetectionResult(){
    	Map<String, String> reMap = new HashMap<String, String>();
    	StringBuilder resultMsg=new StringBuilder("[环境1,环境2,音乐,人声]\n");
    	if(!classifierApi.isCoarseQuiet()){
    		//声音粗分类结果概率
    		double[] coarseProbability = classifierApi.getCoarseProbability();
    		int c_index = this.getMaxIndex(coarseProbability);
    		voice_queue.add(c_index);
    		if (voice_queue.size() > 5){
                voice_queue.poll();
            }
    		//通过贝叶斯分类获得最终结果
    		int voice_result = this.bayesianPredictor(SoundClassifier.VOICE_MATRIX,
    				voice_queue.toArray(new Integer[]{}));
    		resultMsg.append(Arrays.toString(coarseProbability)+": ");
    		resultMsg.append(coarse_arr[voice_result]+"(最近5秒)\n");
    		
//    		//性别分类结果概率
//    		double[] genderProbability = classifierApi.getGenderProbability();
//    		int gender_index = getMaxIndex(genderProbability);
//    		gender_queue.add(gender_index);
//            if (gender_queue.size() > 5){
//                gender_queue.poll();
//            }
//            //通过贝叶斯分类获得最终结果
//            int gender_result = this.bayesianPredictor(SoundClassifier.GENDER_MATRIX,
//            	gender_queue.toArray(new Integer[]{}));
//            resultMsg.append(gender_arr[gender_result]+"\n");

    		reMap.put("msg", resultMsg.toString());
    	}else{
    		reMap.put("msg", "Quiet...");
    	}
    	return reMap;
    }
    
    public void feedSoundWindow(double[] window) {
        try {
            ExtractFeature soundFeatures = new ExtractFeature();
            // Reshape one window to multiple frames for coarse classifier and finer classifier
            double[][] coarseFrames = Preprocess.frameSignal(window, this.SAMPLE_RATE, 0.064, 0.064);
            double[][] fineFrames = Preprocess.frameSignal(window, this.SAMPLE_RATE, 0.032, 0.016);

            // Only calculate the feature if the window is informative
            if (Preprocess.frameAdmin(coarseFrames, -60, 6, 0.0)) {
                // Calculate the coarse classification for every second
                double[] coarseFeatures = soundFeatures.extractFeatures(coarseFrames, 
                		window, this.SAMPLE_RATE);
//                classifierApi.setCoarseProbability(this.clf.classifyInstance(coarseFeatures));
                classifierApi.setCoarseProbability(this.voiceClf.getDistribution(coarseFeatures));
                classifierApi.setCoarseQuiet(false);
            } else {
                classifierApi.setCoarseQuiet(true);
            }
            // Do the finer classification if more than half of the frames are informative
            if (Preprocess.frameAdmin(fineFrames, -60, 6, 0.2)) {
                double[] fineFeatures = soundFeatures.mfccWindowFeatures(fineFrames, this.SAMPLE_RATE);
//                classifierApi.setGenderProbability(this.genderClf.getDistribution(fineFeatures));
////                classifierApi.setSceneProbability(this.sceneClf.classifyInstance(fineFeatures));
//                classifierApi.setGenderQuiet(false);
                
//                classifierApi.setAgeProbability(this.ageClf.getDistribution(fineFeatures));
//                classifierApi.setAgeQuiet(false);
//                
//                double[][] gmmAgeFeatures = soundFeatures.mfccFrameFeatures(fineFrames, this.SAMPLE_RATE);
//                double[] ageScores = new double[3];
//                double maxScore = gmm_young.getScore(gmmAgeFeatures);
//                String age="Young";
//                ageScores[0]=maxScore;
//                
//                double currentScore =  gmm_adult.getScore(gmmAgeFeatures);
//                ageScores[1] = currentScore;
//                if(currentScore > maxScore){
//                	maxScore = currentScore;
//                	age="Adult";
//                }
//                
//                currentScore =  gmm_old.getScore(gmmAgeFeatures);
//                ageScores[2] = currentScore;
//                if(currentScore > maxScore){
//                	maxScore = currentScore;
//                	age="Old";
//                }
//                
//                classifierApi.setAgeScores(ageScores);
//                classifierApi.setAge(age);
            } else {
                classifierApi.setGenderQuiet(true);
                classifierApi.setAgeQuiet(true);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public int bayesianPredictor(double[][] probabilityMatrix, Integer[] observations) {
        // Calculate the logarithm of the probabilities, so that we use adding instead of multiplying later
        double[][] logEmissionMatrix = new double[probabilityMatrix.length][probabilityMatrix[0].length];
        for (int i = 0; i < probabilityMatrix.length; i++) {
            for (int j = 0; j < probabilityMatrix[0].length; j++) {
                logEmissionMatrix[i][j] = Math.log10(probabilityMatrix[i][j]);
            }
        }

        // Calculate the overall probability (in LOG) of each hidden state
        double[] overallProb = new double[probabilityMatrix.length];
        Arrays.fill(overallProb, 0);
        for (int i = 0; i < probabilityMatrix.length; i++) {
            for (int observe : observations) {
                overallProb[i] += logEmissionMatrix[i][observe];
            }
        }
        return Stat.maxIndex(overallProb);
    }
    
    private int getMaxIndex(double[] array){
        int index = 0;
        double max = array[0];
        for (int i = 1 ; i < array.length ; i++){
            if (array[i] > max){
                max = array[i];
                index = i;
            }
        }
        return index;
    }

}
