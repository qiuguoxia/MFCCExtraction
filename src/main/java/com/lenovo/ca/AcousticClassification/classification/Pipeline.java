package com.lenovo.ca.AcousticClassification.classification;


import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

import com.lenovo.ca.AcousticClassification.classification.weka.WekaClassifier;

import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;

 
public class Pipeline {
	
	/**
	 * 训练
	 * @param inputPath
	 * @param savePath
	 * @param option
	 * @throws Exception
	 */
	public static void train(String inputPath, String savePath,String option) throws Exception{
		WekaClassifier classifier = new WekaClassifier();
		System.out.println("start train...");
		classifier.loadTrainSet(inputPath);
		if(null==option){
			classifier.buildClassifierModel(new RandomForest());
		}else{
			classifier.buildClassifierModel(new RandomForest(),option);
		}
		classifier.saveClassifier(savePath);
		System.out.println("end train...");
	}
	
	public static List<String> testPredict() throws Exception{
		String inputPath = "data/coarse8.0.arff";
		String modelfile = "model/voice_model.model";
		Instances testSet = new Instances(new BufferedReader(new FileReader(inputPath)));
		WekaClassifier classifier = new WekaClassifier();
		classifier.loadClassifier(modelfile);
		List<String> results = new ArrayList<String>();
		for(Instance ins:testSet){
			int numValues = ins.numValues() - 1;
			double[] data = new double[numValues];
			for(int i = 0;i<numValues;i++){
				data[i] = ins.value(i);
			}
			String presult = classifier.predict(data);
			results.add(presult);
		}
		return results;
	}
	
	public static List<String> voicePipeline() throws Exception{
		WekaClassifier classifier = new WekaClassifier();
		String inputPath = "data/coarse8.0.arff";
		String savePath = "model/voice_model.model";//模型训练完成后生成的二进制文件
		String option = "-P 100 -I 100 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1";
		classifier.loadTrainSet(inputPath);
		System.out.println("start train...");
		classifier.buildClassifierModel(new RandomForest(),option);
		System.out.println("end train...");
		classifier.saveClassifier(savePath);
//		System.out.println(classifier.evaluateFromFile(inputPath));
		List<String> results = classifier.predictFromFile(inputPath);
		return results;
	}
	
	public static void main(String[] args) throws Exception{
//		List<String> results1 = voicePipeline();
//		List<String> results2 = testPredict();
//		for(int i=0;i<results1.size();i++){
//			if(!results1.get(i).equals(results2.get(i))){
//				System.out.println("error "+i);
//			}
//		}
//		System.out.println("hello");
		String voiceModelPath = "model/voice_model.model";
		String genderModelPath = "";
		InputStream voiceIn = new FileInputStream(voiceModelPath);
		InputStream genderIn = new FileInputStream(genderModelPath);
		SoundClassifier soundClassifier =new SoundClassifier(voiceIn, genderIn);
	}
}
