package com.lenovo.ca.AcousticClassification.classification.weka;


import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;


/**
 * 
 * @author liuck1
 * @time 2017-05-11
 * 
 */
public class WekaClassifier {
	protected Classifier classifier;
	protected Instances dataSet;
	protected Attribute classAttribute;
	
	
	/**
	 * 加载数据
	 */
	public void loadTrainSet(String inputPath) throws IOException {
		dataSet = new Instances(new BufferedReader(new FileReader(inputPath)));
		dataSet.setClassIndex(dataSet.numAttributes() - 1);
		classAttribute = dataSet.attribute(dataSet.numAttributes() - 1);
	}

	/**
	 * 构建模型
	 * @param option:模型参数
	 * @throws Exception
	 */
	public void buildClassifierModel(AbstractClassifier classifier, String option) throws Exception {
		this.classifier = classifier;
		classifier.setOptions(weka.core.Utils.splitOptions(option));
		this.classifier.buildClassifier(dataSet);
	}
	
	/**
	 *  构建模型
	 * @throws Exception
	 */
	public void buildClassifierModel(Classifier classifier) throws Exception {
		this.classifier = classifier;
		this.classifier.buildClassifier(dataSet);
	}
	
	/**
	 * Save the classifier model save as " ***.model" 
	 * @param modelPath
	 * @throws Exception
	 */
	public void saveClassifier(String modelPath) throws Exception {
		//save model
		Map<String, Object> modelMap = new HashMap<String, Object>();
		modelMap.put("model", classifier);
		modelMap.put("config", classAttribute);
		ObjectOutputStream modelOos = new ObjectOutputStream(new FileOutputStream(modelPath));
		modelOos.writeObject(modelMap);
		modelOos.flush();
		modelOos.close();
		
		//save dataSet
//		String configPath = modelPath.substring(0,modelPath.lastIndexOf(".")) + ".config";
//		ObjectOutputStream oos3 = new ObjectOutputStream(new FileOutputStream(configPath));
//		oos3.writeObject(dataSet);
//		oos3.flush();
//		oos3.close();
		
//		ArffSaver saver = new ArffSaver();
//		saver.setInstances(dataSet);
//		saver.setFile(new File(arffPath));
//		saver.writeBatch();
		
	}
	/**
	 * 
	 * @param modelPath:模型路径
	 * @throws Exception
	 */
	public void loadClassifier(String modelPath) throws Exception {
		//read model
		InputStream inModel = new FileInputStream(modelPath);
		this.loadClassifier(inModel);
		
	}
	
	/**
	 * load classifier model
	 * @param inM: model file InputStream
	 * @throws Exception
	 */
	public void loadClassifier(InputStream inModel) throws Exception {
		//read model
		ObjectInputStream modelOis = new ObjectInputStream(inModel);
		Map<String, Object> modelMap = (HashMap<String, Object>) modelOis.readObject();
		classifier = (Classifier) modelMap.get("model");
		classAttribute = (Attribute) modelMap.get("config");
		modelOis.close();
		
	}
	
	/**
	 * load model, 此方法加载模型是为了重新在学习模型
	 * @param modelPath: model file path
	 * @param dataPath:模型训练数据，需要加载原来数据，合并新的数据用于重新训练
	 * @throws Exception
	 */
	public void loadClassifier(String modelPath, String dataPath) throws Exception {
		loadClassifier(modelPath);
		//read dataSet
		ObjectInputStream oisD = new ObjectInputStream(new FileInputStream(dataPath));
		dataSet = (Instances) oisD.readObject();
		oisD.close();
		
	}
	
	/**
	 * load model, 此方法加载模型是为了重新在学习模型
	 * @param inModel: model InputStream
	 * @param inAttr: classAttribute file InputStream
	 * @param inData: data file InputStream,
	 *  need to load the original data and merge the new data for retraining
	 * @throws Exception
	 */
	public void loadClassifier(InputStream inModel, InputStream inData) throws Exception {
		loadClassifier(inModel);
		//read dataSet
		ObjectInputStream oisD = new ObjectInputStream(inData);
		dataSet = (Instances) oisD.readObject();
		oisD.close();
	}
	
	
	/**
	 * evaluate model
	 * @param filePath: data file 
	 * @return
	 * @throws Exception
	 */
	public double evaluateFromFile(String filePath) throws Exception {
		if(null!=classifier){
			Instances testSet = new Instances(new BufferedReader(new FileReader(filePath)));
			//set class index
			testSet.setClassIndex(testSet.numAttributes() - 1);
			//load data
			Evaluation evaluation = new Evaluation(testSet);
			//load model and data
			evaluation.evaluateModel(classifier, testSet);
//			List<Prediction> predictions = evaluation.predictions();
//			predictions.get(0).actual();predictions.get(0).predicted();
			//print confusion matrix
	        System.out.println(evaluation.toMatrixString());
	        //get accuracy
	        double accuracy = (1-evaluation.errorRate())*100;
	        return accuracy;
		}else{
			throw new NullPointerException("classifier is null");
		}
	}
	
	/**
	 * predict from data file
	 * @param filePath:data file
	 * @return results: predict result list
	 * @throws Exception
	 */
	public List<String> predictFromFile(String filePath) throws Exception {
		Instances testSet = new Instances(new BufferedReader(new FileReader(filePath)));
		testSet.setClassIndex(testSet.numAttributes() - 1);
		List<String> results = new ArrayList<String>();
		Attribute attr = testSet.attribute(testSet.classIndex());
		for (Instance ins:testSet) {
			double result = classifier.classifyInstance(ins);
			int classIndex = (int) result;
			String label = attr.value(classIndex);
			results.add(label);
//			double[] d = classifier.distributionForInstance(ins);
//			System.out.println(label+" "+d[0]+" "+d[1]);
		}
		return results;
	}
	
	/**
	 * predict 
	 * @param data
	 * @return resultClass: 
	 * @throws Exception
	 */
	public String predict(double[] data) throws Exception{
		Instance ins = toInstance(data);
		double classIndex = classifier.classifyInstance(ins);
		String resultClass = classAttribute.value((int)classIndex);
		return resultClass;
	}
	
	/**
	 * get probability
	 * @param data
	 * @return predictProbability
	 * @throws Exception 
	 */
	public double[] getDistribution(double[] data) throws Exception{
		Instance ins = toInstance(data);
		double[] predictProbability = classifier.distributionForInstance(ins);
		return predictProbability;
	}
	
	public void train(){
		System.out.println("please overwrite this method");
	}

	/**
	 * change data to Instance
	 * @param data:double[]
	 * @return
	 */
	private Instance toInstance(double[] data){
		Instance ins = new DenseInstance(data.length+1);
		//set data value
		ArrayList<Attribute> atts = new ArrayList<Attribute>();
		for (int i = 0; i < data.length; i++){
			ins.setValue(i, data[i]);
			Attribute att = new Attribute("Att"+i);
			atts.add(att);
		}
		//add class attribute
//		atts.add(dataSet.attribute(dataSet.numAttributes()-1));
		if(null == classAttribute){
			try {
				throw new Exception("must set model classAttribute");
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		atts.add(classAttribute);
		//new a blank Instances
		Instances dataset = new Instances("Data", atts, 0);
		//must set ClassIndex
		dataset.setClassIndex(data.length);
        //must set Dataset
		ins.setDataset(dataset);
		return ins;
	}
	
	public static void main(String[] args) throws Exception{
		String inputPath = "data/iodata44-6.arff";
		String option = "-P 100 -I 20 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1";
		WekaClassifier classifier = new WekaClassifier();
		classifier.loadTrainSet(inputPath);
		classifier.buildClassifierModel(new RandomForest(),option);
		
//		System.out.println(classifier.evaluateFromFile(inputPath));
		classifier.predictFromFile(inputPath);
	}
	
}
