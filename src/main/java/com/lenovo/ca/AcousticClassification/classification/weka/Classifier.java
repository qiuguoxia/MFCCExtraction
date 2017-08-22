package com.lenovo.ca.AcousticClassification.classification.weka;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import com.lenovo.ca.AcousticClassification.preprocessing.ExtractFeature;
import com.lenovo.ca.AcousticClassification.preprocessing.Preprocess;
import com.lenovo.ca.AcousticClassification.utils.*;

import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

public class Classifier {
	public String getClassInfo() {
		return classInfo;
	}
	public void setClassInfo(String classInfo) {
		this.classInfo = classInfo;
	}

	private Instances dataSet;
	private Instances formatFile;
	private RandomForest classifier;
	private int sizeOfDataset;
	private int sizeOfAttribute;
	private String classInfo;
    final List<Integer> list = new ArrayList<Integer>();


	public void loadTrainSet(String inputPath) throws IOException {
		dataSet = new Instances(new BufferedReader(new FileReader(inputPath)));
		setClassInfo(dataSet.attribute("Class").toString());
		setSizeOfDataset(dataSet.numInstances());
		setSizeOfAttribute(dataSet.numAttributes());
		dataSet.setClassIndex(this.sizeOfAttribute - 1);
	}

	public void buildClassifierModel(String option) throws Exception {
		this.classifier = new RandomForest();
		classifier.setOptions(weka.core.Utils.splitOptions(option));
		this.classifier.buildClassifier(dataSet);
	}

	public void buildClassifierModel() throws Exception {
		this.classifier = new RandomForest();
		this.classifier.buildClassifier(dataSet);
	}

    /**
     * Save the classifier model as "modelName.model" and the configuration file as "modelName.config"
     * @param modelName The path to save the model and configuration files. E.g. "output\SoundSense"
     * @throws Exception
     */
	public void saveClassifier(String modelName) throws Exception {
		ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(modelName + ".model"));
		oos.writeObject(classifier);
		oos.flush();
		oos.close();

		// This configuration file stores the format of the dataset
		try (FileOutputStream fos = new FileOutputStream(new File(modelName + ".config"), false);
				OutputStreamWriter osw = new OutputStreamWriter(fos, "UTF-8");
				BufferedWriter bw = new BufferedWriter(osw);) {
			bw.write("@relation MultiLabelData"); bw.newLine();
			bw.newLine();
			for(int i = 1; i <= sizeOfAttribute-1; i++){
				String attrInfo = "@attribute Att" + i + " " + "numeric";
				bw.write(attrInfo); bw.newLine();
			}
			bw.write(classInfo); bw.newLine();
			bw.newLine();
			bw.write("@data"); 
//			bw.newLine();
		} catch (Exception e) {
			System.out.print("write data error!");
			e.printStackTrace();
		}
	}

//	// Overload saveModelFile for empty input
//    public void saveClassifier(String modelName) throws Exception {
//        saveClassifier("", modelName);
//    }

    /**
     * Load the classifier model and the configuration file
     * @param modelName The path to save the model and configuration files. E.g. "output\SoundSense"
     * @throws Exception
     */
    public void loadClassifier(String modelName) throws Exception {
		ObjectInputStream ois = new ObjectInputStream(new FileInputStream(modelName + ".model"));
		classifier = (RandomForest) ois.readObject();
		ois.close();
		
		formatFile = new Instances(new BufferedReader(new FileReader(modelName + ".config")));
		formatFile.setClassIndex(formatFile.numAttributes() - 1);
        setSizeOfAttribute(formatFile.numAttributes());
	}

//	//
//    public void loadClassifier(String modelName) throws Exception {
//	    loadClassifier("", modelName);
//    }

    /**
     * Given a sequence of observations, decode the most likely hidden states.
     * Since the whole sequence should come from the same hidden state, thus only return one state index
     * @param observations a sequence of observed states in the order of {ambient, music, speech}
     */
    public static int bayesianPredictor(double[][] probabilityMatrix, int[] observations) {
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

    /**
     * Store the elements with a fixed size rolling buffer. The old element will be removed if oversized.
     * @param element
     * @param bufferSize
     * @return
     */
    public List<Integer> rollingBuffer(int element, int bufferSize) {
        //final ArrayList<String> list = new ArrayList<String>();
        //final List list = new ArrayList();
        this.list.add(element);
        if (this.list.size() > bufferSize){
            this.list.remove(0);
        }
        return this.list;
    }


    public double[] classifyInstance(double[] testSample) throws Exception {
		Instance ins = new DenseInstance(sizeOfAttribute);//DenseInstance(sizeOfAttribute);
		ins.setDataset(formatFile);
		for (int i = 0; i < ins.numAttributes() -1; i++) {
			ins.setValue(i, testSample[i]);
		}
        double[] distributions = classifier.distributionForInstance(ins);

//        System.out.println(Arrays.toString(distributions));
//        double result = classifier.classifyInstance(ins);
////		System.out.println(result);
		
//		Attribute attr = ins.attribute(ins.classIndex());
//		int classIndex = (int) result;
//		String resultClass = attr.value(classIndex);
//		return resultClass;
		return distributions;
	}
	
	/**
	 * @relation MultiLabelData
	 * @attribute Atti(1<=i<=N) numeric
	 * @attribute Class {SEDENTARY,INCAR,RUNNING,WALKING,INTRAIN,BIKING}
	 * 
	 * @data
	 * <f1,f2,...>,<?>
	 * */
	public ArrayList<String> predictFromFile(String filePath) throws Exception {
		Instances testSet = new Instances(new BufferedReader(new FileReader(filePath)));
		testSet.setClassIndex(testSet.numAttributes() - 1);
		ArrayList<String> list = new ArrayList<>();
		for (int i = 0; i < testSet.numInstances(); i++) {
			double result = classifier.classifyInstance(testSet.instance(i));

			Attribute attr = testSet.attribute(testSet.classIndex());
			int classIndex = (int) result;
			String label = attr.value(classIndex);
			list.add(label);
//			double[] d = classifier.distributionForInstance(testSet.instance(i));
		}
		return list;
	}
	
	/**
	 * @relation MultiLabelData
	 * @attribute Atti(1<=i<=N) numeric
	 * @attribute Class {SEDENTARY,INCAR,RUNNING,WALKING,INTRAIN,BIKING}
	 * 
	 * @data
	 * <f1,f2,...>,<label>
	 * */
	public double evaluateFromFile(String filePath) throws Exception {
		Instances testSet = new Instances(new BufferedReader(new FileReader(filePath)));
		testSet.setClassIndex(testSet.numAttributes() - 1);
		Evaluation evaluation = new Evaluation(testSet);
		evaluation.evaluateModel(classifier, testSet);
		FastVector predictions = new FastVector();
        predictions.appendElements(evaluation.predictions());
        System.out.println(evaluation.toMatrixString());
        double accuracy = calculateAccuracy(predictions);
		return accuracy;
	}
    public static double calculateAccuracy(FastVector predictions) {
        double correct = 0;

        for (int i = 0; i < predictions.size(); i++) {
            NominalPrediction np = (NominalPrediction) predictions.elementAt(i);
            if (np.predicted() == np.actual()) {
                correct++;
            }
        }

        return 100 * correct / predictions.size();
    }
	public int getSizeOfAttribute() {
		return sizeOfAttribute;
	}

	public void setSizeOfAttribute(int sizeOfAttribute) {
		this.sizeOfAttribute = sizeOfAttribute;
	}
	public void setSizeOfDataset(int sizeOfDataset) {
		this.sizeOfDataset = sizeOfDataset;
	}

	public int getSizeOfDataset() {
		return sizeOfDataset;
	}

	// The pipeline to call different functions of the classifier
	public static void pipeline(boolean train, boolean evaluate, boolean predictFile, boolean predictFeature)
            throws Exception {
        Classifier clf = new Classifier();
	    if (train) {
            clf.loadTrainSet("output/feature_files/coarse8.0.arff");
            clf.buildClassifierModel("-P 100 -I 100 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1");
            clf.saveClassifier("output/models/sound_sense_1.0");
        }

        if (evaluate) {
//            clf.loadClassifier("output/models/sound_sense_0.57");
//            double accuracy = clf.evaluateFromFile("output/feature files/test_meeting3.arff");
//            System.out.println("Accuracy is: " + accuracy);
        
            clf.loadClassifier("output/models/sound_sense_1.0");
            double accuracy = clf.evaluateFromFile("output/feature_files/test_office.arff");
            System.out.println("Accuracy is: " + accuracy);
        }

        if (predictFile) {
            clf.loadClassifier("output/models/sound_sense_0.86");
            ArrayList<String> predict = clf.predictFromFile("output/feature_files/test_office.arff");
            for(String s: predict){
                System.out.println(s);
            }
        }

        if (predictFeature) {
			int NEW_SAMPLERATE = 8000;
            clf.loadClassifier("SoundSense");

			String csvFile = "raw_data.csv";
			String line = "";
			String cvsSplitBy = ",";

			try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
				ExtractFeature soundFeatures = new ExtractFeature();
				while ((line = br.readLine()) != null) {

					// use comma as separator
					String[] bufferString = line.split(cvsSplitBy);

					// Convert the string array to double array
					double[] buffer = new double[bufferString.length];
					for (int i = 0; i < buffer.length; i++) {
						buffer[i] = Double.parseDouble(bufferString[i]);
					}
					System.out.println(buffer.length);

					// Reshape one window to multiple frames
					double[][] frames = Preprocess.frameSignal(buffer, NEW_SAMPLERATE, 0.064, 0.064);

					// Only calculate the feature if the window is informative
					if (!Preprocess.frameAdmin(frames)) {
						System.out.println("Quiet");
					} else {
						// Calculate the features for every second
						double[] features = soundFeatures.extractFeatures(frames);
						double[] probability = clf.classifyInstance(features);
						System.out.println(Arrays.toString(probability));
					}
				}
			} catch (IOException e) {
				e.printStackTrace();
			}
        }
    }
	public static void main(String[] args) throws Exception {
	    pipeline(false, true, false, false);
//        int[] orders = {0,0,1,1,2};
//        System.out.println(bayesianPredictor(orders));
//        Classifier clf = new Classifier();
//        for (int i = 0; i < 10; i++) {
//            System.out.println(clf.rollingBuffer(i,5));;
//        }
    }
}
