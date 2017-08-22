package com.lenovo.ca.AcousticClassification.classification.gmm;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import com.lenovo.ca.AcousticClassification.utils.FileUtils;


public class GMMTrain {
	private static final double EPS = 2.2204460492503131e-16;
	private int n_init = 1500;
	private int n_iter = 10;
	private int numOfRows;
	private int numOfCols;
	private int maxIter;
	private double threshold;
	private int numOfComponents;
	private double[][] observations;
	private double min_covar = 0.001;
	private boolean converged = false;
	private double current_log_likelihood = 0;
	private double prev_log_likelihood = Double.NaN;
	private double tol = 0.0001;

	private double[] log_likelihoods = null;
	private double[][] responsibilities = null;

	private double[][] means = null;
	private double[] weights = null;
	private double[][] covars = null;

	private double[][] best_means = null;
	private double[] best_weights = null;
	private double[][] best_covars = null;

	GMMTrain(double[][] data, int compNum) {
		this.observations = data;
		this.numOfRows = data.length;
		this.numOfCols = data[0].length;
		this.numOfComponents = compNum;
		this.means = new double[compNum][data[0].length];
		this.weights = new double[data.length];
		this.covars = new double[compNum][data[0].length];
	}

	GMMTrain(double[][] data, int compNum, int maxIt) {
		this(data, compNum);
		this.maxIter = maxIt;
	}

	GMMTrain(double[][] data, int compNum, int maxIt, double thr) {
		this(data, compNum);
		this.maxIter = maxIt;
		this.threshold = thr;

	}
//EM算法做参数估计
	public void fit() {
		double change = 0;

		try {

			double[][] cv = new double[this.numOfCols][this.numOfCols];
			double max_log_prob = Double.NEGATIVE_INFINITY;

			for (int i = 0; i < this.n_init; i++) {
				KMeans kMeans = new KMeans(this.observations, this.numOfComponents);
				kMeans.fit();
				this.means = kMeans.get_centers();
				this.weights = Matrixes.fillWith(this.weights, (double) 1 / this.numOfComponents);

				this.covars = Matrixes.cov(Matrixes.transpose(this.observations));

				cv = Matrixes.eye(this.observations[0].length, this.min_covar);

				this.covars = Matrixes.addMatrixes(this.covars, cv);
				this.covars = Matrixes.duplicate(Matrixes.chooseDiagonalValues(this.covars), this.numOfComponents);

				for (int j = 0; j < this.n_iter; j++) {
					prev_log_likelihood = current_log_likelihood;
					Score_samples score_samples = new Score_samples(this.observations, this.means, this.covars, this.weights);
					this.log_likelihoods = score_samples.getLogprob();
					this.responsibilities = score_samples.getResponsibilities();
					current_log_likelihood = Statistics.getMean(log_likelihoods);

					if (!Double.isNaN(prev_log_likelihood)) {
						change = Math.abs(current_log_likelihood - prev_log_likelihood);
						if (change < this.tol) {
							this.converged = true;
							break;
						}

					}

					// / do m-step - gmm.py line 509
					do_mstep(this.observations, this.responsibilities);

				}

				if (current_log_likelihood > max_log_prob) {
					max_log_prob = current_log_likelihood;
					this.best_means = this.means;
					this.best_covars = this.covars;
					this.best_weights = this.weights;

				}
			}

			if (Double.isInfinite(max_log_prob))
				System.out.println("EM algorithm was never able to compute a valid likelihood given initial parameters");
		} catch (Exception myEx) {
			myEx.printStackTrace();
			System.exit(1);
		}

	}

	public double[][] get_means() {
		return this.best_means;
	}

	public double[][] get_covars() {
		return this.best_covars;
	}

	public double[] get_weights() {
		return this.best_weights;
	}

	private void do_mstep(double[][] data, double[][] responsibilities) {
		try {
			double[] weights = Matrixes.sum(responsibilities, 0);
			double[][] weighted_X_sum = Matrixes.multiplyByMatrix(Matrixes.transpose(responsibilities), data);
			double[] inverse_weights = Matrixes.invertElements(Matrixes.addValue(weights, 10 * EPS));
			this.weights = Matrixes.addValue(Matrixes.multiplyByValue(weights, 1.0 / (Matrixes.sum(weights) + 10 * EPS)), EPS);
			this.means = Matrixes.multiplyByValue(weighted_X_sum, inverse_weights);
			this.covars = covar_mstep_diag(this.means, data, responsibilities, weighted_X_sum, inverse_weights, this.min_covar);
		} catch (Exception myEx) {
			myEx.printStackTrace();
			System.exit(1);
		}

	}

	private double[][] covar_mstep_diag(double[][] means, double[][] X, double[][] responsibilities, double[][] weighted_X_sum, double[] norm, double min_covar) {
		double[][] temp = null;
		try {
			double[][] avg_X2 = Matrixes.multiplyByValue(Matrixes.multiplyByMatrix(Matrixes.transpose(responsibilities), Matrixes.multiplyMatrixesElByEl(X, X)), norm);
			double[][] avg_means2 = Matrixes.power(means, 2);
			double[][] avg_X_means = Matrixes.multiplyByValue(Matrixes.multiplyMatrixesElByEl(means, weighted_X_sum), norm);
			temp = Matrixes.addValue(Matrixes.addMatrixes(Matrixes.substractMatrixes(avg_X2, Matrixes.multiplyByValue(avg_X_means, 2)), avg_means2), min_covar);
		} catch (Exception myEx) {
			System.out.println("An exception encourred: " + myEx.getMessage());
			myEx.printStackTrace();
			System.exit(1);
		}
		return temp;
	}

	private class Score_samples {
		private double[][] data = null;
		private double[] log_likelihoods = null;
		private double[][] means = null;
		private double[][] covars = null;
		private double[] weights = null;
		/* out matrixes */
		private double[] logprob = null;
		private double[][] responsibilities = null;

		Score_samples(double[][] X, double[][] means, double[][] covars, double[] weights) {
			this.data = X;
			this.log_likelihoods = new double[X.length];
			this.responsibilities = new double[X.length][GMMTrain.this.numOfComponents];
			this.means = means;
			this.covars = covars;
			this.weights = weights;

			try {
				double[][] lpr = log_multivariate_normal_density(this.data, this.means, this.covars);
				lpr = Matrixes.addValue(lpr, Matrixes.makeLog(this.weights));
				this.logprob = Matrixes.logsumexp(lpr);
				// gmm.py line 321
				this.responsibilities = Matrixes.exp(Matrixes.substractValue(lpr, logprob));
			} catch (Exception myEx) {
				myEx.printStackTrace();
				System.exit(1);
			}

		}

		public double[] getLogprob() {
			return this.logprob;
		}

		public double[][] getResponsibilities() {
			return this.responsibilities;
		}

		private double[][] log_multivariate_normal_density(double[][] data, double[][] means, double[][] covars) {
			// diagonal type
			double[][] lpr = new double[data.length][means.length];
			int n_samples = data.length;
			int n_dim = data[0].length;

			try {
				double[] sumLogCov = Matrixes.sum(Matrixes.makeLog(covars), 1);
				double[] sumDivMeanCov = Matrixes.sum(Matrixes.divideElements(Matrixes.power(this.means, 2), this.covars), 1);
				double[][] dotXdivMeanCovT = Matrixes.multiplyByValue(Matrixes.multiplyByMatrix(data, Matrixes.transpose(Matrixes.divideElements(means, covars))), -2);

				double[][] dotXdivOneCovT = Matrixes.multiplyByMatrix(Matrixes.power(data, 2), Matrixes.transpose(Matrixes.invertElements(covars)));

				sumLogCov = Matrixes.addValue(sumLogCov, n_dim * Math.log(2 * Math.PI));
				sumDivMeanCov = Matrixes.addMatrixes(sumDivMeanCov, sumLogCov);
				dotXdivOneCovT = Matrixes.sum(dotXdivOneCovT, dotXdivMeanCovT);
				dotXdivOneCovT = Matrixes.addValue(dotXdivOneCovT, sumDivMeanCov);
				lpr = Matrixes.multiplyByValue(dotXdivOneCovT, -0.5);
			} catch (Exception myEx) {
				System.out.println("An exception encourred: " + myEx.getMessage());
				myEx.printStackTrace();
				System.exit(1);
			}

			return lpr;
		}
	}

	public static void main(String[] args) {
		// List<String> lines = FileUtils.readLines("data/iris.csv");
		List<String> lines = FileUtils.readLines("output/ageFrames.csv");
		Map<String, List<String>> dataMap = new HashMap<String, List<String>>();

		String line = null;
		String label = null;
		for (int i = 0; i < lines.size(); i++) {
			line = lines.get(i);
			label = line.substring(line.lastIndexOf(',') + 1);
			if (!dataMap.containsKey(label)) {
				dataMap.put(label, new ArrayList<String>());
			}

			dataMap.get(label).add(line);
		}

		double[][] datas = null;
		String[] attrs = null;
		double[] feature = null;
		
		long startTime =	0l;
		long endTime =	0l;
		for (String gmmLabel : dataMap.keySet()) {
			startTime =	System.currentTimeMillis();
			datas = new double[dataMap.get(gmmLabel).size()][24];
			for (int i = 0; i < dataMap.get(gmmLabel).size(); i++) {
				line = dataMap.get(gmmLabel).get(i);
				attrs = line.split(",");
				feature = new double[24];

				for (int j = 0; j < attrs.length - 1; j++) {
					feature[j] = Double.parseDouble(attrs[j]);
				}
				datas[i] = feature;
			}
			
			trainGMM(datas, 3, gmmLabel);
			endTime =	System.currentTimeMillis();
			System.out.println("Spend "+(endTime-startTime)/1000 +"s.");
		}
		
		
	/*	List<String> lines = FileUtils.readLines("C:\\dev\\Context_Awareness\\Algorithm\\Acoustic_Classification\\output\\feature files\\femaleFrames.csv");
		double[][] datas = new double[lines.size()][24];
		for (int i = 1; i < lines.size(); i++) {
			String line = lines.get(i);
			String[] attrs = line.split(",");
			double[] feature = new double[24];
			for (int j = 0; j < attrs.length - 1; j++) {
				feature[j] = Double.parseDouble(attrs[j]);
			}
			
			trainGMM(datas, 3, gmmLabel);
			endTime =	System.currentTimeMillis();
			System.out.println("Spend "+(endTime-startTime)/1000 +"s.");
		}*/
	}

	private static void trainGMM(double[][] datas, int compNum, String gmmLabel) {
		System.out.println("############################### " + gmmLabel + "("+datas.length+") ###############################");
		GMMTrain gmm = new GMMTrain(datas, compNum);

		//System.out.println("Start building GMM");
		//GMMTrain gmm = new GMMTrain(datas, 8);
		
		gmm.fit();
		double[] weight = gmm.get_weights();
		System.out.println("weight: ");
		System.out.println(Arrays.toString(weight));
		
		System.out.println("");
		System.out.println("means: ");
		double[][] means = gmm.get_means();
		System.out.println(Arrays.deepToString(means));
		
		System.out.println("");
		System.out.println("covars: ");
		double[][] covars = gmm.get_covars();
		System.out.println(Arrays.deepToString(covars));

	}

}
