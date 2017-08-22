package com.lenovo.ca.AcousticClassification.classification;


import java.io.Serializable;

/**
 * Created by fubo5 on 2017/6/6.
 */
public class ClassifierApi implements Serializable {
    private boolean coarseQuiet;
    private boolean genderQuiet;
    private boolean ageQuiet;

    public boolean isCoarseQuiet() {
        return coarseQuiet;
    }

    public void setCoarseQuiet(boolean coarseQuiet) {
        this.coarseQuiet = coarseQuiet;
    }

    public boolean isGenderQuiet() {
        return genderQuiet;
    }

    public void setGenderQuiet(boolean genderQuiet) {
        this.genderQuiet = genderQuiet;
    }
    
    public boolean isAgeQuiet() {
		return ageQuiet;
	}

	public void setAgeQuiet(boolean ageQuiet) {
		this.ageQuiet = ageQuiet;
	}

	private double[] coarseProbability;
    private double[] genderProbability;
    private double[] sceneProbability;

//    private double[] voiceProbability;

    private double[] ageProbability;
    
    private double[] ageScores;

    private String age;
    
    
    public double[] getCoarseProbability() {
        return coarseProbability;
    }

    public void setCoarseProbability(double[] coarseProbability) {
        this.coarseProbability = coarseProbability;
    }

    public double[] getGenderProbability() {
        return genderProbability;
    }

    public void setGenderProbability(double[] genderProbability) {
        this.genderProbability = genderProbability;
    }

    public double[] getSceneProbability() {
        return sceneProbability;
    }

    public void setSceneProbability(double[] sceneProbability) {
        this.sceneProbability = sceneProbability;
    }

//    public double[] getVoiceProbability() {
//        return voiceProbability;
//    }
//
//    public void setVoiceProbability(double[] voiceProbability) {
//        this.voiceProbability = voiceProbability;
//    }

	public double[] getAgeProbability() {
		return ageProbability;
	}

	public void setAgeProbability(double[] ageProbability) {
		this.ageProbability = ageProbability;
	}

	public double[] getAgeScores() {
		return ageScores;
	}

	public void setAgeScores(double[] ageScores) {
		this.ageScores = ageScores;
	}

	public String getAge() {
		return age;
	}

	public void setAge(String age) {
		this.age = age;
	}
	
}
