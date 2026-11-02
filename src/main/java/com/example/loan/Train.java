package com.example.loan;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.Logistic;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

import java.io.File;
import java.util.Random;

/** Run this file once (right-click > Run 'Train.main()'). It trains 2 models and saves the better one. */
public class Train {
    static final String CSV = "/Users/shubhgupta/Documents/loan-default-predictor/data/loan.csv";
    static final String TARGET = "loan_status";
    static final String MODEL_OUT = "/Users/shubhgupta/Documents/loan-default-predictor/src/main/resources/loan.model";

    public static void main(String[] args) throws Exception {
        // 1. Load data
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(CSV));
        Instances data = loader.getDataSet();

        // 2. Target column: number -> category (0/1 become classes)
        int t = data.attribute(TARGET).index();
        NumericToNominal toNominal = new NumericToNominal();
        toNominal.setAttributeIndices(String.valueOf(t + 1));
        toNominal.setInputFormat(data);
        data = Filter.useFilter(data, toNominal);
        data.setClassIndex(t);
        int pos = Math.max(0, data.classAttribute().indexOfValue("1")); // "default" class

        // 3. Split 80% train / 20% test
        data.randomize(new Random(42));
        int trainSize = (int) Math.round(data.numInstances() * 0.8);
        Instances train = new Instances(data, 0, trainSize);
        Instances test = new Instances(data, trainSize, data.numInstances() - trainSize);
        System.out.println("Train rows: " + train.numInstances() + ", test rows: " + test.numInstances());

        // 4. Two models. Missing values are filled inside the model (no data leakage)
        RandomForest rf = new RandomForest();
        rf.setNumIterations(100);
        rf.setMaxDepth(12);          // keeps the saved model file small
        Classifier[] bases = {new Logistic(), rf};
        String[] names = {"Logistic Regression", "Random Forest"};

        FilteredClassifier best = null;
        double bestAuc = -1;
        System.out.println("\nModel                 AUC     Precision  Recall   F1   (for class 'default')");
        for (int i = 0; i < bases.length; i++) {
            FilteredClassifier model = new FilteredClassifier();
            model.setFilter(new ReplaceMissingValues());
            model.setClassifier(bases[i]);
            model.buildClassifier(train);

            Evaluation eval = new Evaluation(train);
            eval.evaluateModel(model, test);
            double auc = eval.areaUnderROC(pos);
            System.out.printf("%-20s  %.3f   %.3f      %.3f    %.3f%n", names[i], auc,
                    eval.precision(pos), eval.recall(pos), eval.fMeasure(pos));
            System.out.println(eval.toMatrixString("Confusion matrix for " + names[i]));
            if (auc > bestAuc) { bestAuc = auc; best = model; }
        }

        // 5. Save the best model + column info for the web app
        SerializationHelper.writeAll(MODEL_OUT, new Object[]{best, new Instances(train, 0)});
        System.out.println("Saved best model (AUC " + String.format("%.3f", bestAuc) + ") to " + MODEL_OUT);
    }
}
