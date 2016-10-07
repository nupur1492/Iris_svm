
import java.io.*;
import java.util.Map;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.Instance;
import net.sf.javaml.tools.data.FileHandler;
import net.sf.javaml.classification.Classifier;
import weka.core.Instances; 
import net.sf.javaml.tools.weka.ToWekaUtils;
import net.sf.javaml.tools.weka.FromWekaUtils;
import net.sf.javaml.classification.evaluation.EvaluateDataset;
import net.sf.javaml.classification.evaluation.PerformanceMeasure;
import libsvm.LibSVM;

public class svm {
	public static void main(String[] args) throws Exception {
		// Read iris data into javaml Dataset
		Dataset irisData = FileHandler.loadDataset(new File("src/iris.data"), 4, ",");
		
		//Convert javaml to weka instances
		ToWekaUtils weka = new ToWekaUtils(irisData);
		Instances irisInstances = weka.getDataset();
		
		
		int index = irisInstances.numAttributes()-1;
		irisInstances.setClassIndex(index);
		
		//Split data into training and test
		irisInstances.randomize(new java.util.Random());	
		Instances trainData = irisInstances.trainCV(2,0);
		Instances testData = irisInstances.testCV(2, 0);
		
		//Convert weka instances back to javaml
		FromWekaUtils javamlTrain = new FromWekaUtils(trainData); 
		FromWekaUtils javamlTest = new FromWekaUtils(testData);
		
		//Initialize svm classifier
		Classifier svm = new LibSVM();
		
		//Build model using training data
		svm.buildClassifier(javamlTrain.getDataset());
		
		//Get Accuracy using Test Data
		Map<Object, PerformanceMeasure> map = EvaluateDataset.testDataset(svm,javamlTest.getDataset());
		for(Object obj : map.keySet()){
			System.out.println(obj + " : " + map.get(obj).getAccuracy());
		}
		
		int correct = 0, wrong = 0;
		for (Instance inst : javamlTest.getDataset()) {
            Object predictedClassValue = svm.classify(inst);
            Object realClassValue = inst.classValue();
            if (predictedClassValue.equals(realClassValue))
                correct++;
            else
                wrong++;
        }
		System.out.println("Correct predictions  " + correct);
        System.out.println("Wrong predictions " + wrong);
	}
}
