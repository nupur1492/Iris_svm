
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
		Dataset irisData = FileHandler.loadDataset(new File("src/iris.data"), 4, ",");
		ToWekaUtils weka = new ToWekaUtils(irisData);
		Instances irisInstances = weka.getDataset();
		int index = irisInstances.numAttributes()-1;
		irisInstances.setClassIndex(index);
		
		irisInstances.randomize(new java.util.Random());	
		Instances trainData = irisInstances.trainCV(2,0);
		Instances testData = irisInstances.testCV(2, 0);
		
		FromWekaUtils javamlTrain = new FromWekaUtils(trainData); 
		FromWekaUtils javamlTest = new FromWekaUtils(testData);
		
		Classifier svm = new LibSVM();
		svm.buildClassifier(javamlTrain.getDataset());
		
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
