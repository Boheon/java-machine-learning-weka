import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;

public class SimpleWeka {
    public static void main(String[] args) throws Exception{
        int numfolds = 10;
        int numfold = 0;
        int seed = 1;

        Instances data = new Instances(
                new BufferedReader(
                        new FileReader("D:\\IdeaProjects\\wekaex\\src\\main\\data\\titanic2_pre.arff")
                )
        );

        Instances train = data.trainCV(numfolds, numfold, new Random(seed));
        Instances test = data.testCV(numfolds, numfold);

        RandomForest model = new RandomForest();

        train.setClassIndex(train.numAttributes()-1);
        test.setClassIndex(test.numAttributes()-1);

        Evaluation eval = new Evaluation(train);

        eval.crossValidateModel(model, train, numfolds, new Random(seed));

        model.buildClassifier(train);

        eval.evaluateModel(model,test);

        System.out.println(model);
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toMatrixString());

    }
}
