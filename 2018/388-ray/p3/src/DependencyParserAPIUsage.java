import edu.stanford.nlp.parser.nndep.DependencyParser;
import edu.stanford.nlp.parser.nndep.DependencyTree;
import edu.stanford.nlp.util.ScoredObject;
import edu.stanford.nlp.util.ScoredComparator;

import java.util.Properties;
import java.util.List;

import org.apache.commons.cli.*;


/**
 * Created by abhisheksinha on 3/20/17.
 * Modified by Zeyuan Hu on 4/9/18.
 */
public class DependencyParserAPIUsage {
    public static void main(String[] args) {
        Options options = new Options();

        Option trainPath = new Option(
                "trainFile",
                true,
                "Path to training file"
        );
        trainPath.setRequired(true);
        options.addOption(trainPath);

        Option testPath = new Option(
                "testFile",
                true,
                "Path to test file"
        );
        testPath.setRequired(true);
        options.addOption(testPath);

        Option embedFile = new Option(
                "embedFile",
                true,
                "Path to embedding file"
        );
        embedFile.setRequired(true);
        options.addOption(embedFile);

        Option modelSavePath = new Option(
                "model",
                true,
                "Path where model is to be saved"
        );
        modelSavePath.setRequired(false);
        options.addOption(modelSavePath);

        Option outFile = new Option(
                "outFile",
                true,
                "Path where test data annotations are stored"
        );
        outFile.setRequired(false);
        options.addOption(outFile);

        Option maxIter = new Option(
                "maxIter",
                true,
                "maxIter property for Stanford Neural Network Dependency Parser"
        );
        maxIter.setRequired(true);
        options.addOption(maxIter);

        CommandLineParser parser = new DefaultParser();
        HelpFormatter formatter = new HelpFormatter();
        CommandLine cmd;

        try{
            cmd = parser.parse(options, args);
        } catch (ParseException e) {
            System.out.println(e.getMessage());
            formatter.printHelp("java -cp stanford-corenlp-jar/*:src/* DependencyParserAPIUsage.java", options);

            System.exit(1);
            return;
        }
        // Training Data path
        String trainFilePath = cmd.getOptionValue("trainFile");
        System.out.println(trainFilePath);

        // Test Data Path
        String testFilePath = cmd.getOptionValue("testFile");
        System.out.println(testFilePath);

        // Path to embedding vectors file
        String embeddingPath = cmd.getOptionValue("embedFile");
        System.out.println(embeddingPath);

        // Path where model is to be saved
        String modelPath = cmd.getOptionValue("model", "outputs/model1");
        System.out.println(modelPath);

        // Path where test data annotations are stored
        String testAnnotationsPath = cmd.getOptionValue("outFile", "outputs/test_annotation.conllx");
        System.out.println(testAnnotationsPath);

        // Configuring propreties for the parser. A full list of properties can be found
        // here https://nlp.stanford.edu/software/nndep.shtml
        // Alternative view of list of properties:
        // https://nlp.stanford.edu/nlp/javadoc/javanlp-3.5.0/edu/stanford/nlp/parser/nndep/DependencyParser.html
        String maxIterVal = cmd.getOptionValue("maxIter", "50");
        System.out.println(maxIterVal);
        Properties prop = new Properties();
        prop.setProperty("maxIter", maxIterVal);
        // Run full UAS (unlabeled attachment score) evaluation every time we finish this number of iterations.
        // (Only valid if a development treebank is provided with ‑devFile.)
        // We set this property because of "the accuracy of the current learned model is tested on the test set
        // after every batch of labeled data is selected and the model is retrained" from project spec.
        prop.setProperty("evalPerIter", "10");
        DependencyParser p = new DependencyParser(prop);

        // Argument 1 - Training Path
        // Argument 2 - Dev Path (can be null)
        // Argument 3 - Path where model is saved
        // Argument 4 - Path to embedding vectors (can be null)
        // We use "test set" as the dev set in the code because of
        // "the accuracy of the current learned model is tested on the test set
        // after every batch of labeled data is selected and the model is retrained" from project spec.
        p.train(trainFilePath, testFilePath, modelPath, embeddingPath);

        // Load a saved path
        DependencyParser model = DependencyParser.loadFromModelFile(modelPath);

        // Test model on test data, write annotations to testAnnotationsPath
        System.out.println(model.testCoNLL(testFilePath, testAnnotationsPath));

        // returns parse trees for all the sentences in test data using model,
        // this function does not come with default parser and has been written for you
        List<DependencyTree> predictedParses = model.testCoNLLProb(testFilePath);

        // By default NN parser does not give you any probability 
        // https://cs.stanford.edu/~danqi/papers/emnlp2014.pdf explains that
        // the parsing is performed by picking the transition with the highest output in the final layer
        // To get a certainty measure from the final layer output layer, we take use a softmax function.
        // For Raw Probability score We sum the logs of probability of every transition taken in the parse tree to get the following metric
        // For Margin Probability score we sum the log of margin between probabilities assigned to two top transitions at every step
        // Following line prints that probability metrics for 12-th sentence in test data
        // all probabilities in log space to reduce numerical errors. Adjust your code accordingly!
        System.out.printf("Raw Probability: %f\n",predictedParses.get(12).RawScore);
        System.out.printf("Margin Probability: %f\n",predictedParses.get(12).MarginScore);


        // You probably want to use the ScoredObject and scoredComparator classes for this assignment
        // https://nlp.stanford.edu/nlp/javadoc/javanlp-3.6.0/edu/stanford/nlp/util/ScoredObject.html
        // https://nlp.stanford.edu/nlp/javadoc/javanlp/edu/stanford/nlp/util/ScoredComparator.html
    }
}
