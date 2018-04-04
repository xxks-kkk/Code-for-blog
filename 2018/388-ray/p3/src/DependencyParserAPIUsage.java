import edu.stanford.nlp.io.IOUtilsTest;
import edu.stanford.nlp.io.RuntimeIOException;
import edu.stanford.nlp.parser.nndep.DependencyParser;
import edu.stanford.nlp.parser.nndep.DependencyTree;
import edu.stanford.nlp.util.ScoredObject;
import edu.stanford.nlp.util.ScoredComparator;
import edu.stanford.nlp.io.IOUtils;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Writer;
import java.sql.SQLSyntaxErrorException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Properties;
import java.util.List;
import java.util.Comparator;
import java.util.Arrays;

import org.apache.commons.cli.*;


/**
 * Created by abhisheksinha on 3/20/17.
 * Modified by Zeyuan Hu on 4/9/18.
 */
public class DependencyParserAPIUsage {
    public static int generateTrainingFile(String unlabeledTrainingPool, PolicyType policy) {
        int NUM_WORDS_PER_BATCH = 1500;
        List<String> sents = new ArrayList<>();
        List<Integer> wordCnts = new ArrayList<>();
        int totalCnt = 0;
        try (BufferedReader reader = IOUtils.readerFromString(unlabeledTrainingPool)) {
            String sentBuffer = "";
            int wordCnt = 0;
            for (String line : IOUtils.getLineIterable(reader, false)) {
                if (line.isEmpty()) {
                    totalCnt += 1;
                    sents.add(sentBuffer);
                    wordCnts.add(wordCnt);
                    sentBuffer = "";
                    wordCnt = 0;
                    continue;
                }
                sentBuffer = sentBuffer + line + "\n";
                wordCnt += 1;
            }
            System.out.println(totalCnt);
            assert sents.size() == totalCnt;
            assert wordCnts.size() == totalCnt;
        } catch (IOException e) {
            throw new RuntimeIOException(e);
        }
        List<Integer> list = new ArrayList<>();
        for (int i = 0; i < totalCnt; i++) list.add(i);

        if (policy == PolicyType.RANDOM) {
            // We shuffle the training sentences
            Collections.shuffle(list);
        }

        if (policy == PolicyType.LENGTH){
            CustomComparator comparator = new CustomComparator(wordCnts);
            list = comparator.createIndexArray();
            Collections.sort(list, comparator);
        }

        //System.out.println(list.toString());
        // We create the training batches
        int currCnt = 0;
        int batchNum = 0;
        String batchStr = "";
        for (int i = 0; i < totalCnt; i++) {
            if (currCnt > NUM_WORDS_PER_BATCH) {
                String filename = unlabeledTrainingPool + Integer.toString(batchNum);
                try {
                    Writer output = IOUtils.getPrintWriter(filename);
                    output.write(batchStr);
                    output.close();
                } catch (IOException e) {
                    throw new RuntimeIOException(e);
                }
                batchStr = "";
                batchNum += 1;
                currCnt = 0;
            }
            currCnt += wordCnts.get(list.get(i));
            batchStr = batchStr + sents.get(list.get(i)) + "\n";
        }
        return batchNum;
    }

    public static String mergeTrainFiles(String file1, String file2, int i) {
        String OUTPUT_FILENAME = "accum.conllx" + Integer.toString(i);
        try {
            Writer output = IOUtils.getPrintWriter(OUTPUT_FILENAME);
            try (BufferedReader reader = IOUtils.readerFromString(file1)) {
                output.write(IOUtils.slurpReader(reader));
            } catch (IOException e) {
                throw new RuntimeIOException(e);
            }
            try (BufferedReader reader = IOUtils.readerFromString(file2)) {
                output.write(IOUtils.slurpReader(reader));
            } catch (IOException e) {
                throw new RuntimeIOException(e);
            }
            output.close();
        } catch (IOException e) {
            throw new RuntimeIOException(e);
        }
        return OUTPUT_FILENAME;
    }

    public static int countWords(String file1){
        int wordCnt = 0;
        try (BufferedReader reader = IOUtils.readerFromString(file1)) {
            for (String line : IOUtils.getLineIterable(reader, false)) {
                if(!line.isEmpty()){
                    wordCnt += 1;
                }
            }
        } catch (IOException e) {
            throw new RuntimeIOException(e);
        }
        System.out.println(wordCnt);
        return wordCnt;
    }

    public static void main(String[] args) {
        String RESULT_FILENAME = "result.txt";
        PolicyType POLICY_OPT = PolicyType.RANDOM;

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

        Option unlabelTrain = new Option(
                "unlabelTrain",
                true,
                "Path to unlabeled training instances"
        );
        unlabelTrain.setRequired(true);
        options.addOption(unlabelTrain);

        Option policy = new Option(
                "policy",
                true,
                "Selection policy in the active learning"
        );
        policy.setRequired(false);
        options.addOption(policy);

        options.addOption("h", "help", false, "Print out help manual");

        CommandLineParser parser = new DefaultParser();
        HelpFormatter formatter = new HelpFormatter();
        CommandLine cmd;

        try {
            cmd = parser.parse(options, args);
            if (cmd.hasOption("policy")){
                switch (cmd.getOptionValue("policy")){
                    case "random":
                        POLICY_OPT = PolicyType.RANDOM;
                        break;
                    case "length":
                        POLICY_OPT = PolicyType.LENGTH;
                        break;
                    case "raw":
                        POLICY_OPT = PolicyType.RAW;
                        break;
                    case "margin":
                        POLICY_OPT = PolicyType.MARGIN;
                        break;
                    default:
                        POLICY_OPT = PolicyType.RANDOM;
                }
            }
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

        // Path to "unlabeled" training pool
        String unlabeledTrainingPool = cmd.getOptionValue("unlabelTrain", "unlabeled_train.conllx");
        System.out.println(unlabeledTrainingPool);

        // Create the training batches
        int batchNum = 0;
        if (POLICY_OPT == PolicyType.RANDOM)
            batchNum = generateTrainingFile(unlabeledTrainingPool, POLICY_OPT);
        if (POLICY_OPT == PolicyType.LENGTH)
            batchNum = generateTrainingFile(unlabeledTrainingPool, POLICY_OPT);

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
        //prop.setProperty("evalPerIter", "10");
        DependencyParser p = new DependencyParser(prop);

        // Argument 1 - Training Path
        // Argument 2 - Dev Path (can be null)
        // Argument 3 - Path where model is saved
        // Argument 4 - Path to embedding vectors (can be null)
        // We use "test set" as the dev set in the code because of
        // "the accuracy of the current learned model is tested on the test set
        // after every batch of labeled data is selected and the model is retrained" from project spec.
        DependencyParser model = null;
        try {
            Writer result = IOUtils.getPrintWriter(RESULT_FILENAME);
            for (int i = 0; i < Math.min(batchNum, 20); i++) {
                result.write("Iteration: " + Integer.toString(i) + "\n");
                result.write("Number of words: " + Integer.toString(countWords(trainFilePath)) + "\n");
                p.train(trainFilePath, null, modelPath, embeddingPath);
                // Load a saved path
                model = DependencyParser.loadFromModelFile(modelPath);
                // Test model on test data, write annotations to testAnnotationsPath
                double lasScore = model.testCoNLL(testFilePath, testAnnotationsPath);
                result.write("LAS score: " + Double.toString(lasScore) + "\n");
                if (POLICY_OPT == PolicyType.RANDOM || POLICY_OPT == PolicyType.LENGTH) {
                    String filename = unlabeledTrainingPool + Integer.toString(i);
                    trainFilePath = mergeTrainFiles(trainFilePath, filename, i);
                }
            }
            result.close();
        } catch (IOException e) {
            throw new RuntimeIOException(e);
        }

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
        System.out.printf("Raw Probability: %f\n", predictedParses.get(12).RawScore);
        System.out.printf("Margin Probability: %f\n", predictedParses.get(12).MarginScore);


        // You probably want to use the ScoredObject and scoredComparator classes for this assignment
        // https://nlp.stanford.edu/nlp/javadoc/javanlp-3.6.0/edu/stanford/nlp/util/ScoredObject.html
        // https://nlp.stanford.edu/nlp/javadoc/javanlp/edu/stanford/nlp/util/ScoredComparator.html
    }
}

// https://stackoverflow.com/questions/2784514/sort-arraylist-of-custom-objects-by-property
// https://stackoverflow.com/questions/4859261/get-the-indices-of-an-array-after-sorting
class CustomComparator implements Comparator<Integer>{
    private final List<Integer> wordCnts;

    public CustomComparator(List<Integer> wordCnts){
        this.wordCnts = wordCnts;
    }

    public List<Integer> createIndexArray(){
        List<Integer> list = new ArrayList<>();
        for (int i = 0; i < wordCnts.size(); i++)
            list.add(i);
        return list;
    }

    @Override
    public int compare(Integer index1, Integer index2){
        return wordCnts.get(index2) - wordCnts.get(index1);
    }
}

enum PolicyType{
    RANDOM, LENGTH, RAW, MARGIN
}