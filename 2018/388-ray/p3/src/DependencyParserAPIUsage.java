import edu.stanford.nlp.io.RuntimeIOException;
import edu.stanford.nlp.parser.nndep.DependencyParser;
import edu.stanford.nlp.parser.nndep.DependencyTree;
import edu.stanford.nlp.io.IOUtils;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Properties;
import java.util.List;
import java.util.Comparator;

import org.apache.commons.cli.*;

@SuppressWarnings("Duplicates")

/**
 * Created by abhisheksinha on 3/20/17.
 * Modified by Zeyuan Hu on 4/9/18.
 */
public class DependencyParserAPIUsage {
    public static int NUM_WORDS_PER_BATCH = 1500;

    public static int generateTrainingFile(String unlabeledTrainingPool, PolicyType policy) {
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
            //Collections.shuffle(list, new Random(System.currentTimeMillis()));
            Collections.shuffle(list);
        }

        if (policy == PolicyType.LENGTH) {
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

    public static String[] generateTrainingFile2(DependencyParser model,
                                                 String trainFilePath,
                                                 String unlabeledTrainingPool,
                                                 PolicyType policy) {
        List<String> sents = new ArrayList<>();
        List<Integer> wordCnts = new ArrayList<>();
        List<Integer> list;
        String[] result = {"", ""};

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
            System.out.printf("sentences remaining in unlabeledTrainingPool: %d\n", totalCnt);
            int sum = 0;
            for(Integer d: wordCnts)
                sum += d;
            System.out.printf("words in unlabeledTrainingPool: %d\n", sum);
            assert sents.size() == totalCnt;
            assert wordCnts.size() == totalCnt;
        } catch (IOException e) {
            throw new RuntimeIOException(e);
        }

        List<DependencyTree> predictedParses = model.testCoNLLProb(unlabeledTrainingPool);

        List<Double> scores = new ArrayList<>();

        for (int i = 0; i < totalCnt; i++) {
            if (policy == PolicyType.RAW) {
                scores.add(predictedParses.get(i).RawScore / wordCnts.get(i));
            }
            else if (policy == PolicyType.MARGIN) {
                scores.add(predictedParses.get(i).MarginScore /wordCnts.get(i));
            }
        }
        CustomComparator2 comparator = new CustomComparator2(scores);
        list = comparator.createIndexArray();
        Collections.sort(list, comparator);

        int currCnt = 0;
        String batchStr = "";

        int i = 0;
        for (; i < totalCnt; i++) {
            if (currCnt > NUM_WORDS_PER_BATCH) {
                try {
                    PrintWriter output = new PrintWriter(new BufferedWriter(new FileWriter(trainFilePath, true)));
                    output.write(batchStr);
                    output.close();
                } catch (IOException e) {
                    throw new RuntimeIOException(e);
                }
                batchStr = "";
                break;
            }
            currCnt += wordCnts.get(list.get(i));
            batchStr = batchStr + sents.get(list.get(i)) + "\n";
        }
        try{
            Writer output = IOUtils.getPrintWriter(unlabeledTrainingPool);
            for (; i < totalCnt; i++){
                batchStr = batchStr + sents.get(list.get(i)) + "\n";
            }
            output.write(batchStr);
            output.close();
        } catch (IOException e) {
            throw new RuntimeIOException(e);
        }
        result[0] = trainFilePath;
        result[1] = unlabeledTrainingPool;

        return result;
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

    public static int countWords(String file1) {
        int wordCnt = 0;
        try (BufferedReader reader = IOUtils.readerFromString(file1)) {
            for (String line : IOUtils.getLineIterable(reader, false)) {
                if (!line.isEmpty()) {
                    wordCnt += 1;
                }
            }
        } catch (IOException e) {
            throw new RuntimeIOException(e);
        }
        System.out.printf("words in trainFilePath: %d\n", wordCnt);
        return wordCnt;
    }

    public static PrintStream outputFile(String name) {
        FileOutputStream out;
        PrintStream p  = null;
        try{
            out = new FileOutputStream(name);
            p = new PrintStream(new BufferedOutputStream(out), true);
            return p;
        } catch (FileNotFoundException e){
            System.out.println(e.getMessage());
            System.exit(1);
        }
        return p;
    }

    public static void main(String[] args) {
        long startTime = System.currentTimeMillis();

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

        Option resultFileName = new Option(
                "result",
                true,
                "Name of the file that the result is to be saved"
        );
        policy.setRequired(false);
        options.addOption(resultFileName);

        // @Deprecated
        // We want to log the output of the train(...) method but this option is not helpful.
        Option outputFileName = new Option(
                "output",
                true,
                "Name of the file that you want to save the stdout to"
        );
        policy.setRequired(false);
        options.addOption(outputFileName);

        CommandLineParser parser = new DefaultParser();
        HelpFormatter formatter = new HelpFormatter();
        CommandLine cmd;

        try {
            cmd = parser.parse(options, args);
            if (cmd.hasOption("policy")) {
                switch (cmd.getOptionValue("policy")) {
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
            if (cmd.getOptionValue("output") != null){
                // We redirect all the stdout to a file
                // https://stackoverflow.com/questions/2851234/system-out-to-a-file-in-java
                System.setOut(outputFile(cmd.getOptionValue("output")));
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

        RESULT_FILENAME = cmd.getOptionValue("result", "result.txt");
        System.out.println(RESULT_FILENAME);

        // Create the training batches
        int batchNum = 0;
        if (POLICY_OPT == PolicyType.RANDOM)
            batchNum = generateTrainingFile(unlabeledTrainingPool, POLICY_OPT);
        if (POLICY_OPT == PolicyType.LENGTH)
            batchNum = generateTrainingFile(unlabeledTrainingPool, POLICY_OPT);

        String maxIterVal = cmd.getOptionValue("maxIter", "500");
        Properties prop = new Properties();
        prop.setProperty("maxIter", maxIterVal);
        prop.setProperty("trainingThreads", "1");

        DependencyParser p = new DependencyParser(prop);
        if (POLICY_OPT == PolicyType.RANDOM || POLICY_OPT == PolicyType.LENGTH) {
            try {
                Writer result = IOUtils.getPrintWriter(RESULT_FILENAME);
                for (int i = 0; i < Math.min(batchNum, 20); i++) {
                    result.write("Iteration: " + Integer.toString(i) + "\n");
                    result.write("Number of words: " + Integer.toString(countWords(trainFilePath)) + "\n");
                    p.train(trainFilePath, null, modelPath, embeddingPath);
                    // Test model on test data, write annotations to testAnnotationsPath
                    double lasScore = p.testCoNLL(testFilePath, testAnnotationsPath);
                    result.write("LAS score: " + Double.toString(lasScore) + "\n");
                    String filename = unlabeledTrainingPool + Integer.toString(i);
                    trainFilePath = mergeTrainFiles(trainFilePath, filename, i);
                }
                result.close();
            } catch (IOException e) {
                throw new RuntimeIOException(e);
            }
        }

        if (POLICY_OPT == PolicyType.RAW || POLICY_OPT == PolicyType.MARGIN) {
            try {
                Writer result = IOUtils.getPrintWriter(RESULT_FILENAME);
                for (int i = 0; i < 20; i++) {
                    result.write("Iteration: " + Integer.toString(i) + "\n");
                    result.write("Number of words: " + Integer.toString(countWords(trainFilePath)) + "\n");
                    p.train(trainFilePath, null, modelPath, embeddingPath);
                    String[] files = generateTrainingFile2(p, trainFilePath, unlabeledTrainingPool, POLICY_OPT);
                    trainFilePath = files[0];
                    System.out.printf("trainFilePath: %s\n", trainFilePath);
                    unlabeledTrainingPool = files[1];
                    System.out.printf("unlabeledTrainingPool: %s\n", unlabeledTrainingPool);
                    double lasScore = p.testCoNLL(testFilePath, testAnnotationsPath);
                    result.write("LAS score: " + Double.toString(lasScore) + "\n");
                }
                result.close();
            } catch (IOException e) {
                throw new RuntimeIOException(e);
            }
        }
        long endTime = System.currentTimeMillis();
        System.out.println("Takes: " + (endTime - startTime)/1000.0 + " seconds");
    }
}

// https://stackoverflow.com/questions/2784514/sort-arraylist-of-custom-objects-by-property
// https://stackoverflow.com/questions/4859261/get-the-indices-of-an-array-after-sorting
class CustomComparator implements Comparator<Integer> {
    private final List<Integer> wordCnts;

    public CustomComparator(List<Integer> wordCnts) {
        this.wordCnts = wordCnts;
    }

    public List<Integer> createIndexArray() {
        List<Integer> list = new ArrayList<>();
        for (int i = 0; i < wordCnts.size(); i++)
            list.add(i);
        return list;
    }

    @Override
    public int compare(Integer index1, Integer index2) {
        return wordCnts.get(index2) - wordCnts.get(index1);
    }
}


class CustomComparator2 implements Comparator<Integer> {
    private final List<Double> scores;

    public CustomComparator2(List<Double> scores) {
        this.scores = scores;
    }

    public List<Integer> createIndexArray() {
        List<Integer> list = new ArrayList<>();
        for (int i = 0; i < scores.size(); i++)
            list.add(i);
        return list;
    }

    @Override
    public int compare(Integer index1, Integer index2) {
        return scores.get(index1).compareTo(scores.get(index2));
    }
}

enum PolicyType {
    RANDOM, LENGTH, RAW, MARGIN
}