package nlp.lm;

import java.io.*;
import java.util.*;

@SuppressWarnings("Duplicates")

/**
 * @author Ray Mooney
 * @author Zeyuan Hu - zh4378
 * A simple bigram language model that uses simple fixed-weight interpolation
 * with a unigram model for smoothing.
 *
 * Backward model based on the intuition that tokens might be better predicted
 * from their right context rather than their left context.
 */

public class BackwardBigramModel extends Ngram{


    /** Train the reversed sentence on the Forward model */
    public void trainSentence (List<String> sentence, String start_marker, String end_marker) {
        List<String> reverseSentence = new ArrayList<String>(sentence);
        Collections.reverse(reverseSentence);
        super.trainSentence(reverseSentence, start_marker, end_marker);
    }

    /** Overrides Ngrams log-probability for Perplexity.. Now for reverse sentence */
    public double sentenceLogProb (List<String> sentence, String start_marker, String end_marker) {
        List<String> reverseSentence = new ArrayList<String>(sentence);
        Collections.reverse(reverseSentence);
        return super.sentenceLogProb(reverseSentence, start_marker, end_marker);
    }

    /** Overrides log-probability of Ngrams for Word Perplexity.. Now for reverse sentence*/
    public double sentenceLogProb2 (List<String> sentence, String start_marker) {
        List<String> reverseSentence = new ArrayList<String>(sentence);
        Collections.reverse(reverseSentence);
        return super.sentenceLogProb2(reverseSentence, start_marker);
    }

    /** Returns vector of probabilities of predicting each token in the sentence
     * Prediction starts from end-of-sentence to beginning. */
    public double[] sentenceTokenProbs (List<String> sentence, String start_marker, String end_marker) {
        List<String> reverseSentence = new ArrayList<String>(sentence);
        Collections.reverse(reverseSentence);
        return super.sentenceTokenProbs(reverseSentence, start_marker, end_marker);
    }

    /** Train and test a sntence reversal backward bigram model.
     *  Command format: "nlp.lm.BackwardBigramModel [DIR]* [TestFrac]" where DIR
     *  is the name of a file or directory whose LDC POS Tagged files should be
     *  used for input data; and TestFrac is the fraction of the sentences
     *  in this data that should be used for testing, the rest for training.
     *  0 < TestFrac < 1
     *  Uses the last fraction of the data for testing and the first part
     *  for training.
     */
    public static void main(String[] args) throws IOException {
        // All but last arg is a file/directory of LDC tagged input data
        File[] files = new File[args.length - 1];
        for (int i = 0; i < files.length; i++)
            files[i] = new File(args[i]);

        // Last arg is the TestFrac
        double testFraction = Double.valueOf(args[args.length -1]);

        // Get list of sentences from the LDC POS tagged input files
        List<List<String>> sentences = 	POSTaggedFile.convertToTokenLists(files);
        int numSentences = sentences.size();

        // Compute number of test sentences based on TestFrac
        int numTest = (int)Math.round(numSentences * testFraction);

        // Take test sentences from end of data
        List<List<String>> testSentences = sentences.subList(numSentences - numTest, numSentences);

        // Take training sentences from start of data
        List<List<String>> trainSentences = sentences.subList(0, numSentences - numTest);
        System.out.println("# Train Sentences = " + trainSentences.size() +
                " (# words = " + wordCount(trainSentences) +
                ") \n# Test Sentences = " + testSentences.size() +
                " (# words = " + wordCount(testSentences) + ")");

        // Create a backward bigram model and train it.
        BackwardBigramModel model = new BackwardBigramModel();

        System.out.println("Training...");
        model.train(trainSentences, model.end_symbol, model.start_symbol);

        // Test on training data using test and test2
        model.test(trainSentences, model.end_symbol, model.start_symbol);
        model.test2(trainSentences, model.end_symbol);

        System.out.println("Testing...");
        // Test on test data using test and test2
        model.test(testSentences, model.end_symbol, model.start_symbol);
        model.test2(testSentences, model.end_symbol);
    }
}
