package com.robrua.nlp.bert;

import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.net.URL;
import java.nio.IntBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.collect.Lists;
import com.google.common.io.Resources;

/**
 * <p>
 * easy-bert is a dead simple API for using <a href="https://github.com/google-research/bert">Google's high quality BERT language model</a>.
 *
 * The easy-bert Java bindings allow you to run pre-trained BERT models generated with easy-bert's Python tools. You can also used pre-generated models on Maven
 * Central.
 * <br>
 * <br>
 * <p>
 * To load a model from your local filesystem, you can use:
 * 
 * <blockquote>
 * <pre>
 * {@code
 * try(Bert bert = Bert.load(new File("/path/to/your/model/"))) {
 *     // Embed some sequences
 * }
 * }
 * </pre>
 * </blockquote>
 * 
 * If the model is on your classpath (e.g. if you're pulling it in via Maven), you can use:
 * 
 * <blockquote>
 * <pre>
 * {@code
 * try(Bert bert = Bert.load("/resource/path/to/your/model/")) {
 *     // Embed some sequences
 * }
 * }
 * </pre>
 * </blockquote>
 * 
 * See <a href="https://github.com/robrua/easy-bert">the easy-bert GitHub Repository</a> for information about model available via Maven Central.
 * <br>
 * <br>
 * <p>
 * Once you have a BERT model loaded, you can get sequence embeddings using {@link com.robrua.nlp.bert.Bert#embedSequence(String)},
 * {@link com.robrua.nlp.bert.Bert#embedSequences(String...)}, {@link com.robrua.nlp.bert.Bert#embedSequences(Iterable)}, or
 * {@link com.robrua.nlp.bert.Bert#embedSequences(Iterator)}:
 * 
 * <blockquote>
 * <pre>
 * {@code
 * float[] embedding = bert.embedSequence("A sequence");
 * float[][] embeddings = bert.embedSequence("Multiple", "Sequences");
 * }
 * </pre>
 * </blockquote>
 * 
 * If you want per-token embeddings, you can use {@link com.robrua.nlp.bert.Bert#embedTokens(String)}, {@link com.robrua.nlp.bert.Bert#embedTokens(String...)},
 * {@link com.robrua.nlp.bert.Bert#embedTokens(Iterable)}, or {@link com.robrua.nlp.bert.Bert#embedTokens(Iterator)}:
 *
 * <blockquote>
 * <pre>
 * {@code
 * float[][] embedding = bert.embedTokens("A sequence");
 * float[][][] embeddings = bert.embedTokens("Multiple", "Sequences");
 * }
 * </pre>
 * </blockquote>
 *
 * @author Rob Rua (https://github.com/robrua)
 * @version 1.0.3
 * @since 1.0.3
 *
 * @see <a href="https://github.com/robrua/easy-bert">The easy-bert GitHub Repository</a>
 * @see <a href="https://github.com/google-research/bert">Google's BERT GitHub Repository</a>
 */
public class Bert implements AutoCloseable {
    private class Inputs implements AutoCloseable {
        private final Tensor<Integer> inputIds, inputMask, segmentIds;

        public Inputs(final IntBuffer inputIds, final IntBuffer inputMask, final IntBuffer segmentIds, final int count) {
            this.inputIds = Tensor.create(new long[] {count, model.maxSequenceLength}, inputIds);
            this.inputMask = Tensor.create(new long[] {count, model.maxSequenceLength}, inputMask);
            this.segmentIds = Tensor.create(new long[] {count, model.maxSequenceLength}, segmentIds);
        }

        @Override
        public void close() {
            inputIds.close();
            inputMask.close();
            segmentIds.close();
        }
    }

    private static class ModelDetails {
        public boolean doLowerCase;
        public String inputIds, inputMask, segmentIds, pooledOutput, sequenceOutput;
        public int maxSequenceLength;
    }

    private static final int FILE_COPY_BUFFER_BYTES = 1024 * 1024;
    private static final String MODEL_DETAILS = "model.json";
    private static final String SEPARATOR_TOKEN = "[SEP]";
    private static final String START_TOKEN = "[CLS]";
    private static final String VOCAB_FILE = "vocab.txt";

    /**
     * Loads a pre-trained BERT model from a TensorFlow saved model saved by the easy-bert Python utilities
     *
     * @param model
     *        the model to load
     * @return a ready-to-use BERT model
     * @since 1.0.3
     */
    public static Bert load(final File model) {
        return load(Paths.get(model.toURI()));
    }

    /**
     * Loads a pre-trained BERT model from a TensorFlow saved model saved by the easy-bert Python utilities
     *
     * @param path
     *        the path to load the model from
     * @return a ready-to-use BERT model
     * @since 1.0.3
     */
    public static Bert load(Path path) {
        path = path.toAbsolutePath();
        ModelDetails model;
        try {
            model = new ObjectMapper().readValue(path.resolve("assets").resolve(MODEL_DETAILS).toFile(), ModelDetails.class);
        } catch(final IOException e) {
            throw new RuntimeException(e);
        }

        return new Bert(SavedModelBundle.load(path.toString(), "serve"), model, path.resolve("assets").resolve(VOCAB_FILE));
    }

    /**
     * Loads a pre-trained BERT model from a TensorFlow saved model saved by the easy-bert Python utilities. The target resource should be in .zip format.
     *
     * @param resource
     *        the resource path to load the model from - should be in .zip format
     * @return a ready-to-use BERT model
     * @since 1.0.3
     */
    public static Bert load(final String resource) {
        Path directory = null;
        try {
            // Create a temp directory to unpack the zip into
            final URL model = Resources.getResource(resource);
            directory = Files.createTempDirectory("easy-bert-");

            try(ZipInputStream zip = new ZipInputStream(Resources.asByteSource(model).openBufferedStream())) {
                ZipEntry entry;
                // Copy each zip entry into the temp directory
                while((entry = zip.getNextEntry()) != null) {
                    final Path path = directory.resolve(entry.getName());
                    if(entry.getName().endsWith("/")) {
                        Files.createDirectories(path);
                    } else {
                        Files.createFile(path);

                        try(OutputStream output = Files.newOutputStream(path)) {
                            final byte[] buffer = new byte[FILE_COPY_BUFFER_BYTES];
                            int bytes;
                            while((bytes = zip.read(buffer)) > 0) {
                                output.write(buffer, 0, bytes);
                            }
                        }
                    }
                    zip.closeEntry();
                }
            }

            // Load a BERT model from the temp directory
            return Bert.load(directory);
        } catch(final IOException e) {
            throw new RuntimeException(e);
        } finally {
            // Clean up the temp directory
            if(directory != null && Files.exists(directory)) {
                try {
                    Files.walk(directory)
                        .sorted(Comparator.reverseOrder())
                        .forEach((final Path file) -> {
                            try {
                                Files.delete(file);
                            } catch(final IOException e) {
                                throw new RuntimeException(e);
                            }
                        });
                } catch(final IOException e) {
                    throw new RuntimeException(e);
                }
            }
        }
    }

    private final SavedModelBundle bundle;
    private final ModelDetails model;
    private final int separatorTokenId;
    private final int startTokenId;
    private final FullTokenizer tokenizer;

    private Bert(final SavedModelBundle bundle, final ModelDetails model, final Path vocabulary) {
        tokenizer = new FullTokenizer(vocabulary, model.doLowerCase);
        this.bundle = bundle;
        this.model = model;

        final int[] ids = tokenizer.convert(new String[] {START_TOKEN, SEPARATOR_TOKEN});
        startTokenId = ids[0];
        separatorTokenId = ids[1];
    }

    @Override
    public void close() {
        bundle.close();
    }

    /**
     * Gets a pooled BERT embedding for a single sequence. Sequences are usually individual sentences, but don't have to be.
     *
     * @param sequence
     *        the sequence to embed
     * @return the pooled embedding for the sequence
     * @since 1.0.3
     */
    public float[] embedSequence(final String sequence) {
        try(Inputs inputs = getInputs(sequence)) {
            final List<Tensor<?>> output = bundle.session().runner()
                .feed(model.inputIds, inputs.inputIds)
                .feed(model.inputMask, inputs.inputMask)
                .feed(model.segmentIds, inputs.segmentIds)
                .fetch(model.pooledOutput)
                .run();

            try(Tensor<?> embedding = output.get(0)) {
                final float[][] converted = new float[1][(int)embedding.shape()[1]];
                embedding.copyTo(converted);
                return converted[0];
            }
        }
    }

    /**
     * Gets pooled BERT embeddings for multiple sequences. Sequences are usually individual sentences, but don't have to be.
     * The sequences will be processed in parallel as a single batch input to the TensorFlow model.
     *
     * @param sequences
     *        the sequences to embed
     * @return the pooled embeddings for the sequences, in the order the input {@link java.lang.Iterable} provided them
     * @since 1.0.3
     */
    public float[][] embedSequences(final Iterable<String> sequences) {
        final List<String> list = Lists.newArrayList(sequences);
        return embedSequences(list.toArray(new String[list.size()]));
    }

    /**
     * Gets pooled BERT embeddings for multiple sequences. Sequences are usually individual sentences, but don't have to be.
     * The sequences will be processed in parallel as a single batch input to the TensorFlow model.
     *
     * @param sequences
     *        the sequences to embed
     * @return the pooled embeddings for the sequences, in the order the input {@link java.util.Iterator} provided them
     * @since 1.0.3
     */
    public float[][] embedSequences(final Iterator<String> sequences) {
        final List<String> list = Lists.newArrayList(sequences);
        return embedSequences(list.toArray(new String[list.size()]));
    }

    /**
     * Gets pooled BERT embeddings for multiple sequences. Sequences are usually individual sentences, but don't have to be.
     * The sequences will be processed in parallel as a single batch input to the TensorFlow model.
     *
     * @param sequences
     *        the sequences to embed
     * @return the pooled embeddings for the sequences, in the order they were provided
     * @since 1.0.3
     */
    public float[][] embedSequences(final String... sequences) {
        try(Inputs inputs = getInputs(sequences)) {
            final List<Tensor<?>> output = bundle.session().runner()
                .feed(model.inputIds, inputs.inputIds)
                .feed(model.inputMask, inputs.inputMask)
                .feed(model.segmentIds, inputs.segmentIds)
                .fetch(model.pooledOutput)
                .run();

            try(Tensor<?> embedding = output.get(0)) {
                final float[][] converted = new float[sequences.length][(int)embedding.shape()[1]];
                embedding.copyTo(converted);
                return converted;
            }
        }
    }

    /**
     * Gets BERT embeddings for each of the tokens in multiple sequences. Sequences are usually individual sentences, but don't have to be.
     * The sequences will be processed in parallel as a single batch input to the TensorFlow model.
     *
     * @param sequences
     *        the sequences to embed
     * @return the token embeddings for the sequences, in the order the input {@link java.lang.Iterable} provided them
     * @since 1.0.3
     */
    public float[][][] embedTokens(final Iterable<String> sequences) {
        final List<String> list = Lists.newArrayList(sequences);
        return embedTokens(list.toArray(new String[list.size()]));
    }

    /**
     * Gets BERT embeddings for each of the tokens in multiple sequences. Sequences are usually individual sentences, but don't have to be.
     * The sequences will be processed in parallel as a single batch input to the TensorFlow model.
     *
     * @param sequences
     *        the sequences to embed
     * @return the token embeddings for the sequences, in the order the input {@link java.util.Iterator} provided them
     * @since 1.0.3
     */
    public float[][][] embedTokens(final Iterator<String> sequences) {
        final List<String> list = Lists.newArrayList(sequences);
        return embedTokens(list.toArray(new String[list.size()]));
    }

    /**
     * Gets BERT embeddings for each of the tokens in single sequence. Sequences are usually individual sentences, but don't have to be.
     *
     * @param sequence
     *        the sequence to embed
     * @return the token embeddings for the sequence
     * @since 1.0.3
     */
    public float[][] embedTokens(final String sequence) {
        try(Inputs inputs = getInputs(sequence)) {
            final List<Tensor<?>> output = bundle.session().runner()
                .feed(model.inputIds, inputs.inputIds)
                .feed(model.inputMask, inputs.inputMask)
                .feed(model.segmentIds, inputs.segmentIds)
                .fetch(model.sequenceOutput)
                .run();

            try(Tensor<?> embedding = output.get(0)) {
                final float[][][] converted = new float[1][(int)embedding.shape()[1]][(int)embedding.shape()[2]];
                embedding.copyTo(converted);
                return converted[0];
            }
        }
    }

    /**
     * Gets BERT embeddings for each of the tokens in multiple sequences. Sequences are usually individual sentences, but don't have to be.
     * The sequences will be processed in parallel as a single batch input to the TensorFlow model.
     *
     * @param sequences
     *        the sequences to embed
     * @return the token embeddings for the sequences, in the order they were provided
     * @since 1.0.3
     */
    public float[][][] embedTokens(final String... sequences) {
        try(Inputs inputs = getInputs(sequences)) {
            final List<Tensor<?>> output = bundle.session().runner()
                .feed(model.inputIds, inputs.inputIds)
                .feed(model.inputMask, inputs.inputMask)
                .feed(model.segmentIds, inputs.segmentIds)
                .fetch(model.sequenceOutput)
                .run();

            try(Tensor<?> embedding = output.get(0)) {
                final float[][][] converted = new float[sequences.length][(int)embedding.shape()[1]][(int)embedding.shape()[2]];
                embedding.copyTo(converted);
                return converted;
            }
        }
    }

    private Inputs getInputs(final String sequence) {
        final String[] tokens = tokenizer.tokenize(sequence);

        final IntBuffer inputIds = IntBuffer.allocate(model.maxSequenceLength);
        final IntBuffer inputMask = IntBuffer.allocate(model.maxSequenceLength);
        final IntBuffer segmentIds = IntBuffer.allocate(model.maxSequenceLength);

        /*
         * In BERT:
         * inputIds are the indexes in the vocabulary for each token in the sequence
         * inputMask is a binary mask that shows which inputIds have valid data in them
         * segmentIds are meant to distinguish paired sequences during training tasks. Here they're always 0 since we're only doing inference.
         */
        final int[] ids = tokenizer.convert(tokens);
        inputIds.put(startTokenId);
        inputMask.put(1);
        segmentIds.put(0);
        for(int i = 0; i < ids.length && i < model.maxSequenceLength - 2; i++) {
            inputIds.put(ids[i]);
            inputMask.put(1);
            segmentIds.put(0);
        }
        inputIds.put(separatorTokenId);
        inputMask.put(1);
        segmentIds.put(0);

        while(inputIds.position() < model.maxSequenceLength) {
            inputIds.put(0);
            inputMask.put(0);
            segmentIds.put(0);
        }

        inputIds.rewind();
        inputMask.rewind();
        segmentIds.rewind();

        return new Inputs(inputIds, inputMask, segmentIds, 1);
    }

    private Inputs getInputs(final String[] sequences) {
        final String[][] tokens = tokenizer.tokenize(sequences);

        final IntBuffer inputIds = IntBuffer.allocate(sequences.length * model.maxSequenceLength);
        final IntBuffer inputMask = IntBuffer.allocate(sequences.length * model.maxSequenceLength);
        final IntBuffer segmentIds = IntBuffer.allocate(sequences.length * model.maxSequenceLength);

        /*
         * In BERT:
         * inputIds are the indexes in the vocabulary for each token in the sequence
         * inputMask is a binary mask that shows which inputIds have valid data in them
         * segmentIds are meant to distinguish paired sequences during training tasks. Here they're always 0 since we're only doing inference.
         */
        int instance = 1;
        for(final String[] token : tokens) {
            final int[] ids = tokenizer.convert(token);
            inputIds.put(startTokenId);
            inputMask.put(1);
            segmentIds.put(0);
            for(int i = 0; i < ids.length && i < model.maxSequenceLength - 2; i++) {
                inputIds.put(ids[i]);
                inputMask.put(1);
                segmentIds.put(0);
            }
            inputIds.put(separatorTokenId);
            inputMask.put(1);
            segmentIds.put(0);

            while(inputIds.position() < model.maxSequenceLength * instance) {
                inputIds.put(0);
                inputMask.put(0);
                segmentIds.put(0);
            }
            instance++;
        }

        inputIds.rewind();
        inputMask.rewind();
        segmentIds.rewind();

        return new Inputs(inputIds, inputMask, segmentIds, sequences.length);
    }
}
