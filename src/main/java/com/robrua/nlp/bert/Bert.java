package com.robrua.nlp.bert;

import java.io.File;
import java.io.IOException;
import java.nio.IntBuffer;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Iterator;
import java.util.List;

import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.collect.Lists;

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

    private static final String MODEL_DETAILS = "model.json";
    private static final String SEPARATOR_TOKEN = "[SEP]";
    private static final String START_TOKEN = "[CLS]";
    private static final String VOCAB_FILE = "vocab.txt";

    public static Bert load(final File path) {
        return load(path.getAbsolutePath());
    }

    public static Bert load(final Path path) {
        return load(path.toAbsolutePath().toString());
    }

    public static Bert load(final String path) {
        ModelDetails model;
        try {
            model = new ObjectMapper().readValue(Paths.get(path, "assets", MODEL_DETAILS).toFile(), ModelDetails.class);
        } catch(final IOException e) {
            throw new RuntimeException(e);
        }

        return new Bert(SavedModelBundle.load(path, "serve"), model, Paths.get(path, "assets", VOCAB_FILE));
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

    public float[][] embedSequences(final Iterable<String> sequences) {
        final List<String> list = Lists.newArrayList(sequences);
        return embedSequences(list.toArray(new String[list.size()]));
    }

    public float[][] embedSequences(final Iterator<String> sequences) {
        final List<String> list = Lists.newArrayList(sequences);
        return embedSequences(list.toArray(new String[list.size()]));
    }

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

    public float[][][] embedTokens(final Iterable<String> sequences) {
        final List<String> list = Lists.newArrayList(sequences);
        return embedTokens(list.toArray(new String[list.size()]));
    }

    public float[][][] embedTokens(final Iterator<String> sequences) {
        final List<String> list = Lists.newArrayList(sequences);
        return embedTokens(list.toArray(new String[list.size()]));
    }

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
