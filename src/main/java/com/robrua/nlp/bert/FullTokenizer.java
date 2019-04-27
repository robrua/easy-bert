package com.robrua.nlp.bert;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Stream;

import com.google.common.io.Resources;

public class FullTokenizer extends Tokenizer {
    private static final boolean DEFAULT_DO_LOWER_CASE = false;

    private static Map<String, Integer> loadVocabulary(final Path file) {
        final Map<String, Integer> vocabulary = new HashMap<>();
        try(BufferedReader reader = Files.newBufferedReader(file, Charset.forName("UTF-8"))) {
            int index = 0;
            String line;
            while((line = reader.readLine()) != null) {
                vocabulary.put(line.trim(), index++);
            }
        } catch(final IOException e) {
            throw new RuntimeException(e);
        }
        return vocabulary;
    }

    private static Path toPath(final String resource) {
        try {
            return Paths.get(Resources.getResource(resource).toURI());
        } catch(final URISyntaxException e) {
            throw new RuntimeException(e);
        }
    }

    private final BasicTokenizer basic;
    private final Map<String, Integer> vocabulary;
    private final WordpieceTokenizer wordpiece;

    public FullTokenizer(final File vocabulary) {
        this(Paths.get(vocabulary.toURI()), DEFAULT_DO_LOWER_CASE);
    }

    public FullTokenizer(final File vocabulary, final boolean doLowerCase) {
        this(Paths.get(vocabulary.toURI()), doLowerCase);
    }

    public FullTokenizer(final Path vocabulary) {
        this(vocabulary, DEFAULT_DO_LOWER_CASE);
    }

    public FullTokenizer(final Path vocabulary, final boolean doLowerCase) {
        this.vocabulary = loadVocabulary(vocabulary);
        basic = new BasicTokenizer(doLowerCase);
        wordpiece = new WordpieceTokenizer(this.vocabulary);
    }

    public FullTokenizer(final String vocabulary) {
        this(toPath(vocabulary), DEFAULT_DO_LOWER_CASE);
    }

    public FullTokenizer(final String vocabulary, final boolean doLowerCase) {
        this(toPath(vocabulary), doLowerCase);
    }

    public int[] convert(final String[] tokens) {
        return Arrays.stream(tokens).mapToInt(vocabulary::get).toArray();
    }

    @Override
    public String[] tokenize(final String sequence) {
        return Arrays.stream(wordpiece.tokenize(basic.tokenize(sequence)))
            .flatMap(Stream::of)
            .toArray(String[]::new);
    }

    @Override
    public String[][] tokenize(final String... sequences) {
        return Arrays.stream(basic.tokenize(sequences))
            .map((final String[] tokens) -> Arrays.stream(wordpiece.tokenize(tokens))
                .flatMap(Stream::of)
                .toArray(String[]::new))
            .toArray(String[][]::new);
    }
}
