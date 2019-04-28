package com.robrua.nlp.bert;

import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.stream.Stream;

import com.google.common.collect.Lists;

/**
 * A tokenizer that converts text sequences into tokens or sub-tokens for BERT to use
 *
 * @author Rob Rua (https://github.com/robrua)
 * @version 1.0.3
 * @since 1.0.3
 */
public abstract class Tokenizer {
    /**
     * Splits a sequence into tokens based on whitespace
     *
     * @param sequence
     *        the sequence to split
     * @return a stream of the tokens from the stream that were separated by whitespace
     * @since 1.0.3
     */
    protected static Stream<String> whitespaceTokenize(final String sequence) {
        return Arrays.stream(sequence.trim().split("\\s+"));
    }

    /**
     * Tokenizes a multiple sequences
     *
     * @param sequences
     *        the sequences to tokenize
     * @return the tokens in the sequences, in the order the {@link java.lang.Iterable} provided them
     * @since 1.0.3
     */
    public String[][] tokenize(final Iterable<String> sequences) {
        final List<String> list = Lists.newArrayList(sequences);
        return tokenize(list.toArray(new String[list.size()]));
    }

    /**
     * Tokenizes a multiple sequences
     *
     * @param sequences
     *        the sequences to tokenize
     * @return the tokens in the sequences, in the order the {@link java.util.Iterator} provided them
     * @since 1.0.3
     */
    public String[][] tokenize(final Iterator<String> sequences) {
        final List<String> list = Lists.newArrayList(sequences);
        return tokenize(list.toArray(new String[list.size()]));
    }

    /**
     * Tokenizes a single sequence
     *
     * @param sequence
     *        the sequence to tokenize
     * @return the tokens in the sequence
     * @since 1.0.3
     */
    public abstract String[] tokenize(String sequence);

    /**
     * Tokenizes a multiple sequences
     *
     * @param sequences
     *        the sequences to tokenize
     * @return the tokens in the sequences, in the order they were provided
     * @since 1.0.3
     */
    public abstract String[][] tokenize(String... sequences);
}
