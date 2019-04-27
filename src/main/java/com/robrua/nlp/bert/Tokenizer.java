package com.robrua.nlp.bert;

import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.stream.Stream;

import com.google.common.collect.Lists;

public abstract class Tokenizer {
    protected static Stream<String> whitespaceTokenize(final String sequence) {
        return Arrays.stream(sequence.trim().split("\\s+"));
    }

    public String[][] tokenize(final Iterable<String> sequences) {
        final List<String> list = Lists.newArrayList(sequences);
        return tokenize(list.toArray(new String[list.size()]));
    }

    public String[][] tokenize(final Iterator<String> sequences) {
        final List<String> list = Lists.newArrayList(sequences);
        return tokenize(list.toArray(new String[list.size()]));
    }

    public abstract String[] tokenize(String sequence);

    public abstract String[][] tokenize(String... sequences);
}
